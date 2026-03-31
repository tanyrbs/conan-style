import os
import torch
import numpy as np
from typing import Dict, List
import json
import time
from datetime import datetime

from utils.commons.hparams import hparams, set_hparams
from utils.commons.ckpt_utils import load_ckpt, load_ckpt_emformer
from utils.audio import librosa_wav2spec
from utils.audio.io import save_wav
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls

from modules.Conan.Conan import Conan
from modules.Conan.reference_bundle import (
    build_control_kwargs,
    build_reference_bundle_from_inputs,
    reference_bundle_to_model_kwargs,
)
from modules.Conan.reference_cache import reference_cache_to_model_kwargs
from modules.Conan.style_profiles import (
    available_style_profiles,
    resolve_style_profile,
    style_profile_to_runtime_kwargs,
)
# Emformer feature extractor
from modules.Emformer.emformer import EmformerDistillModel
__all__ = ["StreamingVoiceConversion"]

CONDITION_FIELDS = ("emotion", "style", "accent")

class StreamingVoiceConversion:
    """
    Streaming style-transfer inference using Emformer for feature extraction.
    """
    tokens_per_chunk: int = 4  # 4 HuBERT tokens ≈ 80 ms
    
    def __init__(self, hp: Dict):
        self.hparams = hp
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.condition_maps = self._load_condition_maps()
        self.model = self._build_model()
        self.vocoder = self._build_vocoder()
        self.emformer = self._build_emformer()
        self._vocoder_warm_zero()

    def _build_model(self):
        m = Conan(0, self.hparams)
        m.eval()
        load_ckpt(m, self.hparams["work_dir"], strict=False)
        return m.to(self.device)

    def _build_vocoder(self):

        vocoder_cls = get_vocoder_cls(self.hparams["vocoder"])
        if vocoder_cls is None:
            raise ValueError(f"Vocoder '{self.hparams['vocoder']}' is not registered. Check vocoder name and registration.")
        return vocoder_cls()

    def _build_emformer(self):
        emformer = EmformerDistillModel(self.hparams, output_dim=100)
        # load checkpoint
        load_ckpt(emformer, self.hparams["emformer_ckpt"], strict=False)
        emformer.eval()
        return emformer.to(self.device)

    def _vocoder_warm_zero(self):
        _ = self.vocoder.spec2wav(np.zeros((4, 80), dtype=np.float32))

    def _load_condition_maps(self):
        condition_maps = {}
        candidate_dirs = [
            self.hparams.get("binary_data_dir"),
            self.hparams.get("processed_data_dir"),
        ]
        for field in CONDITION_FIELDS:
            vocab = None
            for data_dir in candidate_dirs:
                if not data_dir:
                    continue
                vocab_path = os.path.join(data_dir, f"{field}_set.json")
                if os.path.exists(vocab_path):
                    with open(vocab_path, "r", encoding="utf-8") as f:
                        vocab = json.load(f)
                    break
            if vocab is None:
                vocab = []
            condition_maps[field] = {str(label): idx for idx, label in enumerate(vocab)}
        return condition_maps

    def _resolve_condition_id(self, field: str, value):
        if value is None:
            return None
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, float) and float(value).is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return None
            try:
                return int(stripped)
            except ValueError:
                return self.condition_maps.get(field, {}).get(stripped, None)
        return None
         
    @staticmethod
    def _wav_to_mel(path: str) -> np.ndarray:
        mel = librosa_wav2spec(
            path,
            fft_size=hparams["fft_size"],
            hop_size=hparams["hop_size"],
            win_length=hparams["win_size"],
            num_mels=hparams["audio_num_mel_bins"],
            fmin=hparams["fmin"],
            fmax=hparams["fmax"],
            sample_rate=hparams["audio_sample_rate"],
            loud_norm=hparams["loud_norm"],
        )["mel"]
        return np.clip(mel, hparams["mel_vmin"], hparams["mel_vmax"])

    def _load_reference_mels(self, inp: Dict):
        ref_wav = inp["ref_wav"]
        ref_style_wav = inp.get("ref_style_wav", ref_wav)
        return build_reference_bundle_from_inputs(
            ref=torch.from_numpy(self._wav_to_mel(ref_wav)).float().unsqueeze(0).to(self.device),
            ref_timbre=torch.from_numpy(
                self._wav_to_mel(inp.get("ref_timbre_wav", ref_wav))
            ).float().unsqueeze(0).to(self.device),
            ref_style=torch.from_numpy(self._wav_to_mel(ref_style_wav)).float().unsqueeze(0).to(self.device),
            ref_dynamic_timbre=torch.from_numpy(
                self._wav_to_mel(inp.get("ref_dynamic_timbre_wav", ref_style_wav))
            ).float().unsqueeze(0).to(self.device),
            ref_emotion=torch.from_numpy(
                self._wav_to_mel(inp.get("ref_emotion_wav", ref_style_wav))
            ).float().unsqueeze(0).to(self.device),
            ref_accent=torch.from_numpy(
                self._wav_to_mel(inp.get("ref_accent_wav", ref_style_wav))
            ).float().unsqueeze(0).to(self.device),
            prompt_fallback_to_style=bool(
                inp.get(
                    "prompt_ref_fallback_to_style",
                    self.hparams.get("prompt_ref_fallback_to_style", True),
                )
            ),
            reference_contract_mode=inp.get(
                "reference_contract_mode",
                self.hparams.get("reference_contract_mode", "collapsed_reference"),
            ),
        )

    def _build_model_control_kwargs(self, inp: Dict):
        style_profile = resolve_style_profile(
            inp,
            preset=self.hparams.get("style_profile", "balanced"),
        )
        style_strength = style_profile["style_strength"]
        resolved_controls = {
            "emotion_id": self._resolve_condition_id(
                "emotion", inp.get("emotion_id", inp.get("emotion", None))
            ),
            "style_id": self._resolve_condition_id(
                "style", inp.get("style_id", inp.get("style", None))
            ),
            "accent_id": self._resolve_condition_id(
                "accent", inp.get("accent_id", inp.get("accent", None))
            ),
            "arousal": inp.get("arousal", None),
            "valence": inp.get("valence", None),
            "energy": inp.get("energy", None),
            "style_strength": style_strength,
            "emotion_strength": inp.get("emotion_strength", 1.0),
            "accent_strength": inp.get("accent_strength", 1.0),
            "dynamic_timbre_strength": style_profile["dynamic_timbre_strength"],
        }
        return build_control_kwargs(resolved_controls, style_strength_default=style_strength)

    def _build_style_runtime_kwargs(self, inp: Dict):
        style_profile = style_profile_to_runtime_kwargs(
            inp,
            preset=self.hparams.get("style_profile", "balanced"),
        )
        return {
            "decoder_style_condition_mode": style_profile.get(
                "decoder_style_condition_mode",
                self.hparams.get("decoder_style_condition_mode", "legacy_full"),
            ),
            "global_style_anchor_strength": float(style_profile.get("global_style_anchor_strength", 1.0)),
            "style_trace_mode": style_profile.get(
                "style_trace_mode",
                self.hparams.get("style_trace_mode", "fast"),
            ),
            "style_memory_mode": style_profile.get(
                "style_memory_mode",
                self.hparams.get("style_reference_memory_mode", "fast"),
            ),
            "style_temperature": float(style_profile.get("style_temperature", 1.0)),
            "global_style_trace_blend": float(style_profile.get("global_style_trace_blend", 0.0)),
            "dynamic_timbre_memory_mode": style_profile.get(
                "dynamic_timbre_memory_mode",
                self.hparams.get("dynamic_timbre_reference_memory_mode", "fast"),
            ),
            "dynamic_timbre_temperature": float(style_profile.get("dynamic_timbre_temperature", 1.0)),
            "dynamic_timbre_gate_scale": float(style_profile.get("dynamic_timbre_gate_scale", 1.0)),
            "dynamic_timbre_gate_bias": float(style_profile.get("dynamic_timbre_gate_bias", 0.0)),
            "dynamic_timbre_boundary_suppress_strength": float(
                style_profile.get("dynamic_timbre_boundary_suppress_strength", 0.0)
            ),
            "dynamic_timbre_boundary_radius": int(style_profile.get("dynamic_timbre_boundary_radius", 2)),
            "dynamic_timbre_anchor_preserve_strength": float(
                style_profile.get("dynamic_timbre_anchor_preserve_strength", 0.0)
            ),
        }

    def infer_once(self, inp: Dict):
        # 1. Load reference mel
        ref_mels = self._load_reference_mels(inp)
        control_kwargs = self._build_model_control_kwargs(inp)
        runtime_kwargs = self._build_style_runtime_kwargs(inp)
        with torch.no_grad():
            reference_cache = self.model.prepare_reference_cache(
                reference_bundle=ref_mels,
                spk_embed=None,
                infer=True,
                global_steps=200000,
            )

        # 2. Load src mel
        src_mel_np = self._wav_to_mel(inp["src_wav"])
        src_mel = torch.from_numpy(src_mel_np).unsqueeze(0).to(self.device)  # [1, T, 80]
        total_frames = src_mel.shape[1]
        # 3. Streaming Emformer + main model with proper state management
        chunk_size = self.hparams["chunk_size"] // 20  # frames per chunk (20ms per frame)
        right_context = self.hparams["right_context"]  # frames
        seg = chunk_size
        rc = right_context

        content_code_buffer = []  # list of [emit,] tensors
        mel_chunks = []
        wav_chunks = []
        prev_len = 0
        pos = 0
        state = None  # Emformer state for streaming
        

        while pos < total_frames:
            # 1) How many NEW frames do we want to emit this step?
            emit = min(seg, total_frames - pos)

            # 2) How much genuine look-ahead is still available?
            look = min(rc, total_frames - (pos + emit))
            
            # 3) Build the real chunk (emit + look) … then pad
            real_len = emit + look
            chunk = src_mel[:, pos:pos + real_len, :]  # (1, real_len, 80)

            # Pad so that len(chunk) == seg + rc, as Emformer expects
            need_pad = (seg + rc) - real_len
            if need_pad > 0:
                pad = chunk[:, -1:, :].expand(1, need_pad, src_mel.shape[2])  # repeat last frame
                chunk = torch.cat([chunk, pad], dim=1)  # (1, seg+rc, 80)

            # 4) Run one streaming step (length **includes** the right context)
            lengths = torch.full((1,), chunk.size(1), dtype=torch.long, device=self.device)
            with torch.no_grad():
                chunk_out, _, state = self.emformer.emformer.infer(chunk, lengths, state)
                # Apply projection if needed
                if self.emformer.mode == 'both':
                    chunk_out = self.emformer.proj1(chunk_out)
                else:
                    chunk_out = self.emformer.proj(chunk_out)
                
                # Convert to tokens if needed
                if chunk_out.dim() == 3 and chunk_out.shape[-1] > 1:
                    chunk_out = torch.argmax(chunk_out, dim=-1)  # [1, seg+rc]
                
            # Emformer drops the right context in its output; keep only `emit` frames
            new_codes = chunk_out[:, :emit]  # [1, emit]
            content_code_buffer.append(new_codes.squeeze(0))  # [emit]
            all_codes = torch.cat(content_code_buffer, dim=0).unsqueeze(0)  # [1, T_so_far]
            
            with torch.no_grad():
                out = self.model(
                    content=all_codes,
                    spk_embed=None,
                    target=None,
                    ref=ref_mels["ref"],
                    f0=None,
                    uv=None,
                    infer=True,
                    global_steps=200000,
                    **reference_bundle_to_model_kwargs(ref_mels),
                    **reference_cache_to_model_kwargs(reference_cache),
                    **control_kwargs,
                    **runtime_kwargs,
                )
                mel_out = out["mel_out"][0]
            mel_new = mel_out[prev_len:]
            mel_chunks.append(mel_new)
            prev_len = mel_out.shape[0]
            pos += emit
            # collect mel from start to current pos
            mel_chunks_forvocoder = torch.cat(mel_chunks, dim=0)
            wav_chunk_vocoder = self.vocoder.spec2wav(mel_chunks_forvocoder.cpu().numpy())
            # only keep the wav generated for the current chunk
            hop = self.hparams["hop_size"]
            start_sample = max(0, (pos - emit) * hop)
            end_sample = min(len(wav_chunk_vocoder), pos * hop)
            if end_sample > start_sample:
                wav_chunk = wav_chunk_vocoder[start_sample:end_sample]
                wav_chunks.append(wav_chunk)
        mel_pred = torch.cat(mel_chunks, dim=0)
        if len(wav_chunks) > 0:
            wav_pred = np.concatenate(wav_chunks, axis=0)
        else:
            wav_pred = self.vocoder.spec2wav(mel_pred.cpu().numpy())

        # wav_pred = self.vocoder.spec2wav(mel_pred.cpu().numpy())
        
        
        return wav_pred, mel_pred.cpu().numpy()

    def test_multiple_sentences(self, test_cases: List[Dict]):
        os.makedirs("infer_out_demo", exist_ok=True)
        for i, inp in enumerate(test_cases):
            wav, mel = self.infer_once(inp)
            ref_name = os.path.splitext(os.path.basename(inp["ref_wav"]))[0]
            src_name = os.path.splitext(os.path.basename(inp["src_wav"]))[0]
            save_path = f"infer_out_demo/{ref_name}_{src_name}.wav"
            save_wav(wav, save_path, self.hparams["audio_sample_rate"])
            print(f"Saved output: {save_path}")


if __name__ == "__main__":
    set_hparams()
    # Example usage: update with your own wav paths
    demo = [
        {
            "ref_wav": "path/to/reference_audio.wav",
            "src_wav": "path/to/source_audio.wav",
            "ref_timbre_wav": "path/to/reference_timbre.wav",
            "ref_style_wav": "path/to/reference_style.wav",
            "ref_dynamic_timbre_wav": "path/to/reference_dynamic_timbre.wav",
            "ref_emotion_wav": "path/to/reference_emotion.wav",
            "emotion": "angry",
            "style": "dramatic",
            "accent": "british",
            "style_profile": "strong_style_timbre",
            "emotion_strength": 1.0,
            "accent_strength": 0.6,
            "arousal": 0.9,
            "valence": -0.3,
        },
    ]
    print("Available style profiles:", ", ".join(available_style_profiles()))

    engine = StreamingVoiceConversion(hparams)
    engine.test_multiple_sentences(demo)
