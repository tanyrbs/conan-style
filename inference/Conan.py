import os
import json
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch

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

CONDITION_FIELDS = ("emotion", "accent")

class StreamingVoiceConversion:
    """
    Streaming-style inference front-end.

    Important:
    - reference features are cached once
    - Emformer content extraction is stateful
    - the acoustic model / vocoder path still uses prefix recomputation

    So the current implementation is suitable for streaming-oriented evaluation,
    but it is not yet a fully stateful end-to-end incremental decoder/vocoder stack.
    """
    tokens_per_chunk: int = 4  # 4 HuBERT tokens ≈ 80 ms
    
    def __init__(self, hp: Dict):
        self.hparams = hp
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.streaming_impl = "emformer_stateful_prefix_recompute"
        self.last_infer_metadata = {}
        self._validate_runtime_layout()
        self.condition_maps = self._load_condition_maps()
        self.model = self._build_model()
        self.vocoder = self._build_vocoder()
        self.emformer = self._build_emformer()
        self._vocoder_warm_zero()

    def _validate_runtime_layout(self):
        required_paths = {
            "work_dir": self.hparams.get("work_dir"),
            "emformer_ckpt": self.hparams.get("emformer_ckpt"),
            "vocoder_ckpt": self.hparams.get("vocoder_ckpt"),
        }
        missing = [
            f"{name}={path}"
            for name, path in required_paths.items()
            if not path or not os.path.exists(str(path))
        ]
        if missing:
            raise FileNotFoundError(
                "Conan inference layout is incomplete. Missing required path(s): "
                + ", ".join(missing)
            )
        optional_dirs = [
            self.hparams.get("binary_data_dir"),
            self.hparams.get("processed_data_dir"),
        ]
        if not any(path and os.path.exists(str(path)) for path in optional_dirs):
            warnings.warn(
                "Neither binary_data_dir nor processed_data_dir exists. "
                "Condition-label vocab lookup will be unavailable for inference.",
                stacklevel=2,
            )

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
         
    def _wav_to_mel(self, path: str) -> np.ndarray:
        mel = librosa_wav2spec(
            path,
            fft_size=self.hparams["fft_size"],
            hop_size=self.hparams["hop_size"],
            win_length=self.hparams["win_size"],
            num_mels=self.hparams["audio_num_mel_bins"],
            fmin=self.hparams["fmin"],
            fmax=self.hparams["fmax"],
            sample_rate=self.hparams["audio_sample_rate"],
            loud_norm=self.hparams["loud_norm"],
        )["mel"]
        return np.clip(mel, self.hparams["mel_vmin"], self.hparams["mel_vmax"])

    @staticmethod
    def _has_distinct_split_reference_inputs(inp: Dict):
        ref_wav = inp["ref_wav"]
        split_reference_keys = (
            "ref_timbre_wav",
            "ref_style_wav",
            "ref_dynamic_timbre_wav",
            "ref_emotion_wav",
            "ref_accent_wav",
        )
        return any(inp.get(key) not in (None, "", ref_wav) for key in split_reference_keys)

    def _load_reference_mels(self, inp: Dict):
        ref_wav = inp["ref_wav"]
        split_reference_inputs = bool(inp.get("allow_split_reference_inputs", False))
        has_distinct_split_refs = self._has_distinct_split_reference_inputs(inp)
        if has_distinct_split_refs and not split_reference_inputs:
            warnings.warn(
                "Distinct split reference inputs were provided but allow_split_reference_inputs is false; "
                "the inference path will collapse them back to ref_wav.",
                stacklevel=2,
            )
        ref_mel = torch.from_numpy(self._wav_to_mel(ref_wav)).float().unsqueeze(0).to(self.device)
        if split_reference_inputs:
            ref_timbre_wav = inp.get("ref_timbre_wav", ref_wav)
            ref_style_wav = inp.get("ref_style_wav", ref_wav)
            ref_dynamic_timbre_wav = inp.get("ref_dynamic_timbre_wav", ref_style_wav)
            ref_emotion_wav = inp.get("ref_emotion_wav", ref_style_wav)
            ref_accent_wav = inp.get("ref_accent_wav", ref_style_wav)
            bundle = build_reference_bundle_from_inputs(
                ref=ref_mel,
                ref_timbre=torch.from_numpy(
                    self._wav_to_mel(ref_timbre_wav)
                ).float().unsqueeze(0).to(self.device),
                ref_style=torch.from_numpy(
                    self._wav_to_mel(ref_style_wav)
                ).float().unsqueeze(0).to(self.device),
                ref_dynamic_timbre=torch.from_numpy(
                    self._wav_to_mel(ref_dynamic_timbre_wav)
                ).float().unsqueeze(0).to(self.device),
                ref_emotion=torch.from_numpy(
                    self._wav_to_mel(ref_emotion_wav)
                ).float().unsqueeze(0).to(self.device),
                ref_accent=torch.from_numpy(
                    self._wav_to_mel(ref_accent_wav)
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
        else:
            bundle = build_reference_bundle_from_inputs(
                ref=ref_mel,
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
        return bundle, {
            "split_reference_inputs": bool(split_reference_inputs),
            "has_distinct_split_refs": bool(has_distinct_split_refs),
            "collapsed_split_refs": bool(has_distinct_split_refs and not split_reference_inputs),
        }

    def _build_model_control_kwargs(self, inp: Dict):
        style_profile = resolve_style_profile(
            inp,
            preset=self.hparams.get("style_profile", "strong_style"),
        )
        style_strength = style_profile["style_strength"]
        resolved_controls = {
            "emotion_id": self._resolve_condition_id(
                "emotion", inp.get("emotion_id", inp.get("emotion", None))
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
            "dynamic_timbre_strength": style_profile.get("dynamic_timbre_strength", 1.0),
        }
        return build_control_kwargs(resolved_controls, style_strength_default=style_strength)

    def _build_style_runtime_kwargs(self, inp: Dict):
        style_profile = style_profile_to_runtime_kwargs(
            inp,
            preset=self.hparams.get("style_profile", "strong_style"),
        )
        return {
            "decoder_style_condition_mode": style_profile.get(
                "decoder_style_condition_mode",
                self.hparams.get("decoder_style_condition_mode", "mainline_full"),
            ),
            "global_timbre_to_pitch": bool(
                style_profile.get(
                    "global_timbre_to_pitch",
                    self.hparams.get("global_timbre_to_pitch", False),
                )
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
            "fast_style_strength_scale": float(style_profile.get("fast_style_strength_scale", 1.0)),
            "slow_style_strength_scale": float(style_profile.get("slow_style_strength_scale", 1.0)),
            "style_temperature": float(style_profile.get("style_temperature", 1.0)),
            "global_style_trace_blend": float(style_profile.get("global_style_trace_blend", 0.0)),
            "dynamic_timbre_memory_mode": style_profile.get(
                "dynamic_timbre_memory_mode",
                self.hparams.get("dynamic_timbre_reference_memory_mode", "fast"),
            ),
            "dynamic_timbre_temperature": float(style_profile.get("dynamic_timbre_temperature", 1.0)),
            "dynamic_timbre_style_condition_scale": float(
                style_profile.get("dynamic_timbre_style_condition_scale", 0.5)
            ),
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

    def _normalize_spk_embed(self, spk_emb):
        if spk_emb is None:
            return None
        if not isinstance(spk_emb, torch.Tensor):
            spk_embed = torch.as_tensor(spk_emb, dtype=torch.float32, device=self.device)
        else:
            spk_embed = spk_emb.to(self.device, dtype=torch.float32)
        if spk_embed.dim() == 1:
            spk_embed = spk_embed.unsqueeze(0)
        return spk_embed

    def infer_once(self, inp: Dict, *, spk_emb: Optional[torch.Tensor] = None):
        # 1. Load reference mel
        ref_mels, reference_meta = self._load_reference_mels(inp)
        control_kwargs = self._build_model_control_kwargs(inp)
        runtime_kwargs = self._build_style_runtime_kwargs(inp)
        resolved_profile = resolve_style_profile(
            inp,
            preset=self.hparams.get("style_profile", "strong_style"),
        )
        spk_embed = self._normalize_spk_embed(spk_emb)
        self.last_infer_metadata = {
            "streaming_impl": self.streaming_impl,
            "reference_contract_mode": ref_mels.get(
                "reference_contract_mode",
                self.hparams.get("reference_contract_mode", "collapsed_reference"),
            ),
            "split_reference_inputs": reference_meta["split_reference_inputs"],
            "has_distinct_split_refs": reference_meta["has_distinct_split_refs"],
            "collapsed_split_refs": reference_meta["collapsed_split_refs"],
            "style_profile": resolved_profile.get("style_profile", "strong_style"),
            "style_strength": float(resolved_profile.get("style_strength", 1.0)),
            "dynamic_timbre_strength": float(resolved_profile.get("dynamic_timbre_strength", 1.0)),
            "spk_embed_override": bool(spk_embed is not None),
        }
        with torch.no_grad():
            reference_cache = self.model.prepare_reference_cache(
                reference_bundle=ref_mels,
                spk_embed=spk_embed,
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
        vocoder_left_context = max(0, int(self.hparams.get("streaming_vocoder_left_context_frames", 48)))
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
                    spk_embed=spk_embed,
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
            # bounded-left-context vocoder re-synthesis: still not fully stateful,
            # but avoids re-running the vocoder on the entire mel prefix every step.
            mel_prefix = torch.cat(mel_chunks, dim=0)
            mel_total_frames = int(mel_prefix.size(0))
            mel_new_frames = int(mel_new.size(0))
            vocoder_context_frames = min(vocoder_left_context, max(0, mel_total_frames - mel_new_frames))
            vocoder_start = max(0, mel_total_frames - mel_new_frames - vocoder_context_frames)
            mel_window = mel_prefix[vocoder_start:]
            wav_chunk_vocoder = self.vocoder.spec2wav(mel_window.cpu().numpy())
            hop = self.hparams["hop_size"]
            start_sample = vocoder_context_frames * hop
            end_sample = min(len(wav_chunk_vocoder), start_sample + mel_new_frames * hop)
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
            print(f"Saved output: {save_path} | metadata={self.last_infer_metadata}")


if __name__ == "__main__":
    set_hparams()
    # Example usage: update with your own wav paths
    demo = [
        {
            "ref_wav": "path/to/reference_audio.wav",
            "src_wav": "path/to/source_audio.wav",
            "style_profile": "strong_style",
        },
    ]
    print("Available style profiles:", ", ".join(available_style_profiles()))

    engine = StreamingVoiceConversion(hparams)
    engine.test_multiple_sentences(demo)
