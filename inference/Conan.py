import os
import json
import glob
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from inference.streaming_runtime import (
    PrefixCodeBuffer,
    RollingMelContextBuffer,
    build_streaming_latency_report,
    cumulative_prefix_recompute_multiplier,
    resolve_streaming_layout,
    resolve_vocoder_left_context_frames,
)
from utils.commons.hparams import hparams, set_hparams
from utils.commons.ckpt_utils import load_ckpt
from utils.audio import librosa_wav2spec
from utils.audio.io import save_wav
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls

from modules.Conan.Conan import Conan
from modules.Conan.reference_bundle import (
    build_control_kwargs,
    build_reference_bundle_from_inputs,
    build_style_runtime_kwargs as build_reference_style_runtime_kwargs,
    reference_bundle_to_model_kwargs,
)
from modules.Conan.reference_cache import reference_cache_to_model_kwargs
from modules.Conan.style_profiles import (
    STYLE_PROFILE_KEYS,
    available_mainline_style_profiles,
    resolve_style_profile,
)
from inference.conan_request import (
    ADVANCED_STYLE_RUNTIME_KEYS,
    CONDITION_FIELDS,
    has_distinct_split_reference_inputs,
)
from utils.commons.condition_labels import load_condition_id_maps, resolve_condition_label_id
# Emformer feature extractor
from modules.Emformer.emformer import EmformerDistillModel
__all__ = ["StreamingVoiceConversion"]

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
    tokens_per_chunk: int = 4  # 4 HuBERT tokens ? 80 ms

    @staticmethod
    def _resolve_checkpoint_artifacts(path_value) -> List[str]:
        if path_value is None:
            return []
        path = str(path_value)
        if os.path.isfile(path):
            basename = os.path.basename(path)
            if path.endswith(".ckpt") or basename == "generator_v1":
                return [path]
            return []
        if not os.path.isdir(path):
            return []
        artifacts = []
        for pattern in (
            "model_ckpt_steps_*.ckpt",
            "*.ckpt",
            "generator_v1",
        ):
            artifacts.extend(glob.glob(os.path.join(path, pattern)))
        return sorted({artifact for artifact in artifacts if os.path.isfile(artifact)})
    
    def __init__(self, hp: Dict):
        if hp is None:
            raise ValueError("StreamingVoiceConversion requires a resolved hparams dictionary.")
        resolved_hp = dict(hp)
        vocoder_left_context_frames, vocoder_left_context_source = resolve_vocoder_left_context_frames(
            resolved_hp
        )
        resolved_hp["vocoder_left_context_frames"] = int(vocoder_left_context_frames)
        # Keep the global hparams bridge for legacy modules (notably the vocoder),
        # but make the front-end itself read from an instance-local resolved copy.
        hparams.clear()
        hparams.update(resolved_hp)
        self.hparams = resolved_hp
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.streaming_impl = "emformer_stateful_prefix_recompute"
        self.vocoder_left_context_frames = int(vocoder_left_context_frames)
        self.vocoder_left_context_source = str(vocoder_left_context_source)
        self.streaming_layout = resolve_streaming_layout(resolved_hp)
        self.tokens_per_chunk = int(self.streaming_layout["chunk_frames"])
        self.streaming_capabilities = {
            "frontend_stateful_streaming": True,
            "acoustic_native_streaming": False,
            "acoustic_prefix_recompute": True,
            "vocoder_native_streaming": False,
            "vocoder_stream_api_available": False,
        }
        self.streaming_latency_report = build_streaming_latency_report(resolved_hp)
        self.last_infer_metadata = {}
        print(
            f"| Conan vocoder_left_context_frames={self.vocoder_left_context_frames} "
            f"(source={self.vocoder_left_context_source})"
        )
        self._validate_runtime_layout()
        self.condition_maps = self._load_condition_maps()
        self.model = self._build_model()
        self.vocoder = self._build_vocoder()
        self.streaming_capabilities["vocoder_native_streaming"] = bool(
            self.vocoder.supports_native_streaming()
        )
        self.streaming_capabilities["vocoder_stream_api_available"] = hasattr(
            self.vocoder, "spec2wav_stream"
        )
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
        missing_checkpoint_artifacts = [
            f"{name}={path}"
            for name, path in required_paths.items()
            if not self._resolve_checkpoint_artifacts(path)
        ]
        if missing_checkpoint_artifacts:
            raise FileNotFoundError(
                "Conan inference layout is missing actual checkpoint artifact(s). "
                "Expected a concrete checkpoint file or a directory containing model_ckpt_steps_*.ckpt "
                "(or generator_v1 for legacy NSF vocoders): "
                + ", ".join(missing_checkpoint_artifacts)
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
        return load_condition_id_maps(
            [
                self.hparams.get("binary_data_dir"),
                self.hparams.get("processed_data_dir"),
            ],
            fields=CONDITION_FIELDS,
        )

    @staticmethod
    def _write_json(path: str, payload: Dict):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

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
                resolved = resolve_condition_label_id(
                    self.condition_maps.get(field, {}),
                    stripped,
                    default=-1,
                )
                if resolved >= 0:
                    return int(resolved)
                if self.condition_maps.get(field):
                    known = ", ".join(sorted(self.condition_maps[field].keys())[:12])
                    raise KeyError(
                        f"Unknown {field} label '{stripped}'. Available labels: {known or '<none>'}"
                    )
                return None
        return None

    def _resolve_style_profile(self, inp: Dict):
        return resolve_style_profile(
            inp,
            preset=self.hparams.get("style_profile", "strong_style"),
        )

    @staticmethod
    def _resolved_style_profile_to_runtime_kwargs(resolved_profile: Dict):
        runtime_kwargs = {
            key: resolved_profile.get(key)
            for key in STYLE_PROFILE_KEYS
            if resolved_profile.get(key) is not None
        }
        runtime_kwargs["style_profile"] = resolved_profile["style_profile"]
        return runtime_kwargs
         
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

    def _load_reference_mels(self, inp: Dict):
        ref_wav = inp["ref_wav"]
        mel_cache = {}

        def load_mel_tensor(path: str) -> torch.Tensor:
            cache_key = os.path.abspath(str(path))
            cached = mel_cache.get(cache_key)
            if cached is None:
                cached = torch.from_numpy(self._wav_to_mel(path)).float().unsqueeze(0).to(self.device)
                mel_cache[cache_key] = cached
            return cached

        split_reference_surface_enabled = bool(
            self.hparams.get("allow_split_reference_inputs", False)
        )
        split_reference_inputs = bool(
            inp.get("allow_split_reference_inputs", split_reference_surface_enabled)
        ) and split_reference_surface_enabled
        has_distinct_split_refs = has_distinct_split_reference_inputs(inp)
        if has_distinct_split_refs and not split_reference_inputs:
            warnings.warn(
                "Distinct split reference inputs were provided but the canonical mainline surface keeps "
                "allow_split_reference_inputs disabled; the inference path will collapse them back to ref_wav.",
                stacklevel=2,
            )
        ref_mel = load_mel_tensor(ref_wav)
        if split_reference_inputs:
            ref_timbre_wav = inp.get("ref_timbre_wav", ref_wav)
            ref_style_wav = inp.get("ref_style_wav", ref_wav)
            ref_dynamic_timbre_wav = inp.get("ref_dynamic_timbre_wav", ref_style_wav)
            ref_emotion_wav = inp.get("ref_emotion_wav", ref_style_wav)
            ref_accent_wav = inp.get("ref_accent_wav", ref_style_wav)
            bundle = build_reference_bundle_from_inputs(
                ref=ref_mel,
                ref_timbre=load_mel_tensor(ref_timbre_wav),
                ref_style=load_mel_tensor(ref_style_wav),
                ref_dynamic_timbre=load_mel_tensor(ref_dynamic_timbre_wav),
                ref_emotion=load_mel_tensor(ref_emotion_wav),
                ref_accent=load_mel_tensor(ref_accent_wav),
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
                allow_split_reference_inputs=split_reference_inputs,
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
                allow_split_reference_inputs=split_reference_inputs,
            )
        return bundle, {
            "split_reference_inputs": bool(split_reference_inputs),
            "has_distinct_split_refs": bool(has_distinct_split_refs),
            "collapsed_split_refs": bool(has_distinct_split_refs and not split_reference_inputs),
        }

    def _build_model_control_kwargs(self, inp: Dict, *, resolved_style_profile: Optional[Dict] = None):
        style_profile = resolved_style_profile or self._resolve_style_profile(inp)
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
        }
        explicit_dynamic_timbre_strength = inp.get("dynamic_timbre_strength", None)
        if explicit_dynamic_timbre_strength is not None:
            resolved_controls["dynamic_timbre_strength"] = explicit_dynamic_timbre_strength
        return build_control_kwargs(resolved_controls, style_strength_default=style_strength)

    def _build_style_runtime_kwargs(self, inp: Dict, *, resolved_style_profile: Optional[Dict] = None):
        resolved_style_profile = resolved_style_profile or self._resolve_style_profile(inp)
        runtime_source = dict(self._resolved_style_profile_to_runtime_kwargs(resolved_style_profile))
        for key in ADVANCED_STYLE_RUNTIME_KEYS:
            value = inp.get(key, None)
            if value is not None:
                runtime_source[key] = value
        return {
            key: value
            for key, value in build_reference_style_runtime_kwargs(runtime_source).items()
            if value is not None
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

    def _prepare_inference_runtime(self, inp: Dict, *, spk_emb: Optional[torch.Tensor] = None):
        ref_mels, reference_meta = self._load_reference_mels(inp)
        resolved_profile = self._resolve_style_profile(inp)
        control_kwargs = self._build_model_control_kwargs(inp, resolved_style_profile=resolved_profile)
        runtime_kwargs = self._build_style_runtime_kwargs(inp, resolved_style_profile=resolved_profile)
        try:
            requested_style_strength = float(
                resolved_profile.get(
                    "style_strength_requested",
                    resolved_profile.get("style_strength", 1.0),
                )
            )
        except (TypeError, ValueError):
            requested_style_strength = float(resolved_profile.get("style_strength", 1.0))
        spk_embed = self._normalize_spk_embed(spk_emb)
        latency_report = dict(self.streaming_latency_report)
        metadata = {
            "streaming_impl": self.streaming_impl,
            "streaming_capabilities": dict(self.streaming_capabilities),
            "streaming_latency_report": latency_report,
            "reference_contract_mode": ref_mels.get(
                "reference_contract_mode",
                self.hparams.get("reference_contract_mode", "collapsed_reference"),
            ),
            "split_reference_inputs": reference_meta["split_reference_inputs"],
            "has_distinct_split_refs": reference_meta["has_distinct_split_refs"],
            "collapsed_split_refs": reference_meta["collapsed_split_refs"],
            "style_profile": resolved_profile.get("style_profile", "strong_style"),
            "style_profile_track": str(
                resolved_profile.get("style_profile_track", resolved_profile.get("track", "mainline"))
            ),
            "style_strength": requested_style_strength,
            "style_strength_effective": float(
                resolved_profile.get(
                    "style_strength_effective",
                    resolved_profile.get("style_strength", 1.0),
                )
            ),
            "style_strength_was_clamped": bool(
                resolved_profile.get("style_strength_was_clamped", False)
            ),
            "dynamic_timbre_strength": float(resolved_profile.get("dynamic_timbre_strength", 1.0)),
            "dynamic_timbre_strength_effective": float(
                resolved_profile.get("dynamic_timbre_strength", 1.0)
            ),
            "dynamic_timbre_strength_source": str(
                resolved_profile.get("dynamic_timbre_strength_source", "derived_from_style_strength")
            ),
            "style_to_pitch_residual_include_timbre": bool(
                resolved_profile.get("style_to_pitch_residual_include_timbre", False)
            ),
            "streaming_frontend_stateful": True,
            "acoustic_native_streaming": False,
            "acoustic_prefix_recompute": True,
            "vocoder_native_streaming": bool(
                self.streaming_capabilities.get("vocoder_native_streaming", False)
            ),
            "theoretical_first_packet_latency_ms": float(
                latency_report.get("first_packet_algorithmic_latency_ms", 0.0)
            ),
            "steady_state_vocoder_window_ms": float(
                latency_report.get("steady_state_vocoder_window_ms", 0.0)
            ),
            "steady_state_vocoder_recompute_multiplier": float(
                latency_report.get("steady_state_vocoder_recompute_multiplier", 0.0)
            ),
            "vocoder_left_context_frames": int(self.vocoder_left_context_frames),
            "vocoder_left_context_frames_effective": int(self.vocoder_left_context_frames),
            "vocoder_left_context_source": str(self.vocoder_left_context_source),
            "spk_embed_override": bool(spk_embed is not None),
        }
        with torch.inference_mode():
            reference_cache = self.model.prepare_reference_cache(
                reference_bundle=ref_mels,
                spk_embed=spk_embed,
                infer=True,
                global_steps=200000,
            )
        return {
            "reference_bundle": ref_mels,
            "reference_cache": reference_cache,
            "control_kwargs": control_kwargs,
            "runtime_kwargs": runtime_kwargs,
            "spk_embed": spk_embed,
            "metadata": metadata,
        }

    def _load_source_mel_tensor(self, src_wav: str):
        src_mel_np = self._wav_to_mel(src_wav)
        src_mel = torch.from_numpy(src_mel_np).unsqueeze(0).to(self.device)
        return src_mel, src_mel_np

    def _project_emformer_output_to_codes(self, emformer_out: torch.Tensor):
        if not isinstance(emformer_out, torch.Tensor):
            raise TypeError("Expected a tensor from the Emformer front-end.")
        if emformer_out.dim() == 3 and emformer_out.shape[-1] > 1:
            emformer_out = torch.argmax(emformer_out, dim=-1)
        return emformer_out

    def _extract_offline_content_codes(self, src_mel: torch.Tensor):
        lengths = torch.full((src_mel.size(0),), src_mel.size(1), dtype=torch.long, device=self.device)
        with torch.inference_mode():
            if self.emformer.mode == "both":
                emformer_out, _, _ = self.emformer(src_mel, lengths)
            else:
                emformer_out, _ = self.emformer(src_mel, lengths)
        return self._project_emformer_output_to_codes(emformer_out)

    def _run_model_from_content_codes(self, content_codes: torch.Tensor, runtime: Dict):
        reference_bundle = runtime["reference_bundle"]
        reference_cache = runtime["reference_cache"]
        with torch.inference_mode():
            return self.model(
                content=content_codes,
                spk_embed=runtime["spk_embed"],
                target=None,
                ref=reference_bundle["ref"],
                f0=None,
                uv=None,
                infer=True,
                global_steps=200000,
                **reference_bundle_to_model_kwargs(reference_bundle),
                **reference_cache_to_model_kwargs(reference_cache),
                **runtime["control_kwargs"],
                **runtime["runtime_kwargs"],
            )

    def _infer_prefix_online_from_mel(self, src_mel: torch.Tensor, runtime: Dict):
        total_frames = src_mel.shape[1]
        seg = int(self.streaming_layout["chunk_frames"])
        rc = int(self.streaming_layout["right_context_frames"])

        content_code_buffer = PrefixCodeBuffer(max_length=max(1, int(total_frames)))
        mel_chunks = RollingMelContextBuffer(self.vocoder_left_context_frames)
        wav_chunks = []
        vocoder_left_context = int(self.vocoder_left_context_frames)
        prev_len = 0
        pos = 0
        state = None
        final_out = None
        chunk_count = 0
        self.vocoder.reset_stream()

        while pos < total_frames:
            chunk_count += 1
            emit = min(seg, total_frames - pos)
            look = min(rc, total_frames - (pos + emit))
            real_len = emit + look
            chunk = src_mel[:, pos:pos + real_len, :]
            need_pad = (seg + rc) - real_len
            if need_pad > 0:
                pad = chunk[:, -1:, :].expand(1, need_pad, src_mel.shape[2])
                chunk = torch.cat([chunk, pad], dim=1)

            lengths = torch.full((1,), chunk.size(1), dtype=torch.long, device=self.device)
            with torch.inference_mode():
                chunk_out, _, state = self.emformer.emformer.infer(chunk, lengths, state)
                if self.emformer.mode == "both":
                    chunk_out = self.emformer.proj1(chunk_out)
                else:
                    chunk_out = self.emformer.proj(chunk_out)
            chunk_out = self._project_emformer_output_to_codes(chunk_out)

            tail_context_deficit = max(0, rc - look)
            effective_emit = max(0, emit - tail_context_deficit)
            effective_emit = min(int(chunk_out.size(1)), int(effective_emit))
            if effective_emit <= 0:
                pos += emit
                continue
            new_codes = chunk_out[:, :effective_emit]
            all_codes = content_code_buffer.append(new_codes.squeeze(0)).unsqueeze(0)
            final_out = self._run_model_from_content_codes(all_codes, runtime)
            mel_out = final_out["mel_out"][0]
            mel_new = mel_out[prev_len:]
            prev_len = mel_out.shape[0]
            pos += emit

            if mel_new.size(0) <= 0:
                continue
            mel_new_frames = int(mel_new.size(0))
            mel_window, vocoder_context_frames = mel_chunks.append(mel_new)
            wav_chunk_vocoder = self.vocoder.spec2wav(mel_window.cpu().numpy())
            hop = self.hparams["hop_size"]
            start_sample = vocoder_context_frames * hop
            end_sample = min(len(wav_chunk_vocoder), start_sample + mel_new_frames * hop)
            if end_sample > start_sample:
                wav_chunks.append(wav_chunk_vocoder[start_sample:end_sample])

        if final_out is None:
            raise RuntimeError("Streaming inference produced no decoder output.")
        mel_pred = mel_chunks.full_sequence()
        if mel_pred is None:
            mel_pred = final_out["mel_out"][0]
        wav_pred = (
            np.concatenate(wav_chunks, axis=0)
            if len(wav_chunks) > 0
            else self.vocoder.spec2wav(mel_pred.cpu().numpy())
        )
        estimated_runtime = build_streaming_latency_report(self.hparams, total_frames=int(total_frames))
        stream_meta = {
            "num_chunks": int(chunk_count),
            "tokens_per_chunk": int(seg),
            "mel_frames": int(mel_pred.size(0)),
            "wav_num_samples": int(len(wav_pred)),
            "vocoder_left_context_frames_effective": int(vocoder_left_context),
            "vocoder_left_context_source": str(self.vocoder_left_context_source),
            "acoustic_prefix_recompute_multiplier": float(
                cumulative_prefix_recompute_multiplier(chunk_count)
            ),
            "streaming_latency_report": estimated_runtime,
        }
        return wav_pred, mel_pred, final_out, stream_meta

    def _infer_offline_from_mel(self, src_mel: torch.Tensor, runtime: Dict):
        content_codes = self._extract_offline_content_codes(src_mel)
        out = self._run_model_from_content_codes(content_codes, runtime)
        mel_pred = out["mel_out"][0]
        self.vocoder.reset_stream()
        wav_pred = self.vocoder.spec2wav(mel_pred.cpu().numpy())
        offline_meta = {
            "num_chunks": 1,
            "tokens_per_chunk": int(content_codes.size(1)),
            "mel_frames": int(mel_pred.size(0)),
            "wav_num_samples": int(len(wav_pred)),
        }
        return wav_pred, mel_pred, out, offline_meta

    def _mel_parity_metrics(self, online_mel: torch.Tensor, offline_mel: torch.Tensor):
        metrics = {}
        if not isinstance(online_mel, torch.Tensor) or not isinstance(offline_mel, torch.Tensor):
            return metrics
        min_len = min(int(online_mel.size(0)), int(offline_mel.size(0)))
        if min_len <= 0:
            return metrics
        online_aligned = online_mel[:min_len]
        offline_aligned = offline_mel[:min_len]
        metrics["mel_len_online"] = int(online_mel.size(0))
        metrics["mel_len_offline"] = int(offline_mel.size(0))
        metrics["mel_l1_full"] = float(torch.mean(torch.abs(online_aligned - offline_aligned)).cpu())
        metrics["mel_l2_full"] = float(F.mse_loss(online_aligned, offline_aligned).cpu())
        tail_frames = min(min_len, int(self.hparams.get("streaming_parity_tail_frames", 32)))
        if tail_frames > 0:
            metrics["mel_l1_tail"] = float(
                torch.mean(torch.abs(online_aligned[-tail_frames:] - offline_aligned[-tail_frames:])).cpu()
            )
        return metrics

    @staticmethod
    def _effective_runtime_metadata(model_out: Dict):
        if not isinstance(model_out, dict):
            return {}
        style_mainline = model_out.get("style_mainline")
        if not isinstance(style_mainline, dict):
            style_mainline = {}
        metadata = {}
        float_fields = (
            "style_strength",
            "dynamic_timbre_strength",
            "style_temperature",
            "dynamic_timbre_temperature",
            "style_query_global_summary_scale",
            "dynamic_timbre_coarse_style_context_scale",
            "dynamic_timbre_query_style_condition_scale",
            "expressive_upper_bound_progress",
            "runtime_dynamic_timbre_style_budget_ratio",
            "runtime_dynamic_timbre_style_budget_margin",
        )
        bool_fields = (
            "style_to_pitch_residual_include_timbre",
            "style_to_pitch_residual_uses_timbre_context",
            "style_to_pitch_residual_post_rhythm_requested",
            "style_to_pitch_residual_post_rhythm_available",
            "style_to_pitch_residual_post_rhythm_runtime_available",
            "style_to_pitch_residual_post_rhythm_projection_used",
            "dynamic_timbre_coarse_style_context_applied",
            "dynamic_timbre_tvt_enabled",
            "dynamic_timbre_tvt_deferred_by_curriculum",
            "runtime_dynamic_timbre_style_budget_enabled",
            "runtime_dynamic_timbre_style_budget_applied",
            "upper_bound_curriculum_enabled",
            "dynamic_timbre_style_context_stopgrad",
        )
        string_fields = (
            "style_to_pitch_residual_canvas",
            "style_to_pitch_residual_requested_mode",
            "style_to_pitch_residual_canvas_selection_reason",
            "style_to_pitch_residual_projection_selection_reason",
            "style_to_pitch_residual_canvas_fallback_reason",
            "style_to_pitch_residual_projection_fallback_reason",
            "timbre_query_style_scale_source",
        )
        for key in float_fields:
            value = model_out.get(key, style_mainline.get(key))
            if value is None:
                continue
            try:
                metadata[key] = float(value)
            except (TypeError, ValueError):
                continue
        for key in bool_fields:
            value = model_out.get(key, style_mainline.get(key))
            if value is None:
                continue
            metadata[key] = bool(value)
        for key in string_fields:
            value = model_out.get(key, style_mainline.get(key))
            if value is None:
                continue
            metadata[key] = str(value)
        mode = style_mainline.get("mode")
        if mode is not None:
            metadata["decoder_style_condition_mode"] = str(mode)
        for key in ("style_trace_mode", "style_memory_mode", "dynamic_timbre_memory_mode"):
            value = style_mainline.get(key, model_out.get(key))
            if value is not None:
                metadata[key] = str(value)
        if "dynamic_timbre_strength" in style_mainline:
            metadata["dynamic_timbre_strength_effective"] = float(
                style_mainline["dynamic_timbre_strength"]
            )
        if "style_strength" in style_mainline:
            metadata["style_strength_effective"] = float(style_mainline["style_strength"])
        if style_mainline.get("dynamic_timbre_strength_source") is not None:
            metadata["dynamic_timbre_strength_source"] = str(
                style_mainline["dynamic_timbre_strength_source"]
            )
        return metadata

    def infer_once(self, inp: Dict, *, spk_emb: Optional[torch.Tensor] = None):
        runtime = self._prepare_inference_runtime(inp, spk_emb=spk_emb)
        src_mel, _ = self._load_source_mel_tensor(inp["src_wav"])
        wav_pred, mel_pred, final_out, stream_meta = self._infer_prefix_online_from_mel(src_mel, runtime)
        self.last_infer_metadata = dict(runtime["metadata"])
        self.last_infer_metadata.update(self._effective_runtime_metadata(final_out))
        self.last_infer_metadata.update(
            {
                **stream_meta,
                "mainline_contract": {
                    "query_anchor_split_applied": bool(final_out.get("query_anchor_split_applied", False)),
                    "dynamic_timbre_style_context_owner_safe": bool(
                        final_out.get("dynamic_timbre_style_context_owner_safe", False)
                    ),
                    "global_timbre_to_pitch_applied": bool(
                        final_out.get("global_timbre_to_pitch_applied", False)
                    ),
                },
            }
        )
        return wav_pred, mel_pred.cpu().numpy()

    def infer_parity_once(self, inp: Dict, *, spk_emb: Optional[torch.Tensor] = None):
        runtime = self._prepare_inference_runtime(inp, spk_emb=spk_emb)
        src_mel, _ = self._load_source_mel_tensor(inp["src_wav"])
        wav_online, mel_online, online_out, online_meta = self._infer_prefix_online_from_mel(src_mel, runtime)
        wav_offline, mel_offline, offline_out, offline_meta = self._infer_offline_from_mel(src_mel, runtime)
        metrics = self._mel_parity_metrics(mel_online, mel_offline)
        metrics["wav_num_samples_online"] = int(len(wav_online))
        metrics["wav_num_samples_offline"] = int(len(wav_offline))
        metrics["duration_gap_ms"] = float(
            1000.0 * abs(len(wav_online) - len(wav_offline)) / float(self.hparams["audio_sample_rate"])
        )
        summary = {
            "metadata": dict(runtime["metadata"]),
            "online_meta": online_meta,
            "offline_meta": offline_meta,
            "mainline_contract": {
                "query_anchor_split_applied": bool(online_out.get("query_anchor_split_applied", False)),
                "dynamic_timbre_style_context_owner_safe": bool(
                    online_out.get("dynamic_timbre_style_context_owner_safe", False)
                ),
                "global_timbre_to_pitch_applied_online": bool(
                    online_out.get("global_timbre_to_pitch_applied", False)
                ),
                "global_timbre_to_pitch_applied_offline": bool(
                    offline_out.get("global_timbre_to_pitch_applied", False)
                ),
            },
            "metrics": metrics,
        }
        self.last_infer_metadata = dict(runtime["metadata"])
        self.last_infer_metadata.update(self._effective_runtime_metadata(online_out))
        self.last_infer_metadata.update(
            {
                "online_chunks": int(online_meta.get("num_chunks", 0)),
                "offline_chunks": int(offline_meta.get("num_chunks", 0)),
                **metrics,
            }
        )
        return {
            "wav_online": wav_online,
            "wav_offline": wav_offline,
            "mel_online": mel_online.cpu().numpy(),
            "mel_offline": mel_offline.cpu().numpy(),
            "summary": summary,
        }

    def test_multiple_sentences(self, test_cases: List[Dict]):
        os.makedirs("infer_out_demo", exist_ok=True)
        for i, inp in enumerate(test_cases):
            wav, mel = self.infer_once(inp)
            ref_name = os.path.splitext(os.path.basename(inp["ref_wav"]))[0]
            src_name = os.path.splitext(os.path.basename(inp["src_wav"]))[0]
            save_path = f"infer_out_demo/{ref_name}_{src_name}.wav"
            save_wav(wav, save_path, self.hparams["audio_sample_rate"])
            metadata_path = f"infer_out_demo/{ref_name}_{src_name}.metadata.json"
            self._write_json(metadata_path, self.last_infer_metadata)
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
    print("Available style profiles:", ", ".join(available_mainline_style_profiles()))

    engine = StreamingVoiceConversion(hparams)
    engine.test_multiple_sentences(demo)
