import csv
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from inference.external_metrics import ExternalMetricSuite
from modules.Conan.reference_bundle import build_reference_bundle_from_inputs
from utils.audio import get_energy_librosa, librosa_wav2spec
from utils.audio.pitch_extractors import extract_pitch


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _summary_vector(value):
    if not isinstance(value, torch.Tensor):
        return None
    if value.dim() == 3:
        return value.squeeze(1)
    return value


def _normalize_mask(mask, sequence):
    if not isinstance(mask, torch.Tensor) or not isinstance(sequence, torch.Tensor):
        return None
    if mask.dim() == 3 and mask.size(-1) == 1:
        mask = mask.squeeze(-1)
    if mask.dim() != 2 or tuple(mask.shape) != tuple(sequence.shape[:2]):
        return None
    return mask.bool().to(sequence.device)


def _masked_sequence_mean(sequence, mask=None):
    if not isinstance(sequence, torch.Tensor) or sequence.dim() != 3:
        return None
    if mask is None:
        return sequence.mean(dim=1)
    mask = _normalize_mask(mask, sequence)
    if mask is None:
        return None
    valid = (~mask).unsqueeze(-1).to(sequence.dtype)
    denom = valid.sum(dim=1).clamp_min(1.0)
    pooled = (sequence * valid).sum(dim=1) / denom
    return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


def _cosine_similarity(a, b):
    a = _summary_vector(a)
    b = _summary_vector(b)
    if a is None or b is None:
        return None
    if a.dim() != 2 or b.dim() != 2 or tuple(a.shape) != tuple(b.shape):
        return None
    return F.cosine_similarity(a, b, dim=-1, eps=1e-6).mean().item()


def _mean_abs_diff(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size <= 0 or b.size <= 0:
        return None
    dim = min(len(a), len(b))
    if dim <= 0:
        return None
    return float(np.mean(np.abs(a[:dim] - b[:dim])))


def _voiced_stats(f0):
    f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
    voiced = f0[f0 > 1e-6]
    voiced_frac = float((f0 > 1e-6).mean()) if f0.size > 0 else 0.0
    if voiced.size <= 0:
        return {
            "voiced_frac": voiced_frac,
            "f0_mean": 0.0,
            "f0_std": 0.0,
            "f0_median": 0.0,
        }
    return {
        "voiced_frac": voiced_frac,
        "f0_mean": float(voiced.mean()),
        "f0_std": float(voiced.std()),
        "f0_median": float(np.median(voiced)),
    }


def _lazy_import_streaming_engine():
    try:
        from inference.Conan import StreamingVoiceConversion

        return StreamingVoiceConversion
    except Exception as exc:
        raise RuntimeError(
            "Failed to import the Conan streaming inference stack for profile-sweep evaluation. "
            "Check that torchaudio matches torch and that Emformer runtime dependencies are installed."
        ) from exc


class AudioFeatureCache:
    def __init__(self, hparams):
        self.hparams = hparams
        self.cache = {}
        self._pitch_warning_emitted = False

    def get(self, wav_path):
        wav_path = str(wav_path)
        if wav_path not in self.cache:
            spec = librosa_wav2spec(
                wav_path,
                fft_size=self.hparams["fft_size"],
                hop_size=self.hparams["hop_size"],
                win_length=self.hparams["win_size"],
                num_mels=self.hparams["audio_num_mel_bins"],
                fmin=self.hparams["fmin"],
                fmax=self.hparams["fmax"],
                sample_rate=self.hparams["audio_sample_rate"],
                loud_norm=self.hparams["loud_norm"],
            )
            mel = np.clip(spec["mel"], self.hparams["mel_vmin"], self.hparams["mel_vmax"]).astype(np.float32)
            wav = np.asarray(spec["wav"], dtype=np.float32)
            energy = get_energy_librosa(wav, mel.shape[0], self.hparams).astype(np.float32)
            pitch_available = True
            try:
                f0 = extract_pitch(
                    self.hparams["pitch_extractor"],
                    wav,
                    self.hparams["hop_size"],
                    self.hparams["audio_sample_rate"],
                    f0_min=self.hparams["f0_min"],
                    f0_max=self.hparams["f0_max"],
                ).astype(np.float32)
            except Exception as e:
                pitch_available = False
                if not self._pitch_warning_emitted:
                    warnings.warn(
                        f"Pitch extraction failed in profile sweep evaluation ({e}); "
                        "F0-derived metrics will be zeroed for this run.",
                        stacklevel=2,
                    )
                    self._pitch_warning_emitted = True
                f0 = np.zeros((mel.shape[0],), dtype=np.float32)
            if len(f0) < mel.shape[0]:
                f0 = np.pad(f0, [[0, mel.shape[0] - len(f0)]], mode="constant")
            else:
                f0 = f0[: mel.shape[0]]
            self.cache[wav_path] = {
                "wav_path": wav_path,
                "wav": wav,
                "mel": mel,
                "energy": energy[: mel.shape[0]],
                "f0": f0[: mel.shape[0]],
                "duration_sec": float(len(wav)) / float(self.hparams["audio_sample_rate"]),
                "mel_frames": int(mel.shape[0]),
                "mel_mean": mel.mean(axis=0),
                "mel_std": mel.std(axis=0),
                "energy_mean": float(np.mean(energy[: mel.shape[0]])),
                "energy_std": float(np.std(energy[: mel.shape[0]])),
                "pitch_available": bool(pitch_available),
                **_voiced_stats(f0[: mel.shape[0]]),
            }
        return self.cache[wav_path]


class ModelFeatureCache:
    def __init__(self, engine):
        self.engine = engine
        self.cache = {}

    def _wav_to_mel_tensor(self, wav_path):
        mel = self.engine._wav_to_mel(str(wav_path))
        return torch.from_numpy(mel).float().unsqueeze(0).to(self.engine.device)

    def get(self, wav_path):
        wav_path = str(wav_path)
        if wav_path not in self.cache:
            mel = self._wav_to_mel_tensor(wav_path)
            bundle = build_reference_bundle_from_inputs(
                ref=mel,
                prompt_fallback_to_style=True,
            )
            with torch.no_grad():
                reference_cache = self.engine.model.prepare_reference_cache(
                    reference_bundle=bundle,
                    spk_embed=None,
                    infer=True,
                    global_steps=200000,
                )
            self.cache[wav_path] = {
                "style_global": _summary_vector(
                    reference_cache.get("global_style_summary", reference_cache.get("style_embed"))
                ),
                "timbre_global": _summary_vector(
                    reference_cache.get("global_timbre_anchor", reference_cache.get("style_embed"))
                ),
                "prosody": _masked_sequence_mean(
                    reference_cache.get("prosody_memory_slow", reference_cache.get("prosody_memory")),
                    reference_cache.get("prosody_memory_slow_mask", reference_cache.get("prosody_memory_mask")),
                ),
                "dynamic_timbre": _masked_sequence_mean(
                    reference_cache.get("timbre_memory_slow", reference_cache.get("timbre_memory")),
                    reference_cache.get("timbre_memory_slow_mask", reference_cache.get("timbre_memory_mask")),
                ),
            }
        return self.cache[wav_path]


def evaluate_profile_pair(gen_audio, src_audio, ref_audio):
    metrics = {
        "duration_sec": gen_audio["duration_sec"],
        "mel_frames": gen_audio["mel_frames"],
        "duration_ratio_to_src": gen_audio["duration_sec"] / max(src_audio["duration_sec"], 1e-6),
        "duration_ratio_to_ref": gen_audio["duration_sec"] / max(ref_audio["duration_sec"], 1e-6),
        "mel_mean_l1_to_src": _mean_abs_diff(gen_audio["mel_mean"], src_audio["mel_mean"]),
        "mel_mean_l1_to_ref": _mean_abs_diff(gen_audio["mel_mean"], ref_audio["mel_mean"]),
        "mel_std_l1_to_src": _mean_abs_diff(gen_audio["mel_std"], src_audio["mel_std"]),
        "mel_std_l1_to_ref": _mean_abs_diff(gen_audio["mel_std"], ref_audio["mel_std"]),
        "energy_mean_abs_to_src": abs(gen_audio["energy_mean"] - src_audio["energy_mean"]),
        "energy_mean_abs_to_ref": abs(gen_audio["energy_mean"] - ref_audio["energy_mean"]),
        "energy_std_abs_to_src": abs(gen_audio["energy_std"] - src_audio["energy_std"]),
        "energy_std_abs_to_ref": abs(gen_audio["energy_std"] - ref_audio["energy_std"]),
        "f0_mean_abs_to_src": abs(gen_audio["f0_mean"] - src_audio["f0_mean"]),
        "f0_mean_abs_to_ref": abs(gen_audio["f0_mean"] - ref_audio["f0_mean"]),
        "f0_std_abs_to_src": abs(gen_audio["f0_std"] - src_audio["f0_std"]),
        "f0_std_abs_to_ref": abs(gen_audio["f0_std"] - ref_audio["f0_std"]),
        "voiced_frac_abs_to_src": abs(gen_audio["voiced_frac"] - src_audio["voiced_frac"]),
        "voiced_frac_abs_to_ref": abs(gen_audio["voiced_frac"] - ref_audio["voiced_frac"]),
    }
    return metrics


def evaluate_audio_against_reference(gen_audio, ref_audio, prefix):
    return {
        f"duration_ratio_to_{prefix}": gen_audio["duration_sec"] / max(ref_audio["duration_sec"], 1e-6),
        f"mel_mean_l1_to_{prefix}": _mean_abs_diff(gen_audio["mel_mean"], ref_audio["mel_mean"]),
        f"mel_std_l1_to_{prefix}": _mean_abs_diff(gen_audio["mel_std"], ref_audio["mel_std"]),
        f"energy_mean_abs_to_{prefix}": abs(gen_audio["energy_mean"] - ref_audio["energy_mean"]),
        f"energy_std_abs_to_{prefix}": abs(gen_audio["energy_std"] - ref_audio["energy_std"]),
        f"f0_mean_abs_to_{prefix}": abs(gen_audio["f0_mean"] - ref_audio["f0_mean"]),
        f"f0_std_abs_to_{prefix}": abs(gen_audio["f0_std"] - ref_audio["f0_std"]),
        f"voiced_frac_abs_to_{prefix}": abs(gen_audio["voiced_frac"] - ref_audio["voiced_frac"]),
    }


PROFILE_META_KEYS = (
    "swap_matrix_group",
    "swap_variant",
    "factorized_references_requested",
    "reference_contract_mode",
    "src_speaker",
    "timbre_speaker",
    "style_speaker",
    "dynamic_timbre_speaker",
)


class StyleProfileSweepEvaluator:
    def __init__(
        self,
        hparams,
        sweep_dir,
        *,
        engine=None,
        use_model_metrics=True,
        use_external_metrics=False,
        enable_external_speaker=True,
        enable_external_content=False,
        include_research_metadata=False,
    ):
        self.hparams = hparams
        self.sweep_dir = Path(sweep_dir)
        self.engine = engine
        self.use_model_metrics = bool(use_model_metrics)
        self.include_research_metadata = bool(include_research_metadata)
        self.audio_cache = AudioFeatureCache(hparams)
        self.model_cache = ModelFeatureCache(engine) if engine is not None and self.use_model_metrics else None
        self.external_metrics = (
            ExternalMetricSuite(
                enable_speaker=bool(enable_external_speaker),
                enable_content=bool(enable_external_content),
            )
            if bool(use_external_metrics)
            else None
        )

    def _build_engine(self):
        if self.engine is None:
            StreamingVoiceConversion = _lazy_import_streaming_engine()
            self.engine = StreamingVoiceConversion(self.hparams)
        if self.use_model_metrics and self.model_cache is None:
            self.model_cache = ModelFeatureCache(self.engine)

    def _iter_profile_entries(self):
        manifest_path = self.sweep_dir / "sweep_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                for entry in json.load(f):
                    if entry.get("status") == "ok":
                        yield dict(entry)
            return

        for case_manifest in self.sweep_dir.glob("*/case_manifest.json"):
            with open(case_manifest, "r", encoding="utf-8") as f:
                for entry in json.load(f):
                    if entry.get("status") == "ok":
                        yield dict(entry)

    def _load_profile_meta(self, entry):
        meta_path = self.sweep_dir / entry["meta_path"]
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _evaluate_model_metrics(self, gen_path, src_path, ref_path):
        if self.model_cache is None:
            return {}
        gen_feat = self.model_cache.get(gen_path)
        src_feat = self.model_cache.get(src_path)
        ref_feat = self.model_cache.get(ref_path)
        return {
            "global_cos_to_src": _cosine_similarity(gen_feat["style_global"], src_feat["style_global"]),
            "global_cos_to_ref": _cosine_similarity(gen_feat["style_global"], ref_feat["style_global"]),
            "timbre_global_cos_to_src": _cosine_similarity(gen_feat["timbre_global"], src_feat["timbre_global"]),
            "timbre_global_cos_to_ref": _cosine_similarity(gen_feat["timbre_global"], ref_feat["timbre_global"]),
            "prosody_cos_to_src": _cosine_similarity(gen_feat["prosody"], src_feat["prosody"]),
            "prosody_cos_to_ref": _cosine_similarity(gen_feat["prosody"], ref_feat["prosody"]),
            "dynamic_timbre_cos_to_src": _cosine_similarity(gen_feat["dynamic_timbre"], src_feat["dynamic_timbre"]),
            "dynamic_timbre_cos_to_ref": _cosine_similarity(gen_feat["dynamic_timbre"], ref_feat["dynamic_timbre"]),
        }

    def _evaluate_explicit_reference_metrics(self, gen_path, model_input):
        metrics = {}
        gen_audio = self.audio_cache.get(gen_path)
        explicit_refs = {
            "timbre_ref": model_input.get("ref_timbre_wav"),
            "style_ref": model_input.get("ref_style_wav"),
            "dynamic_timbre_ref": model_input.get("ref_dynamic_timbre_wav"),
            "emotion_ref": model_input.get("ref_emotion_wav"),
            "accent_ref": model_input.get("ref_accent_wav"),
        }
        for prefix, ref_path in explicit_refs.items():
            if not ref_path:
                continue
            ref_audio = self.audio_cache.get(Path(ref_path))
            metrics.update(evaluate_audio_against_reference(gen_audio, ref_audio, prefix))

        if self.model_cache is not None:
            gen_feat = self.model_cache.get(gen_path)
            if explicit_refs["timbre_ref"]:
                timbre_feat = self.model_cache.get(Path(explicit_refs["timbre_ref"]))
                metrics["timbre_global_cos_to_timbre_ref"] = _cosine_similarity(
                    gen_feat["timbre_global"], timbre_feat["timbre_global"]
                )
            if explicit_refs["style_ref"]:
                style_feat = self.model_cache.get(Path(explicit_refs["style_ref"]))
                metrics["global_cos_to_style_ref"] = _cosine_similarity(
                    gen_feat["style_global"], style_feat["style_global"]
                )
                metrics["prosody_cos_to_style_ref"] = _cosine_similarity(
                    gen_feat["prosody"], style_feat["prosody"]
                )
            if explicit_refs["dynamic_timbre_ref"]:
                dynamic_feat = self.model_cache.get(Path(explicit_refs["dynamic_timbre_ref"]))
                metrics["dynamic_timbre_cos_to_dynamic_timbre_ref"] = _cosine_similarity(
                    gen_feat["dynamic_timbre"], dynamic_feat["dynamic_timbre"]
                )

        if self.external_metrics is not None and self.external_metrics.available:
            metrics.update(
                self.external_metrics.evaluate(
                    gen_path=gen_path,
                    src_path=model_input.get("src_wav"),
                    timbre_ref_path=explicit_refs["timbre_ref"],
                    style_ref_path=explicit_refs["style_ref"],
                    dynamic_timbre_ref_path=explicit_refs["dynamic_timbre_ref"],
                )
            )
        return metrics

    def evaluate_entry(self, entry):
        meta = self._load_profile_meta(entry)
        model_input = meta["model_input"]
        gen_path = self.sweep_dir / entry["wav_path"]
        src_path = Path(model_input["src_wav"])
        ref_path = Path(model_input["ref_wav"])

        gen_audio = self.audio_cache.get(gen_path)
        src_audio = self.audio_cache.get(src_path)
        ref_audio = self.audio_cache.get(ref_path)

        metrics = evaluate_profile_pair(gen_audio, src_audio, ref_audio)
        metrics.update(
            {
                "case_name": entry["case_name"],
                "profile": entry["profile"],
                "wav_path": entry["wav_path"],
                "src_wav": str(src_path),
                "ref_wav": str(ref_path),
                "style_profile": meta.get("style_profile", entry["profile"]),
                "elapsed_sec": entry.get("elapsed_sec"),
            }
        )
        explicit_reference_requested = any(
            model_input.get(key)
            for key in (
                "ref_timbre_wav",
                "ref_style_wav",
                "ref_dynamic_timbre_wav",
                "ref_emotion_wav",
                "ref_accent_wav",
            )
        )
        if self.include_research_metadata:
            metrics.update(
                {
                    "swap_matrix_group": model_input.get("swap_matrix_group"),
                    "swap_variant": model_input.get("swap_variant"),
                    "factorized_references_requested": model_input.get("factorized_references_requested"),
                    "ref_timbre_wav": model_input.get("ref_timbre_wav"),
                    "ref_style_wav": model_input.get("ref_style_wav"),
                    "ref_dynamic_timbre_wav": model_input.get("ref_dynamic_timbre_wav"),
                    "ref_emotion_wav": model_input.get("ref_emotion_wav"),
                    "ref_accent_wav": model_input.get("ref_accent_wav"),
                }
            )
            for key in PROFILE_META_KEYS:
                if key not in metrics:
                    metrics[key] = model_input.get(key)
        metrics.update(self._evaluate_model_metrics(gen_path, src_path, ref_path))
        if explicit_reference_requested:
            metrics.update(self._evaluate_explicit_reference_metrics(gen_path, model_input))
        return metrics

    def evaluate_all(self):
        if self.use_model_metrics and self.engine is None:
            self._build_engine()

        entries = list(self._iter_profile_entries())
        if len(entries) <= 0:
            raise ValueError(f"No successful sweep entries found under {self.sweep_dir}.")

        started = time.time()
        results = [self.evaluate_entry(entry) for entry in entries]

        json_path = self.sweep_dir / "evaluation_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        csv_path = self.sweep_dir / "evaluation_summary.csv"
        fieldnames = sorted({key for item in results for key in item.keys()})
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

        report = {
            "sweep_dir": str(self.sweep_dir),
            "num_entries": len(results),
            "elapsed_sec": round(time.time() - started, 4),
            "json_path": str(json_path),
            "csv_path": str(csv_path),
            "use_model_metrics": self.model_cache is not None,
            "use_external_metrics": self.external_metrics is not None and self.external_metrics.available,
        }
        if self.external_metrics is not None and self.external_metrics.init_errors:
            report["external_metric_init_errors"] = dict(self.external_metrics.init_errors)
        with open(self.sweep_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return report, results
