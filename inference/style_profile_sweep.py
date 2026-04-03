"""Single-reference style-profile sweep utilities.

This runner is intended for the current product-style Conan surface:
`src_wav + ref_wav + optional style_profile`.

Factorized multi-reference sweeps should live in separate research tooling.
"""

import csv
import json
import os
import re
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np

from inference.conan_request import ADVANCED_CONTROL_KEYS, has_distinct_split_reference_inputs
from modules.Conan.style_profiles import available_mainline_style_profiles, resolve_style_profile
from utils.audio.io import save_wav


CASE_META_KEYS = {
    "name",
    "profiles",
    "profile_overrides",
}

def slugify_filename(text, default="case"):
    text = str(text or "").strip()
    if text == "":
        return default
    text = re.sub(r"[^\w\-\.]+", "_", text, flags=re.UNICODE)
    text = text.strip("._")
    return text or default


def load_sweep_config(config_or_path):
    if isinstance(config_or_path, dict):
        return deepcopy(config_or_path)
    config_path = Path(config_or_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_dict(base, update):
    result = dict(base or {})
    for key, value in (update or {}).items():
        result[key] = value
    return result


class StyleProfileSweepRunner:
    def __init__(
        self,
        hparams,
        sweep_config,
        *,
        output_dir=None,
        engine=None,
        allow_advanced_controls=False,
        allow_split_reference_inputs=False,
    ):
        self.hparams = hparams
        self.config = load_sweep_config(sweep_config)
        self.output_dir = Path(output_dir or self.config.get("output_dir", self._default_output_dir()))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.engine = engine if engine is not None else self._build_engine()
        self._advanced_control_warning_emitted = False
        self.allow_advanced_controls = bool(allow_advanced_controls)
        self.allow_split_reference_inputs = bool(allow_split_reference_inputs)

    def _build_engine(self):
        from inference.Conan import StreamingVoiceConversion

        return StreamingVoiceConversion(self.hparams)

    def _default_output_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join("infer_out_profiles", timestamp)

    def _resolve_profiles_for_case(self, case):
        profiles = case.get("profiles") or self.config.get("profiles") or available_mainline_style_profiles()
        known_profiles = set(available_mainline_style_profiles())
        profiles = [str(p) for p in profiles]
        unknown_profiles = [p for p in profiles if p not in known_profiles]
        if unknown_profiles:
            raise ValueError(
                f"Unknown style profiles in sweep config: {unknown_profiles}. "
                f"Available profiles: {sorted(known_profiles)}."
            )
        if len(profiles) <= 0:
            profiles = ["strong_style"]
        return profiles

    def _case_name(self, case, case_idx):
        explicit = case.get("name", None)
        if explicit:
            return slugify_filename(explicit, default=f"case_{case_idx:03d}")
        src_name = slugify_filename(Path(case["src_wav"]).stem, default=f"src_{case_idx:03d}")
        ref_name = slugify_filename(Path(case["ref_wav"]).stem, default=f"ref_{case_idx:03d}")
        return f"{case_idx:03d}_{src_name}_to_{ref_name}"

    def _build_profile_input(self, case, profile_name):
        base_input = merge_dict(self.config.get("defaults", {}), case)
        for key in CASE_META_KEYS:
            base_input.pop(key, None)

        profile_overrides = merge_dict(
            self.config.get("profile_overrides", {}).get(profile_name, {}),
            case.get("profile_overrides", {}).get(profile_name, {}),
        )
        base_input = merge_dict(base_input, profile_overrides)
        allow_advanced_controls = bool(
            base_input.get("allow_advanced_controls", self.allow_advanced_controls)
        )
        if not allow_advanced_controls:
            ignored_advanced_keys = [key for key in ADVANCED_CONTROL_KEYS if base_input.get(key) is not None]
            if ignored_advanced_keys and not self._advanced_control_warning_emitted:
                warnings.warn(
                    "Ignoring advanced non-mainline control keys in style_profile_sweep.py: "
                    f"{ignored_advanced_keys}. Set allow_advanced_controls=true only for "
                    "explicit research/ablation sweeps.",
                    stacklevel=2,
                )
                self._advanced_control_warning_emitted = True
            for key in ignored_advanced_keys:
                base_input.pop(key, None)
        has_distinct_split_refs = has_distinct_split_reference_inputs(base_input)
        if has_distinct_split_refs and not self.allow_split_reference_inputs:
            raise ValueError(
                "StyleProfileSweepRunner is the canonical single-reference Conan sweep surface "
                "and does not accept split reference inputs. Use only ref_wav on the mainline path."
            )
        if has_distinct_split_refs:
            base_input["allow_split_reference_inputs"] = True
        base_input["style_profile"] = profile_name
        return base_input

    def _save_profile_outputs(
        self,
        case_dir,
        profile_name,
        wav,
        mel,
        resolved_profile,
        model_input,
        infer_metadata=None,
    ):
        profile_slug = slugify_filename(profile_name, default="strong_style")
        wav_path = case_dir / f"{profile_slug}.wav"
        save_wav(wav, str(wav_path), self.hparams["audio_sample_rate"])

        mel_path = None
        if self.config.get("save_mel_npy", True):
            mel_path = case_dir / f"{profile_slug}.npy"
            np.save(mel_path, np.asarray(mel, dtype=np.float32))

        meta_path = case_dir / f"{profile_slug}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "style_profile": profile_name,
                    "resolved_profile": resolved_profile,
                    "model_input": model_input,
                    "infer_metadata": infer_metadata,
                    "wav_path": wav_path.name,
                    "mel_path": mel_path.name if mel_path is not None else None,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return wav_path, mel_path, meta_path

    def run_case(self, case_idx, case):
        case_name = self._case_name(case, case_idx)
        case_dir = self.output_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        entries = []
        for profile_name in self._resolve_profiles_for_case(case):
            infer_input = self._build_profile_input(case, profile_name)
            resolved_profile = resolve_style_profile(
                infer_input,
                preset=self.hparams.get("style_profile", "strong_style"),
            )
            started = time.time()
            try:
                wav_pred, mel_pred = self.engine.infer_once(infer_input)
                wav_path, mel_path, meta_path = self._save_profile_outputs(
                    case_dir,
                    profile_name,
                    wav_pred,
                    mel_pred,
                    resolved_profile,
                    infer_input,
                    getattr(self.engine, "last_infer_metadata", None),
                )
                entries.append(
                    {
                        "case_name": case_name,
                        "profile": profile_name,
                        "status": "ok",
                        "elapsed_sec": round(time.time() - started, 4),
                        "wav_path": str(wav_path.relative_to(self.output_dir)),
                        "mel_path": str(mel_path.relative_to(self.output_dir)) if mel_path is not None else None,
                        "meta_path": str(meta_path.relative_to(self.output_dir)),
                    }
                )
            except Exception as e:
                entries.append(
                    {
                        "case_name": case_name,
                        "profile": profile_name,
                        "status": "error",
                        "elapsed_sec": round(time.time() - started, 4),
                        "error": str(e),
                    }
                )

        with open(case_dir / "case_manifest.json", "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        return entries

    def _write_summary_files(self, entries):
        manifest_path = self.output_dir / "sweep_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

        summary_csv_path = self.output_dir / "sweep_summary.csv"
        fieldnames = [
            "case_name",
            "profile",
            "status",
            "elapsed_sec",
            "wav_path",
            "mel_path",
            "meta_path",
            "error",
        ]
        with open(summary_csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in entries:
                writer.writerow({key: entry.get(key, None) for key in fieldnames})

    def run_all(self):
        cases = self.config.get("cases", [])
        if len(cases) <= 0:
            raise ValueError("Style profile sweep config must contain at least one case.")

        all_entries = []
        started = time.time()
        for case_idx, case in enumerate(cases):
            all_entries.extend(self.run_case(case_idx, case))

        self._write_summary_files(all_entries)
        success = sum(1 for entry in all_entries if entry.get("status") == "ok")
        failed = sum(1 for entry in all_entries if entry.get("status") != "ok")
        summary = {
            "output_dir": str(self.output_dir),
            "num_cases": len(cases),
            "num_runs": len(all_entries),
            "successful_runs": success,
            "failed_runs": failed,
            "elapsed_sec": round(time.time() - started, 4),
            "profiles": self.config.get("profiles", available_mainline_style_profiles()),
        }
        with open(self.output_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary, all_entries
