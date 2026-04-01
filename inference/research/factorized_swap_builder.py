"""Research-only utilities for factorized multi-reference swap ablations."""

import json
from pathlib import Path

from modules.Conan.style_profiles import available_style_profiles


def collect_libritts_wavs(dataset_root, splits=None):
    dataset_root = Path(dataset_root)
    splits = list(splits or ["dev-clean"])
    speaker_to_wavs = {}
    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        for wav_path in sorted(split_dir.rglob("*.wav")):
            rel_parts = wav_path.relative_to(dataset_root).parts
            if len(rel_parts) < 4:
                continue
            speaker_id = rel_parts[1]
            speaker_to_wavs.setdefault(speaker_id, []).append(str(wav_path))
    return speaker_to_wavs


def _pick_distinct_speakers(speakers, start_idx):
    total = len(speakers)
    if total < 4:
        raise ValueError("Need at least four speakers to build a factorized swap matrix.")
    return (
        speakers[start_idx % total],
        speakers[(start_idx + 1) % total],
        speakers[(start_idx + 2) % total],
        speakers[(start_idx + 3) % total],
    )


def build_factorized_swap_cases(
    speaker_to_wavs,
    *,
    num_groups=4,
    profiles=None,
):
    speakers = sorted([spk for spk, wavs in speaker_to_wavs.items() if len(wavs) > 0])
    if len(speakers) < 4:
        raise ValueError("Need at least four speakers with wavs to build factorized swap cases.")
    profiles = list(profiles or ["strong_style", "extreme"])

    cases = []
    for group_idx in range(int(num_groups)):
        src_spk, timbre_spk, style_spk, dyn_spk = _pick_distinct_speakers(speakers, group_idx)
        src_wavs = speaker_to_wavs[src_spk]
        timbre_wavs = speaker_to_wavs[timbre_spk]
        style_wavs = speaker_to_wavs[style_spk]
        dyn_wavs = speaker_to_wavs[dyn_spk]

        src_wav = src_wavs[group_idx % len(src_wavs)]
        timbre_wav = timbre_wavs[group_idx % len(timbre_wavs)]
        style_wav = style_wavs[group_idx % len(style_wavs)]
        dyn_wav = dyn_wavs[group_idx % len(dyn_wavs)]

        variants = [
            ("collapsed_ref", timbre_wav, timbre_wav, timbre_wav),
            ("style_only", timbre_wav, style_wav, timbre_wav),
            ("dynamic_only", timbre_wav, timbre_wav, dyn_wav),
            ("style_dynamic", timbre_wav, style_wav, dyn_wav),
        ]

        for variant_name, ref_timbre_wav, ref_style_wav, ref_dynamic_timbre_wav in variants:
            cases.append(
                {
                    "name": (
                        f"swap_{group_idx:03d}_{variant_name}_"
                        f"{Path(src_wav).stem}_t{timbre_spk}_s{style_spk}_d{dyn_spk}"
                    ),
                    "profiles": profiles,
                    "src_wav": src_wav,
                    "ref_wav": ref_timbre_wav,
                    "ref_timbre_wav": ref_timbre_wav,
                    "ref_style_wav": ref_style_wav,
                    "ref_dynamic_timbre_wav": ref_dynamic_timbre_wav,
                    "allow_split_reference_inputs": True,
                    "swap_matrix_group": group_idx,
                    "swap_variant": variant_name,
                    "src_speaker": src_spk,
                    "timbre_speaker": timbre_spk,
                    "style_speaker": style_spk,
                    "dynamic_timbre_speaker": dyn_spk,
                    "reference_contract_mode": "collapsed_reference",
                    "factorized_references_requested": True,
                }
            )
    return cases


def build_factorized_swap_config_from_libritts(
    dataset_root,
    *,
    splits=None,
    num_groups=4,
    profiles=None,
    output_dir="infer_out_profiles/research_factorized_swap",
    defaults=None,
):
    profiles = list(profiles or available_style_profiles())
    speaker_to_wavs = collect_libritts_wavs(dataset_root, splits=splits)
    cases = build_factorized_swap_cases(
        speaker_to_wavs,
        num_groups=num_groups,
        profiles=profiles,
    )
    return {
        "dataset_root": str(dataset_root),
        "splits": list(splits or ["dev-clean"]),
        "profiles": profiles,
        "save_mel_npy": True,
        "output_dir": output_dir,
        "defaults": dict(defaults or {}),
        "cases": cases,
    }


def save_swap_config(config, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return output_path
