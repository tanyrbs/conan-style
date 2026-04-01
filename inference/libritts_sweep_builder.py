import json
from pathlib import Path

from modules.Conan.style_profiles import available_mainline_style_profiles


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


def build_cross_speaker_cases(speaker_to_wavs, *, num_cases=8):
    speakers = sorted([spk for spk, wavs in speaker_to_wavs.items() if len(wavs) > 0])
    if len(speakers) < 2:
        raise ValueError("Need at least two speakers with wav files to build LibriTTS sweep cases.")

    cases = []
    for case_idx in range(int(num_cases)):
        src_speaker = speakers[case_idx % len(speakers)]
        ref_speaker = speakers[(case_idx + 1) % len(speakers)]
        if ref_speaker == src_speaker:
            ref_speaker = speakers[(case_idx + 2) % len(speakers)]
        src_wavs = speaker_to_wavs[src_speaker]
        ref_wavs = speaker_to_wavs[ref_speaker]
        src_wav = src_wavs[case_idx % len(src_wavs)]
        ref_wav = ref_wavs[case_idx % len(ref_wavs)]
        cases.append(
            {
                "name": f"libritts_{case_idx:03d}_{src_speaker}_to_{ref_speaker}",
                "src_wav": src_wav,
                "ref_wav": ref_wav,
            }
        )
    return cases


def build_sweep_config_from_libritts(
    dataset_root,
    *,
    splits=None,
    num_cases=8,
    profiles=None,
    output_dir="infer_out_profiles/libritts",
    defaults=None,
):
    profiles = list(profiles or available_mainline_style_profiles())
    speaker_to_wavs = collect_libritts_wavs(dataset_root, splits=splits)
    cases = build_cross_speaker_cases(speaker_to_wavs, num_cases=num_cases)
    return {
        "dataset_root": str(dataset_root),
        "splits": list(splits or ["dev-clean"]),
        "profiles": profiles,
        "save_mel_npy": True,
        "output_dir": output_dir,
        "defaults": dict(defaults or {}),
        "cases": cases,
    }


def save_sweep_config(config, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return output_path
