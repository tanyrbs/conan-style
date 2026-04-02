#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.libritts_sweep_builder import (
    build_sweep_config_from_libritts,
    save_sweep_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a single-reference style-profile sweep config from a LibriTTS root."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root path of LibriTTS, e.g. G:\\streamVC\\LibriTTS_local\\LibriTTS",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="dev-clean",
        help="Comma separated LibriTTS splits, e.g. dev-clean,test-clean",
    )
    parser.add_argument("--num_cases", type=int, default=8, help="Number of cross-speaker cases to generate.")
    parser.add_argument(
        "--output_config",
        type=str,
        default="infer_out_profiles/libritts_style_profile_sweep.generated.json",
        help="Where to save the generated JSON config.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="infer_out_profiles/libritts",
        help="Output directory that the sweep runner should use.",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="",
        help="Comma separated style profiles. Empty means all available profiles.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    profiles = [part.strip() for part in args.profiles.split(",") if part.strip()]
    config = build_sweep_config_from_libritts(
        args.dataset_root,
        splits=splits,
        num_cases=args.num_cases,
        profiles=profiles or None,
        output_dir=args.output_dir,
    )
    output_path = save_sweep_config(config, args.output_config)
    print(f"Generated LibriTTS profile sweep config: {output_path}")
    print(f"Cases: {len(config['cases'])}, Profiles: {config['profiles']}")


if __name__ == "__main__":
    main()
