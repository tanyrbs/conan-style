#!/usr/bin/env python3
"""Research-only factorized swap config builder."""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.factorized_swap_builder import (
    build_factorized_swap_config_from_libritts,
    save_swap_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a research-only LibriTTS factorized swap matrix config."
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--splits", type=str, default="dev-clean")
    parser.add_argument("--num_groups", type=int, default=4)
    parser.add_argument(
        "--output_config",
        type=str,
        default="inference/libritts_factorized_swap.generated.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="infer_out_profiles/factorized_swap",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="strong_style,extreme",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    profiles = [part.strip() for part in args.profiles.split(",") if part.strip()]
    config = build_factorized_swap_config_from_libritts(
        args.dataset_root,
        splits=splits,
        num_groups=args.num_groups,
        profiles=profiles or None,
        output_dir=args.output_dir,
    )
    output_path = save_swap_config(config, args.output_config)
    print(f"Generated factorized swap config: {output_path}")
    print(f"Cases: {len(config['cases'])}, Profiles: {config['profiles']}")


if __name__ == "__main__":
    main()
