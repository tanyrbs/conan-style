#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.style_profile_sweep import StyleProfileSweepRunner
from utils.commons.hparams import hparams, set_hparams

CANONICAL_CONFIG = "egs/conan_mainline_infer.yaml"
CANONICAL_EXP_NAME = "Conan"
DEFAULT_SWEEP_CONFIG = "inference/conan_style_profile_sweep.example.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single-reference Conan style-profile sweep (mainline product surface)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=CANONICAL_CONFIG,
        help="Acoustic config yaml passed through to set_hparams(). Defaults to the canonical Conan mainline infer config.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=CANONICAL_EXP_NAME,
        help="Checkpoint experiment name passed through to set_hparams(). Defaults to the canonical Conan checkpoint.",
    )
    parser.add_argument(
        "-hp",
        "--hparams",
        type=str,
        default="",
        help="Optional hparams override string passed through to set_hparams().",
    )
    parser.add_argument(
        "--sweep_config",
        type=str,
        default=DEFAULT_SWEEP_CONFIG,
        help="JSON config describing single-reference mainline cases and profiles.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Optional output directory override.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_hparams(config=args.config, exp_name=args.exp_name, hparams_str=args.hparams)
    runner = StyleProfileSweepRunner(
        hparams,
        args.sweep_config,
        output_dir=args.output_dir or None,
    )
    summary, _ = runner.run_all()
    print("Style profile sweep finished:", summary)


if __name__ == "__main__":
    main()
