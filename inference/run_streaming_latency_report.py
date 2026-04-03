#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.streaming_runtime import build_streaming_latency_report
from utils.commons.hparams import hparams, set_hparams

CANONICAL_CONFIG = "egs/conan_mainline_infer.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print the canonical Conan streaming latency/recompute report without loading checkpoints."
    )
    parser.add_argument("--config", type=str, default=CANONICAL_CONFIG)
    parser.add_argument("--duration_seconds", type=float, default=5.0)
    parser.add_argument("-hp", "--hparams", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    set_hparams(config=args.config, hparams_str=args.hparams, print_hparams=False)
    report = build_streaming_latency_report(hparams, duration_seconds=args.duration_seconds)
    payload = {
        "config": args.config,
        "streaming_impl": "emformer_stateful_prefix_recompute",
        "frontend_stateful_streaming": True,
        "acoustic_native_streaming": False,
        "acoustic_prefix_recompute": True,
        "vocoder_native_streaming": False,
        "report": report,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
