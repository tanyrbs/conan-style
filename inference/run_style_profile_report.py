#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.profile_sweep_evaluator import StyleProfileSweepEvaluator
from inference.profile_sweep_report import StyleProfileSweepReporter
from utils.commons.hparams import hparams, set_hparams


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HTML/Markdown report for style profile sweeps.")
    parser.add_argument(
        "--sweep_dir",
        type=str,
        required=True,
        help="Sweep directory produced by run_style_profile_sweep.py",
    )
    parser.add_argument(
        "--auto_evaluate",
        action="store_true",
        help="If evaluation_summary.json is missing, run evaluation first.",
    )
    parser.add_argument(
        "--disable_model_metrics",
        action="store_true",
        help="Only used together with --auto_evaluate.",
    )
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    set_hparams()
    sweep_dir = Path(args.sweep_dir)
    if args.auto_evaluate and not (sweep_dir / "evaluation_summary.json").exists():
        evaluator = StyleProfileSweepEvaluator(
            hparams,
            sweep_dir,
            use_model_metrics=not args.disable_model_metrics,
        )
        evaluator.evaluate_all()

    reporter = StyleProfileSweepReporter(sweep_dir)
    report = reporter.write_reports()
    print("Style profile report finished:", report)


if __name__ == "__main__":
    main()
