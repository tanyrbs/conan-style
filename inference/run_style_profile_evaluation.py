#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.factorized_swap_report import write_factorized_swap_report
from inference.profile_sweep_evaluator import StyleProfileSweepEvaluator
from utils.commons.hparams import hparams, set_hparams

CANONICAL_CONFIG = "egs/conan_mainline_infer.yaml"
CANONICAL_EXP_NAME = "Conan"


def write_augmented_evaluation_report(sweep_dir, report):
    report_path = Path(sweep_dir) / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate style profile sweep outputs.")
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
        "--sweep_dir",
        type=str,
        required=True,
        help="Output directory produced by run_style_profile_sweep.py",
    )
    parser.add_argument(
        "--disable_model_metrics",
        action="store_true",
        help="Disable Conan internal embedding similarity metrics and only compute audio statistics.",
    )
    parser.add_argument(
        "--use_external_metrics",
        action="store_true",
        help="Enable optional external metrics (e.g. resemblyzer speaker similarity).",
    )
    parser.add_argument(
        "--disable_external_speaker",
        action="store_true",
        help="Disable external speaker metric when --use_external_metrics is set.",
    )
    parser.add_argument(
        "--enable_external_content",
        action="store_true",
        help="Enable external SSL content metric when --use_external_metrics is set.",
    )
    parser.add_argument(
        "--enable_factorized_report",
        action="store_true",
        help="Research-only: generate a factorized swap report when the sweep contains explicit factorized rows.",
    )
    parser.add_argument(
        "--skip_factorized_report",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_hparams(config=args.config, exp_name=args.exp_name, hparams_str=args.hparams)
    should_write_factorized_report = bool(args.enable_factorized_report and not args.skip_factorized_report)
    evaluator = StyleProfileSweepEvaluator(
        hparams,
        args.sweep_dir,
        use_model_metrics=not args.disable_model_metrics,
        use_external_metrics=args.use_external_metrics,
        enable_external_speaker=not args.disable_external_speaker,
        enable_external_content=args.enable_external_content,
        include_research_metadata=should_write_factorized_report,
    )
    report, rows = evaluator.evaluate_all()
    factorized_report = None
    has_factorized_rows = any(
        bool(row.get("factorized_references_requested")) or row.get("swap_variant") is not None
        for row in rows
    )
    if should_write_factorized_report and has_factorized_rows:
        factorized_report = write_factorized_swap_report(args.sweep_dir)
        report["factorized_swap_report"] = factorized_report
        write_augmented_evaluation_report(args.sweep_dir, report)
    print("Style profile evaluation finished:", report)


if __name__ == "__main__":
    main()
