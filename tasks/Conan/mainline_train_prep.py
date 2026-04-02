#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.Conan.style_mainline import resolve_style_mainline_controls
from modules.Conan.style_profiles import resolve_style_profile
from tasks.Conan.control_schedule import (
    MAINLINE_MINIMAL_CONTROL_LAMBDAS,
    resolve_control_regularization_config,
)
from utils.commons.hparams import hparams, set_hparams


REQUIRED_ZERO_KEYS = (
    "lambda_style_timbre_disentangle",
    "lambda_style_trace_consistency",
    "lambda_style_query_var",
    "lambda_global_style_summary_align",
    "lambda_slow_style_summary_align",
    "lambda_tv_timbre_smooth",
    "lambda_tv_timbre_anchor",
    "lambda_timbre_anchor_cosine",
    "lambda_style_dynamic_timbre_disentangle",
    "lambda_dynamic_timbre_gate",
    "lambda_dynamic_timbre_anchor",
    "lambda_gate_rank",
    "lambda_decoder_late_anchor_budget",
)

REQUIRED_POSITIVE_KEYS = (
    "lambda_output_identity_cosine",
    "lambda_dynamic_timbre_budget",
    "lambda_dynamic_timbre_boundary",
    "lambda_decoder_late_owner",
)

REQUIRED_PRESENT_KEYS = (
    "lambda_output_identity_cosine",
    "lambda_dynamic_timbre_budget",
    "lambda_dynamic_timbre_boundary",
    "lambda_decoder_late_owner",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Conan mainline training config before the first real-dataset run."
    )
    parser.add_argument("--config", type=str, default="egs/conan_emformer.yaml")
    parser.add_argument(
        "--output_path",
        type=str,
        default="smoke_runs/mainline_train_prep.json",
    )
    return parser.parse_args()


def _check_equal(checks, name, actual, expected):
    ok = actual == expected
    checks.append(
        {
            "name": name,
            "ok": bool(ok),
            "actual": actual,
            "expected": expected,
        }
    )


def _check_close(checks, name, actual, expected, tol=1e-8):
    ok = abs(float(actual) - float(expected)) <= float(tol)
    checks.append(
        {
            "name": name,
            "ok": bool(ok),
            "actual": float(actual),
            "expected": float(expected),
            "tolerance": float(tol),
        }
    )


def _check_exists(checks, name, path_value):
    ok = bool(path_value) and os.path.exists(str(path_value))
    checks.append(
        {
            "name": name,
            "ok": bool(ok),
            "path": str(path_value) if path_value is not None else None,
        }
    )


def run_prep(args):
    set_hparams(config=args.config, print_hparams=False)
    controls = resolve_style_mainline_controls(hparams=hparams)
    resolved_profile = resolve_style_profile(
        {
            "style_profile": hparams.get("style_profile", "strong_style"),
            "style_trace_mode": hparams.get("style_trace_mode", None),
            "style_strength": hparams.get("style_strength", None),
            "global_style_trace_blend": hparams.get("global_style_trace_blend", None),
            "style_query_global_summary_scale": hparams.get("style_query_global_summary_scale", None),
            "dynamic_timbre_coarse_style_context_scale": hparams.get(
                "dynamic_timbre_coarse_style_context_scale",
                None,
            ),
            "allow_explicit_dynamic_timbre_strength": hparams.get(
                "allow_explicit_dynamic_timbre_strength",
                False,
            ),
        },
        preset=hparams.get("style_profile", "strong_style"),
    )
    regularization = resolve_control_regularization_config(hparams, global_step=0)

    checks = []
    _check_equal(checks, "reference_contract_mode", hparams.get("reference_contract_mode"), "collapsed_reference")
    _check_equal(checks, "decoder_style_condition_mode", controls.mode, "mainline_full")
    _check_equal(checks, "global_timbre_to_pitch", bool(controls.global_timbre_to_pitch), False)
    _check_equal(checks, "style_trace_mode", controls.style_trace_mode, "slow")
    _check_equal(checks, "style_memory_mode", controls.style_memory_mode, "slow")
    _check_equal(checks, "dynamic_timbre_memory_mode", controls.dynamic_timbre_memory_mode, "slow")
    _check_close(checks, "global_style_trace_blend", hparams.get("global_style_trace_blend", 0.0), 0.0)
    _check_close(
        checks,
        "style_query_global_summary_scale",
        hparams.get("style_query_global_summary_scale", 0.0),
        0.0,
    )
    _check_close(
        checks,
        "dynamic_timbre_coarse_style_context_scale",
        hparams.get("dynamic_timbre_coarse_style_context_scale", 0.0),
        0.0,
    )
    _check_equal(
        checks,
        "dynamic_timbre_style_context_stopgrad",
        bool(hparams.get("dynamic_timbre_style_context_stopgrad", True)),
        True,
    )
    _check_equal(
        checks,
        "allow_explicit_dynamic_timbre_strength",
        bool(hparams.get("allow_explicit_dynamic_timbre_strength", False)),
        False,
    )
    _check_equal(
        checks,
        "allow_split_reference_inputs",
        bool(hparams.get("allow_split_reference_inputs", False)),
        False,
    )
    _check_equal(
        checks,
        "emit_collapsed_reference_aliases",
        bool(hparams.get("emit_collapsed_reference_aliases", False)),
        False,
    )
    _check_equal(checks, "control_loss_profile", regularization.get("control_loss_profile"), "mainline_minimal")
    _check_equal(
        checks,
        "style_profile_track",
        resolved_profile.get("style_profile_track", resolved_profile.get("track")),
        "mainline",
    )
    _check_close(
        checks,
        "runtime_dynamic_timbre_style_budget_ratio",
        hparams.get("runtime_dynamic_timbre_style_budget_ratio", 0.50),
        0.50,
    )
    _check_equal(
        checks,
        "runtime_dynamic_timbre_style_budget_enabled",
        bool(hparams.get("runtime_dynamic_timbre_style_budget_enabled", True)),
        True,
    )
    _check_close(
        checks,
        "runtime_dynamic_timbre_style_budget_margin",
        hparams.get("runtime_dynamic_timbre_style_budget_margin", 0.0),
        0.0,
    )
    _check_close(checks, "dynamic_timbre_budget_ratio", hparams.get("dynamic_timbre_budget_ratio", 0.50), 0.50)
    _check_close(
        checks,
        "decoder_late_timbre_owner_ratio",
        hparams.get("decoder_late_timbre_owner_ratio", 0.55),
        0.55,
    )
    active_mainline_control_keys = tuple(
        key for key in MAINLINE_MINIMAL_CONTROL_LAMBDAS if float(hparams.get(key, 0.0)) > 0.0
    )
    checks.append(
        {
            "name": "mainline_minimal_active_control_loss_count",
            "ok": len(active_mainline_control_keys) <= 4,
            "actual": len(active_mainline_control_keys),
            "expected": "<= 4",
        }
    )
    checks.append(
        {
            "name": "mainline_minimal_active_control_losses",
            "ok": set(active_mainline_control_keys) == set(MAINLINE_MINIMAL_CONTROL_LAMBDAS),
            "actual": list(active_mainline_control_keys),
            "expected": list(MAINLINE_MINIMAL_CONTROL_LAMBDAS),
        }
    )

    for key in REQUIRED_POSITIVE_KEYS:
        checks.append(
            {
                "name": key,
                "ok": float(hparams.get(key, 0.0)) > 0.0,
                "actual": float(hparams.get(key, 0.0)),
                "expected": "> 0",
            }
        )
    for key in REQUIRED_ZERO_KEYS:
        _check_close(checks, key, hparams.get(key, 0.0), 0.0)
    for key in REQUIRED_PRESENT_KEYS:
        checks.append(
            {
                "name": f"{key}_present",
                "ok": key in hparams,
                "actual": key in hparams,
                "expected": True,
            }
        )

    _check_exists(checks, "binary_data_dir_exists", hparams.get("binary_data_dir"))
    _check_exists(checks, "processed_data_dir_exists", hparams.get("processed_data_dir"))

    summary = {
        "config": args.config,
        "mainline_controls": controls.as_dict(),
        "resolved_profile": resolved_profile,
        "checks": checks,
        "ready": bool(all(item["ok"] for item in checks)),
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main():
    args = parse_args()
    summary = run_prep(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["ready"]:
        raise SystemExit("MAINLINE_TRAIN_PREP_NOT_READY")
    print("MAINLINE_TRAIN_PREP_OK")


if __name__ == "__main__":
    main()
