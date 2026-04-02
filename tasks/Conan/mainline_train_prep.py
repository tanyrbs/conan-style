#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.Conan.style_mainline import resolve_style_mainline_controls
from modules.Conan.style_profiles import resolve_style_profile
from modules.Conan.reference_bundle import build_control_kwargs, build_style_runtime_kwargs
from tasks.Conan.dataset import ConanDataset
from tasks.Conan.control_schedule import (
    MAINLINE_MINIMAL_CONTROL_LAMBDAS,
    resolve_control_regularization_config,
)
from tasks.Conan.reference_curriculum import resolve_reference_curriculum
from tasks.Conan.forcing_schedule import resolve_forcing_schedule
from utils.commons.hparams import hparams, set_hparams


REQUIRED_ZERO_KEYS = (
    "lambda_energy",
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
    "lambda_dynamic_timbre_boundary",
    "lambda_dynamic_timbre_anchor",
    "lambda_gate_rank",
    "lambda_decoder_late_anchor_budget",
)

REQUIRED_POSITIVE_KEYS = (
    "lambda_output_identity_cosine",
    "lambda_dynamic_timbre_budget",
    "lambda_pitch_residual_safe",
    "lambda_decoder_late_owner",
)

REQUIRED_PRESENT_KEYS = (
    "lambda_output_identity_cosine",
    "lambda_dynamic_timbre_budget",
    "lambda_pitch_residual_safe",
    "lambda_decoder_late_owner",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Conan mainline training config before the first real-dataset run."
    )
    parser.add_argument("--config", type=str, default="egs/conan_emformer.yaml")
    parser.add_argument(
        "--binary_data_dir",
        type=str,
        default=None,
        help="Optional override for validating the actual binary dataset dir used by the run.",
    )
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


def _check_true(checks, name, condition, *, actual=None, expected=True):
    checks.append(
        {
            "name": name,
            "ok": bool(condition),
            "actual": actual if actual is not None else bool(condition),
            "expected": expected,
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


def _check_npy_count_positive(checks, name, path_value):
    try:
        count = int(len(np.load(str(path_value), mmap_mode='r')))
        ok = count > 0
        checks.append(
            {
                "name": name,
                "ok": bool(ok),
                "actual": count,
                "expected": "> 0",
                "path": str(path_value),
            }
        )
    except Exception as exc:
        checks.append(
            {
                "name": name,
                "ok": False,
                "actual": f"{type(exc).__name__}: {exc}",
                "expected": "> 0",
                "path": str(path_value),
            }
        )


def _check_importable(checks, name, module_name):
    try:
        importlib.import_module(module_name)
        checks.append(
            {
                "name": name,
                "ok": True,
                "actual": f"import ok: {module_name}",
                "expected": "importable",
            }
        )
    except Exception as exc:  # pragma: no cover - surfaced through prep output
        checks.append(
            {
                "name": name,
                "ok": False,
                "actual": f"{type(exc).__name__}: {exc}",
                "expected": "importable",
            }
        )


def _jsonable(value):
    try:
        import torch
    except Exception:  # pragma: no cover - torch is available in normal runs
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def run_prep(args):
    set_hparams(config=args.config, print_hparams=False)
    if getattr(args, "binary_data_dir", None):
        hparams["binary_data_dir"] = str(args.binary_data_dir)
    controls = resolve_style_mainline_controls(hparams=hparams)
    resolved_profile = resolve_style_profile(
        {
            "style_profile": hparams.get("style_profile", "strong_style"),
            "style_trace_mode": hparams.get("style_trace_mode", None),
            "style_strength": hparams.get("style_strength", None),
            "global_style_trace_blend": hparams.get("global_style_trace_blend", None),
            "style_query_global_summary_scale": hparams.get("style_query_global_summary_scale", None),
            "dynamic_timbre_style_condition_scale": hparams.get("dynamic_timbre_style_condition_scale", None),
            "dynamic_timbre_coarse_style_context_scale": hparams.get(
                "dynamic_timbre_coarse_style_context_scale",
                None,
            ),
            "dynamic_timbre_query_style_condition_scale": hparams.get(
                "dynamic_timbre_query_style_condition_scale",
                None,
            ),
            "runtime_dynamic_timbre_style_budget_enabled": hparams.get(
                "runtime_dynamic_timbre_style_budget_enabled",
                None,
            ),
            "runtime_dynamic_timbre_style_budget_ratio": hparams.get(
                "runtime_dynamic_timbre_style_budget_ratio",
                None,
            ),
            "runtime_dynamic_timbre_style_budget_margin": hparams.get(
                "runtime_dynamic_timbre_style_budget_margin",
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
    batch_controls = None
    batch_controls_error = None
    try:
        train_dataset = ConanDataset(prefix="train", shuffle=False)
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            batch = train_dataset.collater([sample])
            batch_kwargs = {}
            batch_kwargs.update(build_control_kwargs(batch))
            batch_kwargs.update(build_style_runtime_kwargs(batch))
            batch_controls = resolve_style_mainline_controls(batch_kwargs, hparams=hparams)
        else:
            batch_controls_error = "empty_train_dataset"
    except Exception as exc:  # pragma: no cover - surfaced through prep output
        batch_controls_error = f"{type(exc).__name__}: {exc}"

    checks = []
    _check_equal(checks, "reference_contract_mode", hparams.get("reference_contract_mode"), "collapsed_reference")
    _check_equal(checks, "decoder_style_condition_mode", controls.mode, "mainline_full")
    _check_equal(checks, "global_timbre_to_pitch", bool(controls.global_timbre_to_pitch), False)
    _check_equal(checks, "style_to_pitch_residual", bool(controls.style_to_pitch_residual), True)
    _check_equal(checks, "style_to_pitch_residual_mode", controls.style_to_pitch_residual_mode, "auto")
    _check_equal(checks, "style_trace_mode", controls.style_trace_mode, "dual")
    _check_equal(checks, "style_router_enabled", bool(controls.style_router_enabled), True)
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
        "style_profile_controls_style_strength",
        controls.style_strength,
        resolved_profile.get("style_strength", controls.style_strength),
    )
    _check_close(
        checks,
        "style_profile_controls_dynamic_timbre_strength",
        controls.dynamic_timbre_strength,
        resolved_profile.get("dynamic_timbre_strength", controls.dynamic_timbre_strength),
    )
    _check_close(
        checks,
        "style_profile_controls_style_temperature",
        controls.style_temperature,
        resolved_profile.get("style_temperature", controls.style_temperature),
    )
    _check_close(
        checks,
        "style_profile_controls_dynamic_timbre_temperature",
        controls.dynamic_timbre_temperature,
        resolved_profile.get("dynamic_timbre_temperature", controls.dynamic_timbre_temperature),
    )
    _check_true(
        checks,
        "train_batch_mainline_controls_resolvable",
        batch_controls is not None,
        actual=batch_controls_error if batch_controls is None else "resolved",
        expected="resolved",
    )
    if batch_controls is not None:
        _check_close(
            checks,
            "train_batch_style_strength_matches_profile",
            batch_controls.style_strength,
            resolved_profile.get("style_strength", batch_controls.style_strength),
            tol=1e-6,
        )
        _check_close(
            checks,
            "train_batch_dynamic_timbre_strength_matches_profile",
            batch_controls.dynamic_timbre_strength,
            resolved_profile.get("dynamic_timbre_strength", batch_controls.dynamic_timbre_strength),
            tol=1e-6,
        )
        _check_close(
            checks,
            "train_batch_style_temperature_matches_profile",
            batch_controls.style_temperature,
            resolved_profile.get("style_temperature", batch_controls.style_temperature),
        )
        _check_close(
            checks,
            "train_batch_dynamic_timbre_temperature_matches_profile",
            batch_controls.dynamic_timbre_temperature,
            resolved_profile.get("dynamic_timbre_temperature", batch_controls.dynamic_timbre_temperature),
        )
    _check_close(
        checks,
        "runtime_dynamic_timbre_style_budget_ratio",
        hparams.get("runtime_dynamic_timbre_style_budget_ratio", 0.40),
        0.40,
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
    _check_close(checks, "dynamic_timbre_budget_ratio", hparams.get("dynamic_timbre_budget_ratio", 0.40), 0.40)
    _check_close(
        checks,
        "decoder_late_timbre_owner_ratio",
        hparams.get("decoder_late_timbre_owner_ratio", 0.50),
        0.50,
    )
    _check_close(
        checks,
        "decoder_dynamic_timbre_late_no_style_scale",
        hparams.get("decoder_dynamic_timbre_late_no_style_scale", 0.0),
        0.0,
    )
    _check_close(
        checks,
        "dynamic_timbre_style_condition_scale",
        hparams.get("dynamic_timbre_style_condition_scale", 0.35),
        0.35,
    )
    _check_close(
        checks,
        "dynamic_timbre_query_style_condition_scale",
        hparams.get("dynamic_timbre_query_style_condition_scale", 0.0),
        0.0,
    )
    _check_equal(
        checks,
        "dynamic_timbre_use_tvt",
        bool(hparams.get("dynamic_timbre_use_tvt", True)),
        True,
    )
    _check_close(
        checks,
        "dynamic_timbre_tvt_prior_scale",
        hparams.get("dynamic_timbre_tvt_prior_scale", 1.0),
        1.0,
    )
    _check_close(
        checks,
        "tv_timbre_gate_bias_init",
        hparams.get("tv_timbre_gate_bias_init", -1.0),
        -1.0,
    )
    _check_equal(
        checks,
        "use_external_speaker_verifier",
        bool(hparams.get("use_external_speaker_verifier", False)),
        False,
    )
    _check_equal(
        checks,
        "speaker_verifier_detach_input",
        bool(hparams.get("speaker_verifier_detach_input", False)),
        False,
    )
    _check_equal(checks, "reference_curriculum_mode", hparams.get("reference_curriculum_mode"), "bernoulli_cosine")
    _check_close(
        checks,
        "reference_curriculum_start_steps",
        hparams.get("reference_curriculum_start_steps", hparams.get("forcing", 0)),
        20000,
    )
    _check_close(
        checks,
        "reference_curriculum_end_steps",
        hparams.get("reference_curriculum_end_steps", hparams.get("random_speaker_steps", 0)),
        100000,
    )
    _check_close(
        checks,
        "reference_curriculum_external_prob_init",
        hparams.get("reference_curriculum_external_prob_init", 0.0),
        0.0,
    )
    _check_close(
        checks,
        "reference_curriculum_external_prob_final",
        hparams.get("reference_curriculum_external_prob_final", 1.0),
        1.0,
    )
    _check_close(
        checks,
        "reference_curriculum_self_ref_floor",
        hparams.get("reference_curriculum_self_ref_floor", 0.0),
        0.0,
    )
    _check_equal(
        checks,
        "reference_curriculum_sample_mode",
        str(hparams.get("reference_curriculum_sample_mode", "batch")).strip().lower(),
        "batch",
    )
    _check_equal(checks, "forcing_schedule_mode", hparams.get("forcing_schedule_mode"), "bernoulli_cosine")
    _check_close(
        checks,
        "forcing_legacy_cut",
        hparams.get("forcing", 0),
        20000,
    )
    _check_close(
        checks,
        "forcing_decay_start_steps",
        hparams.get("forcing_decay_start_steps", hparams.get("forcing", 0)),
        12000,
    )
    _check_close(
        checks,
        "forcing_decay_end_steps",
        hparams.get("forcing_decay_end_steps", hparams.get("forcing", 0)),
        60000,
    )
    _check_close(checks, "forcing_prob_init", hparams.get("forcing_prob_init", 1.0), 1.0)
    _check_close(checks, "forcing_prob_final", hparams.get("forcing_prob_final", 0.0), 0.0)
    _check_importable(checks, "runtime_import_torchaudio", "torchaudio")
    _check_close(
        checks,
        "random_speaker_steps_matches_curriculum_end",
        hparams.get("random_speaker_steps", 0),
        hparams.get("reference_curriculum_end_steps", hparams.get("random_speaker_steps", 0)),
    )
    reference_start = int(hparams.get("reference_curriculum_start_steps", hparams.get("forcing", 0)))
    reference_end = int(hparams.get("reference_curriculum_end_steps", hparams.get("random_speaker_steps", 0)))
    forcing_start = int(hparams.get("forcing_decay_start_steps", hparams.get("forcing", 0)))
    forcing_end = int(hparams.get("forcing_decay_end_steps", hparams.get("forcing", 0)))
    _check_true(
        checks,
        "forcing_decay_starts_no_later_than_reference_curriculum",
        forcing_start <= reference_start,
        actual={"forcing_decay_start_steps": forcing_start, "reference_curriculum_start_steps": reference_start},
        expected="forcing_decay_start_steps <= reference_curriculum_start_steps",
    )
    _check_true(
        checks,
        "forcing_decay_ends_no_later_than_reference_curriculum",
        forcing_end <= reference_end,
        actual={"forcing_decay_end_steps": forcing_end, "reference_curriculum_end_steps": reference_end},
        expected="forcing_decay_end_steps <= reference_curriculum_end_steps",
    )
    reference_preview_40000 = resolve_reference_curriculum(40000, hparams)
    reference_preview_70000 = resolve_reference_curriculum(70000, hparams)
    forcing_preview_20000 = resolve_forcing_schedule(20000, hparams)
    _check_close(
        checks,
        "reference_curriculum_external_prob_step_40000",
        reference_preview_40000.get("external_prob", 0.0),
        0.1464466094,
        tol=1e-6,
    )
    _check_close(
        checks,
        "reference_curriculum_external_prob_step_70000",
        reference_preview_70000.get("external_prob", 0.0),
        0.6913417162,
        tol=1e-6,
    )
    _check_close(
        checks,
        "forcing_prob_step_20000",
        forcing_preview_20000.get("forcing_prob", 0.0),
        0.9330127019,
        tol=1e-6,
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
    binary_data_dir = hparams.get("binary_data_dir")
    if binary_data_dir:
        for filename in (
            "train.data",
            "train.idx",
            "train_lengths.npy",
            "train_spk_ids.npy",
            "valid.data",
            "valid.idx",
            "valid_lengths.npy",
            "valid_spk_ids.npy",
            "test.data",
            "test.idx",
            "test_lengths.npy",
            "test_spk_ids.npy",
        ):
            _check_exists(
                checks,
                f"binary_{filename}_exists",
                os.path.join(str(binary_data_dir), filename),
            )
        for split in ("train", "valid", "test"):
            _check_npy_count_positive(
                checks,
                f"binary_{split}_items_nonempty",
                os.path.join(str(binary_data_dir), f"{split}_lengths.npy"),
            )

    reference_preview_steps = (0, 20000, 40000, 70000, 100000)
    forcing_preview_steps = (0, 12000, 20000, 60000, 100000)
    summary = {
        "config": args.config,
        "mainline_controls": _jsonable(controls.as_dict()),
        "train_batch_mainline_controls": None if batch_controls is None else _jsonable(batch_controls.as_dict()),
        "resolved_profile": _jsonable(resolved_profile),
        "reference_curriculum_preview": [
            _jsonable({"step": int(step), **resolve_reference_curriculum(step, hparams)})
            for step in reference_preview_steps
        ],
        "forcing_schedule_preview": [
            _jsonable({"step": int(step), **resolve_forcing_schedule(step, hparams)})
            for step in forcing_preview_steps
        ],
        "checks": _jsonable(checks),
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
