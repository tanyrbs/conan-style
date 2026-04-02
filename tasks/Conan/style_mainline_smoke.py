#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tasks.Conan.smoke_utils import (
    build_conan_training_task,
    build_pseudo_style_batch,
    build_pseudo_style_dataset,
    default_smoke_binary_data_dir,
    is_finite_scalar,
    resolve_smoke_batch_shape,
    scalarize_logs,
    scalarize_value,
    select_speaker_batch_indices,
)


EXPECTED_MODE_FLAGS = {
    "legacy_full": {"style_trace": True, "dynamic_timbre": True, "global_anchor": True},
    "mainline_full": {"style_trace": True, "dynamic_timbre": True, "global_anchor": True},
    "global_style_dynamic_timbre": {"style_trace": False, "dynamic_timbre": True, "global_anchor": True},
    "global_only": {"style_trace": False, "dynamic_timbre": False, "global_anchor": True},
    "dynamic_timbre_only": {"style_trace": False, "dynamic_timbre": True, "global_anchor": False},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Conan strong-style mainline contract and query split.")
    parser.add_argument("--config", type=str, default="egs/conan_mainline_infer.yaml")
    parser.add_argument(
        "--binary_data_dir",
        type=str,
        default=default_smoke_binary_data_dir(),
        help="Binary dataset dir. Defaults to the first available built-in smoke dataset.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="mainline_full",
    )
    parser.add_argument("--speakers_per_batch", type=int, default=2)
    parser.add_argument("--items_per_speaker", type=int, default=2)
    parser.add_argument("--output_path", type=str, default="smoke_runs/style_mainline_smoke_mainline_full.json")
    return parser.parse_args()


def _sum_loss(losses):
    total = None
    for value in losses.values():
        if isinstance(value, torch.Tensor) and value.requires_grad:
            total = value if total is None else (total + value)
    return total


def _cosine_mean(a, b):
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return None
    if a.dim() == 3:
        a = a.squeeze(1)
    if b.dim() == 3:
        b = b.squeeze(1)
    if a.dim() != 2 or b.dim() != 2 or tuple(a.shape) != tuple(b.shape):
        return None
    return torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-6).mean()


def _safe_abs_mean(value):
    if not isinstance(value, torch.Tensor):
        return None
    return value.abs().mean()


def _validate_mode_output(mode, output):
    expected = EXPECTED_MODE_FLAGS[mode]
    if bool(output.get("style_trace_applied", False)) != expected["style_trace"]:
        raise AssertionError(
            f"{mode}: style_trace_applied mismatch, expected {expected['style_trace']}, "
            f"got {output.get('style_trace_applied')}"
        )
    if bool(output.get("dynamic_timbre_applied", False)) != expected["dynamic_timbre"]:
        raise AssertionError(
            f"{mode}: dynamic_timbre_applied mismatch, expected {expected['dynamic_timbre']}, "
            f"got {output.get('dynamic_timbre_applied')}"
        )
    if bool(output.get("global_style_anchor_applied", False)) != expected["global_anchor"]:
        raise AssertionError(
            f"{mode}: global_style_anchor_applied mismatch, expected {expected['global_anchor']}, "
            f"got {output.get('global_style_anchor_applied')}"
        )
    if str(output.get("reference_contract_mode")) != "collapsed_reference":
        raise AssertionError(
            f"{mode}: unexpected reference_contract_mode={output.get('reference_contract_mode')}"
        )
    reference_contract = output.get("reference_contract", {}) or {}
    if bool(reference_contract.get("factorization_guaranteed", True)):
        raise AssertionError(f"{mode}: factorization_guaranteed unexpectedly true for mainline.")
    if str(reference_contract.get("factorization_semantics")) != "single_reference_weak_internal_factorization":
        raise AssertionError(
            f"{mode}: unexpected factorization_semantics={reference_contract.get('factorization_semantics')}"
        )
    if not isinstance(output.get("style_query_inp"), torch.Tensor):
        raise AssertionError(f"{mode}: style_query_inp missing.")
    if not isinstance(output.get("timbre_query_inp"), torch.Tensor):
        raise AssertionError(f"{mode}: timbre_query_inp missing.")
    if not isinstance(output.get("style_query_base"), torch.Tensor):
        raise AssertionError(f"{mode}: style_query_base missing.")
    if not isinstance(output.get("timbre_query_base"), torch.Tensor):
        raise AssertionError(f"{mode}: timbre_query_base missing.")
    if output["style_query_inp"].shape != output["timbre_query_inp"].shape:
        raise AssertionError(f"{mode}: style/timbre query shape mismatch.")
    if not bool(output.get("query_anchor_split_applied", False)):
        raise AssertionError(f"{mode}: query_anchor_split_applied is false.")
    expected_owner_safe = bool(expected["style_trace"])
    if bool(output.get("dynamic_timbre_style_context_owner_safe", False)) != expected_owner_safe:
        raise AssertionError(
            f"{mode}: dynamic_timbre_style_context_owner_safe mismatch, "
            f"expected {expected_owner_safe}, got {output.get('dynamic_timbre_style_context_owner_safe')}"
        )
    if expected_owner_safe:
        bridge_flag = str(output.get("dynamic_timbre_style_context_bridge", ""))
        stopgrad_flag = bool(output.get("dynamic_timbre_style_context_stopgrad", True))
        expected_bridge = "layernorm_stopgrad" if stopgrad_flag else "layernorm"
        if bridge_flag != expected_bridge:
            raise AssertionError(
                f"{mode}: dynamic_timbre_style_context_bridge expected {expected_bridge}, got {bridge_flag}"
            )
        style_raw = output.get("dynamic_timbre_style_context_raw")
        style_dec = output.get("style_decoder_residual")
        if isinstance(style_raw, torch.Tensor) and isinstance(style_dec, torch.Tensor):
            if not torch.allclose(style_raw, style_dec, atol=1e-5, rtol=1e-4):
                raise AssertionError(
                    f"{mode}: dynamic_timbre_style_context_raw must match style_decoder_residual for owner-aware conditioning."
                )
    coarse_style_scale = float(output.get("dynamic_timbre_coarse_style_context_scale", 0.0))
    query_style_scale_config = output.get("dynamic_timbre_query_style_condition_scale", None)
    if query_style_scale_config is None:
        query_style_scale_config = output.get("dynamic_timbre_query_style_condition_scale_runtime", None)
    style_condition_scale = float(output.get("dynamic_timbre_style_condition_scale_runtime", 0.0))
    actual_query_style_scale = float(output.get("timbre_query_style_scale", 0.0))
    actual_query_style_scale_source = str(output.get("timbre_query_style_scale_source", ""))
    if query_style_scale_config is not None:
        expected_query_scale = float(query_style_scale_config)
        expected_source = "query_style_condition" if expected_query_scale != 0.0 else "disabled"
        if abs(actual_query_style_scale - expected_query_scale) > 1e-12:
            raise AssertionError(
                f"{mode}: timbre_query_style_scale must match explicit query_style_condition scale."
            )
        if actual_query_style_scale_source != expected_source:
            raise AssertionError(
                f"{mode}: expected timbre_query_style_scale_source={expected_source}, got {actual_query_style_scale_source}"
            )
    elif coarse_style_scale != 0.0:
        if abs(actual_query_style_scale - coarse_style_scale) > 1e-12:
            raise AssertionError(
                f"{mode}: timbre_query_style_scale must match explicit coarse style scale."
            )
        if actual_query_style_scale_source != "coarse_style_context":
            raise AssertionError(
                f"{mode}: expected timbre_query_style_scale_source=coarse_style_context, got {actual_query_style_scale_source}"
            )
    else:
        if abs(actual_query_style_scale) > 1e-12 or actual_query_style_scale_source != "disabled":
            raise AssertionError(
                f"{mode}: timbre_query_style_scale should be disabled when explicit query conditioning is off."
            )
    expected_coarse_applied = bool(expected_owner_safe and coarse_style_scale != 0.0)
    if bool(output.get("dynamic_timbre_coarse_style_context_applied", False)) != expected_coarse_applied:
        raise AssertionError(
            f"{mode}: dynamic_timbre_coarse_style_context_applied mismatch, "
            f"expected {expected_coarse_applied}, got {output.get('dynamic_timbre_coarse_style_context_applied')}"
        )
    expected_query_style_applied = bool(expected_owner_safe and actual_query_style_scale != 0.0)
    if bool(output.get("timbre_query_style_context_applied", False)) != expected_query_style_applied:
        raise AssertionError(
            f"{mode}: timbre_query_style_context_applied mismatch, "
            f"expected {expected_query_style_applied}, got {output.get('timbre_query_style_context_applied')}"
        )
    expected_gate_style_conditioned = bool(expected_owner_safe and style_condition_scale != 0.0)
    if bool(output.get("dynamic_timbre_style_conditioned", False)) != expected_gate_style_conditioned:
        raise AssertionError(
            f"{mode}: dynamic_timbre_style_conditioned mismatch, "
            f"expected {expected_gate_style_conditioned}, got {output.get('dynamic_timbre_style_conditioned')}"
        )
    if expected["dynamic_timbre"]:
        gate_calibration = str(output.get("dynamic_timbre_gate_calibration", ""))
        if gate_calibration != "logit_affine":
            raise AssertionError(
                f"{mode}: dynamic_timbre_gate_calibration expected logit_affine, got {gate_calibration}"
            )
    if not bool(output.get("decoder_style_adapter_enabled", False)):
        raise AssertionError(f"{mode}: decoder_style_adapter_enabled is false.")
    if not isinstance(output.get("output_identity_embed"), torch.Tensor):
        raise AssertionError(f"{mode}: output_identity_embed missing.")
    if not isinstance(output.get("output_identity_anchor_target"), torch.Tensor):
        raise AssertionError(f"{mode}: output_identity_anchor_target missing.")
    decoder_style_bundle = output.get("decoder_style_bundle")
    if not isinstance(decoder_style_bundle, dict):
        raise AssertionError(f"{mode}: decoder_style_bundle missing.")
    if "effective_signal_epsilon" not in decoder_style_bundle:
        raise AssertionError(f"{mode}: decoder_style_bundle.effective_signal_epsilon missing.")
    if "decoder_style_bundle_effective_signal_epsilon" not in output:
        raise AssertionError(f"{mode}: decoder_style_bundle_effective_signal_epsilon missing in output.")
    if abs(
        float(decoder_style_bundle.get("effective_signal_epsilon", 0.0))
        - float(output.get("decoder_style_bundle_effective_signal_epsilon", -1.0))
    ) > 1e-12:
        raise AssertionError(f"{mode}: decoder style bundle effective_signal_epsilon mismatch.")
    if not bool(decoder_style_bundle.get("decoder_only", False)):
        raise AssertionError(f"{mode}: decoder_style_bundle.decoder_only is false.")
    if bool(decoder_style_bundle.get("planner_writeback_allowed", True)):
        raise AssertionError(f"{mode}: decoder_style_bundle unexpectedly enables planner writeback.")
    if bool(decoder_style_bundle.get("projector_writeback_allowed", True)):
        raise AssertionError(f"{mode}: decoder_style_bundle unexpectedly enables projector writeback.")
    if str(decoder_style_bundle.get("bundle_variant")) != mode:
        raise AssertionError(f"{mode}: unexpected decoder bundle variant {decoder_style_bundle.get('bundle_variant')}")
    if expected["dynamic_timbre"] and not isinstance(decoder_style_bundle.get("M_timbre"), torch.Tensor):
        raise AssertionError(f"{mode}: M_timbre missing from decoder bundle.")
    if not expected["dynamic_timbre"] and decoder_style_bundle.get("M_timbre") is not None:
        raise AssertionError(f"{mode}: M_timbre should be None when dynamic timbre is disabled.")
    if expected["style_trace"] and not (
        isinstance(decoder_style_bundle.get("M_style"), torch.Tensor)
        or isinstance(decoder_style_bundle.get("slow_style_trace"), torch.Tensor)
    ):
        raise AssertionError(f"{mode}: style bundle missing M_style/slow_style_trace.")
    if not expected["style_trace"] and decoder_style_bundle.get("M_style") is not None:
        raise AssertionError(f"{mode}: M_style should be None when style trace is disabled.")
    if not expected["global_anchor"] and decoder_style_bundle.get("global_timbre_anchor_runtime") is not None:
        raise AssertionError(f"{mode}: global_timbre_anchor_runtime should be None when global anchor is disabled.")
    if mode == "mainline_full" and expected["style_trace"]:
        M_style = decoder_style_bundle.get("M_style")
        if not isinstance(M_style, torch.Tensor):
            raise AssertionError(f"{mode}: M_style missing from decoder bundle.")
        if decoder_style_bundle.get("slow_style_trace") is not None:
            raise AssertionError(f"{mode}: slow_style_trace should not be a parallel owner in decoder bundle.")
        if not torch.allclose(output["style_decoder_residual"], M_style, atol=1e-5, rtol=1e-4):
            raise AssertionError(f"{mode}: style_decoder_residual does not match decoder bundle M_style.")
        if not torch.allclose(output["main_style_owner_residual"], output["style_decoder_residual"], atol=1e-5, rtol=1e-4):
            raise AssertionError(f"{mode}: main_style_owner_residual does not match style_decoder_residual.")
        if not torch.allclose(output["pitch_condition_inp"], output["query_condition_inp"], atol=1e-5, rtol=1e-4):
            raise AssertionError(f"{mode}: pitch_condition_inp should equal query_condition_inp when global_timbre_to_pitch is off.")
    expected_global_timbre_to_pitch = False
    if bool(output.get("global_timbre_to_pitch_applied", False)) != expected_global_timbre_to_pitch:
        raise AssertionError(
            f"{mode}: global_timbre_to_pitch_applied mismatch, expected {expected_global_timbre_to_pitch}, "
            f"got {output.get('global_timbre_to_pitch_applied')}"
        )
    stage_outputs = output.get("decoder_style_adapter_stages") or {}
    late_stage = stage_outputs.get("late") if isinstance(stage_outputs, dict) else None
    if isinstance(stage_outputs, dict):
        if not expected["style_trace"]:
            for stage_name, stage_meta in stage_outputs.items():
                if not isinstance(stage_meta, dict):
                    continue
                if stage_meta.get("style_trace_delta") is not None or stage_meta.get("slow_style_delta") is not None:
                    raise AssertionError(f"{mode}: style branch leaked into stage '{stage_name}' despite being disabled.")
        if not expected["dynamic_timbre"]:
            for stage_name, stage_meta in stage_outputs.items():
                if not isinstance(stage_meta, dict):
                    continue
                if stage_meta.get("dynamic_timbre_delta") is not None:
                    raise AssertionError(f"{mode}: dynamic timbre branch leaked into stage '{stage_name}' despite being disabled.")
        if not expected["global_anchor"]:
            for stage_name, stage_meta in stage_outputs.items():
                if not isinstance(stage_meta, dict):
                    continue
                if stage_meta.get("global_timbre_delta") is not None:
                    raise AssertionError(f"{mode}: global anchor branch leaked into stage '{stage_name}' despite being disabled.")
    if expected["dynamic_timbre"]:
        if "runtime_dynamic_timbre_style_budget_enabled" not in output:
            raise AssertionError(f"{mode}: runtime_dynamic_timbre_style_budget_enabled missing.")
        if "runtime_dynamic_timbre_style_budget_clip_frac" not in output:
            raise AssertionError(f"{mode}: runtime_dynamic_timbre_style_budget_clip_frac missing.")
    if mode == "mainline_full" and expected["style_trace"] and isinstance(late_stage, dict):
        if not bool(late_stage.get("global_style_skipped_due_to_local_owner", False)):
            raise AssertionError(
                f"{mode}: late-stage global_style_summary was not skipped despite local style owner."
            )


def run_smoke(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.binary_data_dir:
        raise ValueError(
            "No smoke binary_data_dir was provided and no default smoke dataset was found."
        )
    modes = [item.strip() for item in str(args.modes).split(",") if item.strip()]
    results = []
    batch_shape_info = None
    for mode in modes:
        dataset, _ = build_pseudo_style_dataset(
            config_path=args.config,
            binary_data_dir=args.binary_data_dir,
            extra_hparams={
                "reference_contract_mode": "collapsed_reference",
                "decoder_style_condition_mode": mode,
                "lambda_mel_adv": 0.0,
            },
        )
        if batch_shape_info is None:
            batch_shape_info = resolve_smoke_batch_shape(
                dataset,
                speakers_per_batch=int(args.speakers_per_batch),
                items_per_speaker=int(args.items_per_speaker),
            )
        task = build_conan_training_task(device)
        optimizer, _ = task.build_optimizer(task.model)
        indices = select_speaker_batch_indices(
            dataset,
            0,
            speakers_per_batch=batch_shape_info["speakers_per_batch"],
            items_per_speaker=batch_shape_info["items_per_speaker"],
        )
        batch = build_pseudo_style_batch(dataset, indices, device, ensure_factorized_refs=False)
        optimizer.zero_grad(set_to_none=True)
        losses, output = task.run_model(batch, infer=False)
        total_loss = _sum_loss(losses)
        if total_loss is None:
            raise RuntimeError(f"{mode}: no differentiable losses were produced.")
        if not is_finite_scalar(total_loss):
            raise RuntimeError(f"{mode}: non-finite total loss {total_loss}")
        total_loss.backward()
        task.on_before_optimization(0)
        optimizer.step()
        _validate_mode_output(mode, output)
        scalar_losses = scalarize_logs(losses)
        results.append(
            {
                "mode": mode,
                "total_loss": scalarize_value(total_loss),
                "reference_contract_mode": output.get("reference_contract_mode"),
                "reference_contract": output.get("reference_contract"),
                "style_trace_applied": bool(output.get("style_trace_applied", False)),
                "dynamic_timbre_applied": bool(output.get("dynamic_timbre_applied", False)),
                "global_style_anchor_applied": bool(output.get("global_style_anchor_applied", False)),
                "style_trace_skip_reason": output.get("style_trace_skip_reason"),
                "dynamic_timbre_skip_reason": output.get("dynamic_timbre_skip_reason"),
                "style_mainline_surface": output.get("style_mainline_surface"),
                "decoder_style_adapter_enabled": bool(output.get("decoder_style_adapter_enabled", False)),
                "decoder_style_adapter_stages": sorted(list((output.get("decoder_style_adapter_stages") or {}).keys())),
                "late_stage_global_style_skipped_due_to_local_owner": bool(
                    ((output.get("decoder_style_adapter_stages") or {}).get("late") or {}).get(
                        "global_style_skipped_due_to_local_owner",
                        False,
                    )
                ),
                "decoder_inp_l1": scalarize_value(output["decoder_inp"].abs().mean()),
                "pitch_inp_l1": scalarize_value(output["pitch_embed"].abs().mean()),
                "style_decoder_residual_l1": scalarize_value(
                    _safe_abs_mean(output.get("style_decoder_residual"))
                ),
                "dynamic_timbre_decoder_residual_l1": scalarize_value(
                    _safe_abs_mean(output.get("dynamic_timbre_decoder_residual"))
                ),
                "runtime_dynamic_timbre_style_budget_applied": bool(
                    output.get("runtime_dynamic_timbre_style_budget_applied", False)
                ),
                "runtime_dynamic_timbre_style_budget_clip_frac": scalarize_value(
                    output.get("runtime_dynamic_timbre_style_budget_clip_frac")
                ),
                "dynamic_timbre_style_context_owner_safe": bool(
                    output.get("dynamic_timbre_style_context_owner_safe", False)
                ),
                "output_identity_anchor_cos": scalarize_value(
                    _cosine_mean(output.get("output_identity_embed"), output.get("output_identity_anchor_target"))
                ),
                "batch_shape": batch_shape_info,
                "losses": scalar_losses,
            }
        )

    summary = {
        "device": device,
        "config": args.config,
        "binary_data_dir": args.binary_data_dir,
        "batch_shape": batch_shape_info,
        "results": results,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main():
    args = parse_args()
    summary = run_smoke(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("STYLE_MAINLINE_SMOKE_OK")


if __name__ == "__main__":
    main()
