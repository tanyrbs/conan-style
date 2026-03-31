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
    is_finite_scalar,
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
    parser.add_argument("--config", type=str, default="egs/conan_emformer.yaml")
    parser.add_argument("--binary_data_dir", type=str, required=True)
    parser.add_argument(
        "--modes",
        type=str,
        default="legacy_full,mainline_full,global_style_dynamic_timbre,global_only,dynamic_timbre_only",
    )
    parser.add_argument("--speakers_per_batch", type=int, default=2)
    parser.add_argument("--items_per_speaker", type=int, default=2)
    parser.add_argument("--output_path", type=str, default="smoke_runs/style_mainline_smoke.json")
    return parser.parse_args()


def _sum_loss(losses):
    return sum(
        value
        for value in losses.values()
        if isinstance(value, torch.Tensor) and value.requires_grad
    )


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
    reference_contract = output.get("reference_contract", {})
    if not bool(reference_contract.get("factorization_guaranteed", False)):
        raise AssertionError(f"{mode}: strict factorized reference contract was not guaranteed.")
    if str(output.get("reference_contract_mode")) != "strict_factorized":
        raise AssertionError(f"{mode}: unexpected reference_contract_mode={output.get('reference_contract_mode')}")
    if not isinstance(output.get("style_query_inp"), torch.Tensor):
        raise AssertionError(f"{mode}: style_query_inp missing.")
    if not isinstance(output.get("timbre_query_inp"), torch.Tensor):
        raise AssertionError(f"{mode}: timbre_query_inp missing.")
    if output["style_query_inp"].shape != output["timbre_query_inp"].shape:
        raise AssertionError(f"{mode}: style/timbre query shape mismatch.")
    if not bool(output.get("decoder_style_adapter_enabled", False)):
        raise AssertionError(f"{mode}: decoder_style_adapter_enabled is false.")
    decoder_style_bundle = output.get("decoder_style_bundle")
    if not isinstance(decoder_style_bundle, dict):
        raise AssertionError(f"{mode}: decoder_style_bundle missing.")
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
    if expected["style_trace"] and not (
        isinstance(decoder_style_bundle.get("M_style"), torch.Tensor)
        or isinstance(decoder_style_bundle.get("slow_style_trace"), torch.Tensor)
    ):
        raise AssertionError(f"{mode}: style bundle missing M_style/slow_style_trace.")
    expected_global_timbre_to_pitch = (mode == "legacy_full")
    if bool(output.get("global_timbre_to_pitch_applied", False)) != expected_global_timbre_to_pitch:
        raise AssertionError(
            f"{mode}: global_timbre_to_pitch_applied mismatch, expected {expected_global_timbre_to_pitch}, "
            f"got {output.get('global_timbre_to_pitch_applied')}"
        )


def run_smoke(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modes = [item.strip() for item in str(args.modes).split(",") if item.strip()]
    results = []
    for mode in modes:
        dataset, _ = build_pseudo_style_dataset(
            config_path=args.config,
            binary_data_dir=args.binary_data_dir,
            extra_hparams={
                "reference_contract_mode": "strict_factorized",
                "decoder_style_condition_mode": mode,
                "lambda_mel_adv": 0.0,
            },
        )
        task = build_conan_training_task(device)
        optimizer, _ = task.build_optimizer(task.model)
        indices = select_speaker_batch_indices(
            dataset,
            0,
            speakers_per_batch=int(args.speakers_per_batch),
            items_per_speaker=int(args.items_per_speaker),
        )
        batch = build_pseudo_style_batch(dataset, indices, device, ensure_factorized_refs=True)
        optimizer.zero_grad(set_to_none=True)
        losses, output = task.run_model(batch, infer=False)
        total_loss = _sum_loss(losses)
        if total_loss is None or not is_finite_scalar(total_loss):
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
                "decoder_inp_l1": scalarize_value(output["decoder_inp"].abs().mean()),
                "pitch_inp_l1": scalarize_value(output["pitch_embed"].abs().mean()),
                "style_decoder_residual_l1": scalarize_value(output["style_decoder_residual"].abs().mean()),
                "dynamic_timbre_decoder_residual_l1": scalarize_value(
                    output["dynamic_timbre_decoder_residual"].abs().mean()
                ),
                "losses": scalar_losses,
            }
        )

    summary = {
        "device": device,
        "config": args.config,
        "binary_data_dir": args.binary_data_dir,
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
