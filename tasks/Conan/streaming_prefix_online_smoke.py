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
    scalarize_value,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Conan offline mel against prefix-online chunked mel on the canonical single-reference mainline."
    )
    parser.add_argument("--config", type=str, default="egs/conan_mainline_infer.yaml")
    parser.add_argument(
        "--binary_data_dir",
        type=str,
        default=default_smoke_binary_data_dir(),
        help="Binary dataset dir. Defaults to the first available built-in smoke dataset.",
    )
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--tokens_per_chunk", type=int, default=4)
    parser.add_argument("--max_mel_l1", type=float, default=0.10)
    parser.add_argument("--max_tail_mel_l1", type=float, default=0.05)
    parser.add_argument("--max_boundary_mel_l1", type=float, default=0.05)
    parser.add_argument("--max_boundary_mel_l1_max", type=float, default=0.10)
    parser.add_argument(
        "--output_path",
        type=str,
        default="smoke_runs/streaming_prefix_online_smoke.json",
    )
    return parser.parse_args()


def run_smoke(args):
    if not args.binary_data_dir:
        raise ValueError("No smoke binary_data_dir was provided and no default smoke dataset was found.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset, _ = build_pseudo_style_dataset(
        config_path=args.config,
        binary_data_dir=args.binary_data_dir,
        extra_hparams={
            "reference_contract_mode": "collapsed_reference",
            "decoder_style_condition_mode": "mainline_full",
            "lambda_mel_adv": 0.0,
        },
    )
    sample_index = int(args.sample_index) % len(dataset)
    batch = build_pseudo_style_batch(dataset, [sample_index], device, ensure_factorized_refs=False)
    task = build_conan_training_task(device, global_step=200000)
    task.model.eval()
    task.mel_disc.eval()

    with torch.no_grad():
        offline_output = task.run_model(batch, infer=True, test=True)[1]
        streaming_output = task._run_prefix_online_inference(
            batch,
            tokens_per_chunk=int(args.tokens_per_chunk),
        )
        parity_metrics = task._streaming_parity_metrics(offline_output, streaming_output)

    if not parity_metrics:
        raise RuntimeError("Failed to compute online/offline parity metrics.")
    for key, value in parity_metrics.items():
        if not is_finite_scalar(value):
            raise RuntimeError(f"Non-finite parity metric: {key}={value}")

    mel_l1 = float(scalarize_value(parity_metrics["streaming_mel_l1"]))
    tail_l1 = float(scalarize_value(parity_metrics.get("streaming_tail_mel_l1", parity_metrics["streaming_mel_l1"])))
    boundary_l1 = float(
        scalarize_value(parity_metrics.get("streaming_boundary_mel_l1", parity_metrics["streaming_mel_l1"]))
    )
    boundary_l1_max = float(
        scalarize_value(parity_metrics.get("streaming_boundary_mel_l1_max", parity_metrics["streaming_mel_l1"]))
    )
    if mel_l1 > float(args.max_mel_l1):
        raise AssertionError(f"streaming_mel_l1 too large: {mel_l1} > {args.max_mel_l1}")
    if tail_l1 > float(args.max_tail_mel_l1):
        raise AssertionError(f"streaming_tail_mel_l1 too large: {tail_l1} > {args.max_tail_mel_l1}")
    if boundary_l1 > float(args.max_boundary_mel_l1):
        raise AssertionError(f"streaming_boundary_mel_l1 too large: {boundary_l1} > {args.max_boundary_mel_l1}")
    if boundary_l1_max > float(args.max_boundary_mel_l1_max):
        raise AssertionError(
            f"streaming_boundary_mel_l1_max too large: {boundary_l1_max} > {args.max_boundary_mel_l1_max}"
        )
    if not bool(streaming_output.get("query_anchor_split_applied", False)):
        raise AssertionError("query_anchor_split_applied is false on the prefix-online path.")
    if not bool(streaming_output.get("dynamic_timbre_style_context_owner_safe", False)):
        raise AssertionError("dynamic_timbre_style_context_owner_safe is false on the prefix-online path.")
    if not bool(offline_output.get("query_anchor_split_applied", False)):
        raise AssertionError("query_anchor_split_applied is false on the offline path.")
    if not bool(offline_output.get("dynamic_timbre_style_context_owner_safe", False)):
        raise AssertionError("dynamic_timbre_style_context_owner_safe is false on the offline path.")
    if bool(streaming_output.get("global_timbre_to_pitch_applied", False)):
        raise AssertionError("global_timbre_to_pitch_applied unexpectedly true on the prefix-online path.")
    if bool(offline_output.get("global_timbre_to_pitch_applied", False)):
        raise AssertionError("global_timbre_to_pitch_applied unexpectedly true on the offline path.")
    if int(streaming_output.get("streaming_total_chunks", 0)) <= 0:
        raise AssertionError("streaming_total_chunks must be positive.")
    if streaming_output.get("streaming_eval_mode") != "prefix_online_content_chunked":
        raise AssertionError(
            "streaming_eval_mode must be prefix_online_content_chunked when using prefix-online smoke."
        )
    if int(streaming_output.get("streaming_chunk_tokens", 0)) != int(args.tokens_per_chunk):
        raise AssertionError(
            "streaming_chunk_tokens must match the requested chunk size in prefix-online smoke."
        )

    summary = {
        "device": device,
        "config": args.config,
        "binary_data_dir": args.binary_data_dir,
        "sample_index": int(sample_index),
        "tokens_per_chunk": int(args.tokens_per_chunk),
        "streaming_eval_mode": streaming_output.get("streaming_eval_mode"),
        "streaming_total_chunks": int(streaming_output.get("streaming_total_chunks", 0)),
        "streaming_chunk_tokens": int(streaming_output.get("streaming_chunk_tokens", args.tokens_per_chunk)),
        "offline_mel_shape": list(offline_output["mel_out"].shape),
        "streaming_mel_shape": list(streaming_output["mel_out"].shape),
        "query_anchor_split_applied": bool(streaming_output.get("query_anchor_split_applied", False)),
        "dynamic_timbre_style_context_owner_safe": bool(
            streaming_output.get("dynamic_timbre_style_context_owner_safe", False)
        ),
        "offline_query_anchor_split_applied": bool(offline_output.get("query_anchor_split_applied", False)),
        "offline_dynamic_timbre_style_context_owner_safe": bool(
            offline_output.get("dynamic_timbre_style_context_owner_safe", False)
        ),
        "global_timbre_to_pitch_applied_online": bool(streaming_output.get("global_timbre_to_pitch_applied", False)),
        "global_timbre_to_pitch_applied_offline": bool(offline_output.get("global_timbre_to_pitch_applied", False)),
        "parity_metrics": {key: scalarize_value(value) for key, value in parity_metrics.items()},
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
    print("STREAMING_PREFIX_ONLINE_SMOKE_OK")


if __name__ == "__main__":
    main()
