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
    scalarize_value,
    scalarize_logs,
    select_speaker_batch_indices,
)
from utils.commons.hparams import hparams


def parse_args():
    parser = argparse.ArgumentParser(description="Run pseudo-style training smoke for Conan.")
    parser.add_argument("--config", type=str, default="egs/conan_emformer.yaml")
    parser.add_argument("--binary_data_dir", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=3)
    parser.add_argument("--speakers_per_batch", type=int, default=2)
    parser.add_argument("--items_per_speaker", type=int, default=2)
    parser.add_argument("--lambda_style_cls", type=float, default=1.0)
    parser.add_argument("--lambda_style_contrastive", type=float, default=0.5)
    parser.add_argument("--lambda_style_trace_consistency", type=float, default=0.1)
    parser.add_argument("--lambda_style_timbre_disentangle", type=float, default=0.05)
    parser.add_argument("--lambda_style_dynamic_timbre_disentangle", type=float, default=0.05)
    parser.add_argument("--lambda_dynamic_timbre_gate", type=float, default=0.02)
    parser.add_argument("--lambda_mel_adv", type=float, default=0.0)
    return parser.parse_args()

def run_smoke(args):
    dataset, num_styles = build_pseudo_style_dataset(
        config_path=args.config,
        binary_data_dir=args.binary_data_dir,
        extra_hparams={
            "lambda_style_cls": args.lambda_style_cls,
            "lambda_style_contrastive": args.lambda_style_contrastive,
            "lambda_style_trace_consistency": args.lambda_style_trace_consistency,
            "lambda_style_timbre_disentangle": args.lambda_style_timbre_disentangle,
            "lambda_style_dynamic_timbre_disentangle": args.lambda_style_dynamic_timbre_disentangle,
            "lambda_dynamic_timbre_gate": args.lambda_dynamic_timbre_gate,
            "lambda_mel_adv": args.lambda_mel_adv,
        },
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    task = build_conan_training_task(device)

    opt_gen, _ = task.build_optimizer(task.model)
    history = []
    for step_idx in range(int(args.num_steps)):
        indices = select_speaker_batch_indices(
            dataset,
            step_idx,
            speakers_per_batch=int(args.speakers_per_batch),
            items_per_speaker=int(args.items_per_speaker),
        )
        batch = build_pseudo_style_batch(dataset, indices, device)
        opt_gen.zero_grad(set_to_none=True)
        total_loss, logs = task._training_step(batch, step_idx, 0)
        if total_loss is None or not is_finite_scalar(total_loss):
            raise RuntimeError(f"Non-finite generator loss at step {step_idx}: {total_loss}")
        total_loss.backward()
        task.on_before_optimization(0)
        opt_gen.step()

        scalar_logs = scalarize_logs(logs)
        history.append(
            {
                "step": step_idx,
                "indices": [int(idx) for idx in indices],
                "pseudo_style_ids": [int(idx) for idx in batch["style_ids"].detach().cpu().tolist()],
                "total_loss": scalarize_value(total_loss),
                "style_cls": scalar_logs.get("style_cls"),
                "style_contrastive": scalar_logs.get("style_contrastive"),
                "style_trace_consistency": scalar_logs.get("style_trace_consistency"),
                "style_timbre_dis": scalar_logs.get("style_timbre_dis"),
                "style_dynamic_timbre_dis": scalar_logs.get("style_dynamic_timbre_dis"),
                "dynamic_timbre_gate_reg": scalar_logs.get("dynamic_timbre_gate_reg"),
                "l1": scalar_logs.get("l1"),
                "ssim": scalar_logs.get("ssim"),
                "fdiff": scalar_logs.get("fdiff"),
                "uv": scalar_logs.get("uv"),
            }
        )
        task.global_step += 1

    summary = {
        "device": device,
        "binary_data_dir": hparams["binary_data_dir"],
        "num_styles": num_styles,
        "num_steps": int(args.num_steps),
        "speakers_per_batch": int(args.speakers_per_batch),
        "items_per_speaker": int(args.items_per_speaker),
        "style_losses_active": {
            "lambda_style_cls": float(hparams.get("lambda_style_cls", 0.0)),
            "lambda_style_contrastive": float(hparams.get("lambda_style_contrastive", 0.0)),
            "lambda_style_trace_consistency": float(hparams.get("lambda_style_trace_consistency", 0.0)),
            "lambda_style_timbre_disentangle": float(hparams.get("lambda_style_timbre_disentangle", 0.0)),
            "lambda_style_dynamic_timbre_disentangle": float(
                hparams.get("lambda_style_dynamic_timbre_disentangle", 0.0)
            ),
            "lambda_dynamic_timbre_gate": float(hparams.get("lambda_dynamic_timbre_gate", 0.0)),
        },
        "history": history,
    }
    return summary


def main():
    args = parse_args()
    summary = run_smoke(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("PSEUDO_STYLE_SMOKE_OK")


if __name__ == "__main__":
    main()
