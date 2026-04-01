#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tasks.Conan.smoke_utils import (
    build_conan_training_task,
    build_pseudo_style_batch,
    build_pseudo_style_dataset,
    compare_model_state_dicts,
    is_finite_scalar,
    scalarize_value,
    scalarize_logs,
    select_speaker_batch_indices,
)
from utils.commons.hparams import hparams


DEFAULT_TRACKED_KEYS = (
    "total_loss",
    "style_trace_consistency",
    "style_timbre_dis",
    "style_dynamic_timbre_dis",
    "dynamic_timbre_gate_reg",
    "l1",
    "ssim",
    "fdiff",
    "uv",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run longer pseudo-style smoke with checkpoint resume validation.")
    parser.add_argument("--config", type=str, default="egs/conan_emformer.yaml")
    parser.add_argument("--binary_data_dir", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=6)
    parser.add_argument("--checkpoint_step", type=int, default=3)
    parser.add_argument("--speakers_per_batch", type=int, default=2)
    parser.add_argument("--items_per_speaker", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="smoke_runs/pseudo_style_long")
    parser.add_argument("--lambda_style_trace_consistency", type=float, default=0.1)
    parser.add_argument("--lambda_style_timbre_disentangle", type=float, default=0.05)
    parser.add_argument("--lambda_style_dynamic_timbre_disentangle", type=float, default=0.0)
    parser.add_argument("--lambda_dynamic_timbre_gate", type=float, default=0.0)
    parser.add_argument("--lambda_mel_adv", type=float, default=0.0)
    return parser.parse_args()


def _snapshot_loss(task, batch):
    model_training = task.model.training
    disc_training = task.mel_disc.training
    task.model.eval()
    task.mel_disc.eval()
    try:
        with torch.enable_grad():
            total_loss, logs = task._training_step(batch, 0, 0)
        scalar_logs = scalarize_logs(logs)
        return scalarize_value(total_loss), scalar_logs
    finally:
        task.model.train(model_training)
        task.mel_disc.train(disc_training)


def _optimizer_summary(optimizer):
    state_dict = optimizer.state_dict()
    return {
        "num_param_groups": len(state_dict.get("param_groups", [])),
        "num_state_entries": len(state_dict.get("state", {})),
    }


def _run_generator_step(task, optimizer, batch, step_idx, indices):
    optimizer.zero_grad(set_to_none=True)
    total_loss, logs = task._training_step(batch, step_idx, 0)
    if total_loss is None or not is_finite_scalar(total_loss):
        raise RuntimeError(f"Non-finite generator loss at step {step_idx}: {total_loss}")
    total_loss.backward()
    task.on_before_optimization(0)
    optimizer.step()
    scalar_logs = scalarize_logs(logs)
    return {
        "step": int(step_idx),
        "indices": [int(idx) for idx in indices],
        "pseudo_style_ids": [],
        "total_loss": scalarize_value(total_loss),
        **{key: scalar_logs.get(key) for key in DEFAULT_TRACKED_KEYS if key != "total_loss"},
    }


def _save_history(history, output_dir):
    json_path = output_dir / "history.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    csv_path = output_dir / "history.csv"
    fieldnames = sorted({key for item in history for key in item.keys()})
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    return json_path, csv_path


def _plot_history(history, output_dir):
    tracked = [key for key in DEFAULT_TRACKED_KEYS if any(item.get(key) is not None for item in history)]
    if len(tracked) <= 0:
        return None
    fig, axes = plt.subplots(len(tracked), 1, figsize=(10, 2.6 * len(tracked)), sharex=True)
    if len(tracked) == 1:
        axes = [axes]
    steps = [item["step"] for item in history]
    for axis, key in zip(axes, tracked):
        values = [item.get(key) for item in history]
        axis.plot(steps, values, marker="o")
        axis.set_ylabel(key)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    plot_path = output_dir / "loss_curves.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def run_long_smoke(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, num_styles = build_pseudo_style_dataset(
        config_path=args.config,
        binary_data_dir=args.binary_data_dir,
        extra_hparams={
            "lambda_style_trace_consistency": args.lambda_style_trace_consistency,
            "lambda_style_timbre_disentangle": args.lambda_style_timbre_disentangle,
            "lambda_style_dynamic_timbre_disentangle": args.lambda_style_dynamic_timbre_disentangle,
            "lambda_dynamic_timbre_gate": args.lambda_dynamic_timbre_gate,
            "lambda_mel_adv": args.lambda_mel_adv,
        },
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = build_conan_training_task(device)
    optimizer, _ = task.build_optimizer(task.model)

    checkpoint_step = max(1, min(int(args.checkpoint_step), int(args.num_steps) - 1))
    checkpoint_path = output_dir / "resume_ckpt.pt"
    history = []
    resume_validation = {}

    for step_idx in range(int(args.num_steps)):
        indices = select_speaker_batch_indices(
            dataset,
            step_idx,
            speakers_per_batch=int(args.speakers_per_batch),
            items_per_speaker=int(args.items_per_speaker),
        )
        batch = build_pseudo_style_batch(dataset, indices, device)
        history.append(_run_generator_step(task, optimizer, batch, step_idx, indices))
        task.global_step += 1

        if step_idx + 1 == checkpoint_step:
            torch.save(
                {
                    "model_state": task.model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "global_step": int(task.global_step),
                    "hparams_snapshot": {
                        "binary_data_dir": hparams["binary_data_dir"],
                        "num_styles": int(hparams.get("num_styles", num_styles)),
                    },
                },
                checkpoint_path,
            )

            next_indices = select_speaker_batch_indices(
                dataset,
                step_idx + 1,
                speakers_per_batch=int(args.speakers_per_batch),
                items_per_speaker=int(args.items_per_speaker),
            )
            next_batch = build_pseudo_style_batch(dataset, next_indices, device)

            resumed_task = build_conan_training_task(device, global_step=task.global_step)
            resumed_optimizer, _ = resumed_task.build_optimizer(resumed_task.model)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            resumed_task.model.load_state_dict(checkpoint["model_state"], strict=True)
            resumed_optimizer.load_state_dict(checkpoint["optimizer_state"])
            resumed_task.global_step = int(checkpoint["global_step"])

            state_max_abs_diff = compare_model_state_dicts(task.model, resumed_task.model)
            current_loss, current_logs = _snapshot_loss(task, next_batch)
            resumed_loss, resumed_logs = _snapshot_loss(resumed_task, next_batch)
            resumed_step_log = _run_generator_step(
                resumed_task,
                resumed_optimizer,
                next_batch,
                step_idx + 1,
                next_indices,
            )

            compare_keys = [
                "style_trace_consistency",
                "style_timbre_dis",
                "style_dynamic_timbre_dis",
                "dynamic_timbre_gate_reg",
                "l1",
                "fdiff",
                "uv",
            ]
            log_diffs = {
                key: abs(float(current_logs.get(key, 0.0)) - float(resumed_logs.get(key, 0.0)))
                for key in compare_keys
                if current_logs.get(key) is not None and resumed_logs.get(key) is not None
            }
            resume_validation = {
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_step": int(checkpoint_step),
                "state_max_abs_diff": float(state_max_abs_diff),
                "optimizer_summary_before": _optimizer_summary(optimizer),
                "optimizer_summary_after": _optimizer_summary(resumed_optimizer),
                "current_snapshot_loss": scalarize_value(current_loss),
                "resumed_snapshot_loss": scalarize_value(resumed_loss),
                "snapshot_loss_abs_diff": abs(float(current_loss) - float(resumed_loss)),
                "snapshot_log_abs_diff": log_diffs,
                "resumed_step_total_loss": scalarize_value(resumed_step_log["total_loss"]),
                "resume_next_indices": [int(idx) for idx in next_indices],
            }

    history_json_path, history_csv_path = _save_history(history, output_dir)
    plot_path = _plot_history(history, output_dir)

    summary = {
        "device": device,
        "binary_data_dir": hparams["binary_data_dir"],
        "num_styles": int(num_styles),
        "num_steps": int(args.num_steps),
        "checkpoint_step": int(checkpoint_step),
        "history_json": str(history_json_path),
        "history_csv": str(history_csv_path),
        "plot_path": str(plot_path) if plot_path is not None else None,
        "resume_validation": resume_validation,
        "style_losses_active": {
            "lambda_style_trace_consistency": float(hparams.get("lambda_style_trace_consistency", 0.0)),
            "lambda_style_timbre_disentangle": float(hparams.get("lambda_style_timbre_disentangle", 0.0)),
            "lambda_style_dynamic_timbre_disentangle": float(
                hparams.get("lambda_style_dynamic_timbre_disentangle", 0.0)
            ),
            "lambda_dynamic_timbre_gate": float(hparams.get("lambda_dynamic_timbre_gate", 0.0)),
        },
        "history": history,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


def main():
    args = parse_args()
    summary = run_long_smoke(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("PSEUDO_STYLE_LONG_SMOKE_OK")


if __name__ == "__main__":
    main()
