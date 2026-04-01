#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tasks.Conan.mainline_train_prep import run_prep as run_train_prep
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a minimal Conan mainline CPU dry run before the first real-dataset training launch."
    )
    parser.add_argument("--config", type=str, default="egs/conan_emformer.yaml")
    parser.add_argument(
        "--binary_data_dir",
        type=str,
        default=default_smoke_binary_data_dir(),
        help="Smoke binary dataset dir. Defaults to the first available built-in smoke dataset.",
    )
    parser.add_argument("--num_steps", type=int, default=2)
    parser.add_argument("--speakers_per_batch", type=int, default=2)
    parser.add_argument("--items_per_speaker", type=int, default=2)
    parser.add_argument(
        "--output_path",
        type=str,
        default="smoke_runs/mainline_cpu_dry_run.json",
    )
    return parser.parse_args()


def _run_train_steps(args):
    if not args.binary_data_dir:
        raise ValueError("No smoke binary_data_dir was provided and no default smoke dataset was found.")
    dataset, num_styles = build_pseudo_style_dataset(
        config_path=args.config,
        binary_data_dir=args.binary_data_dir,
        extra_hparams={"lambda_mel_adv": 0.0},
    )
    batch_shape = resolve_smoke_batch_shape(
        dataset,
        speakers_per_batch=int(args.speakers_per_batch),
        items_per_speaker=int(args.items_per_speaker),
    )
    task = build_conan_training_task("cpu")
    optimizer, _ = task.build_optimizer(task.model)
    history = []
    for step_idx in range(int(args.num_steps)):
        indices = select_speaker_batch_indices(
            dataset,
            step_idx,
            speakers_per_batch=batch_shape["speakers_per_batch"],
            items_per_speaker=batch_shape["items_per_speaker"],
        )
        batch = build_pseudo_style_batch(dataset, indices, "cpu")
        optimizer.zero_grad(set_to_none=True)
        total_loss, logs = task._training_step(batch, step_idx, 0)
        if total_loss is None or not is_finite_scalar(total_loss):
            raise RuntimeError(f"Non-finite generator loss at step {step_idx}: {total_loss}")
        total_loss.backward()
        task.on_before_optimization(0)
        optimizer.step()
        scalar_logs = scalarize_logs(logs)
        history.append(
            {
                "step": int(step_idx),
                "indices": [int(idx) for idx in indices],
                "total_loss": scalarize_value(total_loss),
                "style_trace_consistency": scalar_logs.get("style_trace_consistency"),
                "output_identity_cosine": scalar_logs.get("output_identity_cosine"),
                "dynamic_timbre_budget": scalar_logs.get("dynamic_timbre_budget"),
                "decoder_late_owner": scalar_logs.get("decoder_late_owner"),
                "l1": scalar_logs.get("l1"),
                "ssim": scalar_logs.get("ssim"),
                "fdiff": scalar_logs.get("fdiff"),
                "uv": scalar_logs.get("uv"),
            }
        )
        task.global_step += 1
    return {
        "device": "cpu",
        "binary_data_dir": args.binary_data_dir,
        "num_styles": int(num_styles),
        "batch_shape": batch_shape,
        "num_steps": int(args.num_steps),
        "history": history,
    }


def run_cpu_dry_run(args):
    prep_summary = run_train_prep(
        argparse.Namespace(
            config=args.config,
            output_path=str(Path(args.output_path).with_name("mainline_train_prep.from_dry_run.json")),
        )
    )
    train_summary = _run_train_steps(args)
    summary = {
        "config": args.config,
        "prep": prep_summary,
        "train": train_summary,
        "ready": bool(prep_summary.get("ready", False)),
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main():
    args = parse_args()
    summary = run_cpu_dry_run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["ready"]:
        raise SystemExit("MAINLINE_CPU_DRY_RUN_NOT_READY")
    print("MAINLINE_CPU_DRY_RUN_OK")


if __name__ == "__main__":
    main()
