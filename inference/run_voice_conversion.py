#!/usr/bin/env python3
"""Batch runner for the canonical Conan single-reference inference path."""

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.Conan import StreamingVoiceConversion
from inference.conan_request import build_mainline_request_input, has_distinct_split_reference_inputs
from utils.audio.io import save_wav
from utils.commons.hparams import hparams, set_hparams

CANONICAL_CONFIG = "egs/conan_mainline_infer.yaml"
CANONICAL_EXP_NAME = "Conan"
DEFAULT_PAIR_CONFIG = "inference/conan_single_reference_demo.example.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch runner for the canonical Conan single-reference inference path."
    )
    parser.add_argument("--config", type=str, default=CANONICAL_CONFIG)
    parser.add_argument("--exp_name", type=str, default=CANONICAL_EXP_NAME)
    parser.add_argument("-hp", "--hparams", type=str, default="")
    parser.add_argument("--pair_config", type=str, default=DEFAULT_PAIR_CONFIG)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Progress checkpoint interval only; this runner still executes model inference pair-by-pair on the canonical streaming path.",
    )
    parser.add_argument(
        "--allow_advanced_controls",
        action="store_true",
        help="Opt-in passthrough for research-only emotion/accent/arousal/valence/energy controls.",
    )
    return parser.parse_args()


class VoiceConversionRunner:
    """Run single-reference Conan VC jobs from a JSON manifest."""

    def __init__(
        self,
        config_file=DEFAULT_PAIR_CONFIG,
        hparams=None,
        *,
        output_dir=None,
        allow_advanced_controls=False,
    ):
        self.config_file = config_file
        self.config = self.load_config()
        self.output_dir = self.setup_output_dir(output_dir=output_dir)
        self.allow_advanced_controls = bool(
            allow_advanced_controls or self.config.get("allow_advanced_controls", False)
        )
        self._advanced_control_warning_emitted = False

        print("Initializing StreamingVoiceConversion engine...")
        self.engine = StreamingVoiceConversion(hparams)
        print("Engine initialized successfully.")

    def load_config(self):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file {self.config_file} not found")

        with open(self.config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        if config.get("allow_split_reference_inputs"):
            raise ValueError(
                "run_voice_conversion.py is the canonical single-reference Conan runner and "
                "does not accept allow_split_reference_inputs=true."
            )

        conversion_pairs = config.get("conversion_pairs", [])
        if not isinstance(conversion_pairs, list):
            raise TypeError("'conversion_pairs' must be a JSON list.")
        config["total_pairs"] = int(config.get("total_pairs", len(conversion_pairs)))
        if config["total_pairs"] != len(conversion_pairs):
            warnings.warn(
                f"Manifest total_pairs={config['total_pairs']} does not match actual conversion_pairs={len(conversion_pairs)}; using the actual list length for execution.",
                stacklevel=2,
            )
            config["total_pairs"] = len(conversion_pairs)
        print(f"Loaded {config['total_pairs']} conversion pairs")
        return config

    def setup_output_dir(self, *, output_dir=None):
        output_dir = output_dir or self.config.get("output_dir", "voice_conversion_output")
        os.makedirs(output_dir, exist_ok=True)
        resolved_output_dir = os.path.abspath(output_dir)
        print(f"Output directory: {resolved_output_dir}")
        return resolved_output_dir

    def _validate_conversion_pair(self, pair, pair_idx):
        if not isinstance(pair, dict):
            raise TypeError(f"Pair {pair_idx} must be a JSON object, got {type(pair).__name__}.")
        for key in ("ref_wav", "src_wav"):
            value = pair.get(key)
            if not value:
                raise ValueError(f"Pair {pair_idx} is missing required field '{key}'.")
            if not os.path.exists(str(value)):
                raise FileNotFoundError(f"Pair {pair_idx} references missing {key}: {value}")

    def _resolve_output_path(self, output_name, pair_idx):
        raw_name = str(output_name or f"pair_{pair_idx:05d}.wav").strip()
        if not raw_name:
            raw_name = f"pair_{pair_idx:05d}.wav"
        normalized = os.path.normpath(raw_name)
        if os.path.isabs(normalized) or normalized.startswith(".."):
            sanitized_name = f"pair_{pair_idx:05d}.wav"
            warnings.warn(
                f"Unsafe output_name='{raw_name}' for pair {pair_idx}; falling back to '{sanitized_name}'.",
                stacklevel=2,
            )
            normalized = sanitized_name
        if normalized in ("", "."):
            normalized = f"pair_{pair_idx:05d}.wav"
        if not os.path.splitext(normalized)[1]:
            normalized = f"{normalized}.wav"
        output_path = os.path.abspath(os.path.join(self.output_dir, normalized))
        if os.path.commonpath([self.output_dir, output_path]) != self.output_dir:
            raise ValueError(
                f"Resolved output path escapes output_dir for pair {pair_idx}: {output_path}"
            )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return normalized, output_path

    def _build_infer_input(self, pair):
        if pair.get("allow_split_reference_inputs") or has_distinct_split_reference_inputs(pair):
            raise ValueError(
                "run_voice_conversion.py is the canonical single-reference Conan runner and "
                "does not accept split reference inputs. Use only ref_wav on the mainline path."
            )
        allow_advanced_controls = bool(
            pair.get("allow_advanced_controls", self.allow_advanced_controls)
        )
        inp, ignored_advanced_keys = build_mainline_request_input(
            pair,
            allow_advanced_controls=allow_advanced_controls,
        )
        if ignored_advanced_keys and not self._advanced_control_warning_emitted:
            warnings.warn(
                "Ignoring advanced non-mainline control keys in run_voice_conversion.py: "
                f"{ignored_advanced_keys}. Set allow_advanced_controls=true in the manifest "
                "only for explicit research/ablation runs.",
                stacklevel=2,
            )
            self._advanced_control_warning_emitted = True
        return inp

    def run_single_conversion(self, pair, pair_idx):
        try:
            self._validate_conversion_pair(pair, pair_idx)
            inp = self._build_infer_input(pair)
            wav_pred, _ = self.engine.infer_once(inp)

            output_name, output_path = self._resolve_output_path(pair.get("output_name"), pair_idx)
            save_wav(wav_pred, output_path, hparams["audio_sample_rate"])
            infer_metadata = getattr(self.engine, "last_infer_metadata", None)
            if infer_metadata is not None:
                meta_path = os.path.splitext(output_path)[0] + ".json"
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "pair_index": int(pair_idx),
                            "output_name": output_name,
                            "model_input": inp,
                            "infer_metadata": infer_metadata,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            return True, {"output_path": output_path, "output_name": output_name}
        except Exception as e:
            print(f"ERROR in pair {pair_idx}: {e}")
            return False, {
                "message": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc(),
            }

    def run_all_conversions(self, start_idx=0, end_idx=None, batch_size=50):
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer.")
        progress_flush_interval = int(batch_size)
        if progress_flush_interval > 1:
            print(
                "Note: run_voice_conversion.py uses --batch_size only for periodic progress writes; "
                "the canonical streaming inference loop remains sequential per pair."
            )
        pairs = self.config["conversion_pairs"]
        total_pairs = len(pairs)
        if end_idx is None or int(end_idx) < 0:
            end_idx = total_pairs
        if end_idx <= start_idx:
            raise ValueError("end_idx must be greater than start_idx.")

        print(f"Running conversions from {start_idx} to {end_idx} (total: {end_idx - start_idx})")

        successful = 0
        failed = 0
        start_time = time.time()
        errors = []
        error_details = []
        progress_file = os.path.join(self.output_dir, "conversion_progress.json")

        for i in range(start_idx, end_idx):
            pair = pairs[i]
            if isinstance(pair, dict):
                pair_name = pair.get("output_name", f"pair_{i:05d}.wav")
                ref_name = os.path.basename(str(pair.get("ref_wav", "")))
                src_name = os.path.basename(str(pair.get("src_wav", "")))
            else:
                pair_name = f"pair_{i:05d}.wav"
                ref_name = "<invalid-pair>"
                src_name = "<invalid-pair>"
            print(f"\nProcessing [{i + 1}/{total_pairs}] {pair_name}")
            print(f"  Ref: {ref_name}")
            print(f"  Src: {src_name}")

            success, result = self.run_single_conversion(pair, i)
            if success:
                successful += 1
                print(f"  Saved: {result['output_path']}")
            else:
                failed += 1
                error_message = result["message"] if isinstance(result, dict) else str(result)
                errors.append(f"Pair {i}: {error_message}")
                error_details.append(
                    {
                        "pair_index": int(i),
                        "output_name": pair.get("output_name") if isinstance(pair, dict) else None,
                        **(result if isinstance(result, dict) else {"message": error_message}),
                    }
                )
                print(f"  Failed: {error_message}")

            processed = i - start_idx + 1
            if processed % progress_flush_interval == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / processed
                remaining = (end_idx - i - 1) * avg_time
                progress = {
                    "processed": processed,
                    "total": end_idx - start_idx,
                    "successful": successful,
                    "failed": failed,
                    "elapsed_time": elapsed,
                    "estimated_remaining": remaining,
                    "current_batch_end": i,
                    "errors": errors,
                }
                with open(progress_file, "w", encoding="utf-8") as f:
                    json.dump(progress, f, ensure_ascii=False, indent=2)

                print("\n--- Progress Update ---")
                print(f"Processed: {processed}/{end_idx - start_idx}")
                print(f"Success rate: {successful}/{processed} ({100 * successful / processed:.1f}%)")
                print(f"Elapsed: {elapsed / 60:.1f}min, Remaining: {remaining / 60:.1f}min")
                print(f"Avg time per file: {avg_time:.1f}s")

        total_processed = end_idx - start_idx
        total_time = time.time() - start_time
        print("\n=== Final Summary ===")
        print(f"Total processed: {total_processed}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {100 * successful / total_processed:.1f}%")
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Average time per file: {total_time / total_processed:.1f}s")
        print(f"Output directory: {self.output_dir}")

        final_report = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "total_processed": total_processed,
            "successful": successful,
            "failed": failed,
            "success_rate": 100 * successful / total_processed,
            "total_time_minutes": total_time / 60,
            "avg_time_per_file_seconds": total_time / total_processed,
            "output_directory": self.output_dir,
            "errors": errors,
            "error_details": error_details,
            "timestamp": datetime.now().isoformat(),
            "streaming_impl": getattr(self.engine, "streaming_impl", None),
        }
        with open(os.path.join(self.output_dir, "final_report.json"), "w", encoding="utf-8") as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)

        if failed > 0:
            print("\nErrors encountered:")
            for error in errors[:10]:
                print(f"  {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")


def main():
    args = parse_args()
    set_hparams(config=args.config, exp_name=args.exp_name, hparams_str=args.hparams)
    runner = VoiceConversionRunner(
        args.pair_config,
        hparams=hparams,
        output_dir=args.output_dir or None,
        allow_advanced_controls=bool(args.allow_advanced_controls),
    )
    runner.run_all_conversions(
        start_idx=int(args.start_idx),
        end_idx=None if int(args.end_idx) < 0 else int(args.end_idx),
        batch_size=int(args.batch_size),
    )


if __name__ == "__main__":
    main()
