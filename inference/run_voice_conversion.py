#!/usr/bin/env python3
"""Batch runner for the single-reference Conan inference path."""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.Conan import StreamingVoiceConversion
from utils.audio.io import save_wav
from utils.commons.hparams import hparams, set_hparams


PUBLIC_CONTROL_KEYS = (
    "style_profile",
    "style_strength",
    "emotion",
    "emotion_id",
    "emotion_strength",
    "accent",
    "accent_id",
    "accent_strength",
    "arousal",
    "valence",
    "energy",
)

SPLIT_REFERENCE_KEYS = (
    "ref_timbre_wav",
    "ref_style_wav",
    "ref_dynamic_timbre_wav",
    "ref_emotion_wav",
    "ref_accent_wav",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch runner for the single-reference Conan inference path."
    )
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("-hp", "--hparams", type=str, default="")
    parser.add_argument("--pair_config", type=str, default="voice_conversion_config.json")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=50)
    return parser.parse_args()


class VoiceConversionRunner:
    """Run single-reference VC jobs from a JSON manifest."""

    def __init__(self, config_file="voice_conversion_config.json", hparams=None, *, output_dir=None):
        self.config_file = config_file
        self.config = self.load_config()
        self.output_dir = self.setup_output_dir(output_dir=output_dir)

        print("Initializing StreamingVoiceConversion engine...")
        self.engine = StreamingVoiceConversion(hparams)
        print("Engine initialized successfully.")

    def load_config(self):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file {self.config_file} not found")

        with open(self.config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        conversion_pairs = config.get("conversion_pairs", [])
        config["total_pairs"] = int(config.get("total_pairs", len(conversion_pairs)))
        print(f"Loaded {config['total_pairs']} conversion pairs")
        return config

    def setup_output_dir(self, *, output_dir=None):
        output_dir = output_dir or self.config.get("output_dir", "voice_conversion_output")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
        return output_dir

    @staticmethod
    def _has_distinct_split_refs(pair):
        ref_wav = pair.get("ref_wav")
        return any(pair.get(key) not in (None, "", ref_wav) for key in SPLIT_REFERENCE_KEYS)

    def _build_infer_input(self, pair):
        inp = {
            "ref_wav": pair["ref_wav"],
            "src_wav": pair["src_wav"],
        }
        for key in PUBLIC_CONTROL_KEYS:
            if pair.get(key) is not None:
                inp[key] = pair[key]
        if pair.get("allow_split_reference_inputs"):
            inp["allow_split_reference_inputs"] = True
            for key in SPLIT_REFERENCE_KEYS:
                if pair.get(key):
                    inp[key] = pair[key]
        elif self._has_distinct_split_refs(pair):
            for key in SPLIT_REFERENCE_KEYS:
                if pair.get(key):
                    inp[key] = pair[key]
        return inp

    def run_single_conversion(self, pair, pair_idx):
        try:
            inp = self._build_infer_input(pair)
            wav_pred, _ = self.engine.infer_once(inp)

            output_name = pair.get("output_name") or f"pair_{pair_idx:05d}.wav"
            output_path = os.path.join(self.output_dir, output_name)
            save_wav(wav_pred, output_path, hparams["audio_sample_rate"])
            return True, output_path
        except Exception as e:
            print(f"ERROR in pair {pair_idx}: {e}")
            return False, str(e)

    def run_all_conversions(self, start_idx=0, end_idx=None, batch_size=50):
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
        progress_file = os.path.join(self.output_dir, "conversion_progress.json")

        for i in range(start_idx, end_idx):
            pair = pairs[i]
            print(f"\nProcessing [{i + 1}/{total_pairs}] {pair.get('output_name', f'pair_{i:05d}.wav')}")
            print(f"  Ref: {os.path.basename(pair['ref_wav'])}")
            print(f"  Src: {os.path.basename(pair['src_wav'])}")

            success, result = self.run_single_conversion(pair, i)
            if success:
                successful += 1
                print(f"  Saved: {result}")
            else:
                failed += 1
                errors.append(f"Pair {i}: {result}")
                print(f"  Failed: {result}")

            processed = i - start_idx + 1
            if processed % batch_size == 0:
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
    )
    runner.run_all_conversions(
        start_idx=int(args.start_idx),
        end_idx=None if int(args.end_idx) < 0 else int(args.end_idx),
        batch_size=int(args.batch_size),
    )


if __name__ == "__main__":
    main()
