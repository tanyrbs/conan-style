#!/usr/bin/env python3
"""Research-only batch runner for Conan with external speaker embedding override."""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.Conan import StreamingVoiceConversion
from utils.audio.io import save_wav
from utils.commons.hparams import hparams, set_hparams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Research-only Conan batch runner with external speaker embeddings."
    )
    parser.add_argument("--config", type=str, default="egs/extract_spk_emb.yaml")
    parser.add_argument("--exp_name", type=str, default="render_nvae")
    parser.add_argument("--pair_config", type=str, default="voice_conversion_config.json")
    parser.add_argument("--emb_path", type=str, required=True, help="Path to a .npy speaker embedding array.")
    parser.add_argument("--output_dir", type=str, default="test_output_nvae_conan")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=50)
    return parser.parse_args()


class VoiceConversionRunner:
    """Run research-only VC jobs with an external speaker embedding override."""

    def __init__(self, config_file, hparams, *, output_dir):
        self.config_file = config_file
        self.config = self.load_config()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        print("Initializing StreamingVoiceConversion engine...")
        self.engine = StreamingVoiceConversion(hparams)
        print("Engine initialized successfully.")

    def load_config(self):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file {self.config_file} not found")
        with open(self.config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        config["total_pairs"] = int(config.get("total_pairs", len(config.get("conversion_pairs", []))))
        print(f"Loaded {config['total_pairs']} conversion pairs")
        return config

    def run_single_conversion(self, pair, pair_idx, *, spk_emb):
        try:
            inp = {
                "ref_wav": pair["ref_wav"],
                "src_wav": pair["src_wav"],
            }
            wav_pred, _ = self.engine.infer_once(inp, spk_emb=spk_emb)
            output_name = pair.get("output_name") or f"{pair.get('src_corpus', 'src')}_{pair.get('src_utt_id', pair_idx)}.wav"
            output_path = os.path.join(self.output_dir, output_name)
            save_wav(wav_pred, output_path, hparams["audio_sample_rate"])
            return True, output_path
        except Exception as e:
            print(f"ERROR in pair {pair_idx}: {e}")
            return False, str(e)

    def run_all_conversions(self, *, start_idx, end_idx, batch_size, embs):
        pairs = self.config["conversion_pairs"]
        total_pairs = len(pairs)
        if end_idx < 0 or end_idx > total_pairs:
            end_idx = total_pairs
        if end_idx <= start_idx:
            raise ValueError("end_idx must be greater than start_idx.")
        if len(embs) < end_idx:
            raise ValueError(f"Embedding count {len(embs)} is smaller than required end_idx {end_idx}.")

        successful = 0
        failed = 0
        errors = []
        start_time = time.time()
        progress_file = os.path.join(self.output_dir, "conversion_progress.json")

        for i in range(start_idx, end_idx):
            pair = pairs[i]
            print(f"\nProcessing [{i + 1}/{total_pairs}] {pair.get('output_name', f'pair_{i:05d}.wav')}")
            print(f"  Src: {os.path.basename(pair['src_wav'])}")
            success, result = self.run_single_conversion(pair, i, spk_emb=embs[i])

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

        total_processed = end_idx - start_idx
        total_time = time.time() - start_time
        final_report = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "total_processed": total_processed,
            "successful": successful,
            "failed": failed,
            "success_rate": 100.0 * successful / total_processed,
            "total_time_minutes": total_time / 60.0,
            "avg_time_per_file_seconds": total_time / total_processed,
            "output_directory": self.output_dir,
            "errors": errors,
            "timestamp": datetime.now().isoformat(),
            "research_only": True,
            "spk_embed_override": True,
        }
        with open(os.path.join(self.output_dir, "final_report.json"), "w", encoding="utf-8") as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)
        return final_report


def main():
    args = parse_args()
    set_hparams(config=args.config, exp_name=args.exp_name)
    embs = np.load(args.emb_path)
    runner = VoiceConversionRunner(args.pair_config, hparams, output_dir=args.output_dir)
    summary = runner.run_all_conversions(
        start_idx=int(args.start_idx),
        end_idx=int(args.end_idx),
        batch_size=int(args.batch_size),
        embs=embs,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
