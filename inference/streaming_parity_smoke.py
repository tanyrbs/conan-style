#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.Conan import StreamingVoiceConversion
from utils.commons.hparams import hparams, set_hparams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Conan mainline online prefix streaming vs offline full-pass inference."
    )
    parser.add_argument("--config", type=str, default="egs/conan_mainline_infer.yaml")
    parser.add_argument("--exp_name", type=str, default="Conan")
    parser.add_argument("--src_wav", type=str, required=True)
    parser.add_argument("--ref_wav", type=str, required=True)
    parser.add_argument("--style_profile", type=str, default="strong_style")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--max_mel_l1", type=float, default=None)
    parser.add_argument("--max_mel_l1_tail", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_hparams(
        config=args.config,
        exp_name=args.exp_name,
        print_hparams=False,
    )
    engine = StreamingVoiceConversion(hparams)
    result = engine.infer_parity_once(
        {
            "src_wav": args.src_wav,
            "ref_wav": args.ref_wav,
            "style_profile": args.style_profile,
        }
    )
    summary = result["summary"]
    metrics = summary.get("metrics", {})
    contract = summary.get("mainline_contract", {})
    mel_l1 = metrics.get("mel_l1_full")
    mel_l1_tail = metrics.get("mel_l1_tail", mel_l1)
    if args.max_mel_l1 is not None and mel_l1 is not None and float(mel_l1) > float(args.max_mel_l1):
        raise AssertionError(f"mel_l1_full too large: {mel_l1} > {args.max_mel_l1}")
    if args.max_mel_l1_tail is not None and mel_l1_tail is not None and float(mel_l1_tail) > float(args.max_mel_l1_tail):
        raise AssertionError(f"mel_l1_tail too large: {mel_l1_tail} > {args.max_mel_l1_tail}")
    if not bool(contract.get("query_anchor_split_applied", False)):
        raise AssertionError("mainline_contract.query_anchor_split_applied is false.")
    if not bool(contract.get("dynamic_timbre_style_context_owner_safe", False)):
        raise AssertionError("mainline_contract.dynamic_timbre_style_context_owner_safe is false.")
    if bool(contract.get("global_timbre_to_pitch_applied_online", False)):
        raise AssertionError("mainline_contract.global_timbre_to_pitch_applied_online unexpectedly true.")
    if bool(contract.get("global_timbre_to_pitch_applied_offline", False)):
        raise AssertionError("mainline_contract.global_timbre_to_pitch_applied_offline unexpectedly true.")
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("STREAMING_PARITY_SMOKE_OK")


if __name__ == "__main__":
    main()
