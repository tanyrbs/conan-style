#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.Conan.decoder_style_adapter import ConanDecoderStyleAdapter
from modules.Conan.decoder_style_bundle import build_decoder_style_bundle


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke-test decoder style adapter effective-signal filtering and owner hierarchy."
    )
    parser.add_argument("--hidden_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--effective_signal_epsilon", type=float, default=1.0e-4)
    parser.add_argument(
        "--output_path",
        type=str,
        default="smoke_runs/decoder_style_adapter_contract_smoke.json",
    )
    return parser.parse_args()


def _assert_close(name, actual, expected, tol=1.0e-8):
    if not torch.allclose(actual, expected, atol=tol, rtol=0.0):
        raise AssertionError(f"{name}: tensors differ beyond tolerance {tol}.")


def _scalarize(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return {
            "shape": list(value.shape),
            "mean": float(value.detach().float().mean().cpu().item()),
            "max_abs": float(value.detach().float().abs().max().cpu().item()),
        }
    return value


def run_smoke(args):
    torch.manual_seed(7)
    hidden_size = int(args.hidden_size)
    batch_size = int(args.batch_size)
    seq_len = int(args.seq_len)
    eps = float(args.effective_signal_epsilon)

    adapter = ConanDecoderStyleAdapter(
        hidden_size,
        effective_signal_epsilon=eps,
        gate_bias=-0.25,
    )
    adapter.eval()

    hidden = torch.randn(batch_size, seq_len, hidden_size)
    nonpadding = torch.ones(batch_size, seq_len, 1)

    near_zero_value = eps * 0.1
    zero_like_bundle = build_decoder_style_bundle(
        global_timbre_anchor=torch.full((batch_size, 1, hidden_size), near_zero_value),
        global_timbre_anchor_runtime=torch.full((batch_size, 1, hidden_size), near_zero_value),
        global_style_summary=torch.full((batch_size, 1, hidden_size), near_zero_value),
        M_style=torch.full((batch_size, seq_len, hidden_size), near_zero_value),
        M_timbre=torch.full((batch_size, seq_len, hidden_size), near_zero_value),
        bundle_variant="mainline_full",
        effective_signal_epsilon=eps,
    )
    zero_conditioned, zero_meta = adapter.forward_stage(
        "late",
        hidden.clone(),
        style_bundle=zero_like_bundle,
        nonpadding=nonpadding,
    )
    _assert_close("near_zero_hard_no_op", zero_conditioned, hidden)
    if bool(zero_meta.get("applied", True)):
        raise AssertionError("near_zero_hard_no_op: adapter reported applied=True for near-zero bundle.")
    if bool(zero_meta.get("local_style_owner_present", True)):
        raise AssertionError("near_zero_hard_no_op: local_style_owner_present should be false.")
    if bool(zero_meta.get("global_style_present", True)):
        raise AssertionError("near_zero_hard_no_op: global_style_present should be false.")

    nonzero_style = torch.randn(batch_size, seq_len, hidden_size) * 0.5
    nonzero_global = torch.randn(batch_size, 1, hidden_size) * 0.5
    owner_bundle = build_decoder_style_bundle(
        global_style_summary=nonzero_global,
        M_style=nonzero_style,
        bundle_variant="mainline_full",
        effective_signal_epsilon=eps,
    )
    _, owner_meta = adapter.forward_stage(
        "late",
        hidden.clone(),
        style_bundle=owner_bundle,
        nonpadding=nonpadding,
    )
    if not bool(owner_meta.get("local_style_owner_present", False)):
        raise AssertionError("owner_hierarchy: local_style_owner_present should be true when M_style is active.")
    if not bool(owner_meta.get("global_style_skipped_due_to_local_owner", False)):
        raise AssertionError("owner_hierarchy: global summary should be skipped when local owner exists.")
    if bool(owner_meta.get("global_style_applied", False)):
        raise AssertionError("owner_hierarchy: global summary should not be applied alongside active local owner.")
    if not bool(owner_meta.get("style_trace_applied", False)):
        raise AssertionError("owner_hierarchy: style_trace branch should be applied for active M_style.")
    if float(owner_meta.get("effective_signal_epsilon", 0.0)) != eps:
        raise AssertionError("owner_hierarchy: decoded metadata should preserve effective_signal_epsilon.")

    fallback_bundle = build_decoder_style_bundle(
        global_style_summary=nonzero_global,
        bundle_variant="global_only",
        effective_signal_epsilon=eps,
    )

    for bundle_name, bundle in [
        ("zero_like_bundle", zero_like_bundle),
        ("owner_bundle", owner_bundle),
        ("fallback_bundle", fallback_bundle),
    ]:
        if float(bundle.get("effective_signal_epsilon", -1.0)) != eps:
            raise AssertionError(
                f"{bundle_name}: effective_signal_epsilon {bundle.get('effective_signal_epsilon')} != {eps}"
            )
    _, fallback_meta = adapter.forward_stage(
        "late",
        hidden.clone(),
        style_bundle=fallback_bundle,
        nonpadding=nonpadding,
    )
    if bool(fallback_meta.get("local_style_owner_present", True)):
        raise AssertionError("global_fallback: local_style_owner_present should be false without local style owner.")
    if not bool(fallback_meta.get("global_style_applied", False)):
        raise AssertionError("global_fallback: global summary should be applied as fallback.")
    if not bool(fallback_meta.get("global_style_candidate", False)):
        raise AssertionError("global_fallback: fallback branch should be marked as candidate.")

    summary = {
        "effective_signal_epsilon": eps,
        "near_zero_hard_no_op": {
            "applied": bool(zero_meta.get("applied", False)),
            "local_style_owner_present": bool(zero_meta.get("local_style_owner_present", False)),
            "global_style_present": bool(zero_meta.get("global_style_present", False)),
            "delta_max_abs": _scalarize((zero_conditioned - hidden).abs().max()),
        },
        "owner_hierarchy": {
            "local_style_owner_present": bool(owner_meta.get("local_style_owner_present", False)),
            "global_style_skipped_due_to_local_owner": bool(
                owner_meta.get("global_style_skipped_due_to_local_owner", False)
            ),
            "global_style_applied": bool(owner_meta.get("global_style_applied", False)),
            "style_trace_applied": bool(owner_meta.get("style_trace_applied", False)),
        },
        "global_fallback": {
            "local_style_owner_present": bool(fallback_meta.get("local_style_owner_present", False)),
            "global_style_applied": bool(fallback_meta.get("global_style_applied", False)),
            "global_style_candidate": bool(fallback_meta.get("global_style_candidate", False)),
        },
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
    print("DECODER_STYLE_ADAPTER_CONTRACT_SMOKE_OK")


if __name__ == "__main__":
    main()
