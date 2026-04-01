from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

from modules.Conan.decoder_style_bundle import (
    canonicalize_decoder_style_bundle,
    ensure_decoder_style_bundle_respects_timing,
)


def _is_sequence_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.dim() == 3 and value.size(1) > 0


class ConanDecoderStyleAdapter(nn.Module):
    """
    Lightweight decoder-side style/timbre adapter.

    Design goals:
    - keep timing ownership outside the style module
    - apply already-aligned style/timbre states only on decoder-side hidden states
    - split stage responsibilities:
      - mid: slow style trace + dynamic timbre
      - late: global style summary + slow/fast style trace + dynamic timbre
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        gate_hidden: Optional[int] = None,
        stage_splits=None,
        global_timbre_scale_early: float = 0.18,
        global_timbre_scale_mid: float = 0.12,
        global_timbre_scale_late: float = 0.08,
        global_style_scale: float = 0.25,
        slow_style_scale: float = 0.55,
        style_trace_scale: float = 0.9,
        dynamic_timbre_scale: float = 0.9,
        dynamic_timbre_scale_mid: Optional[float] = None,
        dynamic_timbre_scale_late: Optional[float] = None,
        dynamic_timbre_late_no_style_scale: float = 0.25,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.stage_splits = stage_splits
        self.global_timbre_scale_early = float(global_timbre_scale_early)
        self.global_timbre_scale_mid = float(global_timbre_scale_mid)
        self.global_timbre_scale_late = float(global_timbre_scale_late)
        self.global_style_scale = float(global_style_scale)
        self.slow_style_scale = float(slow_style_scale)
        self.style_trace_scale = float(style_trace_scale)
        self.dynamic_timbre_scale = float(dynamic_timbre_scale)
        self.dynamic_timbre_scale_mid = float(
            dynamic_timbre_scale if dynamic_timbre_scale_mid is None else dynamic_timbre_scale_mid
        )
        self.dynamic_timbre_scale_late = float(
            (self.dynamic_timbre_scale_mid * 0.6)
            if dynamic_timbre_scale_late is None
            else dynamic_timbre_scale_late
        )
        self.dynamic_timbre_late_no_style_scale = float(dynamic_timbre_late_no_style_scale)
        gate_hidden = max(1, int(gate_hidden or hidden_size))

        self.global_timbre_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.global_style_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.slow_style_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.style_trace_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.dynamic_timbre_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())

        self.global_timbre_gate = self._build_gate(hidden_size * 2, gate_hidden, gate_bias)
        self.global_style_gate = self._build_gate(hidden_size * 2, gate_hidden, gate_bias)
        self.slow_style_gate = self._build_gate(hidden_size * 2, gate_hidden, gate_bias)
        self.style_trace_gate = self._build_gate(hidden_size * 2, gate_hidden, gate_bias)
        self.dynamic_timbre_gate = self._build_gate(hidden_size * 2, gate_hidden, gate_bias)

        self.stage_norm = nn.LayerNorm(hidden_size)
        self._gate_bias = float(gate_bias)
        self._gate_bias_version = 0

    @staticmethod
    def _build_gate(in_dim: int, hidden_dim: int, bias: float):
        gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_dim // 2),
            nn.Sigmoid(),
        )
        nn.init.zeros_(gate[2].weight)
        nn.init.constant_(gate[2].bias, float(bias))
        return gate

    def set_gate_bias(self, bias: float):
        bias = float(bias)
        if self._gate_bias == bias:
            return
        for gate in (
            self.global_timbre_gate,
            self.global_style_gate,
            self.slow_style_gate,
            self.style_trace_gate,
            self.dynamic_timbre_gate,
        ):
            if isinstance(gate, nn.Sequential) and len(gate) >= 3:
                nn.init.constant_(gate[2].bias, bias)
        self._gate_bias = bias
        self._gate_bias_version += 1

    def gate_bias_state(self):
        return {"gate_bias": self._gate_bias, "gate_bias_version": self._gate_bias_version}

    def resolve_stage_end_indices(self, num_blocks: int) -> dict[int, str]:
        num_blocks = max(1, int(num_blocks))
        configured = self.stage_splits
        if isinstance(configured, str):
            configured = [item.strip() for item in configured.split(",") if item.strip()]
        if isinstance(configured, (list, tuple)) and len(configured) == 3:
            try:
                early_blocks, mid_blocks, late_blocks = [max(0, int(v)) for v in configured]
            except (TypeError, ValueError):
                early_blocks = mid_blocks = late_blocks = 0
            if early_blocks + mid_blocks + late_blocks == num_blocks and num_blocks > 0:
                mapping = {}
                cursor = 0
                if early_blocks > 0:
                    mapping[cursor + early_blocks - 1] = "early"
                    cursor += early_blocks
                if mid_blocks > 0:
                    mapping[cursor + mid_blocks - 1] = "mid"
                    cursor += mid_blocks
                if late_blocks > 0:
                    mapping[cursor + late_blocks - 1] = "late"
                return mapping

        if num_blocks == 1:
            return {0: "late"}
        if num_blocks == 2:
            return {0: "mid", 1: "late"}
        early_blocks = max(1, num_blocks // 4)
        late_blocks = max(1, num_blocks // 4)
        mid_blocks = max(1, num_blocks - early_blocks - late_blocks)
        while early_blocks + mid_blocks + late_blocks > num_blocks and mid_blocks > 1:
            mid_blocks -= 1
        while early_blocks + mid_blocks + late_blocks < num_blocks:
            mid_blocks += 1
        return {
            early_blocks - 1: "early",
            early_blocks + mid_blocks - 1: "mid",
            num_blocks - 1: "late",
        }

    @staticmethod
    def _normalize_nonpadding(nonpadding, reference):
        if nonpadding is None:
            return None
        if not isinstance(nonpadding, torch.Tensor):
            return None
        if nonpadding.dim() == 2:
            nonpadding = nonpadding.unsqueeze(-1)
        if nonpadding.dim() != 3 or tuple(nonpadding.shape[:2]) != tuple(reference.shape[:2]):
            return None
        return nonpadding.to(device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _expand_single(sequence: Optional[torch.Tensor], target_len: int):
        if not isinstance(sequence, torch.Tensor):
            return None
        if sequence.dim() == 2:
            sequence = sequence.unsqueeze(1)
        if sequence.dim() != 3:
            return None
        if sequence.size(1) == target_len:
            return sequence
        if sequence.size(1) == 1:
            return sequence.expand(-1, target_len, -1)
        return None

    @staticmethod
    def _apply_mask(value: torch.Tensor, nonpadding_mask: Optional[torch.Tensor]):
        if nonpadding_mask is None:
            return value
        return value * nonpadding_mask

    def _apply_branch(
        self,
        hidden_btc: torch.Tensor,
        branch: Optional[torch.Tensor],
        *,
        projector: nn.Module,
        gate: nn.Module,
        scale: float,
        nonpadding_mask: Optional[torch.Tensor],
    ):
        if not isinstance(branch, torch.Tensor) or float(scale) == 0.0:
            return hidden_btc, None, None
        branch = projector(branch)
        branch = self._apply_mask(branch, nonpadding_mask)
        branch_gate = gate(torch.cat([hidden_btc, branch], dim=-1))
        branch_gate = self._apply_mask(branch_gate, nonpadding_mask)
        hidden_btc = hidden_btc + branch_gate * branch * float(scale)
        return hidden_btc, branch, branch_gate

    def forward_stage(
        self,
        stage_name: str,
        hidden_btc: torch.Tensor,
        *,
        style_bundle: Optional[Mapping[str, Any]] = None,
        nonpadding: Optional[torch.Tensor] = None,
    ):
        stage_name = str(stage_name or "late").strip().lower() or "late"
        style_bundle = canonicalize_decoder_style_bundle(style_bundle)
        if not isinstance(style_bundle, Mapping):
            return hidden_btc, {"stage_name": stage_name, "applied": False}
        ensure_decoder_style_bundle_respects_timing(style_bundle)

        nonpadding_mask = self._normalize_nonpadding(nonpadding, hidden_btc)
        global_timbre_anchor = self._expand_single(
            style_bundle.get("global_timbre_anchor_runtime", style_bundle.get("global_timbre")),
            hidden_btc.size(1),
        )
        slow_style_trace = style_bundle.get("slow_style_trace")
        style_trace = style_bundle.get("M_style", style_bundle.get("style_trace"))
        dynamic_timbre = style_bundle.get("M_timbre", style_bundle.get("dynamic_timbre"))
        global_style_summary = self._expand_single(
            style_bundle.get("global_style_summary"),
            hidden_btc.size(1),
        )

        global_timbre_scale = {
            "early": self.global_timbre_scale_early,
            "mid": self.global_timbre_scale_mid,
            "late": self.global_timbre_scale_late,
        }.get(stage_name, 0.0)
        apply_global_timbre = global_timbre_scale > 0.0
        apply_global = stage_name == "late"
        apply_slow_style = stage_name in {"mid", "late"}
        apply_style = stage_name == "late"
        style_owner_present = bool(
            global_style_summary is not None
            or _is_sequence_tensor(slow_style_trace)
            or _is_sequence_tensor(style_trace)
        )
        dynamic_timbre_scale = {
            "mid": self.dynamic_timbre_scale_mid,
            "late": self.dynamic_timbre_scale_late,
        }.get(stage_name, 0.0)
        if stage_name == "late" and not style_owner_present:
            dynamic_timbre_scale = dynamic_timbre_scale * self.dynamic_timbre_late_no_style_scale
        apply_timbre = dynamic_timbre_scale > 0.0

        conditioned = hidden_btc
        metadata = {
            "stage_name": stage_name,
            "applied": False,
            "style_owner_present": bool(style_owner_present),
            "dynamic_timbre_scale_used": float(dynamic_timbre_scale),
        }

        if apply_global_timbre and global_timbre_anchor is not None:
            conditioned, global_timbre_ctx, global_timbre_gate = self._apply_branch(
                conditioned,
                global_timbre_anchor,
                projector=self.global_timbre_proj,
                gate=self.global_timbre_gate,
                scale=global_timbre_scale,
                nonpadding_mask=nonpadding_mask,
            )
            metadata["global_timbre_ctx"] = global_timbre_ctx
            metadata["global_timbre_gate"] = global_timbre_gate

        if apply_global and global_style_summary is not None:
            conditioned, global_ctx, global_gate = self._apply_branch(
                conditioned,
                global_style_summary,
                projector=self.global_style_proj,
                gate=self.global_style_gate,
                scale=self.global_style_scale,
                nonpadding_mask=nonpadding_mask,
            )
            metadata["global_style_ctx"] = global_ctx
            metadata["global_style_gate"] = global_gate

        if apply_slow_style and _is_sequence_tensor(slow_style_trace):
            conditioned, slow_style_ctx, slow_style_gate = self._apply_branch(
                conditioned,
                slow_style_trace,
                projector=self.slow_style_proj,
                gate=self.slow_style_gate,
                scale=self.slow_style_scale,
                nonpadding_mask=nonpadding_mask,
            )
            metadata["slow_style_ctx"] = slow_style_ctx
            metadata["slow_style_gate"] = slow_style_gate

        if apply_style and _is_sequence_tensor(style_trace):
            conditioned, style_ctx, style_gate = self._apply_branch(
                conditioned,
                style_trace,
                projector=self.style_trace_proj,
                gate=self.style_trace_gate,
                scale=self.style_trace_scale,
                nonpadding_mask=nonpadding_mask,
            )
            metadata["style_trace_ctx"] = style_ctx
            metadata["style_trace_gate"] = style_gate

        if apply_timbre and _is_sequence_tensor(dynamic_timbre):
            conditioned, timbre_ctx, timbre_gate = self._apply_branch(
                conditioned,
                dynamic_timbre,
                projector=self.dynamic_timbre_proj,
                gate=self.dynamic_timbre_gate,
                scale=dynamic_timbre_scale,
                nonpadding_mask=nonpadding_mask,
            )
            metadata["dynamic_timbre_ctx"] = timbre_ctx
            metadata["dynamic_timbre_gate"] = timbre_gate

        delta = conditioned - hidden_btc
        applied = bool(delta.abs().sum().item() > 0)
        if applied:
            conditioned = self.stage_norm(conditioned)
            conditioned = self._apply_mask(conditioned, nonpadding_mask)
        metadata["applied"] = applied
        metadata["delta"] = delta
        return conditioned, metadata


__all__ = ["ConanDecoderStyleAdapter"]
