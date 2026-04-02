from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


def expand_anchor_sequence(anchor: torch.Tensor, length: int) -> torch.Tensor:
    if anchor.dim() == 2:
        anchor = anchor.unsqueeze(1)
    if anchor.size(1) == length:
        return anchor
    return anchor.expand(anchor.size(0), length, anchor.size(-1))


def spherical_interpolate(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a_norm = F.normalize(a, dim=-1, eps=eps)
    b_norm = F.normalize(b, dim=-1, eps=eps)
    dot = (a_norm * b_norm).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega).clamp_min(eps)
    t = t.clamp(0.0, 1.0)
    scale_a = torch.sin((1.0 - t) * omega) / sin_omega
    scale_b = torch.sin(t * omega) / sin_omega
    return scale_a * a + scale_b * b


class GlobalTimbreMemory(nn.Module):
    def __init__(self, hidden_size: int, num_facets: int = 8, facet_hidden: Optional[int] = None):
        super().__init__()
        facet_hidden = int(hidden_size if facet_hidden is None else facet_hidden)
        self.hidden_size = int(hidden_size)
        self.num_facets = int(num_facets)
        self.to_facets = nn.Sequential(
            nn.Linear(self.hidden_size, facet_hidden),
            nn.ReLU(),
            nn.Linear(facet_hidden, self.num_facets * self.hidden_size),
        )

    def forward(self, anchor: torch.Tensor) -> torch.Tensor:
        if anchor.dim() == 3:
            anchor = anchor.mean(dim=1)
        facets = self.to_facets(anchor)
        facets = facets.view(anchor.size(0), self.num_facets, self.hidden_size)
        return facets


class ContentSynchronousTimbreFuser(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        gate_hidden: Optional[int] = None,
        mix_bias: float = -0.25,
        variation_bias: float = -1.0,
    ):
        super().__init__()
        gate_hidden = int(hidden_size if gate_hidden is None else gate_hidden)
        self.hidden_size = int(hidden_size)
        self.mix_bias = float(mix_bias)
        self.variation_bias = float(variation_bias)
        self.query_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.mix_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 4, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )
        self.variation_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 4, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )

    def forward(
        self,
        *,
        query: torch.Tensor,
        global_anchor: torch.Tensor,
        global_memory: torch.Tensor,
        local_absolute: torch.Tensor,
        style_context: Optional[torch.Tensor] = None,
        prior_scale: float = 1.0,
        gate_scale: float = 1.0,
        gate_bias: float = 0.0,
    ) -> dict:
        if style_context is None:
            style_context = torch.zeros_like(query)
        anchor_seq = expand_anchor_sequence(global_anchor, query.size(1))

        q = self.query_proj(query)
        k = self.key_proj(global_memory)
        v = self.value_proj(global_memory)
        attn_logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(float(self.hidden_size))
        attn = torch.softmax(attn_logits, dim=-1)
        prior_delta = self.out_proj(torch.matmul(attn, v)) * float(prior_scale)
        prior_absolute = anchor_seq + prior_delta

        mix_logit = self.mix_gate(torch.cat([query, prior_absolute, local_absolute, style_context], dim=-1))
        mix = torch.sigmoid(mix_logit + float(self.mix_bias))
        candidate_absolute = spherical_interpolate(prior_absolute, local_absolute, mix)

        variation_logit = self.variation_gate(torch.cat([query, candidate_absolute, anchor_seq, style_context], dim=-1))
        variation_gate = torch.sigmoid(variation_logit * float(gate_scale) + float(gate_bias) + float(self.variation_bias))
        candidate_delta = candidate_absolute - anchor_seq

        return {
            "attn": attn,
            "prior_delta": prior_delta,
            "prior_absolute": prior_absolute,
            "mix": mix,
            "candidate_absolute": candidate_absolute,
            "candidate_delta": candidate_delta,
            "variation_logit": variation_logit,
            "variation_gate_raw": variation_gate,
            "variation_gate": variation_gate,
        }


__all__ = [
    "ContentSynchronousTimbreFuser",
    "GlobalTimbreMemory",
    "expand_anchor_sequence",
    "spherical_interpolate",
]
