from __future__ import annotations

from typing import Any, Mapping

import torch

from modules.Conan.control.separation_metrics import resolve_dynamic_timbre_frame_weight


def resolve_dynamic_timbre_budget_support_weight(
    sample: Mapping[str, Any] | None,
    output: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    reference,
):
    if not isinstance(reference, torch.Tensor):
        return None
    sample = sample if isinstance(sample, Mapping) else {}
    output = output if isinstance(output, Mapping) else {}
    return resolve_dynamic_timbre_frame_weight(
        sample.get("uv"),
        sample.get("energy"),
        reference,
        mask=output.get("dynamic_timbre_mask", output.get("style_trace_mask")),
        uv_floor=float(config.get("dynamic_timbre_budget_uv_floor", 0.25)),
        energy_floor=float(config.get("dynamic_timbre_budget_energy_floor", 0.10)),
        energy_power=float(config.get("dynamic_timbre_budget_energy_power", 0.5)),
        energy_quantile=float(config.get("dynamic_timbre_budget_energy_quantile", 0.90)),
    )


__all__ = ["resolve_dynamic_timbre_budget_support_weight"]
