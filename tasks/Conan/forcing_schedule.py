import math
from typing import Any, Dict, Mapping, Optional

import torch

from utils.commons.hparams import hparams


FORCING_SCHEDULE_ALIASES = {
    "off": "legacy_hard",
    "disabled": "legacy_hard",
    "hard": "legacy_hard",
    "legacy": "legacy_hard",
    "legacy_hard": "legacy_hard",
    "linear": "bernoulli_linear",
    "bernoulli_linear": "bernoulli_linear",
    "cosine": "bernoulli_cosine",
    "bernoulli_cosine": "bernoulli_cosine",
}


def _cfg(config: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
    return config if isinstance(config, Mapping) else hparams


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def resolve_forcing_schedule(
    global_step: int,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = _cfg(config)

    mode = str(cfg.get("forcing_schedule_mode", "legacy_hard")).strip().lower()
    mode = FORCING_SCHEDULE_ALIASES.get(mode, "legacy_hard")

    legacy_cut = max(int(cfg.get("forcing", 0)), 0)
    start = max(int(cfg.get("forcing_decay_start_steps", legacy_cut)), 0)
    end = max(int(cfg.get("forcing_decay_end_steps", legacy_cut)), 0)
    p_init = float(cfg.get("forcing_prob_init", 1.0))
    p_final = float(cfg.get("forcing_prob_final", 0.0))

    if end < start:
        end = start

    if mode == "legacy_hard":
        forcing_prob = 1.0 if int(global_step) < legacy_cut else 0.0
        progress = 0.0 if forcing_prob > 0.0 else 1.0
    else:
        if end <= start:
            alpha = 1.0 if int(global_step) >= end else 0.0
        elif int(global_step) <= start:
            alpha = 0.0
        elif int(global_step) >= end:
            alpha = 1.0
        else:
            alpha = float(global_step - start) / float(max(1, end - start))

        if mode == "bernoulli_linear":
            shaped = alpha
        else:
            shaped = 0.5 * (1.0 - math.cos(math.pi * alpha))

        forcing_prob = p_init + (p_final - p_init) * shaped
        forcing_prob = _clamp01(forcing_prob)
        progress = _clamp01(alpha)

    return {
        "mode": mode,
        "progress": progress,
        "forcing_prob": forcing_prob,
        "legacy_cut": legacy_cut,
        "start_steps": start,
        "end_steps": end,
    }


def sample_forcing_flag(
    global_step: int,
    *,
    config: Optional[Mapping[str, Any]] = None,
    device=None,
) -> Dict[str, Any]:
    state = resolve_forcing_schedule(global_step, config=config)
    prob = float(state["forcing_prob"])
    if prob <= 0.0:
        enabled = False
    elif prob >= 1.0:
        enabled = True
    else:
        enabled = bool(torch.rand((), device=device).item() < prob)
    state["forcing_enabled"] = enabled
    return state
