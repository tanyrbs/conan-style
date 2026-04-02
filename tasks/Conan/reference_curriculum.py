import math
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from utils.commons.hparams import hparams


REFERENCE_CURRICULUM_ALIASES = {
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


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _curriculum_mode(config: Optional[Mapping[str, Any]] = None) -> str:
    mode = str(_cfg(config).get("reference_curriculum_mode", "legacy_hard")).strip().lower()
    return REFERENCE_CURRICULUM_ALIASES.get(mode, "legacy_hard")


def _curriculum_window(config: Optional[Mapping[str, Any]] = None) -> Tuple[int, int]:
    cfg = _cfg(config)
    start = int(cfg.get("reference_curriculum_start_steps", cfg.get("forcing", 0)))
    end = int(cfg.get("reference_curriculum_end_steps", cfg.get("random_speaker_steps", start)))
    if end < start:
        end = start
    return max(start, 0), max(end, 0)


def curriculum_progress(global_step: int, config: Optional[Mapping[str, Any]] = None) -> float:
    start, end = _curriculum_window(config)
    step = max(int(global_step), 0)
    if step <= start:
        return 0.0
    if end <= start:
        return 1.0
    return _clamp01(float(step - start) / float(end - start))


def resolve_reference_curriculum(
    global_step: int,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = _cfg(config)
    mode = _curriculum_mode(cfg)
    start, end = _curriculum_window(cfg)
    progress = curriculum_progress(global_step, cfg)

    if mode == "legacy_hard":
        external_prob = 1.0 if max(int(global_step), 0) >= end else 0.0
    else:
        initial = _clamp01(float(cfg.get("reference_curriculum_external_prob_init", 0.0)))
        final = _clamp01(float(cfg.get("reference_curriculum_external_prob_final", 1.0)))
        if mode == "bernoulli_linear":
            shaped = progress
        else:
            shaped = 0.5 - 0.5 * math.cos(math.pi * progress)
        external_prob = initial + (final - initial) * shaped

    self_ref_floor = _clamp01(float(cfg.get("reference_curriculum_self_ref_floor", 0.0)))
    external_prob = min(_clamp01(external_prob), 1.0 - self_ref_floor)
    self_prob = 1.0 - external_prob
    return {
        "mode": mode,
        "start_steps": int(start),
        "end_steps": int(end),
        "progress": float(progress),
        "external_prob": float(external_prob),
        "self_prob": float(self_prob),
        "self_ref_floor": float(self_ref_floor),
    }


def sample_training_reference_source(
    global_step: int,
    *,
    config: Optional[Mapping[str, Any]] = None,
    device=None,
) -> Dict[str, Any]:
    cfg = _cfg(config)
    state = resolve_reference_curriculum(global_step, cfg)
    sample_mode = str(cfg.get("reference_curriculum_sample_mode", "batch")).strip().lower()
    if sample_mode not in {"batch", "batchwise", "global", "bernoulli_batch"}:
        raise ValueError(
            f"Unsupported reference_curriculum_sample_mode '{sample_mode}'. "
            "Current low-risk mainline implementation only supports batchwise sampling."
        )
    p_external = float(state["external_prob"])
    if p_external <= 0.0:
        use_external = False
    elif p_external >= 1.0:
        use_external = True
    else:
        rand = torch.rand((), device=device)
        use_external = bool(rand.item() < p_external)
    state["sample_mode"] = "batch"
    state["use_external_ref"] = bool(use_external)
    state["use_self_ref"] = not bool(use_external)
    state["gloss_scale"] = 0.0 if bool(use_external) else 1.0
    state["reference_source"] = "external_ref" if bool(use_external) else "self_target"
    return state
