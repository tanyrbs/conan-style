from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F


REFERENCE_CACHE_MASK_FIELDS = {
    "prosody_memory": ("prosody_memory_mask", "prosody_key_padding_mask"),
    "prosody_memory_slow": ("prosody_memory_slow_mask", "prosody_key_padding_mask_slow"),
    "timbre_memory": ("timbre_memory_mask", "timbre_key_padding_mask"),
    "timbre_memory_slow": ("timbre_memory_slow_mask", "timbre_key_padding_mask_slow"),
}


def first_present(mapping: Optional[Mapping[str, Any]], *keys: str, default=None):
    if mapping is None:
        return default
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def is_sequence_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.dim() == 3


def _normalize_global_anchor(value):
    if not isinstance(value, torch.Tensor):
        return value
    if value.dim() == 2:
        return value.unsqueeze(1)
    if value.dim() == 3 and value.size(-1) == 1:
        return value.transpose(1, 2)
    return value


def _coerce_mask(mask: Any, reference: Optional[torch.Tensor]):
    if not isinstance(mask, torch.Tensor) or not is_sequence_tensor(reference):
        return None
    if mask.dim() == 3 and mask.size(-1) == 1:
        mask = mask.squeeze(-1)
    if mask.dim() != 2 or tuple(mask.shape[:2]) != tuple(reference.shape[:2]):
        return None
    return mask.bool().to(device=reference.device)


def resolve_sequence_mask(reference_cache: Optional[Mapping[str, Any]], memory_key: str):
    if not isinstance(reference_cache, Mapping):
        return None
    reference = reference_cache.get(memory_key)
    if not is_sequence_tensor(reference):
        return None
    canonical_key, alias_key = REFERENCE_CACHE_MASK_FIELDS.get(memory_key, (None, None))
    if canonical_key is None:
        return None
    return _coerce_mask(
        first_present(reference_cache, canonical_key, alias_key),
        reference,
    )


def masked_sequence_mean(sequence: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None, *, keepdim: bool = False):
    if not is_sequence_tensor(sequence):
        return None
    resolved_mask = _coerce_mask(mask, sequence)
    if resolved_mask is None:
        pooled = sequence.mean(dim=1, keepdim=keepdim)
    else:
        valid = (~resolved_mask).unsqueeze(-1).to(device=sequence.device, dtype=sequence.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (sequence * valid).sum(dim=1, keepdim=True) / denom
        if not keepdim:
            pooled = pooled.squeeze(1)
    return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


def pool_reference_memory(memory: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None, *, pool_size: int = 1):
    if not is_sequence_tensor(memory):
        return memory, mask
    pool_size = max(1, int(pool_size))
    resolved_mask = _coerce_mask(mask, memory)
    if pool_size <= 1 or memory.size(1) <= 0:
        return memory.clone(), resolved_mask.clone() if isinstance(resolved_mask, torch.Tensor) else resolved_mask

    batch_size, seq_len, channels = memory.shape
    padded_len = ((seq_len + pool_size - 1) // pool_size) * pool_size
    padded_memory = memory
    pooled_mask = resolved_mask
    if padded_len != seq_len:
        padded_memory = F.pad(memory, (0, 0, 0, padded_len - seq_len))
        if isinstance(resolved_mask, torch.Tensor):
            pooled_mask = F.pad(resolved_mask.float(), (0, padded_len - seq_len), value=1.0).bool()

    reshaped = padded_memory.reshape(batch_size, padded_len // pool_size, pool_size, channels)
    if not isinstance(pooled_mask, torch.Tensor):
        return reshaped.mean(dim=2), None

    valid = (~pooled_mask).reshape(batch_size, padded_len // pool_size, pool_size, 1).to(memory.dtype)
    denom = valid.sum(dim=2).clamp_min(1.0)
    pooled_memory = (reshaped * valid).sum(dim=2) / denom
    pooled_mask = valid.squeeze(-1).sum(dim=2).le(0.0)
    pooled_memory = pooled_memory * (~pooled_mask).unsqueeze(-1).to(pooled_memory.dtype)
    pooled_memory = torch.where(torch.isfinite(pooled_memory), pooled_memory, torch.zeros_like(pooled_memory))
    return pooled_memory, pooled_mask


def clone_reference_cache(reference_cache: Optional[Mapping[str, Any]], *, detach: bool = False):
    if reference_cache is None:
        return None
    cloned: Dict[str, Any] = {}
    for key, value in dict(reference_cache).items():
        if isinstance(value, torch.Tensor):
            value = value.detach() if detach else value
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def detach_reference_cache(reference_cache: Optional[Mapping[str, Any]]):
    return clone_reference_cache(reference_cache, detach=True)


def merge_reference_cache(
    base_cache: Optional[Mapping[str, Any]] = None,
    updates: Optional[Mapping[str, Any]] = None,
):
    merged: Dict[str, Any] = {}
    if isinstance(base_cache, Mapping):
        merged.update(dict(base_cache))
    if isinstance(updates, Mapping):
        for key, value in updates.items():
            if value is not None:
                merged[key] = value
    return canonicalize_reference_cache(merged)


def canonicalize_reference_cache(reference_cache: Optional[Mapping[str, Any]] = None):
    if reference_cache is None:
        return None
    normalized: Dict[str, Any] = dict(reference_cache)
    global_timbre_anchor = _normalize_global_anchor(
        first_present(normalized, "global_timbre_anchor")
    )
    if global_timbre_anchor is not None:
        normalized["global_timbre_anchor"] = global_timbre_anchor
    global_style_summary = _normalize_global_anchor(
        first_present(normalized, "global_style_summary")
    )
    if global_style_summary is not None:
        normalized["global_style_summary"] = global_style_summary

    for memory_key, (canonical_mask_key, alias_mask_key) in REFERENCE_CACHE_MASK_FIELDS.items():
        memory = normalized.get(memory_key)
        if not is_sequence_tensor(memory):
            continue
        mask = _coerce_mask(
            first_present(normalized, canonical_mask_key, alias_mask_key),
            memory,
        )
        if mask is not None:
            normalized[canonical_mask_key] = mask
            normalized[alias_mask_key] = mask
    return normalized


def validate_reference_cache(reference_cache: Optional[Mapping[str, Any]]):
    if reference_cache is None:
        return
    normalized = canonicalize_reference_cache(reference_cache)
    for key in ("global_timbre_anchor", "global_style_summary"):
        global_anchor = normalized.get(key)
        if global_anchor is not None:
            if not isinstance(global_anchor, torch.Tensor) or global_anchor.dim() != 3:
                raise ValueError(f"reference_cache['{key}'] must be a [B, 1, H] tensor.")
    for memory_key in REFERENCE_CACHE_MASK_FIELDS:
        memory = normalized.get(memory_key)
        if memory is None:
            continue
        if not is_sequence_tensor(memory):
            raise ValueError(f"reference_cache['{memory_key}'] must be a [B, T, H] tensor.")
        mask = resolve_sequence_mask(normalized, memory_key)
        if mask is not None and tuple(mask.shape[:2]) != tuple(memory.shape[:2]):
            raise ValueError(
                f"reference_cache mask shape mismatch for {memory_key}: "
                f"memory={tuple(memory.shape)}, mask={tuple(mask.shape)}"
            )


def select_cached_sequence(
    reference_cache: Optional[Mapping[str, Any]],
    primary_key: str,
    slow_key: Optional[str] = None,
    *,
    prefer_slow: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
    if not isinstance(reference_cache, Mapping):
        return None, None, "none"
    keys = []
    if prefer_slow and slow_key is not None:
        keys.extend([slow_key, primary_key])
    else:
        keys.append(primary_key)
        if slow_key is not None:
            keys.append(slow_key)
    for key in keys:
        value = reference_cache.get(key)
        if is_sequence_tensor(value):
            return value, resolve_sequence_mask(reference_cache, key), key
    return None, None, "none"


def resolve_reference_cache(source: Optional[Mapping[str, Any]] = None):
    source = source or {}
    cache = first_present(source, "reference_cache", default=None)
    if isinstance(cache, Mapping):
        return canonicalize_reference_cache(cache)
    return None


def reference_cache_to_model_kwargs(reference_cache: Optional[Mapping[str, Any]], **extra_kwargs):
    model_kwargs: Dict[str, Any] = {}
    if reference_cache is not None:
        model_kwargs["reference_cache"] = canonicalize_reference_cache(reference_cache)
    for key, value in extra_kwargs.items():
        if value is not None:
            model_kwargs[key] = value
    return model_kwargs
