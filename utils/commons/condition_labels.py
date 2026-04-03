from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np


CONDITION_FIELDS = ("emotion", "style", "accent")
UNKNOWN_CONDITION_LABEL = "<UNK>"


def _read_json_if_exists(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_label_key(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return None
        numeric = float(value)
        return str(int(numeric)) if numeric.is_integer() else str(numeric)
    text = str(value).strip()
    return text or None


def _normalize_condition_id(value, *, default: int = -1) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _condition_map_from_vocab(vocab) -> Dict[str, int]:
    if not isinstance(vocab, Sequence) or isinstance(vocab, (str, bytes)):
        return {}
    labels: List[Optional[str]] = []
    for raw in vocab:
        if raw is None:
            labels.append(None)
            continue
        label = _normalize_label_key(raw)
        labels.append(label)
    if labels and labels[0] == UNKNOWN_CONDITION_LABEL:
        return {
            label: idx - 1
            for idx, label in enumerate(labels)
            if idx > 0 and label not in (None, "", UNKNOWN_CONDITION_LABEL)
        }
    return {
        label: idx
        for idx, label in enumerate(labels)
        if label not in (None, "")
    }


def load_condition_id_maps(
    candidate_dirs: Iterable[Optional[str]],
    *,
    fields: Sequence[str] = CONDITION_FIELDS,
) -> Dict[str, Dict[str, int]]:
    condition_maps: Dict[str, Dict[str, int]] = {}
    normalized_dirs = [str(path) for path in candidate_dirs if path]
    for field in fields:
        mapping: Dict[str, int] = {}
        for data_dir in normalized_dirs:
            map_payload = _read_json_if_exists(os.path.join(data_dir, f"{field}_map.json"))
            if isinstance(map_payload, Mapping):
                normalized_mapping = {}
                for key, value in map_payload.items():
                    label = _normalize_label_key(key)
                    if label is None:
                        continue
                    normalized_mapping[label] = _normalize_condition_id(value, default=-1)
                mapping = normalized_mapping
                break
            vocab_payload = _read_json_if_exists(os.path.join(data_dir, f"{field}_set.json"))
            if vocab_payload is not None:
                mapping = _condition_map_from_vocab(vocab_payload)
                break
        condition_maps[field] = mapping
    return condition_maps


def resolve_condition_label_id(mapping: Mapping[str, int], value, *, default: int = -1) -> int:
    if value is None:
        return int(default)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if np.isnan(numeric):
            return int(default)
        return int(numeric) if numeric.is_integer() else int(default)
    label = _normalize_label_key(value)
    if label is None:
        return int(default)
    try:
        return int(label)
    except (TypeError, ValueError):
        pass
    return int(mapping.get(label, default))


def build_condition_vocab(
    *,
    field: str,
    mapping: Optional[Mapping[str, int]] = None,
    items: Optional[Sequence[Mapping[str, object]]] = None,
) -> List[Optional[str]]:
    explicit_ids = set()
    if items is not None:
        explicit_key = f"{field}_id"
        for item in items:
            explicit_id = _normalize_condition_id(item.get(explicit_key), default=-1)
            if explicit_id >= 0:
                explicit_ids.add(explicit_id)
    normalized_mapping: Dict[str, int] = {}
    if isinstance(mapping, Mapping):
        for label, value in mapping.items():
            normalized_label = _normalize_label_key(label)
            normalized_id = _normalize_condition_id(value, default=-1)
            if normalized_label is None or normalized_id < 0:
                continue
            normalized_mapping[normalized_label] = normalized_id
            explicit_ids.add(normalized_id)
    if not explicit_ids:
        return []
    vocab: List[Optional[str]] = [None] * (max(explicit_ids) + 1)
    names_by_id: MutableMapping[int, List[str]] = {}
    for label, label_id in normalized_mapping.items():
        names_by_id.setdefault(int(label_id), []).append(label)
    for label_id, label_names in names_by_id.items():
        vocab[int(label_id)] = sorted(set(label_names))[0]
    if items is not None:
        explicit_key = f"{field}_id"
        name_key = f"{field}_name"
        for item in items:
            explicit_id = _normalize_condition_id(item.get(explicit_key), default=-1)
            if explicit_id < 0:
                continue
            normalized_name = _normalize_label_key(item.get(name_key))
            if normalized_name is not None:
                if vocab[explicit_id] is None:
                    vocab[explicit_id] = normalized_name
                continue
            if vocab[explicit_id] is None:
                vocab[explicit_id] = str(explicit_id)
    return vocab


def write_condition_artifacts(
    *,
    target_dirs: Iterable[Optional[str]],
    label_maps: Mapping[str, Mapping[str, int]],
    items: Sequence[Mapping[str, object]],
    fields: Sequence[str] = CONDITION_FIELDS,
) -> None:
    normalized_targets = sorted({str(path) for path in target_dirs if path})
    for target_dir in normalized_targets:
        os.makedirs(target_dir, exist_ok=True)
    for field in fields:
        mapping = dict(label_maps.get(field, {})) if isinstance(label_maps, Mapping) else {}
        vocab = build_condition_vocab(field=field, mapping=mapping, items=items)
        for target_dir in normalized_targets:
            with open(os.path.join(target_dir, f"{field}_map.json"), "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)
            with open(os.path.join(target_dir, f"{field}_set.json"), "w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)


__all__ = [
    "CONDITION_FIELDS",
    "UNKNOWN_CONDITION_LABEL",
    "build_condition_vocab",
    "load_condition_id_maps",
    "resolve_condition_label_id",
    "write_condition_artifacts",
]
