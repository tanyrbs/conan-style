#!/usr/bin/env python3
import argparse
import importlib
from importlib import metadata as importlib_metadata
import json
import os
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import utils.commons.single_thread_env  # NOQA
import numpy as np
import torch
import yaml

from modules.Conan.style_mainline import (
    resolve_expressive_upper_bound_progress,
    resolve_style_mainline_controls,
)
from modules.Conan.style_profiles import resolve_style_profile
from modules.Conan.reference_bundle import build_control_kwargs, build_style_runtime_kwargs
from modules.Conan.control.style_success import (
    STYLE_SUCCESS_MEMORY_FALLBACK_SCALE,
    resolve_style_success_bool_flag,
    resolve_style_success_negative_masks,
    resolve_style_success_rank_source_scale,
    resolve_style_success_rank_support_state,
)
from modules.Conan.dynamic_timbre_control import build_dynamic_timbre_boundary_mask
from tasks.Conan.dataset import ConanDataset
from tasks.Conan.control_schedule import (
    MAINLINE_MINIMAL_CONTROL_LAMBDAS,
    resolve_control_regularization_config,
)
from tasks.Conan.reference_curriculum import resolve_reference_curriculum
from tasks.Conan.forcing_schedule import resolve_forcing_schedule
from utils.commons.condition_labels import CONDITION_FIELDS, load_condition_id_maps, resolve_condition_label_id
from utils.commons.hparams import hparams, set_hparams


REQUIRED_ZERO_KEYS = (
    "lambda_energy",
    "lambda_style_timbre_disentangle",
    "lambda_style_trace_consistency",
    "lambda_style_query_var",
    "lambda_global_style_summary_align",
    "lambda_slow_style_summary_align",
    "lambda_tv_timbre_smooth",
    "lambda_tv_timbre_anchor",
    "lambda_timbre_anchor_cosine",
    "lambda_style_dynamic_timbre_disentangle",
    "lambda_dynamic_timbre_gate",
    "lambda_dynamic_timbre_boundary",
    "lambda_dynamic_timbre_anchor",
    "lambda_gate_rank",
    "lambda_decoder_late_anchor_budget",
    "lambda_style_timbre_runtime_overlap",
)

REQUIRED_POSITIVE_KEYS = (
    "lambda_output_identity_cosine",
    "lambda_dynamic_timbre_budget",
    "lambda_pitch_residual_safe",
    "lambda_decoder_late_owner",
    "lambda_style_success_rank",
)

REQUIRED_PRESENT_KEYS = (
    "lambda_output_identity_cosine",
    "lambda_dynamic_timbre_budget",
    "lambda_pitch_residual_safe",
    "lambda_decoder_late_owner",
    "lambda_style_success_rank",
)

CANONICAL_REMOVED_CONFIG_KEYS = (
    "disc_norm",
    "disc_reduction",
    "dur_level",
    "speaker_verifier_backend",
    "speaker_verifier_ckpt",
    "timbre_reg_start_steps",
    "timbre_reg_warmup_steps",
    "timbre_reg_init_scale",
    "timbre_reg_final_scale",
    "use_reference_bundle",
    "use_reference_cache",
    "use_spk_prompt",
    "valid_plot_prob",
)


def _is_boollike_value(value):
    if value is None or isinstance(value, (bool, int, float)):
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "0", "true", "false", "yes", "no", "y", "n", "on", "off"}
    return False

TOP_LEVEL_KEY_PATTERN = re.compile(r"^([A-Za-z0-9_]+):")


def _classify_check_name(name):
    name = str(name or "").strip()
    if (
        name.startswith("runtime_import_")
        or name.startswith("runtime_nltk_")
        or name == "runtime_python_version_compatible_with_pinned_torchaudio"
        or name.endswith("_matches_requirements_pin")
    ):
        return "runtime_dependencies"
    if name in {"binary_data_dir_exists", "processed_data_dir_exists"}:
        return "data_staging"
    if name.startswith("binary_") or name.startswith("condition_artifacts_"):
        return "data_staging"
    if name in {
        "train_batch_mainline_controls_resolvable",
        "dynamic_timbre_boundary_preview_available",
        "dynamic_timbre_boundary_not_global_on_real_data",
        "style_success_runtime_preview",
        "train_batch_style_strength_matches_profile",
        "train_batch_dynamic_timbre_strength_matches_profile",
        "train_batch_dynamic_timbre_strength_source_matches_profile",
        "train_batch_style_temperature_matches_profile",
        "train_batch_dynamic_timbre_temperature_matches_profile",
        "train_batch_dynamic_timbre_use_tvt_matches_profile",
        "train_batch_dynamic_timbre_tvt_prior_scale_matches_profile",
    }:
        return "data_dependent_preview"
    if name.startswith("config_"):
        return "config_surface"
    if name in {
        "reference_contract_mode",
        "decoder_style_condition_mode",
        "global_timbre_to_pitch",
        "style_to_pitch_residual",
        "style_to_pitch_residual_include_timbre",
        "style_to_pitch_residual_mode",
        "style_trace_mode",
        "style_router_enabled",
        "style_memory_mode",
        "dynamic_timbre_memory_mode",
        "style_profile_track",
        "style_profile",
        "control_loss_profile",
        "identity_loss_constraint_mode",
        "mainline_minimal_active_control_loss_count",
        "mainline_minimal_active_control_losses",
    } or name.startswith("lambda_"):
        return "mainline_contract"
    return "mainline_contract"


def _load_repo_requirements(root_dir: Path):
    requirements = {}
    req_path = Path(root_dir) / "requirements.txt"
    if not req_path.exists():
        return requirements
    try:
        for raw_line in req_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "==" not in line:
                continue
            name, version = line.split("==", 1)
            requirements[name.strip().lower()] = version.strip()
    except Exception:
        return {}
    return requirements


def _normalize_requirement_version(version):
    version = str(version or "").strip()
    if not version:
        return None
    return version.split("+", 1)[0]


def _installed_distribution_version(package_name):
    try:
        return importlib_metadata.version(str(package_name))
    except Exception:
        return None


def _check_runtime_requirement_pins(checks, requirements_path, package_names):
    pins = _load_repo_requirements(Path(requirements_path).parent)
    for package_name in package_names:
        normalized_name = str(package_name).strip().lower()
        pinned_version = pins.get(normalized_name)
        installed_version = _installed_distribution_version(normalized_name)
        checks.append(
            {
                "name": f"runtime_{normalized_name}_matches_requirements_pin",
                "ok": bool(
                    pinned_version is not None
                    and installed_version is not None
                    and _normalize_requirement_version(installed_version)
                    == _normalize_requirement_version(pinned_version)
                ),
                "actual": installed_version,
                "expected": pinned_version if pinned_version is not None else "exact pin present",
            }
        )


def _resolve_torchaudio_python_range(version: str):
    version = str(version or "").strip()
    if not version:
        return None
    major_minor = ".".join(version.split(".")[:2])
    compat = {
        "2.6": ((3, 9), (3, 13)),
        "2.5": ((3, 8), (3, 11)),
        "2.4": ((3, 8), (3, 11)),
        "2.3": ((3, 8), (3, 11)),
        "2.2": ((3, 8), (3, 11)),
        "2.1": ((3, 8), (3, 11)),
        "2.0": ((3, 8), (3, 11)),
    }
    return compat.get(major_minor)


def _summarize_failed_checks(checks):
    failed = []
    category_map = {}
    for item in checks:
        if bool(item.get("ok", False)):
            continue
        name = item.get("name")
        category = _classify_check_name(name)
        failed_entry = {
            "name": name,
            "category": category,
            "actual": _jsonable(item.get("actual")),
            "expected": _jsonable(item.get("expected")),
            "path": item.get("path"),
        }
        failed.append(failed_entry)
        category_state = category_map.setdefault(category, {"count": 0, "checks": []})
        category_state["count"] += 1
        category_state["checks"].append(name)
    return {
        "failed_count": len(failed),
        "failed_checks": failed,
        "blocking_categories": category_map,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Conan mainline training config before the first real-dataset run."
    )
    parser.add_argument("--config", type=str, default="egs/conan_emformer.yaml")
    parser.add_argument(
        "--binary_data_dir",
        type=str,
        default=None,
        help="Optional override for validating the actual binary dataset dir used by the run.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="smoke_runs/mainline_train_prep.json",
    )
    return parser.parse_args()


def _check_equal(checks, name, actual, expected):
    ok = actual == expected
    checks.append(
        {
            "name": name,
            "ok": bool(ok),
            "actual": actual,
            "expected": expected,
        }
    )


def _check_close(checks, name, actual, expected, tol=1e-8):
    ok = abs(float(actual) - float(expected)) <= float(tol)
    checks.append(
        {
            "name": name,
            "ok": bool(ok),
            "actual": float(actual),
            "expected": float(expected),
            "tolerance": float(tol),
        }
    )


def _check_true(checks, name, condition, *, actual=None, expected=True):
    checks.append(
        {
            "name": name,
            "ok": bool(condition),
            "actual": actual if actual is not None else bool(condition),
            "expected": expected,
        }
    )


def _check_exists(checks, name, path_value):
    ok = bool(path_value) and os.path.exists(str(path_value))
    checks.append(
        {
            "name": name,
            "ok": bool(ok),
            "path": str(path_value) if path_value is not None else None,
        }
    )


def _check_npy_count_positive(checks, name, path_value):
    try:
        count = int(len(np.load(str(path_value), mmap_mode='r')))
        ok = count > 0
        checks.append(
            {
                "name": name,
                "ok": bool(ok),
                "actual": count,
                "expected": "> 0",
                "path": str(path_value),
            }
        )
    except Exception as exc:
        checks.append(
            {
                "name": name,
                "ok": False,
                "actual": f"{type(exc).__name__}: {exc}",
                "expected": "> 0",
                "path": str(path_value),
            }
        )


def _load_1d_npy_array(path_value):
    array = np.load(str(path_value), mmap_mode='r')
    return np.asarray(array).reshape(-1)


def _check_binary_sidecar_consistency(checks, binary_data_dir, split):
    if not binary_data_dir:
        return
    paths = {
        "lengths": os.path.join(str(binary_data_dir), f"{split}_lengths.npy"),
        "ref_indices": os.path.join(str(binary_data_dir), f"{split}_ref_indices.npy"),
        "spk_ids": os.path.join(str(binary_data_dir), f"{split}_spk_ids.npy"),
    }
    missing = [name for name, path in paths.items() if not os.path.exists(path)]
    if missing:
        checks.append(
            {
                "name": f"binary_{split}_sidecars_consistent",
                "ok": False,
                "actual": {"missing": missing},
                "expected": "lengths/ref_indices/spk_ids sidecars all present and aligned",
            }
        )
        return
    try:
        lengths = _load_1d_npy_array(paths["lengths"])
        ref_indices = _load_1d_npy_array(paths["ref_indices"]).astype(np.int64, copy=False)
        spk_ids = _load_1d_npy_array(paths["spk_ids"]).astype(np.int64, copy=False)
    except Exception as exc:
        checks.append(
            {
                "name": f"binary_{split}_sidecars_consistent",
                "ok": False,
                "actual": f"{type(exc).__name__}: {exc}",
                "expected": "lengths/ref_indices/spk_ids sidecars load successfully",
            }
        )
        return

    num_items = int(lengths.shape[0])
    counts_match = bool(ref_indices.shape[0] == num_items == spk_ids.shape[0])
    nonnegative_lengths = bool(np.all(lengths >= 0))
    ref_indices_in_range = bool(
        num_items > 0
        and np.all(ref_indices >= 0)
        and np.all(ref_indices < num_items)
    )
    same_speaker_refs = False
    same_speaker_frac = None
    if counts_match and ref_indices_in_range and num_items > 0:
        ref_speakers = spk_ids[ref_indices]
        speaker_matches = ref_speakers == spk_ids
        same_speaker_refs = bool(np.all(speaker_matches))
        same_speaker_frac = float(np.mean(speaker_matches.astype(np.float32)))

    checks.append(
        {
            "name": f"binary_{split}_sidecars_consistent",
            "ok": bool(
                counts_match
                and nonnegative_lengths
                and ref_indices_in_range
                and same_speaker_refs
            ),
            "actual": {
                "num_items": num_items,
                "counts": {
                    "lengths": int(lengths.shape[0]),
                    "ref_indices": int(ref_indices.shape[0]),
                    "spk_ids": int(spk_ids.shape[0]),
                },
                "nonnegative_lengths": bool(nonnegative_lengths),
                "ref_indices_in_range": bool(ref_indices_in_range),
                "same_speaker_refs": bool(same_speaker_refs),
                "same_speaker_ref_frac": same_speaker_frac,
                "ref_index_min": int(ref_indices.min()) if ref_indices.size > 0 else None,
                "ref_index_max": int(ref_indices.max()) if ref_indices.size > 0 else None,
            },
            "expected": {
                "counts_match": True,
                "nonnegative_lengths": True,
                "ref_indices_in_range": True,
                "same_speaker_refs": True,
            },
        }
    )


def _check_importable(checks, name, module_name):
    try:
        importlib.import_module(module_name)
        checks.append(
            {
                "name": name,
                "ok": True,
                "actual": f"import ok: {module_name}",
                "expected": "importable",
            }
        )
    except Exception as exc:  # pragma: no cover - surfaced through prep output
        checks.append(
            {
                "name": name,
                "ok": False,
                "actual": f"{type(exc).__name__}: {exc}",
                "expected": "importable",
            }
        )


def _check_runtime_callable(checks, name, fn, *, expected="callable succeeded"):
    try:
        actual = fn()
        checks.append(
            {
                "name": name,
                "ok": True,
                "actual": actual if actual is not None else "ok",
                "expected": expected,
            }
        )
    except Exception as exc:  # pragma: no cover - surfaced through prep output
        checks.append(
            {
                "name": name,
                "ok": False,
                "actual": f"{type(exc).__name__}: {exc}",
                "expected": expected,
            }
        )


def _check_nltk_tagger_for_g2p_en():
    from nltk import pos_tag

    tagged = pos_tag(["test"])
    return {"tagged_example": tagged[:1]}


def _check_nltk_cmudict_for_g2p_en():
    from nltk.corpus import cmudict

    entries = cmudict.entries()
    return {"entries": int(len(entries))}


def _check_local_conan_model_init(config_path):
    from modules.Conan.Conan import Conan
    from utils.commons.hparams import hparams as global_hparams_store
    from utils.commons.hparams import set_hparams as _set_hparams

    backup = dict(global_hparams_store)
    try:
        global_hparams_store.clear()
        local_hparams = _set_hparams(
            config=str(config_path),
            print_hparams=False,
            global_hparams=False,
        )
        model = Conan(0, local_hparams)
        return {
            "model_class": type(model).__name__,
            "global_hparams_len_during_local_init": int(len(global_hparams_store)),
            "style_enabled": bool(local_hparams.get("style", False)),
        }
    finally:
        global_hparams_store.clear()
        global_hparams_store.update(backup)


def _normalize_condition_label_text(value):
    text = str(value or "").strip()
    return text or None


def _normalize_condition_mapping_payload(payload):
    if not isinstance(payload, dict):
        return {}
    normalized = {}
    for key, value in payload.items():
        label = _normalize_condition_label_text(key)
        if label is None:
            continue
        try:
            normalized[label] = int(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _mapping_from_condition_set_payload(payload):
    if not isinstance(payload, list):
        return {}
    normalized = {}
    offset = 1 if payload and str(payload[0]).strip() == "<UNK>" else 0
    for idx, raw in enumerate(payload[offset:], start=0):
        label = _normalize_condition_label_text(raw)
        if label is None or label == "<UNK>":
            continue
        normalized[label] = int(idx)
    return normalized


def _check_condition_artifacts(checks, candidate_dirs):
    resolved_maps = load_condition_id_maps(candidate_dirs, fields=CONDITION_FIELDS)
    for field in CONDITION_FIELDS:
        per_dir_states = []
        map_paths = [
            os.path.join(str(data_dir), f"{field}_map.json")
            for data_dir in candidate_dirs
            if data_dir
        ]
        set_paths = [
            os.path.join(str(data_dir), f"{field}_set.json")
            for data_dir in candidate_dirs
            if data_dir
        ]
        map_exists = any(os.path.exists(path) for path in map_paths)
        set_exists = any(os.path.exists(path) for path in set_paths)
        mapping = resolved_maps.get(field, {})
        if not map_exists and not set_exists:
            checks.append(
                {
                    "name": f"condition_artifacts_{field}",
                    "ok": True,
                    "actual": "not_present",
                    "expected": "optional",
                }
            )
            continue
        normalized_by_dir = []
        for data_dir in candidate_dirs:
            if not data_dir:
                continue
            state = {
                "dir": str(data_dir),
                "map_exists": False,
                "set_exists": False,
                "map_matches_set": True,
                "map_num_labels": 0,
                "set_num_labels": 0,
            }
            map_path = os.path.join(str(data_dir), f"{field}_map.json")
            set_path = os.path.join(str(data_dir), f"{field}_set.json")
            map_payload = None
            set_payload = None
            if os.path.exists(map_path):
                state["map_exists"] = True
                try:
                    with open(map_path, "r", encoding="utf-8") as f:
                        map_payload = json.load(f)
                except Exception as exc:
                    state["map_error"] = f"{type(exc).__name__}: {exc}"
            if os.path.exists(set_path):
                state["set_exists"] = True
                try:
                    with open(set_path, "r", encoding="utf-8") as f:
                        set_payload = json.load(f)
                except Exception as exc:
                    state["set_error"] = f"{type(exc).__name__}: {exc}"
            normalized_map = _normalize_condition_mapping_payload(map_payload)
            normalized_set = _mapping_from_condition_set_payload(set_payload)
            state["map_num_labels"] = int(len(normalized_map))
            state["set_num_labels"] = int(len(normalized_set))
            if state["map_exists"] and state["set_exists"]:
                state["map_matches_set"] = normalized_map == normalized_set
            if state["map_exists"] or state["set_exists"]:
                normalized_by_dir.append(
                    normalized_map if normalized_map else normalized_set
                )
            per_dir_states.append(state)
        roundtrip_ok = True
        roundtrip_mismatches = []
        for label, expected_id in mapping.items():
            actual_id = resolve_condition_label_id(mapping, label, default=-1)
            if int(actual_id) != int(expected_id):
                roundtrip_ok = False
                roundtrip_mismatches.append(
                    {
                        "label": label,
                        "expected": int(expected_id),
                        "actual": int(actual_id),
                    }
                )
        cross_dir_consistent = True
        if len(normalized_by_dir) > 1:
            first_mapping = normalized_by_dir[0]
            cross_dir_consistent = all(candidate == first_mapping for candidate in normalized_by_dir[1:])
        checks.append(
            {
                "name": f"condition_artifacts_{field}",
                "ok": bool(
                    (not set_exists or map_exists)
                    and roundtrip_ok
                    and cross_dir_consistent
                    and all(state.get("map_matches_set", True) for state in per_dir_states)
                ),
                "actual": {
                    "map_exists": bool(map_exists),
                    "set_exists": bool(set_exists),
                    "num_labels": int(len(mapping)),
                    "roundtrip_mismatches": roundtrip_mismatches[:8],
                    "cross_dir_consistent": bool(cross_dir_consistent),
                    "per_dir": per_dir_states,
                },
                "expected": {
                    "map_exists_if_set_exists": True,
                    "roundtrip_consistent": True,
                    "cross_dir_consistent": True,
                    "map_matches_set_when_both_exist": True,
                },
            }
        )
    return resolved_maps


def _style_success_runtime_preview(batch, hparams):
    if not isinstance(batch, dict):
        return {
            "available": False,
            "reason": "missing_batch",
        }
    mel_lengths = batch.get("mel_lengths")
    if isinstance(mel_lengths, torch.Tensor):
        batch_size = int(mel_lengths.numel())
        device = mel_lengths.device
    else:
        batch_size = int(batch.get("nsamples", 0) or 0)
        device = torch.device("cpu")
    if batch_size <= 1:
        return {
            "available": False,
            "reason": "batch_size_le_1",
            "batch_size": batch_size,
        }
    negative_state = resolve_style_success_negative_masks(
        batch,
        batch_size=batch_size,
        device=device,
        proxy_threshold=float(hparams.get("style_success_proxy_negative_threshold", 1.25) or 1.25),
        proxy_min_count=int(hparams.get("style_success_proxy_negative_min_count", 2) or 2),
        proxy_min_batch=int(hparams.get("style_success_proxy_min_batch", 4) or 0),
        use_rate_proxy=resolve_style_success_bool_flag(
            hparams.get("style_success_proxy_use_rate_proxy", False),
            default=False,
        ),
    )
    preview = {
        "available": True,
        "batch_size": batch_size,
        "negative_source": str(negative_state.get("source", "none")),
        "proxy_use_rate_proxy": bool(
            resolve_style_success_bool_flag(
                hparams.get("style_success_proxy_use_rate_proxy", False),
                default=False,
            )
        ),
        "proxy_min_batch": int(hparams.get("style_success_proxy_min_batch", 4) or 0),
    }
    preview["proxy_batch_gate_passed"] = bool(negative_state.get("proxy_batch_gate_passed", True))
    preview["proxy_disabled_reason"] = str(negative_state.get("proxy_disabled_reason", "active"))
    support_state = resolve_style_success_rank_support_state(
        negative_state,
        hparams,
        device=device,
    )
    for state_key, preview_key in (
        ("label_valid_rows", "label_negative_row_frac"),
        ("proxy_valid_rows", "proxy_negative_row_frac"),
        ("valid_rows", "negative_row_frac"),
        ("negative_pair_density", "negative_pair_density"),
        ("negative_row_density", "negative_row_density"),
        ("mean_negatives_per_row", "mean_negatives_per_row"),
        ("mean_negatives_per_valid_row", "mean_negatives_per_valid_row"),
    ):
        rows = negative_state.get(state_key)
        if isinstance(rows, torch.Tensor):
            preview[preview_key] = float(rows.float().mean().detach().cpu())
    negative_mask = negative_state.get("negative_mask")
    if isinstance(negative_mask, torch.Tensor):
        preview["negative_pair_frac"] = float(negative_mask.float().mean().detach().cpu())
    preview["proxy_feature_count"] = int(negative_state.get("proxy_feature_count", 0) or 0)
    preview["proxy_informative_feature_count"] = int(
        negative_state.get("proxy_informative_feature_count", 0) or 0
    )
    preview["proxy_informative_feature_names"] = list(
        negative_state.get("proxy_informative_feature_names", ()) or ()
    )
    preview["rank_source_scale"] = float(
        resolve_style_success_rank_source_scale(preview["negative_source"], hparams)
    )
    for state_key, preview_key in (
        ("support_scale", "rank_support_scale"),
        ("effective_support", "rank_effective_support"),
        ("gate_passed", "rank_gate_passed"),
        ("rank_term_active", "rank_term_active"),
    ):
        value = support_state.get(state_key)
        if isinstance(value, torch.Tensor):
            preview[preview_key] = float(value.detach().float().mean().cpu())
    preview["rank_disabled_reason"] = str(support_state.get("disabled_reason", "unknown"))
    return preview


def _style_success_supervision_summary(hparams, resolved_condition_maps, runtime_preview=None):
    style_success_lambda = float(hparams.get("lambda_style_success_rank", 0.0) or 0.0)
    self_ref_scale = float(hparams.get("style_success_self_ref_scale", 0.35) or 0.35)
    memory_fallback_scale = float(
        hparams.get(
            "style_success_memory_fallback_scale",
            STYLE_SUCCESS_MEMORY_FALLBACK_SCALE,
        )
        or STYLE_SUCCESS_MEMORY_FALLBACK_SCALE
    )
    proxy_negative_threshold = float(hparams.get("style_success_proxy_negative_threshold", 1.25) or 1.25)
    proxy_negative_min_count = int(hparams.get("style_success_proxy_negative_min_count", 2) or 0)
    proxy_min_batch = int(hparams.get("style_success_proxy_min_batch", 4) or 0)
    proxy_use_rate_proxy = bool(
        resolve_style_success_bool_flag(
            hparams.get("style_success_proxy_use_rate_proxy", False),
            default=False,
        )
    )
    label_rank_scale = float(hparams.get("style_success_label_rank_scale", 1.0))
    label_plus_proxy_backfill_scale = float(
        hparams.get("style_success_label_plus_proxy_backfill_scale", 0.75)
    )
    proxy_rank_scale = float(
        hparams.get(
            "style_success_proxy_rank_scale",
            hparams.get("style_success_proxy_only_rank_row_scale", 0.5),
        )
    )
    weak_label_counts = {
        field: int(len(resolved_condition_maps.get(field, {})))
        for field in ("emotion", "accent")
    }
    usable_weak_label_fields = {
        field: count
        for field, count in weak_label_counts.items()
        if int(count) > 1
    }
    pair_enabled = style_success_lambda > 0.0
    weak_label_ranking_available = pair_enabled and len(usable_weak_label_fields) > 0
    mode = (
        "disabled"
        if not pair_enabled
        else (
            "paired_plus_weak_label_ranking"
            if weak_label_ranking_available
            else "paired_only"
        )
    )
    return {
        "lambda_style_success_rank": style_success_lambda,
        "mode": mode,
        "pair_enabled": bool(pair_enabled),
        "weak_label_ranking_available_from_artifacts": bool(weak_label_ranking_available),
        "usable_weak_label_fields": usable_weak_label_fields,
        "weak_label_label_counts": weak_label_counts,
        "proxy_negative_fallback_enabled": bool(pair_enabled),
        "self_ref_scale": self_ref_scale,
        "memory_fallback_scale": memory_fallback_scale,
        "proxy_negative_threshold": proxy_negative_threshold,
        "proxy_negative_min_count": proxy_negative_min_count,
        "proxy_min_batch": proxy_min_batch,
        "proxy_use_rate_proxy": proxy_use_rate_proxy,
        "label_rank_scale": label_rank_scale,
        "label_plus_proxy_backfill_scale": label_plus_proxy_backfill_scale,
        "proxy_rank_scale": proxy_rank_scale,
        "proxy_feature_family": (
            "acoustic_prosodic_plus_rate_proxy"
            if proxy_use_rate_proxy
            else "acoustic_prosodic_only"
        ),
        "runtime_negative_strategy": (
            "disabled"
            if not pair_enabled
            else (
                "label_priority_with_proxy_backfill_plus_rate_proxy"
                if proxy_use_rate_proxy
                else "label_priority_with_proxy_backfill_acoustic_only"
            )
        ),
        "runtime_preview": runtime_preview
        if isinstance(runtime_preview, dict)
        else {"available": False, "reason": "preview_unavailable"},
        "source": "condition_artifacts",
        "note": (
            "artifact-level label summary plus optional small-batch runtime preview; actual rank negatives still depend "
            "on per-item batch composition, and label negatives remain authoritative while proxy negatives only backfill "
            "rows that still fall below the proxy minimum count. Canonical mainline also keeps a proxy min-batch gate and source-aware rank "
            "downscaling so small batches and proxy-only negatives stay conservative. The text/content-length rate "
            "proxy remains disabled by default so the fallback negatives stay acoustic/prosodic unless research "
            "overrides explicitly opt in."
        ),
    }


def _resolve_config_chain(config_path, loaded=None):
    if not config_path:
        return []
    normalized = os.path.normpath(str(config_path))
    if not os.path.exists(normalized):
        return []
    if loaded is None:
        loaded = set()
    if normalized in loaded:
        return []
    loaded.add(normalized)
    chain = []
    try:
        with open(normalized, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
    except Exception:
        config_data = {}
    base_configs = config_data.get("base_config")
    if base_configs:
        if not isinstance(base_configs, list):
            base_configs = [base_configs]
        for base_config in base_configs:
            resolved_base = str(base_config)
            if resolved_base.startswith('.'):
                resolved_base = os.path.normpath(os.path.join(os.path.dirname(normalized), resolved_base))
            chain.extend(_resolve_config_chain(resolved_base, loaded=loaded))
    chain.append(normalized)
    return chain


def _scan_top_level_config_keys(config_path):
    seen = {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if line.startswith((" ", "\t")):
                    continue
                match = TOP_LEVEL_KEY_PATTERN.match(line)
                if match is None:
                    continue
                key = match.group(1)
                seen.setdefault(key, []).append(lineno)
    except FileNotFoundError:
        return {}, {}
    duplicates = {key: lines for key, lines in seen.items() if len(lines) > 1}
    return duplicates, seen


def _scan_config_chain(config_path):
    config_chain = _resolve_config_chain(config_path)
    duplicate_occurrences = {}
    top_level_occurrences = {}
    for path in config_chain:
        duplicates, seen = _scan_top_level_config_keys(path)
        if duplicates:
            duplicate_occurrences[path] = duplicates
        for key, line_numbers in seen.items():
            entries = top_level_occurrences.setdefault(key, [])
            for line_number in line_numbers:
                entries.append({"path": path, "line": int(line_number)})
    return config_chain, duplicate_occurrences, top_level_occurrences


def _jsonable(value):
    try:
        import torch
    except Exception:  # pragma: no cover - torch is available in normal runs
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def run_prep(args):
    set_hparams(config=args.config, print_hparams=False)
    config_chain, config_duplicates, config_top_level_keys = _scan_config_chain(args.config)
    if getattr(args, "binary_data_dir", None):
        hparams["binary_data_dir"] = str(args.binary_data_dir)
    controls = resolve_style_mainline_controls(hparams=hparams)
    resolved_profile = resolve_style_profile(
        {
            "style_profile": hparams.get("style_profile", "strong_style"),
            "style_trace_mode": hparams.get("style_trace_mode", None),
            "style_strength": hparams.get("style_strength", None),
            "style_temperature": hparams.get("style_temperature", None),
            "style_to_pitch_residual_include_timbre": hparams.get(
                "style_to_pitch_residual_include_timbre",
                None,
            ),
            "global_style_trace_blend": hparams.get("global_style_trace_blend", None),
            "style_query_global_summary_scale": hparams.get("style_query_global_summary_scale", None),
            "dynamic_timbre_style_condition_scale": hparams.get("dynamic_timbre_style_condition_scale", None),
            "dynamic_timbre_temperature": hparams.get("dynamic_timbre_temperature", None),
            "dynamic_timbre_coarse_style_context_scale": hparams.get(
                "dynamic_timbre_coarse_style_context_scale",
                None,
            ),
            "dynamic_timbre_query_style_condition_scale": hparams.get(
                "dynamic_timbre_query_style_condition_scale",
                None,
            ),
            "dynamic_timbre_use_tvt": hparams.get("dynamic_timbre_use_tvt", None),
            "dynamic_timbre_tvt_prior_scale": hparams.get(
                "dynamic_timbre_tvt_prior_scale",
                None,
            ),
            "runtime_dynamic_timbre_style_budget_enabled": hparams.get(
                "runtime_dynamic_timbre_style_budget_enabled",
                None,
            ),
            "runtime_dynamic_timbre_style_budget_ratio": hparams.get(
                "runtime_dynamic_timbre_style_budget_ratio",
                None,
            ),
            "runtime_dynamic_timbre_style_budget_margin": hparams.get(
                "runtime_dynamic_timbre_style_budget_margin",
                None,
            ),
            "allow_explicit_dynamic_timbre_strength": hparams.get(
                "allow_explicit_dynamic_timbre_strength",
                False,
            ),
        },
        preset=hparams.get("style_profile", "strong_style"),
    )
    regularization = resolve_control_regularization_config(hparams, global_step=0)
    use_external_speaker_verifier = bool(hparams.get("use_external_speaker_verifier", False))
    freeze_internal_identity_encoder_for_loss = bool(
        hparams.get("freeze_internal_identity_encoder_for_loss", True)
    )
    identity_loss_constraint_mode = (
        "external_speaker_verifier"
        if use_external_speaker_verifier
        else (
            "internal_encoder_frozen_for_loss"
            if freeze_internal_identity_encoder_for_loss
            else "internal_encoder_trainable_for_loss"
        )
    )
    upper_bound_preview_0 = resolve_expressive_upper_bound_progress(0, hparams=hparams)
    upper_bound_preview_20000 = resolve_expressive_upper_bound_progress(20000, hparams=hparams)
    upper_bound_preview_50000 = resolve_expressive_upper_bound_progress(50000, hparams=hparams)
    upper_bound_preview_80000 = resolve_expressive_upper_bound_progress(80000, hparams=hparams)
    batch_controls = None
    batch_controls_error = None
    style_success_runtime_preview = {
        "available": False,
        "reason": "dataset_preview_unavailable",
    }
    boundary_preview = []
    try:
        train_dataset = ConanDataset(prefix="train", shuffle=False)
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            batch = train_dataset.collater([sample])
            batch_kwargs = {}
            batch_kwargs.update(build_control_kwargs(batch))
            batch_kwargs.update(build_style_runtime_kwargs(batch))
            batch_controls = resolve_style_mainline_controls(batch_kwargs, hparams=hparams)
            preview_batch_size = min(4, len(train_dataset))
            if preview_batch_size > 1:
                preview_samples = [train_dataset[idx] for idx in range(preview_batch_size)]
                preview_batch = train_dataset.collater(preview_samples)
                style_success_runtime_preview = _style_success_runtime_preview(preview_batch, hparams)
            else:
                style_success_runtime_preview = {
                    "available": False,
                    "reason": "dataset_size_le_1",
                    "batch_size": int(preview_batch_size),
                }
            preview_count = min(5, len(train_dataset))
            content_padding_idx = int(hparams.get("content_padding_idx", 101))
            silent_token = hparams.get("silent_token", None)
            boundary_radius = int(hparams.get("dynamic_timbre_boundary_radius", 2))
            for preview_idx in range(preview_count):
                preview_sample = train_dataset[preview_idx]
                content = preview_sample["content"].unsqueeze(0)
                padding_mask = content.eq(content_padding_idx)
                boundary_mask, boundary_meta = build_dynamic_timbre_boundary_mask(
                    content,
                    padding_mask=padding_mask,
                    padding_idx=content_padding_idx,
                    silent_token=silent_token,
                    radius=boundary_radius,
                    return_metadata=True,
                )
                valid = (~padding_mask).float()
                valid_count = float(valid.sum().item())
                if valid_count > 0 and isinstance(boundary_mask, torch.Tensor):
                    boundary_coverage = float(
                        ((boundary_mask.squeeze(-1) * valid).sum() / valid.sum().clamp_min(1.0)).item()
                    )
                else:
                    boundary_coverage = None
                transition_rate = boundary_meta.get("transition_rate") if isinstance(boundary_meta, dict) else None
                dense_units = boundary_meta.get("dense_units_detected") if isinstance(boundary_meta, dict) else None
                boundary_preview.append(
                    {
                        "idx": int(preview_idx),
                        "length": int(valid_count),
                        "boundary_coverage": boundary_coverage,
                        "transition_rate": (
                            float(transition_rate[0].item()) if transition_rate is not None else None
                        ),
                        "dense_units_detected": (
                            bool(dense_units[0].item()) if dense_units is not None else None
                        ),
                    }
                )
        else:
            batch_controls_error = "empty_train_dataset"
            style_success_runtime_preview = {
                "available": False,
                "reason": "empty_train_dataset",
                "batch_size": 0,
            }
    except Exception as exc:  # pragma: no cover - surfaced through prep output
        batch_controls_error = f"{type(exc).__name__}: {exc}"
        style_success_runtime_preview = {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    checks = []
    _check_true(
        checks,
        "config_top_level_duplicate_keys_absent",
        len(config_duplicates) == 0,
        actual=config_duplicates if config_duplicates else {},
        expected="no duplicate top-level keys",
    )
    for key in CANONICAL_REMOVED_CONFIG_KEYS:
        locations = config_top_level_keys.get(key, [])
        _check_true(
            checks,
            f"config_removed_key_absent::{key}",
            len(locations) == 0,
            actual={"present": bool(locations), "locations": locations},
            expected={"present": False},
        )
    _check_equal(checks, "reference_contract_mode", hparams.get("reference_contract_mode"), "collapsed_reference")
    _check_equal(checks, "decoder_style_condition_mode", controls.mode, "mainline_full")
    _check_equal(checks, "global_timbre_to_pitch", bool(controls.global_timbre_to_pitch), False)
    _check_equal(checks, "style_to_pitch_residual", bool(controls.style_to_pitch_residual), True)
    _check_equal(
        checks,
        "style_to_pitch_residual_include_timbre",
        bool(getattr(controls, "style_to_pitch_residual_include_timbre", False)),
        False,
    )
    _check_equal(checks, "style_to_pitch_residual_mode", controls.style_to_pitch_residual_mode, "auto")
    _check_equal(checks, "style_trace_mode", controls.style_trace_mode, "dual")
    _check_equal(checks, "style_router_enabled", bool(controls.style_router_enabled), True)
    _check_equal(checks, "style_memory_mode", controls.style_memory_mode, "slow")
    _check_equal(checks, "dynamic_timbre_memory_mode", controls.dynamic_timbre_memory_mode, "slow")
    _check_close(checks, "global_style_trace_blend", hparams.get("global_style_trace_blend", 0.0), 0.0)
    _check_close(
        checks,
        "style_query_global_summary_scale",
        hparams.get("style_query_global_summary_scale", 0.0),
        0.0,
    )
    _check_close(
        checks,
        "dynamic_timbre_coarse_style_context_scale",
        hparams.get("dynamic_timbre_coarse_style_context_scale", 0.0),
        0.0,
    )
    _check_equal(
        checks,
        "dynamic_timbre_style_context_stopgrad",
        bool(hparams.get("dynamic_timbre_style_context_stopgrad", True)),
        True,
    )
    _check_equal(
        checks,
        "allow_explicit_dynamic_timbre_strength",
        bool(hparams.get("allow_explicit_dynamic_timbre_strength", False)),
        False,
    )
    _check_equal(
        checks,
        "allow_split_reference_inputs",
        bool(hparams.get("allow_split_reference_inputs", False)),
        False,
    )
    _check_equal(
        checks,
        "emit_collapsed_reference_aliases",
        bool(hparams.get("emit_collapsed_reference_aliases", False)),
        False,
    )
    _check_equal(checks, "control_loss_profile", regularization.get("control_loss_profile"), "mainline_minimal")
    _check_equal(
        checks,
        "style_profile_track",
        resolved_profile.get("style_profile_track", resolved_profile.get("track")),
        "mainline",
    )
    _check_equal(
        checks,
        "style_profile",
        hparams.get("style_profile", "strong_style"),
        "strong_style",
    )
    _check_close(
        checks,
        "style_profile_controls_style_strength",
        controls.style_strength,
        resolved_profile.get("style_strength", 1.35),
        tol=1e-6,
    )
    _check_close(
        checks,
        "style_profile_controls_dynamic_timbre_strength",
        controls.dynamic_timbre_strength,
        resolved_profile.get("dynamic_timbre_strength", controls.dynamic_timbre_strength),
        tol=1e-6,
    )
    _check_equal(
        checks,
        "style_profile_controls_dynamic_timbre_strength_source",
        getattr(controls, "dynamic_timbre_strength_source", None),
        "derived_from_style_strength",
    )
    _check_close(
        checks,
        "style_profile_controls_style_temperature",
        controls.style_temperature,
        resolved_profile.get("style_temperature", 1.2),
        tol=1e-6,
    )
    _check_close(
        checks,
        "style_profile_controls_dynamic_timbre_temperature",
        controls.dynamic_timbre_temperature,
        resolved_profile.get("dynamic_timbre_temperature", 1.0),
        tol=1e-6,
    )
    _check_equal(
        checks,
        "style_profile_controls_dynamic_timbre_use_tvt",
        bool(controls.dynamic_timbre_use_tvt),
        bool(resolved_profile.get("dynamic_timbre_use_tvt", True)),
    )
    _check_close(
        checks,
        "style_profile_controls_dynamic_timbre_tvt_prior_scale",
        controls.dynamic_timbre_tvt_prior_scale,
        resolved_profile.get("dynamic_timbre_tvt_prior_scale", 1.0),
        tol=1e-6,
    )
    _check_equal(
        checks,
        "upper_bound_curriculum_enabled",
        bool(hparams.get("upper_bound_curriculum_enabled", True)),
        True,
    )
    _check_close(
        checks,
        "upper_bound_curriculum_start_steps",
        hparams.get("upper_bound_curriculum_start_steps", 20000),
        20000,
    )
    _check_close(
        checks,
        "upper_bound_curriculum_end_steps",
        hparams.get("upper_bound_curriculum_end_steps", 80000),
        80000,
    )
    _check_close(
        checks,
        "upper_bound_progress_step_0",
        upper_bound_preview_0,
        0.0,
        tol=1e-6,
    )
    _check_close(
        checks,
        "upper_bound_progress_step_20000",
        upper_bound_preview_20000,
        0.0,
        tol=1e-6,
    )
    _check_close(
        checks,
        "upper_bound_progress_step_50000",
        upper_bound_preview_50000,
        0.5,
        tol=1e-6,
    )
    _check_close(
        checks,
        "upper_bound_progress_step_80000",
        upper_bound_preview_80000,
        1.0,
        tol=1e-6,
    )
    _check_true(
        checks,
        "train_batch_mainline_controls_resolvable",
        batch_controls is not None,
        actual=batch_controls_error if batch_controls is None else "resolved",
        expected="resolved",
    )
    preview_coverages = [
        float(item["boundary_coverage"])
        for item in boundary_preview
        if item.get("boundary_coverage") is not None
    ]
    _check_true(
        checks,
        "dynamic_timbre_boundary_preview_available",
        len(preview_coverages) > 0,
        actual=len(preview_coverages),
        expected="> 0",
    )
    if preview_coverages:
        _check_true(
            checks,
            "dynamic_timbre_boundary_not_global_on_real_data",
            float(max(preview_coverages)) < 0.50,
            actual={
                "max_boundary_coverage": float(max(preview_coverages)),
                "mean_boundary_coverage": float(sum(preview_coverages) / len(preview_coverages)),
            },
            expected="max_boundary_coverage < 0.50 on sampled real-data batches",
        )
    if batch_controls is not None:
        _check_close(
            checks,
            "train_batch_style_strength_matches_profile",
            batch_controls.style_strength,
            resolved_profile.get("style_strength", batch_controls.style_strength),
            tol=1e-6,
        )
        _check_close(
            checks,
            "train_batch_dynamic_timbre_strength_matches_profile",
            batch_controls.dynamic_timbre_strength,
            resolved_profile.get("dynamic_timbre_strength", batch_controls.dynamic_timbre_strength),
            tol=1e-6,
        )
        _check_equal(
            checks,
            "train_batch_dynamic_timbre_strength_source_matches_profile",
            getattr(batch_controls, "dynamic_timbre_strength_source", None),
            "derived_from_style_strength",
        )
        _check_close(
            checks,
            "train_batch_style_temperature_matches_profile",
            batch_controls.style_temperature,
            resolved_profile.get("style_temperature", batch_controls.style_temperature),
        )
        _check_close(
            checks,
            "train_batch_dynamic_timbre_temperature_matches_profile",
            batch_controls.dynamic_timbre_temperature,
            resolved_profile.get("dynamic_timbre_temperature", batch_controls.dynamic_timbre_temperature),
        )
        _check_equal(
            checks,
            "train_batch_dynamic_timbre_use_tvt_matches_profile",
            bool(batch_controls.dynamic_timbre_use_tvt),
            bool(resolved_profile.get("dynamic_timbre_use_tvt", batch_controls.dynamic_timbre_use_tvt)),
        )
        _check_close(
            checks,
            "train_batch_dynamic_timbre_tvt_prior_scale_matches_profile",
            batch_controls.dynamic_timbre_tvt_prior_scale,
            resolved_profile.get(
                "dynamic_timbre_tvt_prior_scale",
                batch_controls.dynamic_timbre_tvt_prior_scale,
            ),
        )
    _check_close(
        checks,
        "runtime_dynamic_timbre_style_budget_ratio",
        hparams.get("runtime_dynamic_timbre_style_budget_ratio", 0.40),
        0.40,
    )
    _check_equal(
        checks,
        "runtime_dynamic_timbre_style_budget_enabled",
        bool(hparams.get("runtime_dynamic_timbre_style_budget_enabled", True)),
        True,
    )
    _check_close(
        checks,
        "runtime_dynamic_timbre_style_budget_margin",
        hparams.get("runtime_dynamic_timbre_style_budget_margin", 0.0),
        0.0,
    )
    _check_close(
        checks,
        "runtime_dynamic_timbre_style_budget_slow_style_weight",
        hparams.get("runtime_dynamic_timbre_style_budget_slow_style_weight", 1.0),
        1.0,
    )
    _check_close(
        checks,
        "runtime_dynamic_timbre_style_budget_epsilon",
        hparams.get("runtime_dynamic_timbre_style_budget_epsilon", 1e-6),
        1e-6,
        tol=1e-12,
    )
    _check_close(checks, "dynamic_timbre_budget_ratio", hparams.get("dynamic_timbre_budget_ratio", 0.40), 0.40)
    _check_close(
        checks,
        "decoder_late_timbre_owner_ratio",
        hparams.get("decoder_late_timbre_owner_ratio", 0.50),
        0.50,
    )
    _check_close(
        checks,
        "decoder_dynamic_timbre_late_no_style_scale",
        hparams.get("decoder_dynamic_timbre_late_no_style_scale", 0.0),
        0.0,
    )
    _check_close(
        checks,
        "dynamic_timbre_style_condition_scale",
        hparams.get("dynamic_timbre_style_condition_scale", 0.35),
        0.35,
    )
    _check_close(
        checks,
        "dynamic_timbre_query_style_condition_scale",
        hparams.get("dynamic_timbre_query_style_condition_scale", 0.0),
        0.0,
    )
    _check_equal(
        checks,
        "dynamic_timbre_use_tvt",
        bool(hparams.get("dynamic_timbre_use_tvt", True)),
        True,
    )
    _check_close(
        checks,
        "dynamic_timbre_tvt_prior_scale",
        hparams.get("dynamic_timbre_tvt_prior_scale", 1.0),
        1.0,
    )
    _check_close(
        checks,
        "tv_timbre_gate_bias_init",
        hparams.get("tv_timbre_gate_bias_init", -1.0),
        -1.0,
    )
    _check_equal(
        checks,
        "allow_item_style_strength_override",
        bool(hparams.get("allow_item_style_strength_override", False)),
        False,
    )
    _check_equal(
        checks,
        "use_external_speaker_verifier",
        use_external_speaker_verifier,
        False,
    )
    _check_equal(
        checks,
        "freeze_internal_identity_encoder_for_loss",
        freeze_internal_identity_encoder_for_loss,
        True,
    )
    _check_equal(
        checks,
        "identity_loss_constraint_mode",
        identity_loss_constraint_mode,
        "internal_encoder_frozen_for_loss",
    )
    _check_equal(
        checks,
        "speaker_verifier_detach_input",
        bool(hparams.get("speaker_verifier_detach_input", False)),
        False,
    )
    _check_equal(checks, "reference_curriculum_mode", hparams.get("reference_curriculum_mode"), "bernoulli_cosine")
    _check_close(
        checks,
        "reference_curriculum_start_steps",
        hparams.get("reference_curriculum_start_steps", hparams.get("forcing", 0)),
        20000,
    )
    _check_close(
        checks,
        "reference_curriculum_end_steps",
        hparams.get("reference_curriculum_end_steps", hparams.get("random_speaker_steps", 0)),
        100000,
    )
    _check_close(
        checks,
        "reference_curriculum_external_prob_init",
        hparams.get("reference_curriculum_external_prob_init", 0.0),
        0.0,
    )
    _check_close(
        checks,
        "reference_curriculum_external_prob_final",
        hparams.get("reference_curriculum_external_prob_final", 1.0),
        1.0,
    )
    _check_close(
        checks,
        "reference_curriculum_self_ref_floor",
        hparams.get("reference_curriculum_self_ref_floor", 0.0),
        0.0,
    )
    _check_equal(
        checks,
        "reference_curriculum_sample_mode",
        str(hparams.get("reference_curriculum_sample_mode", "batch")).strip().lower(),
        "batch",
    )
    _check_equal(checks, "forcing_schedule_mode", hparams.get("forcing_schedule_mode"), "bernoulli_cosine")
    _check_close(
        checks,
        "forcing_legacy_cut",
        hparams.get("forcing", 0),
        20000,
    )
    _check_close(
        checks,
        "forcing_decay_start_steps",
        hparams.get("forcing_decay_start_steps", hparams.get("forcing", 0)),
        12000,
    )
    _check_close(
        checks,
        "forcing_decay_end_steps",
        hparams.get("forcing_decay_end_steps", hparams.get("forcing", 0)),
        60000,
    )
    _check_close(checks, "forcing_prob_init", hparams.get("forcing_prob_init", 1.0), 1.0)
    _check_close(checks, "forcing_prob_final", hparams.get("forcing_prob_final", 0.0), 0.0)
    _check_runtime_requirement_pins(
        checks,
        ROOT_DIR / "requirements.txt",
        package_names=("torch", "torchaudio", "torchdyn", "textgrid", "g2p_en", "nltk"),
    )
    repo_requirements = _load_repo_requirements(ROOT_DIR)
    pinned_torchaudio = repo_requirements.get("torchaudio")
    pinned_python_range = _resolve_torchaudio_python_range(pinned_torchaudio)
    if pinned_python_range is not None:
        py_version = tuple(int(part) for part in sys.version_info[:2])
        py_min, py_max = pinned_python_range
        _check_true(
            checks,
            "runtime_python_version_compatible_with_pinned_torchaudio",
            py_min <= py_version <= py_max,
            actual={
                "python_version": f"{py_version[0]}.{py_version[1]}",
                "pinned_torchaudio": pinned_torchaudio,
            },
            expected={
                "python_min": f"{py_min[0]}.{py_min[1]}",
                "python_max": f"{py_max[0]}.{py_max[1]}",
                "pinned_torchaudio": pinned_torchaudio,
            },
        )
    _check_importable(checks, "runtime_import_torchaudio", "torchaudio")
    _check_importable(checks, "runtime_import_textgrid", "textgrid")
    _check_importable(checks, "runtime_import_torchdyn", "torchdyn")
    _check_importable(checks, "runtime_import_g2p_en", "g2p_en")
    _check_importable(checks, "runtime_import_tasks_Conan_Conan", "tasks.Conan.Conan")
    _check_runtime_callable(
        checks,
        "runtime_local_modules_Conan_Conan_init",
        lambda: _check_local_conan_model_init(args.config),
        expected="local Conan(0, hparams) init succeeds with global_hparams=False",
    )
    _check_runtime_callable(
        checks,
        "runtime_nltk_tagger_for_g2p_en",
        _check_nltk_tagger_for_g2p_en,
        expected="nltk averaged perceptron tagger available for g2p_en",
    )
    _check_runtime_callable(
        checks,
        "runtime_nltk_cmudict_for_g2p_en",
        _check_nltk_cmudict_for_g2p_en,
        expected="nltk cmudict available for g2p_en",
    )
    condition_artifact_dirs = [hparams.get("binary_data_dir"), hparams.get("processed_data_dir")]
    resolved_condition_maps = _check_condition_artifacts(
        checks,
        condition_artifact_dirs,
    )
    style_success_supervision = _style_success_supervision_summary(
        hparams,
        resolved_condition_maps,
        runtime_preview=style_success_runtime_preview,
    )
    checks.append(
        {
            "name": "style_success_supervision_mode",
            "ok": True,
            "actual": _jsonable(style_success_supervision),
            "expected": "informational",
        }
    )
    checks.append(
        {
            "name": "style_success_runtime_preview",
            "ok": True,
            "actual": _jsonable(style_success_runtime_preview),
            "expected": "informational",
        }
    )
    _check_close(
        checks,
        "random_speaker_steps_matches_curriculum_end",
        hparams.get("random_speaker_steps", 0),
        hparams.get("reference_curriculum_end_steps", hparams.get("random_speaker_steps", 0)),
    )
    reference_start = int(hparams.get("reference_curriculum_start_steps", hparams.get("forcing", 0)))
    reference_end = int(hparams.get("reference_curriculum_end_steps", hparams.get("random_speaker_steps", 0)))
    forcing_start = int(hparams.get("forcing_decay_start_steps", hparams.get("forcing", 0)))
    forcing_end = int(hparams.get("forcing_decay_end_steps", hparams.get("forcing", 0)))
    _check_true(
        checks,
        "forcing_decay_starts_no_later_than_reference_curriculum",
        forcing_start <= reference_start,
        actual={"forcing_decay_start_steps": forcing_start, "reference_curriculum_start_steps": reference_start},
        expected="forcing_decay_start_steps <= reference_curriculum_start_steps",
    )
    _check_true(
        checks,
        "forcing_decay_ends_no_later_than_reference_curriculum",
        forcing_end <= reference_end,
        actual={"forcing_decay_end_steps": forcing_end, "reference_curriculum_end_steps": reference_end},
        expected="forcing_decay_end_steps <= reference_curriculum_end_steps",
    )
    upper_bound_start = int(hparams.get("upper_bound_curriculum_start_steps", 20000))
    upper_bound_end = int(hparams.get("upper_bound_curriculum_end_steps", 80000))
    _check_true(
        checks,
        "upper_bound_curriculum_starts_no_earlier_than_forcing_decay",
        upper_bound_start >= forcing_start,
        actual={
            "upper_bound_curriculum_start_steps": upper_bound_start,
            "forcing_decay_start_steps": forcing_start,
        },
        expected="upper_bound_curriculum_start_steps >= forcing_decay_start_steps",
    )
    _check_true(
        checks,
        "upper_bound_curriculum_ends_no_later_than_reference_curriculum",
        upper_bound_end <= reference_end,
        actual={
            "upper_bound_curriculum_end_steps": upper_bound_end,
            "reference_curriculum_end_steps": reference_end,
        },
        expected="upper_bound_curriculum_end_steps <= reference_curriculum_end_steps",
    )
    reference_preview_40000 = resolve_reference_curriculum(40000, hparams)
    reference_preview_70000 = resolve_reference_curriculum(70000, hparams)
    forcing_preview_20000 = resolve_forcing_schedule(20000, hparams)
    _check_close(
        checks,
        "reference_curriculum_external_prob_step_40000",
        reference_preview_40000.get("external_prob", 0.0),
        0.1464466094,
        tol=1e-6,
    )
    _check_close(
        checks,
        "reference_curriculum_external_prob_step_70000",
        reference_preview_70000.get("external_prob", 0.0),
        0.6913417162,
        tol=1e-6,
    )
    _check_close(
        checks,
        "forcing_prob_step_20000",
        forcing_preview_20000.get("forcing_prob", 0.0),
        0.9330127019,
        tol=1e-6,
    )
    active_mainline_control_keys = tuple(
        key for key in MAINLINE_MINIMAL_CONTROL_LAMBDAS if float(hparams.get(key, 0.0)) > 0.0
    )
    checks.append(
        {
            "name": "mainline_minimal_active_control_loss_count",
            "ok": len(active_mainline_control_keys) <= len(MAINLINE_MINIMAL_CONTROL_LAMBDAS),
            "actual": len(active_mainline_control_keys),
            "expected": f"<= {len(MAINLINE_MINIMAL_CONTROL_LAMBDAS)}",
        }
    )
    checks.append(
        {
            "name": "mainline_minimal_active_control_losses",
            "ok": set(active_mainline_control_keys) == set(MAINLINE_MINIMAL_CONTROL_LAMBDAS),
            "actual": list(active_mainline_control_keys),
            "expected": list(MAINLINE_MINIMAL_CONTROL_LAMBDAS),
        }
    )

    for key in REQUIRED_POSITIVE_KEYS:
        checks.append(
            {
                "name": key,
                "ok": float(hparams.get(key, 0.0)) > 0.0,
                "actual": float(hparams.get(key, 0.0)),
                "expected": "> 0",
            }
        )
    for key in REQUIRED_ZERO_KEYS:
        _check_close(checks, key, hparams.get(key, 0.0), 0.0)
    for key in REQUIRED_PRESENT_KEYS:
        checks.append(
            {
                "name": f"{key}_present",
                "ok": key in hparams,
                "actual": key in hparams,
                "expected": True,
            }
        )

    overlap_margin = float(hparams.get("style_timbre_runtime_overlap_margin", 0.10))
    checks.append(
        {
            "name": "style_timbre_runtime_overlap_margin_range",
            "ok": 0.0 <= overlap_margin < 1.0,
            "actual": overlap_margin,
            "expected": "[0.0, 1.0)",
        }
    )
    overlap_lambda = float(hparams.get("lambda_style_timbre_runtime_overlap", 0.0))
    checks.append(
        {
            "name": "lambda_style_timbre_runtime_overlap_nonnegative",
            "ok": overlap_lambda >= 0.0,
            "actual": overlap_lambda,
            "expected": ">= 0.0",
        }
    )
    overlap_use_abs = hparams.get("style_timbre_runtime_overlap_use_abs", True)
    checks.append(
        {
            "name": "style_timbre_runtime_overlap_use_abs_boollike",
            "ok": isinstance(overlap_use_abs, (bool, int, float)),
            "actual": overlap_use_abs,
            "expected": "bool-like",
        }
    )
    pitch_residual_huber_delta = float(hparams.get("pitch_residual_huber_delta", 0.02))
    checks.append(
        {
            "name": "pitch_residual_huber_delta_positive",
            "ok": pitch_residual_huber_delta > 0.0,
            "actual": pitch_residual_huber_delta,
            "expected": "> 0.0",
        }
    )
    pitch_residual_budget_weight = float(hparams.get("pitch_residual_budget_weight", 0.15))
    checks.append(
        {
            "name": "pitch_residual_budget_weight_nonnegative",
            "ok": pitch_residual_budget_weight >= 0.0,
            "actual": pitch_residual_budget_weight,
            "expected": ">= 0.0",
        }
    )
    pitch_residual_budget_margin = float(hparams.get("pitch_residual_budget_margin", 0.015))
    checks.append(
        {
            "name": "pitch_residual_budget_margin_nonnegative",
            "ok": pitch_residual_budget_margin >= 0.0,
            "actual": pitch_residual_budget_margin,
            "expected": ">= 0.0",
        }
    )
    dynamic_timbre_budget_uv_floor = float(hparams.get("dynamic_timbre_budget_uv_floor", 0.25))
    checks.append(
        {
            "name": "dynamic_timbre_budget_uv_floor_range",
            "ok": 0.0 <= dynamic_timbre_budget_uv_floor <= 1.0,
            "actual": dynamic_timbre_budget_uv_floor,
            "expected": "[0.0, 1.0]",
        }
    )
    dynamic_timbre_budget_energy_floor = float(hparams.get("dynamic_timbre_budget_energy_floor", 0.10))
    checks.append(
        {
            "name": "dynamic_timbre_budget_energy_floor_range",
            "ok": 0.0 <= dynamic_timbre_budget_energy_floor <= 1.0,
            "actual": dynamic_timbre_budget_energy_floor,
            "expected": "[0.0, 1.0]",
        }
    )
    dynamic_timbre_budget_energy_power = float(hparams.get("dynamic_timbre_budget_energy_power", 0.5))
    checks.append(
        {
            "name": "dynamic_timbre_budget_energy_power_positive",
            "ok": dynamic_timbre_budget_energy_power > 0.0,
            "actual": dynamic_timbre_budget_energy_power,
            "expected": "> 0.0",
        }
    )
    dynamic_timbre_budget_energy_quantile = float(hparams.get("dynamic_timbre_budget_energy_quantile", 0.90))
    checks.append(
        {
            "name": "dynamic_timbre_budget_energy_quantile_range",
            "ok": 0.0 < dynamic_timbre_budget_energy_quantile <= 1.0,
            "actual": dynamic_timbre_budget_energy_quantile,
            "expected": "(0.0, 1.0]",
        }
    )
    for key, default in (
        ("dynamic_timbre_budget_prebudget_term_weight", 1.0),
        ("dynamic_timbre_budget_mid_term_weight", 0.5),
        ("dynamic_timbre_budget_late_term_weight", 0.75),
    ):
        value = float(hparams.get(key, default))
        checks.append(
            {
                "name": f"{key}_nonnegative",
                "ok": value >= 0.0,
                "actual": value,
                "expected": ">= 0.0",
            }
        )
    dynamic_timbre_budget_term_weight_sum = sum(
        max(float(hparams.get(key, default)), 0.0)
        for key, default in (
            ("dynamic_timbre_budget_prebudget_term_weight", 1.0),
            ("dynamic_timbre_budget_mid_term_weight", 0.5),
            ("dynamic_timbre_budget_late_term_weight", 0.75),
        )
    )
    checks.append(
        {
            "name": "dynamic_timbre_budget_term_weight_sum_positive",
            "ok": dynamic_timbre_budget_term_weight_sum > 0.0,
            "actual": dynamic_timbre_budget_term_weight_sum,
            "expected": "> 0.0",
        }
    )
    style_success_proxy_negative_threshold = float(hparams.get("style_success_proxy_negative_threshold", 1.25))
    checks.append(
        {
            "name": "style_success_proxy_negative_threshold_nonnegative",
            "ok": style_success_proxy_negative_threshold >= 0.0,
            "actual": style_success_proxy_negative_threshold,
            "expected": ">= 0.0",
        }
    )
    style_success_proxy_negative_min_count = int(hparams.get("style_success_proxy_negative_min_count", 2))
    checks.append(
        {
            "name": "style_success_proxy_negative_min_count_nonnegative",
            "ok": style_success_proxy_negative_min_count >= 0,
            "actual": style_success_proxy_negative_min_count,
            "expected": ">= 0",
        }
    )
    style_success_proxy_use_rate_proxy = hparams.get("style_success_proxy_use_rate_proxy", False)
    checks.append(
        {
            "name": "style_success_proxy_use_rate_proxy_boollike",
            "ok": _is_boollike_value(style_success_proxy_use_rate_proxy),
            "actual": style_success_proxy_use_rate_proxy,
            "expected": "bool-like",
        }
    )
    style_success_proxy_min_batch = int(hparams.get("style_success_proxy_min_batch", 4))
    checks.append(
        {
            "name": "style_success_proxy_min_batch_nonnegative",
            "ok": style_success_proxy_min_batch >= 0,
            "actual": style_success_proxy_min_batch,
            "expected": ">= 0",
        }
    )
    style_success_label_rank_scale = float(
        hparams.get("style_success_label_rank_scale", 1.0)
    )
    checks.append(
        {
            "name": "style_success_label_rank_scale_range",
            "ok": 0.0 <= style_success_label_rank_scale <= 1.0,
            "actual": style_success_label_rank_scale,
            "expected": "[0.0, 1.0]",
        }
    )
    style_success_label_plus_proxy_backfill_scale = float(
        hparams.get("style_success_label_plus_proxy_backfill_scale", 0.75)
    )
    checks.append(
        {
            "name": "style_success_label_plus_proxy_backfill_scale_range",
            "ok": 0.0 <= style_success_label_plus_proxy_backfill_scale <= 1.0,
            "actual": style_success_label_plus_proxy_backfill_scale,
            "expected": "[0.0, 1.0]",
        }
    )
    style_success_self_ref_scale = float(
        hparams.get("style_success_self_ref_scale", 0.35)
    )
    checks.append(
        {
            "name": "style_success_self_ref_scale_range",
            "ok": 0.0 <= style_success_self_ref_scale <= 1.0,
            "actual": style_success_self_ref_scale,
            "expected": "[0.0, 1.0]",
        }
    )
    style_success_proxy_rank_scale = float(
        hparams.get(
            "style_success_proxy_rank_scale",
            hparams.get("style_success_proxy_only_rank_row_scale", 0.5),
        )
    )
    checks.append(
        {
            "name": "style_success_proxy_rank_scale_range",
            "ok": 0.0 <= style_success_proxy_rank_scale <= 1.0,
            "actual": style_success_proxy_rank_scale,
            "expected": "[0.0, 1.0]",
        }
    )
    style_success_memory_fallback_scale = float(
        hparams.get(
            "style_success_memory_fallback_scale",
            STYLE_SUCCESS_MEMORY_FALLBACK_SCALE,
        )
    )
    checks.append(
        {
            "name": "style_success_memory_fallback_scale_range",
            "ok": 0.0 <= style_success_memory_fallback_scale <= 1.0,
            "actual": style_success_memory_fallback_scale,
            "expected": "[0.0, 1.0]",
        }
    )
    if "style_success_proxy_only_rank_row_scale" in hparams:
        legacy_proxy_only_rank_row_scale = float(
            hparams.get("style_success_proxy_only_rank_row_scale", 0.35)
        )
        checks.append(
            {
                "name": "style_success_proxy_only_rank_row_scale_range",
                "ok": 0.0 <= legacy_proxy_only_rank_row_scale <= 1.0,
                "actual": legacy_proxy_only_rank_row_scale,
                "expected": "[0.0, 1.0]",
            }
        )
    style_success_rank_min_negative_row_frac = float(
        hparams.get("style_success_rank_min_negative_row_frac", 0.25)
    )
    checks.append(
        {
            "name": "style_success_rank_min_negative_row_frac_range",
            "ok": 0.0 <= style_success_rank_min_negative_row_frac <= 1.0,
            "actual": style_success_rank_min_negative_row_frac,
            "expected": "[0.0, 1.0]",
        }
    )
    style_success_rank_min_mean_negatives_per_row = float(
        hparams.get("style_success_rank_min_mean_negatives_per_row", 2.0)
    )
    checks.append(
        {
            "name": "style_success_rank_min_mean_negatives_per_row_nonnegative",
            "ok": style_success_rank_min_mean_negatives_per_row >= 0.0,
            "actual": style_success_rank_min_mean_negatives_per_row,
            "expected": ">= 0.0",
        }
    )
    style_success_rank_min_effective_support = float(
        hparams.get("style_success_rank_min_effective_support", 0.20)
    )
    checks.append(
        {
            "name": "style_success_rank_min_effective_support_range",
            "ok": 0.0 <= style_success_rank_min_effective_support <= 1.0,
            "actual": style_success_rank_min_effective_support,
            "expected": "[0.0, 1.0]",
        }
    )
    style_success_proxy_min_informative_features = int(
        hparams.get("style_success_proxy_min_informative_features", 2)
    )
    checks.append(
        {
            "name": "style_success_proxy_min_informative_features_nonnegative",
            "ok": style_success_proxy_min_informative_features >= 0,
            "actual": style_success_proxy_min_informative_features,
            "expected": ">= 0",
        }
    )
    decoder_late_style_floor_ratio = float(hparams.get("decoder_late_style_floor_ratio", 0.15))
    checks.append(
        {
            "name": "decoder_late_style_floor_ratio_range",
            "ok": 0.0 <= decoder_late_style_floor_ratio <= 1.0,
            "actual": decoder_late_style_floor_ratio,
            "expected": "[0.0, 1.0]",
        }
    )
    decoder_late_style_floor_margin = float(hparams.get("decoder_late_style_floor_margin", 0.0))
    checks.append(
        {
            "name": "decoder_late_style_floor_margin_nonnegative",
            "ok": decoder_late_style_floor_margin >= 0.0,
            "actual": decoder_late_style_floor_margin,
            "expected": ">= 0.0",
        }
    )
    decoder_late_style_floor_weight = float(hparams.get("decoder_late_style_floor_weight", 0.35))
    checks.append(
        {
            "name": "decoder_late_style_floor_weight_nonnegative",
            "ok": decoder_late_style_floor_weight >= 0.0,
            "actual": decoder_late_style_floor_weight,
            "expected": ">= 0.0",
        }
    )

    _check_exists(checks, "binary_data_dir_exists", hparams.get("binary_data_dir"))
    _check_exists(checks, "processed_data_dir_exists", hparams.get("processed_data_dir"))
    binary_data_dir = hparams.get("binary_data_dir")
    if binary_data_dir:
        for filename in (
            "train.data",
            "train.idx",
            "train_lengths.npy",
            "train_ref_indices.npy",
            "train_spk_ids.npy",
            "valid.data",
            "valid.idx",
            "valid_lengths.npy",
            "valid_ref_indices.npy",
            "valid_spk_ids.npy",
            "test.data",
            "test.idx",
            "test_lengths.npy",
            "test_ref_indices.npy",
            "test_spk_ids.npy",
        ):
            _check_exists(
                checks,
                f"binary_{filename}_exists",
                os.path.join(str(binary_data_dir), filename),
            )
        for split in ("train", "valid", "test"):
            _check_npy_count_positive(
                checks,
                f"binary_{split}_items_nonempty",
                os.path.join(str(binary_data_dir), f"{split}_lengths.npy"),
            )
            _check_binary_sidecar_consistency(checks, binary_data_dir, split)

    reference_preview_steps = (0, 20000, 40000, 70000, 100000)
    forcing_preview_steps = (0, 12000, 20000, 60000, 100000)
    failed_summary = _summarize_failed_checks(checks)
    non_contract_categories = {"runtime_dependencies", "data_staging", "data_dependent_preview"}
    environment_ready = all(
        item.get("ok", False)
        for item in checks
        if _classify_check_name(item.get("name")) == "runtime_dependencies"
    )
    data_ready = all(
        item.get("ok", False)
        for item in checks
        if _classify_check_name(item.get("name")) == "data_staging"
    )
    data_dependent_preview_ready = all(
        item.get("ok", False)
        for item in checks
        if _classify_check_name(item.get("name")) == "data_dependent_preview"
    )
    code_contract_ready = all(
        item.get("ok", False) or _classify_check_name(item.get("name")) in non_contract_categories
        for item in checks
    )
    all_ready = bool(all(item["ok"] for item in checks))
    summary = {
        "config": args.config,
        "config_chain": config_chain,
        "identity_loss_constraint_mode": identity_loss_constraint_mode,
        "style_strength_override_mode": (
            "dataset_item_override_enabled"
            if bool(hparams.get("allow_item_style_strength_override", False))
            else "profile_locked"
        ),
        "mainline_controls": _jsonable(controls.as_dict()),
        "train_batch_mainline_controls": None if batch_controls is None else _jsonable(batch_controls.as_dict()),
        "resolved_profile": _jsonable(resolved_profile),
        "style_success_supervision": _jsonable(style_success_supervision),
        "reference_curriculum_preview": [
            _jsonable({"step": int(step), **resolve_reference_curriculum(step, hparams)})
            for step in reference_preview_steps
        ],
        "forcing_schedule_preview": [
            _jsonable({"step": int(step), **resolve_forcing_schedule(step, hparams)})
            for step in forcing_preview_steps
        ],
        "upper_bound_curriculum_preview": [
            {
                "step": int(step),
                "progress": resolve_expressive_upper_bound_progress(step, hparams=hparams),
            }
            for step in (0, 20000, 50000, 80000, 100000)
        ],
        "dynamic_timbre_boundary_preview": _jsonable(boundary_preview),
        "checks": _jsonable(checks),
        "failed_summary": failed_summary,
        "failed_checks_by_category": _jsonable(failed_summary.get("blocking_categories", {})),
        "code_contract_ready": bool(code_contract_ready),
        "environment_ready": bool(environment_ready),
        "data_ready": bool(data_ready),
        "data_dependent_preview_ready": bool(data_dependent_preview_ready),
        "ready": all_ready,
        "train_ready_now": all_ready,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main():
    args = parse_args()
    summary = run_prep(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["ready"]:
        raise SystemExit("MAINLINE_TRAIN_PREP_NOT_READY")
    print("MAINLINE_TRAIN_PREP_OK")


if __name__ == "__main__":
    main()
