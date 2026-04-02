from pathlib import Path

import numpy as np
import torch

from tasks.Conan.Conan import ConanTask
from tasks.Conan.dataset import ConanDataset
from utils.commons.hparams import hparams, set_hparams
from utils.commons.tensor_utils import move_to_cuda


DEFAULT_SMOKE_HPARAMS = {
    "ds_workers": 0,
    "use_spk_id": True,
    "random_speaker_steps": 0,
    "max_sentences": 4,
    "max_tokens": 4096,
    "max_valid_sentences": 4,
    "max_valid_tokens": 4096,
    "lambda_mel_adv": 0.0,
}

# Generic training-oriented smoke helpers intentionally default to a post-warmup
# step so VQ / regularization / mainline losses are active unless a caller
# explicitly requests a different point on the schedule. Contract-style smokes
# that must assert true step-0 behavior should pass `global_step=0`.
DEFAULT_SMOKE_TRAIN_GLOBAL_STEP = 50000


def default_smoke_binary_data_dir():
    candidates = (
        Path("data/binary/libritts_single_smoke"),
        Path("data/binary/libritts_single_small3"),
        Path("data/binary/libritts_single_small2"),
        Path("data/binary/libritts_single_small"),
        Path("data/binary/libritts_single"),
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def dataset_num_styles(binary_data_dir):
    spk_id_path = Path(binary_data_dir) / "train_spk_ids.npy"
    if not spk_id_path.exists():
        return 8
    spk_ids = np.load(spk_id_path)
    return max(int(np.max(spk_ids)) + 1, 2)


def configure_smoke_hparams(
    *,
    config_path,
    binary_data_dir,
    extra_hparams=None,
):
    extra_hparams = dict(extra_hparams or {})
    merged = dict(DEFAULT_SMOKE_HPARAMS)
    merged.update(extra_hparams)
    merged["binary_data_dir"] = str(binary_data_dir)
    merged.setdefault("num_styles", dataset_num_styles(binary_data_dir))
    merged = {key: value for key, value in merged.items() if value is not None}
    hparam_overrides = ",".join([f"{key}={value}" for key, value in merged.items()])
    set_hparams(config=config_path, hparams_str=hparam_overrides, print_hparams=False)
    return hparams


def build_pseudo_style_dataset(
    *,
    config_path,
    binary_data_dir,
    extra_hparams=None,
):
    configure_smoke_hparams(
        config_path=config_path,
        binary_data_dir=binary_data_dir,
        extra_hparams=extra_hparams,
    )
    dataset = ConanDataset(prefix="train", shuffle=False)
    dataset._build_speaker_map()
    return dataset, int(hparams.get("num_styles", 0))


def select_speaker_batch_indices(dataset, step_idx, *, speakers_per_batch=2, items_per_speaker=2):
    speaker_ids = sorted([sid for sid, bucket in dataset.spk2indices.items() if len(bucket) >= items_per_speaker])
    if len(speaker_ids) < speakers_per_batch:
        raise ValueError(
            f"Need at least {speakers_per_batch} speakers with {items_per_speaker} items, got {len(speaker_ids)}."
        )
    chosen_speakers = [
        speaker_ids[(step_idx + offset) % len(speaker_ids)]
        for offset in range(speakers_per_batch)
    ]

    indices = []
    for speaker_offset, speaker_id in enumerate(chosen_speakers):
        bucket = list(dataset.spk2indices[speaker_id])
        start = (step_idx + speaker_offset) % len(bucket)
        for inner in range(items_per_speaker):
            indices.append(int(bucket[(start + inner) % len(bucket)]))
    return indices


def resolve_smoke_batch_shape(dataset, *, speakers_per_batch=2, items_per_speaker=2):
    requested_speakers = max(1, int(speakers_per_batch))
    requested_items = max(1, int(items_per_speaker))
    bucket_sizes = [len(bucket) for bucket in dataset.spk2indices.values() if len(bucket) > 0]
    if len(bucket_sizes) <= 0:
        raise ValueError("Smoke dataset has no speaker buckets.")

    best = None
    for items in range(requested_items, 0, -1):
        available_speakers = sum(1 for size in bucket_sizes if size >= items)
        if available_speakers <= 0:
            continue
        candidate = (min(requested_speakers, available_speakers), items, available_speakers)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    if best is None:
        raise ValueError("Unable to resolve a valid smoke batch shape from the dataset.")

    used_speakers, used_items, available_speakers = best
    return {
        "requested_speakers_per_batch": requested_speakers,
        "requested_items_per_speaker": requested_items,
        "speakers_per_batch": int(used_speakers),
        "items_per_speaker": int(used_items),
        "available_speakers_for_items": int(available_speakers),
    }


def build_pseudo_style_batch(dataset, indices, device, *, ensure_factorized_refs=False):
    samples = [dataset[idx] for idx in indices]
    batch = dataset.collater(samples)
    if device == "cuda":
        batch = move_to_cuda(batch)
    return batch


def scalarize_value(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return value


def is_finite_scalar(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value))
    return False


def scalarize_logs(logs):
    result = {}
    for key, value in logs.items():
        result[key] = scalarize_value(value)
    return result


def build_conan_training_task(device, *, global_step=None):
    task = ConanTask()
    task.build_tts_model()
    task.model.to(device)
    task.mel_disc.to(device)
    task.model.train()
    task.global_step = (
        int(global_step)
        if global_step is not None
        else max(int(hparams.get("vq_start", 0)) + 1, DEFAULT_SMOKE_TRAIN_GLOBAL_STEP)
    )
    return task


def compare_model_state_dicts(model_a, model_b):
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    keys_a = list(state_a.keys())
    keys_b = list(state_b.keys())
    if keys_a != keys_b:
        raise ValueError(
            f"State dict key mismatch: len_a={len(keys_a)}, len_b={len(keys_b)}, "
            f"first_diff={next(((a, b) for a, b in zip(keys_a, keys_b) if a != b), None)}"
        )
    max_abs_diff = 0.0
    for (name_a, value_a), (name_b, value_b) in zip(state_a.items(), state_b.items()):
        if name_a != name_b:
            raise ValueError(f"State dict key mismatch: {name_a} vs {name_b}")
        if value_a.shape != value_b.shape:
            raise ValueError(f"State dict shape mismatch at {name_a}: {value_a.shape} vs {value_b.shape}")
        diff = (value_a.detach().float().cpu() - value_b.detach().float().cpu()).abs().max().item()
        max_abs_diff = max(max_abs_diff, float(diff))
    return max_abs_diff
