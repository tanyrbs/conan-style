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
        else max(int(hparams.get("vq_start", 0)) + 1, 50000)
    )
    return task


def compare_model_state_dicts(model_a, model_b):
    max_abs_diff = 0.0
    for (name_a, value_a), (name_b, value_b) in zip(
        model_a.state_dict().items(),
        model_b.state_dict().items(),
    ):
        if name_a != name_b:
            raise ValueError(f"State dict key mismatch: {name_a} vs {name_b}")
        if value_a.shape != value_b.shape:
            raise ValueError(f"State dict shape mismatch at {name_a}: {value_a.shape} vs {value_b.shape}")
        diff = (value_a.detach().float().cpu() - value_b.detach().float().cpu()).abs().max().item()
        max_abs_diff = max(max_abs_diff, float(diff))
    return max_abs_diff
