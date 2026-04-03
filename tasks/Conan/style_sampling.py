import numpy as np
import torch
import torch.distributed as dist

from utils.commons.dataset_utils import (
    BaseConcatDataset,
    batch_by_size,
    partition_batches_for_ddp,
    resolve_dataloader_kwargs,
)


def _get_condition_ids(dataset, field):
    if hasattr(dataset, "get_local_condition_ids"):
        return np.asarray(dataset.get_local_condition_ids(field), dtype=np.int64)
    if isinstance(dataset, BaseConcatDataset):
        parts = [_get_condition_ids(sub_dataset, field) for sub_dataset in dataset.datasets]
        if len(parts) <= 0:
            return np.zeros((0,), dtype=np.int64)
        return np.concatenate(parts, axis=0)
    return np.full(len(dataset), -1, dtype=np.int64)


def _resolve_bucket_size(max_sentences, bucket_size, bucket_factor, group_size):
    bucket_size = int(bucket_size)
    if bucket_size > 0:
        return max(bucket_size, int(group_size))
    max_sentences = 32 if max_sentences is None else int(max_sentences)
    bucket_factor = max(1, int(bucket_factor))
    return max(max_sentences * bucket_factor, int(group_size))


def reorder_indices_by_condition(
    indices,
    condition_ids,
    *,
    group_size=2,
    bucket_size=0,
    bucket_factor=8,
    include_unlabeled=True,
):
    indices = np.asarray(indices, dtype=np.int64)
    condition_ids = np.asarray(condition_ids, dtype=np.int64)
    if indices.size <= 0:
        return indices

    group_size = max(1, int(group_size))
    bucket_size = _resolve_bucket_size(
        None,
        bucket_size=bucket_size,
        bucket_factor=bucket_factor,
        group_size=group_size,
    )

    reordered = []
    for start in range(0, len(indices), bucket_size):
        bucket = indices[start:start + bucket_size].tolist()
        label_queues = {}
        unlabeled = []
        for idx in bucket:
            label = int(condition_ids[idx]) if 0 <= int(idx) < len(condition_ids) else -1
            if label < 0:
                if include_unlabeled:
                    unlabeled.append(int(idx))
                continue
            label_queues.setdefault(label, []).append(int(idx))

        active_labels = list(label_queues.keys())
        if len(active_labels) > 1:
            np.random.shuffle(active_labels)

        while len(active_labels) > 0:
            next_active_labels = []
            for label in active_labels:
                queue = label_queues[label]
                take = min(group_size, len(queue))
                reordered.extend(queue[:take])
                del queue[:take]
                if len(queue) > 0:
                    next_active_labels.append(label)
            if len(next_active_labels) > 1:
                np.random.shuffle(next_active_labels)
            active_labels = next_active_labels

        reordered.extend(unlabeled)

    return np.asarray(reordered, dtype=np.int64)


def build_condition_balanced_dataloader(
    dataset,
    *,
    condition_field,
    shuffle,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=-1,
    endless=False,
    batch_by_size_enabled=True,
    bucket_size=0,
    bucket_factor=8,
    group_size=2,
    include_unlabeled=True,
    use_ddp=False,
):
    if use_ddp and dist.is_available() and dist.is_initialized():
        devices_cnt = max(1, int(dist.get_world_size()))
    else:
        devices_cnt = 1
    if required_batch_size_multiple == -1:
        required_batch_size_multiple = devices_cnt

    def shuffle_batches(batches):
        np.random.shuffle(batches)
        return batches

    if max_tokens is not None:
        max_tokens *= devices_cnt
    if max_sentences is not None:
        max_sentences *= devices_cnt

    indices = dataset.ordered_indices()
    condition_ids = _get_condition_ids(dataset, condition_field)
    indices = reorder_indices_by_condition(
        indices,
        condition_ids,
        group_size=group_size,
        bucket_size=_resolve_bucket_size(
            max_sentences,
            bucket_size=bucket_size,
            bucket_factor=bucket_factor,
            group_size=group_size,
        ),
        bucket_factor=bucket_factor,
        include_unlabeled=include_unlabeled,
    )

    if batch_by_size_enabled:
        batch_sampler = batch_by_size(
            indices,
            dataset.num_tokens,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )
    else:
        batch_sampler = []
        step = max_sentences if max_sentences is not None else 1
        for idx in range(0, len(indices), step):
            batch_sampler.append(indices[idx:idx + step].tolist())

    if shuffle:
        batches = shuffle_batches(list(batch_sampler))
        if endless:
            batches = [b for _ in range(1000) for b in shuffle_batches(list(batch_sampler))]
    else:
        batches = list(batch_sampler)
        if endless:
            batches = [b for _ in range(1000) for b in batches]

    num_workers = dataset.num_workers
    if use_ddp:
        batches = partition_batches_for_ddp(batches)

    dataloader_kwargs = resolve_dataloader_kwargs(num_workers)
    return torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collater,
        batch_sampler=batches,
        num_workers=num_workers,
        **dataloader_kwargs,
    )
