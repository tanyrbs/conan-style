from collections import deque

import numpy as np

from utils.commons.dataset_utils import batch_by_size


def get_dataset_condition_ids(dataset, field):
    getter = getattr(dataset, "get_local_condition_ids", None)
    if callable(getter):
        values = getter(field)
        return np.asarray(values, dtype=np.int64)

    values = []
    for local_idx in range(len(dataset)):
        item_getter = getattr(dataset, "_get_item", None)
        if callable(item_getter):
            item = item_getter(local_idx)
            values.append(int(item.get(f"{field}_id", -1)))
        else:
            values.append(-1)
    return np.asarray(values, dtype=np.int64)


def reorder_indices_by_condition_window(indices, labels, *, group_size=2, window_size=128, shuffle_styles=True):
    if indices is None:
        return []
    indices = list(indices)
    if len(indices) <= 1:
        return indices
    if labels is None or len(labels) <= 0:
        return indices

    group_size = max(1, int(group_size))
    window_size = max(group_size, int(window_size))
    reordered = []

    for start in range(0, len(indices), window_size):
        window = indices[start:start + window_size]
        label_buckets = {}
        unlabeled = []
        for idx in window:
            label = int(labels[idx]) if 0 <= int(idx) < len(labels) else -1
            if label < 0:
                unlabeled.append(idx)
                continue
            label_buckets.setdefault(label, []).append(idx)

        active_labels = list(label_buckets.keys())
        if shuffle_styles and len(active_labels) > 1:
            np.random.shuffle(active_labels)
        active = deque(active_labels)
        while active:
            label = active.popleft()
            bucket = label_buckets[label]
            take = min(group_size, len(bucket))
            reordered.extend(bucket[:take])
            del bucket[:take]
            if len(bucket) > 0:
                active.append(label)
        reordered.extend(unlabeled)
    return reordered


def build_condition_balanced_batches(
    dataset,
    indices,
    *,
    field="style",
    group_size=2,
    window_batches=6,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
):
    labels = get_dataset_condition_ids(dataset, field)
    valid_labels = labels[labels >= 0]
    if len(valid_labels) <= 1:
        return batch_by_size(
            indices,
            dataset.num_tokens,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

    window_batch_size = max_sentences or 32
    window_size = max(int(window_batches), 1) * max(int(window_batch_size), 1)
    reordered_indices = reorder_indices_by_condition_window(
        indices,
        labels,
        group_size=group_size,
        window_size=window_size,
    )
    return batch_by_size(
        reordered_indices,
        dataset.num_tokens,
        max_tokens=max_tokens,
        max_sentences=max_sentences,
        required_batch_size_multiple=required_batch_size_multiple,
    )
