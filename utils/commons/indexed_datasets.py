import pickle
import sys
from copy import deepcopy
import importlib

import numpy as np


def _install_numpy_pickle_compat_aliases():
    """Support loading dataset artifacts pickled under NumPy 2.x in NumPy 1.x envs.

    Some existing binary dataset artifacts were produced in environments where
    NumPy pickled module references under ``numpy._core``. Older NumPy releases
    (for example 1.24.x) only expose ``numpy.core``. Register only the missing
    legacy-compat names instead of rewriting modern NumPy's canonical import
    path; otherwise ``numpy.core`` can be forced to point at ``numpy._core``
    early enough to confuse NumPy/SciPy lazy loaders and trigger recursive
    imports in unrelated downstream modules.
    """

    try:
        importlib.import_module("numpy._core")
        # Modern NumPy already exposes the canonical module path. Do not alias
        # ``numpy.core`` here: NumPy owns that compatibility surface and may
        # resolve it lazily.
        return
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) != "numpy._core":
            return
    except Exception:
        return

    try:
        numpy_core = importlib.import_module("numpy.core")
    except Exception:
        return

    aliases = {"numpy._core": numpy_core}
    for submodule in ("multiarray", "numeric", "_multiarray_umath", "umath"):
        try:
            target = importlib.import_module(f"numpy.core.{submodule}")
        except Exception:
            target = None
        if target is not None:
            aliases.setdefault(f"numpy._core.{submodule}", target)
    for name, module in aliases.items():
        sys.modules.setdefault(name, module)


_install_numpy_pickle_compat_aliases()


def _normalize_offsets(payload):
    if isinstance(payload, np.ndarray) and payload.dtype.kind in {"i", "u"}:
        return np.asarray(payload, dtype=np.int64).reshape(-1)
    if isinstance(payload, dict):
        offsets = payload.get('offsets')
        if offsets is not None:
            return np.asarray(offsets, dtype=np.int64).reshape(-1)
    raise ValueError("Unsupported indexed dataset offset payload.")


def _load_offsets(path):
    idx_path = f"{path}.idx"
    try:
        payload = np.load(idx_path, allow_pickle=False)
        return _normalize_offsets(payload)
    except Exception:
        payload = np.load(idx_path, allow_pickle=True)
        if isinstance(payload, np.ndarray) and payload.dtype == object and payload.shape == ():
            payload = payload.item()
        return _normalize_offsets(payload)


class IndexedDataset:
    def __init__(self, path, num_cache=1):
        super().__init__()
        self.path = path
        self.data_file = None
        self.data_offsets = _load_offsets(path)
        self.data_file = open(f"{path}.data", 'rb', buffering=-1)
        self.cache = []
        self.num_cache = num_cache

    def check_index(self, i):
        if i < 0 or i >= len(self.data_offsets) - 1:
            raise IndexError('index out of range')

    def close(self):
        if self.data_file is not None:
            self.data_file.close()
            self.data_file = None

    def __del__(self):
        self.close()

    def __getitem__(self, i):
        self.check_index(i)
        if self.data_file is None:
            self.data_file = open(f"{self.path}.data", 'rb', buffering=-1)
        if self.num_cache > 0:
            for c in self.cache:
                if c[0] == i:
                    return c[1]
        self.data_file.seek(self.data_offsets[i])
        b = self.data_file.read(self.data_offsets[i + 1] - self.data_offsets[i])
        item = pickle.loads(b)
        if self.num_cache > 0:
            self.cache = [(i, deepcopy(item))] + self.cache[:-1]
        return item

    def __len__(self):
        return len(self.data_offsets) - 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class IndexedDatasetBuilder:
    def __init__(self, path):
        self.path = path
        self.out_file = open(f"{path}.data", 'wb')
        self.byte_offsets = [0]

    def close(self):
        if self.out_file is not None:
            self.out_file.close()
            self.out_file = None

    def add_item(self, item):
        s = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
        bytes = self.out_file.write(s)
        self.byte_offsets.append(self.byte_offsets[-1] + bytes)

    def finalize(self):
        self.close()
        with open(f"{self.path}.idx", 'wb') as index_file:
            np.save(index_file, np.asarray(self.byte_offsets, dtype=np.int64))

    def __del__(self):
        self.close()


if __name__ == "__main__":
    import random
    from tqdm import tqdm
    ds_path = '/tmp/indexed_ds_example'
    size = 100
    items = [{"a": np.random.normal(size=[10000, 10]),
              "b": np.random.normal(size=[10000, 10])} for i in range(size)]
    builder = IndexedDatasetBuilder(ds_path)
    for i in tqdm(range(size)):
        builder.add_item(items[i])
    builder.finalize()
    ds = IndexedDataset(ds_path)
    for i in tqdm(range(10000)):
        idx = random.randint(0, size - 1)
        assert (ds[idx]['a'] == items[idx]['a']).all()
