import random
import zlib
import numpy as np
import torch
from utils.commons.dataset_utils import BaseDataset, collate_1d_or_2d
from utils.commons.indexed_datasets import IndexedDataset
from utils.audio.pitch.utils import norm_interp_f0


def _to_float_tensor(value):
    return torch.as_tensor(value, dtype=torch.float32)


def _to_long_tensor(value):
    return torch.as_tensor(value, dtype=torch.long)


class BaseSpeechDataset(BaseDataset):
    """Dataset that always draws a *reference mel* from the same speaker via
    ``spk_id`` while keeping the original public interfaces intact (notably
    ``_get_item``).

    * `spk_id` is **always** required in the binary data for reference sampling.
    * Whether the ID is exposed to the model remains controlled by
      `hparams['use_spk_id']`.
    * ``IndexedDataset`` is lazily opened inside the worker to prevent pickling
      errors.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(shuffle)
        from utils.commons.hparams import hparams
        self.hparams = hparams
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.indexed_ds = None  # ⚠️ lazy open in worker

        # ------------------------------------------------------------------
        # Indices & utterance lengths
        # ------------------------------------------------------------------
        if items is not None:
            self.indexed_ds = items
            self.sizes = [1] * len(items)
            self.avail_idxs = list(range(len(self.sizes)))
        else:
            self.sizes = np.load(f"{self.data_dir}/{self.prefix}_lengths.npy")
            if prefix == 'test' and len(hparams['test_ids']) > 0:
                self.avail_idxs = hparams['test_ids']
            else:
                self.avail_idxs = list(range(len(self.sizes)))
            if prefix == 'train' and hparams['min_frames'] > 0:
                self.avail_idxs = [i for i in self.avail_idxs if self.sizes[i] >= hparams['min_frames']]
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

        # Speaker map will be built lazily in each worker
        self.spk2indices = None  # type: dict[int, list[int]] | None
        self._spk_map_ready = False
        self.cond2indices = {}
        self.cond_ids = {}
        self._cond_map_ready = set()
        self.fixed_ref_indices = None
        self._fixed_ref_ready = False

    def _rng_for_map(self, key):
        base_seed = int(self.hparams.get('seed', 1234))
        digest = zlib.crc32(f"{self.prefix}:{key}".encode('utf-8')) & 0xFFFFFFFF
        return np.random.default_rng((base_seed + digest) % (2 ** 32))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _open_indexed_ds_if_needed(self):
        """Open the mmap dataset after the worker process starts."""
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f"{self.data_dir}/{self.prefix}")

    def _get_item(self, local_idx):
        """**Public** helper – keep original signature & semantics.
        Accepts *local* index (0 ≤ idx < len(avail_idxs)).
        """
        global_idx = self.avail_idxs[local_idx] if self.avail_idxs is not None else local_idx
        self._open_indexed_ds_if_needed()
        return self.indexed_ds[global_idx]

    def _get_item_by_global_idx(self, global_idx):
        self._open_indexed_ds_if_needed()
        return self.indexed_ds[int(global_idx)]

    def _build_fixed_ref_map(self):
        if self._fixed_ref_ready:
            return
        import os
        fixed_ref_path = f"{self.data_dir}/{self.prefix}_ref_indices.npy"
        if os.path.exists(fixed_ref_path):
            self.fixed_ref_indices = np.load(fixed_ref_path, mmap_mode='r')
        else:
            self.fixed_ref_indices = None
        self._fixed_ref_ready = True

    def _build_speaker_map(self):
        """
        Build spk_id → [local_idx] mapping, but never fetch complete samples from disk mmap again.

        - If {prefix}_spk_ids.npy exists, complete in O(N) pure memory;
        - Otherwise fallback to old implementation (iterating through _get_item).
        - Keep at most hparams['max_samples_per_spk'] entries per speaker (default 100).
        """
        if self._spk_map_ready:
            return

        import os
        from collections import defaultdict

        max_per_spk = int(self.hparams.get('max_samples_per_spk', 100))
        spk_ids_path = f"{self.data_dir}/{self.prefix}_spk_ids.npy"
        self.spk2indices = defaultdict(list)
        rng = self._rng_for_map('speaker_map')

        if os.path.exists(spk_ids_path):
            # ---------- Fast path ----------
            # mmap loading, almost no memory usage, speed is in seconds
            spk_ids = np.load(spk_ids_path, mmap_mode='r')
            # Only take current avail_idxs subset
            local_spk_ids = spk_ids[self.avail_idxs]

            # Shuffle, so truncation doesn't bias towards early entries
            for local_idx in rng.permutation(len(local_spk_ids)):
                sid = int(local_spk_ids[local_idx])
                bucket = self.spk2indices[sid]
                if len(bucket) < max_per_spk:
                    bucket.append(local_idx)
        else:
            # ---------- Slow path for legacy data compatibility ----------
            for local_idx in rng.permutation(len(self.avail_idxs)):
                sid = int(self._get_item(local_idx)['spk_id'])
                bucket = self.spk2indices[sid]
                if len(bucket) < max_per_spk:
                    bucket.append(local_idx)

        self._spk_map_ready = True

    def _build_condition_map(self, field):
        if field in self._cond_map_ready:
            return

        from collections import defaultdict

        max_per_label = int(self.hparams.get(f'max_samples_per_{field}', self.hparams.get('max_samples_per_spk', 100)))
        field2indices = defaultdict(list)
        local_ids = self.get_local_condition_ids(field)
        rng = self._rng_for_map(f'condition_map:{field}')
        for local_idx in rng.permutation(len(local_ids)):
            cid = int(local_ids[local_idx])
            if cid < 0:
                continue
            bucket = field2indices[cid]
            if len(bucket) < max_per_label:
                bucket.append(local_idx)

        self.cond2indices[field] = field2indices
        self._cond_map_ready.add(field)

    def get_local_condition_ids(self, field):
        if field in self.cond_ids:
            return self.cond_ids[field]

        import os

        ids_path = f"{self.data_dir}/{self.prefix}_{field}_ids.npy"
        if os.path.exists(ids_path):
            cond_ids = np.load(ids_path, mmap_mode='r')
            local_ids = np.asarray(cond_ids[self.avail_idxs], dtype=np.int64)
        else:
            local_ids = np.asarray(
                [
                    int(self._get_item(local_idx).get(f'{field}_id', -1))
                    for local_idx in range(len(self.avail_idxs))
                ],
                dtype=np.int64,
            )
        self.cond_ids[field] = local_ids
        return local_ids

    def _trim_ref_spec(self, item):
        ref_spec = _to_float_tensor(item['mel'])[:self.hparams['max_frames']]
        ref_spec = ref_spec[: (ref_spec.shape[0] // self.hparams['frames_multiple']) * self.hparams['frames_multiple']]
        return ref_spec

    @staticmethod
    def _sample_from_bucket(bucket, exclude_idx):
        if len(bucket) <= 0:
            return None
        candidates = [idx for idx in bucket if idx != exclude_idx]
        if len(candidates) <= 0:
            return bucket[0]
        return random.choice(candidates)

    @staticmethod
    def _clamp_weight(value, default=1.0):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = float(default)
        if not np.isfinite(value):
            value = float(default)
        return max(1e-3, float(value))

    def _weighted_sample_from_indices(self, indices, *, exclude_idx=None, score_fn=None):
        candidates = [idx for idx in indices if idx != exclude_idx]
        if len(candidates) <= 0:
            return None
        if score_fn is None:
            return random.choice(candidates)
        weights = []
        filtered = []
        for idx in candidates:
            item = self._get_item(idx)
            weight = self._clamp_weight(score_fn(item), default=1.0)
            filtered.append(idx)
            weights.append(weight)
        if len(filtered) <= 0:
            return None
        total = float(sum(weights))
        if total <= 0:
            return random.choice(filtered)
        probs = [w / total for w in weights]
        chosen = int(np.random.choice(len(filtered), p=probs))
        return filtered[chosen]

    @staticmethod
    def _sample_energy_variation(item):
        energy = item.get('energy', None)
        if energy is not None:
            try:
                energy = np.asarray(energy, dtype=np.float32).reshape(-1)
                if energy.size > 1:
                    return float(np.std(energy))
            except Exception:
                pass
        mel = item.get('mel', None)
        if mel is None:
            return 0.0
        try:
            mel = np.asarray(mel, dtype=np.float32)
            if mel.ndim == 2 and mel.shape[0] > 1:
                return float(np.std(np.abs(mel).mean(axis=-1)))
        except Exception:
            return 0.0
        return 0.0

    def _sample_reference_by_condition(self, field, condition_id, exclude_idx, fallback_item):
        self._build_condition_map(field)
        bucket = self.cond2indices.get(field, {}).get(int(condition_id), [])
        local_idx = self._sample_from_bucket(bucket, exclude_idx)
        if local_idx is None:
            return self._trim_ref_spec(fallback_item)
        return self._trim_ref_spec(self._get_item(local_idx))

    def _sample_fixed_speaker_reference(self, index, fallback_item):
        self._build_fixed_ref_map()
        if self.fixed_ref_indices is None:
            return None, None
        global_idx = self.avail_idxs[index] if self.avail_idxs is not None else index
        if int(global_idx) >= len(self.fixed_ref_indices):
            return None, None
        fixed_global_idx = int(self.fixed_ref_indices[int(global_idx)])
        if fixed_global_idx < 0:
            return None, None
        try:
            ref_item = self._get_item_by_global_idx(fixed_global_idx)
            return self._trim_ref_spec(ref_item), ref_item
        except Exception:
            return None, None
    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __getitem__(self, index):
        self._build_speaker_map()
        item = self._get_item(index)
        hparams = self.hparams

        # 1) Main mel
        max_frames = hparams['max_frames']
        spec = _to_float_tensor(item['mel'])[:max_frames]
        max_frames = spec.shape[0] // hparams['frames_multiple'] * hparams['frames_multiple']
        spec = spec[:max_frames]

        # 2) Reference mel from same speaker
        spk_id = int(item['spk_id'])
        cand_locals = self.spk2indices[spk_id]
        ref_spec, ref_item = (None, None)
        if hparams.get('use_fixed_timbre_reference', True):
            ref_spec, ref_item = self._sample_fixed_speaker_reference(index, fallback_item=item)
        if ref_spec is None or ref_item is None:
            ref_local = random.choice([l for l in cand_locals if l != index]) if len(cand_locals) > 1 else index
            ref_item = self._get_item(ref_local)
            ref_spec = _to_float_tensor(ref_item['mel'])[:hparams['max_frames']]
            ref_spec = ref_spec[: (ref_spec.shape[0] // hparams['frames_multiple']) * hparams['frames_multiple']]

        sample = {
            'id': index,
            'item_name': item['item_name'],
            'mel': spec,
            'mel_nonpadding': spec.abs().sum(-1) > 0,
            'ref_mel': ref_spec,
            'ref_timbre_mel': ref_spec,
        }

        # Optional speaker embedding
        if hparams.get('use_spk_embed', False):
            embed = item['spk_embed']
            if isinstance(embed, str):
                embed = _to_float_tensor([float(x) for x in embed.split()])
            else:
                embed = _to_float_tensor(embed)
            sample['spk_embed'] = embed

        # Provide spk_id to model only if enabled
        if hparams.get('use_spk_id', False):
            sample['spk_id'] = spk_id

        for field in ('emotion', 'accent'):
            condition_id = int(item.get(f'{field}_id', -1))
            if condition_id >= 0:
                sample[f'{field}_id'] = condition_id
                sample[f'{field}_label'] = str(item.get(field, item.get(f'{field}_name', '')))
                if hparams.get(f'sample_{field}_reference', False):
                    sample[f'{field}_ref_mel'] = self._sample_reference_by_condition(
                        field=field,
                        condition_id=condition_id,
                        exclude_idx=index,
                        fallback_item=ref_item,
                    )

        for field in ('arousal', 'valence'):
            if field in item:
                sample[field] = float(item.get(field, 0.0))

        return sample

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------
    def collater(self, samples):
        if not samples:
            return {}
        hparams = self.hparams

        ids = torch.tensor([s['id'] for s in samples], dtype=torch.long)
        names = [s['item_name'] for s in samples]
        mels = collate_1d_or_2d([s['mel'] for s in samples], 0.0)
        ref_mels = collate_1d_or_2d([s['ref_mel'] for s in samples], 0.0)
        mel_lens = torch.tensor([s['mel'].shape[0] for s in samples], dtype=torch.long)
        ref_lens = torch.tensor([s['ref_mel'].shape[0] for s in samples], dtype=torch.long)

        batch = {
            'id': ids,
            'item_name': names,
            'nsamples': len(samples),
            'mels': mels,
            'mel_lengths': mel_lens,
            'ref_mels': ref_mels,
            'ref_mel_lengths': ref_lens,
        }
        if all('ref_timbre_mel' in s for s in samples):
            if all(s['ref_timbre_mel'] is s['ref_mel'] for s in samples):
                batch['ref_timbre_mels'] = ref_mels
            else:
                batch['ref_timbre_mels'] = collate_1d_or_2d([s['ref_timbre_mel'] for s in samples], 0.0)

        if hparams.get('use_spk_embed', False):
            batch['spk_embed'] = torch.stack([s['spk_embed'] for s in samples])
        if hparams.get('use_spk_id', False):
            batch['spk_ids'] = torch.tensor([s['spk_id'] for s in samples], dtype=torch.long)
        for field in ('emotion', 'accent'):
            id_key = f'{field}_id'
            if all(id_key in s for s in samples):
                batch[f'{field}_ids'] = torch.tensor([s[id_key] for s in samples], dtype=torch.long)
            ref_key = f'{field}_ref_mel'
            if all(ref_key in s for s in samples):
                batch[f'{field}_ref_mels'] = collate_1d_or_2d([s[ref_key] for s in samples], 0.0)
        for field in ('arousal', 'valence'):
            if all(field in s for s in samples):
                batch[field] = torch.tensor([s[field] for s in samples], dtype=torch.float32)

        return batch


class FastSpeechDataset(BaseSpeechDataset):
    """Dataset for FastSpeech-like models with ref mels & f0/uv."""

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams

        # Align mel length with f0 length
        if 'f0' in item:
            T = min(sample['mel'].shape[0], len(item['f0']))
        else:
            T = sample['mel'].shape[0]
        sample['mel'] = sample['mel'][:T]

        if hparams.get('use_pitch_embed', False):
            if 'f0' in item:
                f0, uv = norm_interp_f0(item['f0'][:T])
                sample['f0'] = _to_float_tensor(f0)
                sample['uv'] = _to_float_tensor(uv)
            else:
                f0, uv = None, None
        else:
            sample['f0'], sample['uv'] = None, None

        if 'energy' in item:
            sample['energy'] = _to_float_tensor(np.asarray(item['energy'])[:T])

        return sample

    def collater(self, samples):
        batch = super().collater(samples)
        hparams = self.hparams
        if hparams.get('use_pitch_embed', False):
            batch['f0'] = collate_1d_or_2d([s['f0'] for s in samples], 0.0)
            batch['uv'] = collate_1d_or_2d([s['uv'] for s in samples])
        if all('energy' in s for s in samples):
            batch['energy'] = collate_1d_or_2d([s['energy'] for s in samples], 0.0)
        return batch


class FastSpeechWordDataset(FastSpeechDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        item = self._get_item(index)
        max_frames = sample['mel'].shape[0]
        if 'word' in item:
            sample['words'] = item['word']
            sample["ph_words"] = item["ph_gb_word"]
            sample["word_tokens"] = _to_long_tensor(item["word_token"])
        else:
            sample['words'] = item['words']
            sample["ph_words"] = " ".join(item["ph_words"])
            sample["word_tokens"] = _to_long_tensor(item["word_tokens"])
        sample["mel2word"] = _to_long_tensor(item.get("mel2word"))[:max_frames]
        sample["ph2word"] = _to_long_tensor(item['ph2word'][:self.hparams['max_input_tokens']])
        return sample

    def collater(self, samples):
        batch = super().collater(samples)
        ph_words = [s['ph_words'] for s in samples]
        batch['ph_words'] = ph_words
        word_tokens = collate_1d_or_2d([s['word_tokens'] for s in samples], 0)
        batch['word_tokens'] = word_tokens
        mel2word = collate_1d_or_2d([s['mel2word'] for s in samples], 0)
        batch['mel2word'] = mel2word
        ph2word = collate_1d_or_2d([s['ph2word'] for s in samples], 0)
        batch['ph2word'] = ph2word
        batch['words'] = [s['words'] for s in samples]
        batch['word_lengths'] = torch.LongTensor([len(s['word_tokens']) for s in samples])
        if self.hparams['use_word_input']:
            batch['txt_tokens'] = batch['word_tokens']
            batch['txt_lengths'] = torch.LongTensor([s['word_tokens'].numel() for s in samples])
            batch['mel2ph'] = batch['mel2word']
        return batch
