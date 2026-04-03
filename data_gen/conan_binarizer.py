import hashlib
import json
import logging
import os
import random
from copy import deepcopy
from functools import partial

import numpy as np
from tqdm import tqdm

from data_gen.tts.base_binarizer import BaseBinarizer as CanonicalBaseBinarizer, BinarizationError
from utils.commons.hparams import hparams
from utils.commons.indexed_datasets import IndexedDatasetBuilder
from utils.commons.multiprocess_utils import multiprocess_run_tqdm

np.seterr(divide='ignore', invalid='ignore')


def _build_voice_encoder():
    try:
        from resemblyzer import VoiceEncoder
    except ImportError as exc:
        raise ImportError(
            'Speaker embedding extraction requires Resemblyzer. '
            'Install it or disable binarization_args.with_spk_embed.'
        ) from exc
    encoder = VoiceEncoder()
    try:
        import torch
        if torch.cuda.is_available():
            encoder = encoder.cuda()
    except Exception:
        pass
    return encoder


def _stringify_optional(value):
    if value is None:
        return ""
    return str(value)


def _parse_hubert_units(value, *, item_name=None):
    if isinstance(value, np.ndarray):
        raw_units = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        raw_units = list(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            raise BinarizationError(f"Empty hubert units for {item_name or '<unknown>'}")
        raw_units = stripped.replace(',', ' ').split()
    else:
        raise BinarizationError(
            f"Unsupported hubert field type for {item_name or '<unknown>'}: {type(value).__name__}"
        )
    try:
        parsed = [int(float(token)) for token in raw_units]
    except (TypeError, ValueError) as exc:
        raise BinarizationError(
            f"Failed to parse hubert units for {item_name or '<unknown>'}"
        ) from exc
    if len(parsed) <= 0:
        raise BinarizationError(f"Empty hubert units for {item_name or '<unknown>'}")
    return parsed


def _resolve_f0_cache_path(wav_fn):
    stem = os.path.splitext(os.path.basename(wav_fn))[0]
    return os.path.join(os.path.dirname(wav_fn) + '_f0', f'{stem}_f0.npy')


def _load_cached_f0(wav_fn, *, item_name=None, max_frames=None):
    f0_path = _resolve_f0_cache_path(wav_fn)
    if not os.path.exists(f0_path):
        raise FileNotFoundError(
            f"Missing cached f0 for {item_name or wav_fn}: {f0_path}. "
            "Run `python utils/extract_f0_rmvpe.py --config egs/conan_emformer.yaml --pe-ckpt <rmvpe.pt>` first."
        )
    f0 = np.asarray(np.load(f0_path), dtype=np.float32).reshape(-1)
    if f0.size <= 0:
        raise BinarizationError(f"Empty cached f0 for {item_name or wav_fn}: {f0_path}")
    if max_frames is not None:
        f0 = f0[:int(max_frames)]
    return f0


def _coerce_frame_series(value, target_length, default_value=0.0):
    target_length = int(target_length)
    if target_length <= 0:
        return np.zeros((0,), dtype=np.float32)
    series = _coerce_1d_float_array(value)
    if series is None or series.size <= 0:
        return np.full((target_length,), float(default_value), dtype=np.float32)
    series = series.astype(np.float32, copy=False).reshape(-1)
    if series.shape[0] < target_length:
        if series.shape[0] == 1:
            series = np.full((target_length,), float(series[0]), dtype=np.float32)
        else:
            series = np.pad(series, (0, target_length - series.shape[0]), mode='edge')
    return series[:target_length]


STYLE_LABEL_FIELD_CANDIDATES = {
    "emotion": ("emotion", "emotion_label", "emo", "mood"),
    "style": ("style", "style_label", "speaking_style", "style_tag"),
    "accent": ("accent", "accent_label", "dialect", "accent_name"),
}


def _normalize_label_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return int(value) if float(value).is_integer() else float(value)
    return str(value).strip() or None


def _coerce_float(value, default=0.0):
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_1d_float_array(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).reshape(-1)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        return np.asarray([float(x) for x in stripped.split()], dtype=np.float32).reshape(-1)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32).reshape(-1)
    if isinstance(value, (float, int, np.floating, np.integer)):
        return np.asarray([float(value)], dtype=np.float32)
    return None


def _item_name_speaker_key(item_name):
    token = str(item_name).strip()
    if token == "":
        return token
    for sep in ("/", "\\", "_"):
        if sep in token:
            return token.split(sep, 1)[0]
    return token


def _stable_bucket_seed(value):
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _normalize_prefixes(prefixes):
    normalized = []
    for prefix in prefixes or []:
        prefix = str(prefix).strip()
        if prefix != "":
            normalized.append(prefix)
    return normalized


def _match_by_speaker_prefix(item_names, prefixes):
    prefix_set = set(_normalize_prefixes(prefixes))
    if len(prefix_set) <= 0:
        return []
    return [name for name in item_names if _item_name_speaker_key(name) in prefix_set]


def _fallback_holdout_by_speaker(
        item_names,
        *,
        valid_item_names=None,
        test_item_names=None,
        fallback_valid_items_per_speaker=2,
        fallback_test_items_per_speaker=2):
    from collections import defaultdict

    per_speaker = defaultdict(list)
    for name in item_names:
        per_speaker[_item_name_speaker_key(name)].append(name)

    reserved_valid = set(valid_item_names or [])
    reserved_test = set(test_item_names or [])
    used = reserved_valid | reserved_test

    fallback_valid = max(1, int(fallback_valid_items_per_speaker))
    fallback_test = max(1, int(fallback_test_items_per_speaker))

    for _, speaker_items in sorted(per_speaker.items()):
        speaker_items = sorted(speaker_items)
        available = [name for name in speaker_items if name not in used]
        if len(available) <= 2:
            continue

        test_take = min(fallback_test, max(0, len(available) - 2))
        speaker_test = available[:test_take]
        reserved_test.update(speaker_test)
        used.update(speaker_test)
        available = available[test_take:]

        valid_take = min(fallback_valid, max(0, len(available) - 1))
        speaker_valid = available[:valid_take]
        reserved_valid.update(speaker_valid)
        used.update(speaker_valid)

    return sorted(reserved_valid - reserved_test), sorted(reserved_test)


CONDITION_FIELDS = ("emotion", "style", "accent")


def _safe_int(value, default=-1):
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value, default=0.0):
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class ConanBaseBinarizer(CanonicalBaseBinarizer):
    """
    Thin compatibility layer for Conan-specific binarizers.

    The canonical binarization primitives now live in
    data_gen.tts.base_binarizer.BaseBinarizer. Keeping Conan on that shared
    implementation prevents silent drift between the generic TTS pipeline and the
    Conan VC pipeline.
    """


class VCBinarizer(ConanBaseBinarizer):
    _spker_map_cache = None
    _spker_map_cache_key = None
    _condition_maps = None
    _condition_maps_key = None

    def __init__(self, processed_data_dir=None):
        super().__init__(processed_data_dir=processed_data_dir)
        self.label_maps = {}

    @staticmethod
    def _build_reference_indices(items):
        from collections import defaultdict

        speaker_buckets = defaultdict(list)
        for idx, item in enumerate(items):
            speaker_buckets[int(item['spk_id'])].append(idx)

        ref_indices = np.full((len(items),), -1, dtype=np.int32)
        for idx, item in enumerate(items):
            bucket = speaker_buckets.get(int(item['spk_id']), [])
            if len(bucket) <= 1:
                ref_indices[idx] = idx
                continue
            candidates = [candidate for candidate in bucket if candidate != idx]
            if len(candidates) <= 0:
                ref_indices[idx] = idx
                continue
            stable_seed = _stable_bucket_seed(item.get('item_name', idx))
            ref_indices[idx] = int(candidates[stable_seed % len(candidates)])
        return ref_indices

    @classmethod
    def _load_spker_map(cls):
        processed_data_dir = hparams.get("processed_data_dir", None)
        if cls._spker_map_cache is not None and cls._spker_map_cache_key == processed_data_dir:
            return cls._spker_map_cache
        if not processed_data_dir:
            raise KeyError(
                "processed_data_dir is not set in hparams. Call set_hparams(...) before using the binarizer."
            )
        spker_path = os.path.join(processed_data_dir, "spker_set.json")
        if not os.path.exists(spker_path):
            raise FileNotFoundError(f"Speaker map not found: {spker_path}")
        with open(spker_path, "r", encoding="utf-8") as f:
            cls._spker_map_cache = json.load(f)
        cls._spker_map_cache_key = processed_data_dir
        return cls._spker_map_cache

    @classmethod
    def _load_condition_maps(cls):
        cache_key = (hparams.get("binary_data_dir"), hparams.get("processed_data_dir"))
        if cls._condition_maps is not None and cls._condition_maps_key == cache_key:
            return cls._condition_maps
        condition_maps = {}
        for field in CONDITION_FIELDS:
            field_map = None
            for candidate_dir in [hparams.get("binary_data_dir"), hparams.get("processed_data_dir")]:
                if not candidate_dir:
                    continue
                map_candidate = os.path.join(candidate_dir, f"{field}_map.json")
                if os.path.exists(map_candidate):
                    field_map = {
                        str(label): int(idx)
                        for label, idx in json.load(open(map_candidate, encoding="utf-8")).items()
                    }
                    break
                set_candidate = os.path.join(candidate_dir, f"{field}_set.json")
                if os.path.exists(set_candidate):
                    vocab = json.load(open(set_candidate, encoding="utf-8"))
                    field_map = {
                        str(label): idx
                        for idx, label in enumerate(vocab)
                        if label not in (None, "")
                    }
                    break
            condition_maps[field] = field_map or {}
        cls._condition_maps = condition_maps
        cls._condition_maps_key = cache_key
        return condition_maps

    @classmethod
    def _resolve_condition_id(cls, item, field):
        explicit_key = f"{field}_id"
        explicit_value = item.get(explicit_key, None)
        if explicit_value is not None:
            return _safe_int(explicit_value, default=-1)
        raw_value = item.get(field, None)
        if raw_value is None:
            return -1
        if isinstance(raw_value, (int, np.integer)):
            return int(raw_value)
        condition_maps = cls._load_condition_maps()
        return int(condition_maps.get(field, {}).get(str(raw_value), -1))

    @classmethod
    def _resolve_speaker_id(cls, item_name):
        spker_map = cls._load_spker_map()
        speaker_key = _item_name_speaker_key(item_name)
        if speaker_key not in spker_map:
            raise KeyError(
                f"[{cls.__name__}] Speaker key '{speaker_key}' for {item_name} is missing from spker_set.json."
            )
        return int(spker_map[speaker_key])

    @staticmethod
    def _register_condition_label(field, label_to_id, id_to_label, label, idx):
        label = str(label)
        idx = int(idx)
        prev_idx = label_to_id.get(label, None)
        if prev_idx is not None and int(prev_idx) != idx:
            raise BinarizationError(
                f"Conflicting {field} ids for label '{label}': {prev_idx} vs {idx}"
            )
        prev_label = id_to_label.get(idx, None)
        if prev_label is not None and str(prev_label) != label:
            raise BinarizationError(
                f"Conflicting {field} labels for id {idx}: '{prev_label}' vs '{label}'"
            )
        label_to_id[label] = idx
        id_to_label[idx] = label

    @staticmethod
    def _next_available_condition_id(used_ids):
        next_id = 0
        while next_id in used_ids:
            next_id += 1
        return next_id

    @staticmethod
    def _resolve_optional_label(item, base_key):
        id_keys = [f'{base_key}_id', f'{base_key}_idx']
        alias_keys = STYLE_LABEL_FIELD_CANDIDATES.get(base_key, ())
        name_keys = []
        for key in [*alias_keys, base_key, f'{base_key}_label', f'{base_key}_name']:
            if key not in name_keys:
                name_keys.append(key)
        for key in id_keys:
            if key in item and item[key] not in (None, ''):
                try:
                    label_value = None
                    for name_key in name_keys:
                        if name_key in item and item[name_key] not in (None, ''):
                            label_value = _normalize_label_value(item[name_key])
                            break
                    return int(item[key]), None if label_value is None else str(label_value)
                except (TypeError, ValueError):
                    pass
        for key in name_keys:
            if key in item and item[key] not in (None, ''):
                normalized = _normalize_label_value(item[key])
                if normalized is not None:
                    return normalized, str(normalized)
        return None, None

    def _metadata_path(self):
        candidates = [
            os.path.join(self.processed_data_dir, 'metadata_vctk_librittsr_gt.json'),
            os.path.join(self.processed_data_dir, 'metadata.json'),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f'No metadata file found in {self.processed_data_dir}. Tried: {candidates}'
        )

    def _prepare_condition_metadata(self, items_list):
        condition_maps = {}
        condition_sets = {}
        for field in CONDITION_FIELDS:
            label_to_id = {}
            id_to_label = {}
            pending_labels = []

            for item in items_list:
                raw_value, raw_name = self._resolve_optional_label(item, field)
                if raw_value is None:
                    continue
                resolved_label = str(raw_name) if raw_name is not None else None
                if isinstance(raw_value, str):
                    pending_labels.append(resolved_label or str(raw_value))
                    continue
                explicit_id = int(raw_value)
                self._register_condition_label(
                    field,
                    label_to_id,
                    id_to_label,
                    resolved_label or str(explicit_id),
                    explicit_id,
                )

            for label in sorted(set(pending_labels)):
                if label in label_to_id:
                    continue
                assigned_id = self._next_available_condition_id(set(id_to_label.keys()))
                self._register_condition_label(field, label_to_id, id_to_label, label, assigned_id)

            for item in items_list:
                raw_value, raw_name = self._resolve_optional_label(item, field)
                if raw_value is None:
                    continue
                if isinstance(raw_value, str):
                    resolved_label = str(raw_name) if raw_name is not None else str(raw_value)
                    assigned_id = label_to_id[resolved_label]
                else:
                    assigned_id = int(raw_value)
                    resolved_label = str(raw_name) if raw_name is not None else id_to_label[assigned_id]
                item[f'{field}_id'] = int(assigned_id)
                item[f'{field}_name'] = resolved_label
                item[field] = resolved_label
                strength_key = f'{field}_strength'
                if strength_key in item and item[strength_key] not in (None, ''):
                    item[strength_key] = float(item[strength_key])

            max_id = max(id_to_label.keys(), default=-1)
            vocab = [None] * (max_id + 1)
            for idx, label in id_to_label.items():
                vocab[int(idx)] = str(label)
            condition_maps[field] = {
                str(label): int(idx)
                for label, idx in sorted(label_to_id.items(), key=lambda kv: (int(kv[1]), str(kv[0])))
            }
            condition_sets[field] = vocab

        for item in items_list:
            if 'energy' in item and isinstance(item['energy'], str):
                item['energy'] = [float(x) for x in item['energy'].split()]
        return condition_maps, condition_sets

    def split_train_test_set(self, item_names):
        deterministic_item_names = sorted(deepcopy(item_names))
        valid_prefixes = _normalize_prefixes(hparams.get('valid_prefixes', []))
        test_prefixes = _normalize_prefixes(hparams.get('test_prefixes', []))

        test_item_names = _match_by_speaker_prefix(deterministic_item_names, test_prefixes)
        valid_item_names = _match_by_speaker_prefix(deterministic_item_names, valid_prefixes)

        fallback_needed = len(valid_item_names) <= 0 or len(test_item_names) <= 0
        if fallback_needed:
            valid_item_names, test_item_names = _fallback_holdout_by_speaker(
                deterministic_item_names,
                valid_item_names=valid_item_names,
                test_item_names=test_item_names,
                fallback_valid_items_per_speaker=hparams.get('fallback_valid_items_per_speaker', 2),
                fallback_test_items_per_speaker=hparams.get('fallback_test_items_per_speaker', 2),
            )
            log_fn = logging.warning if valid_prefixes or test_prefixes else logging.info
            log_fn(
                "Using deterministic per-speaker utterance holdout for valid/test splits. "
                f"valid={len(valid_item_names)}, test={len(test_item_names)}"
            )

        test_set = set(test_item_names)
        valid_set = set(valid_item_names)
        train_item_names = [
            name for name in deterministic_item_names
            if name not in test_set and name not in valid_set
        ]
        if self.binarization_args.get('shuffle', False):
            random.seed(1234)
            random.shuffle(train_item_names)

        logging.info(f"train {len(train_item_names)}")
        logging.info(f"valid {len(valid_item_names)}")
        logging.info(f"test  {len(test_item_names)}")
        return train_item_names, test_item_names, valid_item_names


    def load_meta_data(self):
        metadata_path = self._metadata_path()
        items_list = json.load(open(metadata_path, encoding='utf-8'))
        self.label_maps, self.condition_sets = self._prepare_condition_metadata(items_list)
        for r in tqdm(items_list, desc='Loading meta data.'):
            item_name = r['item_name']
            self.items[item_name] = r
            self.item_names.append(item_name)
        self._write_condition_artifacts()
        self._train_item_names, self._test_item_names, self._valid_item_names = self.split_train_test_set(self.item_names)

    def _write_condition_artifacts(self):
        target_dirs = {self.processed_data_dir, hparams['binary_data_dir']}
        for target_dir in target_dirs:
            os.makedirs(target_dir, exist_ok=True)
            for field in CONDITION_FIELDS:
                json.dump(
                    self.label_maps.get(field, {}),
                    open(os.path.join(target_dir, f"{field}_map.json"), "w", encoding='utf-8'),
                    ensure_ascii=False,
                    indent=2,
                )
                json.dump(
                    self.condition_sets.get(field, []),
                    open(os.path.join(target_dir, f"{field}_set.json"), "w", encoding='utf-8'),
                    ensure_ascii=False,
                    indent=2,
                )

    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._valid_item_names

    @property
    def test_item_names(self):
        return self._test_item_names

    def process(self):
        self.load_meta_data()
        self.word_encoder = None
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')

    def meta_data(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            yield self.items[item_name]

    def process_data(self, prefix):
        items = self._collect_processed_items(prefix)
        self._attach_spk_embeddings(items, prefix)
        self._serialize_items(prefix, items)

    def _collect_processed_items(self, prefix):
        meta_data = list(self.meta_data(prefix))
        process_item = partial(self.process_item, binarization_args=self.binarization_args)
        args = [{'item': item} for item in meta_data]
        if self.num_workers <= 1:
            iterator = (
                (item_id, process_item(item=meta_item))
                for item_id, meta_item in enumerate(tqdm(meta_data, desc=f'Processing {prefix}'))
            )
        else:
            iterator = multiprocess_run_tqdm(process_item, args, desc=f'Processing {prefix}')
        items = []
        for _, item in iterator:
            if item is not None:
                items.append(item)
        return items

    def _attach_spk_embeddings(self, items, prefix):
        if not self.binarization_args['with_spk_embed'] or len(items) <= 0:
            return
        if self.num_workers <= 1:
            voice_encoder = _build_voice_encoder()
            for item in tqdm(items, desc=f'Extracting {prefix} spk embed'):
                item['spk_embed'] = np.asarray(
                    self.get_spk_embed(item['wav'], {'voice_encoder': voice_encoder}),
                    dtype=np.float32,
                )
            return
        spk_args = [{'wav': item['wav']} for item in items]
        for item_id, spk_embed in multiprocess_run_tqdm(
                self.get_spk_embed, spk_args,
                init_ctx_func=lambda wid: {'voice_encoder': _build_voice_encoder()},
                num_workers=min(max(1, self.num_workers), 4),
                desc=f'Extracting {prefix} spk embed'):
            items[item_id]['spk_embed'] = np.asarray(spk_embed, dtype=np.float32)

    def _serialize_items(self, prefix, items):
        data_dir = hparams['binary_data_dir']
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        lengths, spk_ids = [], []
        optional_ids = {field: [] for field in CONDITION_FIELDS}
        total_sec = 0.0
        for item in items:
            if not self.binarization_args['with_wav'] and 'wav' in item:
                del item['wav']
            builder.add_item(item)
            lengths.append(int(item['len']))
            spk_ids.append(int(item['spk_id']))
            for field in CONDITION_FIELDS:
                optional_ids[field].append(int(item.get(f'{field}_id', -1)))
            total_sec += float(item['sec'])
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', np.asarray(lengths, dtype=np.int32))
        np.save(f'{data_dir}/{prefix}_spk_ids.npy', np.asarray(spk_ids, dtype=np.int32))
        if items:
            np.save(f'{data_dir}/{prefix}_ref_indices.npy', self._build_reference_indices(items))
        for field, values in optional_ids.items():
            if any(v >= 0 for v in values):
                np.save(f'{data_dir}/{prefix}_{field}_ids.npy', np.asarray(values, dtype=np.int32))
        print(f"| {prefix} total duration: {total_sec:.2f}s, #items: {len(lengths)}")

    @classmethod
    def _load_frame_features(cls, item, *, item_name, wav_fn, mel, binarization_args):
        del item, item_name, wav_fn, mel, binarization_args
        return {}

    @classmethod
    def _finalize_common_item(cls, item, *, wav, mel, content, frame_features):
        lengths = [len(content), int(mel.shape[0])]
        for feature_name, feature_value in frame_features.items():
            feature_array = np.asarray(feature_value, dtype=np.float32).reshape(-1)
            frame_features[feature_name] = feature_array
            lengths.append(int(feature_array.shape[0]))
        min_length = min(lengths)
        if min_length <= 0:
            raise BinarizationError(f'No aligned frames remained for {item["item_name"]}')
        item['mel'] = mel = mel[:min_length]
        item['wav'] = wav[:min_length * hparams['hop_size']]
        item['hubert'] = content[:min_length]
        item['len'] = int(min_length)
        for feature_name, feature_value in frame_features.items():
            item[feature_name] = feature_value[:min_length]
        default_energy = np.abs(mel).mean(axis=-1)
        item['energy'] = _coerce_frame_series(item.get('energy', default_energy), min_length, default_value=0.0)
        for field in CONDITION_FIELDS:
            item[f'{field}_id'] = cls._resolve_condition_id(item, field)
            item[field] = _stringify_optional(item.get(field, ''))
        item['arousal'] = _safe_float(item.get('arousal', 0.0), default=0.0)
        item['valence'] = _safe_float(item.get('valence', 0.0), default=0.0)
        return item

    @classmethod
    def process_item(cls, item, binarization_args):
        item_name = item['item_name']
        wav_fn = item['wav_fn']
        wav, mel = cls.process_audio(wav_fn, item, binarization_args)
        mel = item['mel']
        wav = item['wav']
        if item.get('hubert', None) in (None, ''):
            raise KeyError(
                f"[{cls.__name__}] Missing `hubert` units for {item_name}. "
                "Run offline content extraction before binarization."
            )
        content = _parse_hubert_units(item.get('hubert', None), item_name=item_name)
        item['spk_id'] = cls._resolve_speaker_id(item_name)
        frame_features = cls._load_frame_features(
            item,
            item_name=item_name,
            wav_fn=wav_fn,
            mel=mel,
            binarization_args=binarization_args,
        )
        return cls._finalize_common_item(
            item,
            wav=wav,
            mel=mel,
            content=content,
            frame_features=frame_features,
        )


class ConanBinarizer(VCBinarizer):
    @classmethod
    def _load_frame_features(cls, item, *, item_name, wav_fn, mel, binarization_args):
        del item, binarization_args
        return {
            'f0': _load_cached_f0(wav_fn, item_name=item_name, max_frames=mel.shape[0]),
        }


class EmformerBinarizer(VCBinarizer):
    pass
