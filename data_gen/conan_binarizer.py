from resemblyzer import VoiceEncoder
from utils.audio import librosa_wav2spec, get_energy_librosa, norm_energy
import hashlib
import shutil
import random, os, json
import traceback
from copy import deepcopy
import logging
from data_gen.tts.base_binarizer import BaseBinarizer, BinarizationError
from utils.commons.hparams import hparams
from utils.commons.indexed_datasets import IndexedDatasetBuilder
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from functools import partial
import numpy as np
from tqdm import tqdm
from utils.audio.align import get_mel2ph, mel2token_to_dur
from utils.text.text_encoder import build_token_encoder
from utils.audio.pitch.utils import f0_to_coarse
np.seterr(divide='ignore', invalid='ignore')


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


def _coerce_int(value, default=0):
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


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


class BinarizationError(Exception):
    pass

class BaseBinarizer:
    def __init__(self, processed_data_dir=None):
        if processed_data_dir is None:
            processed_data_dir = hparams['processed_data_dir']
        self.processed_data_dir = processed_data_dir
        self.binarization_args = hparams['binarization_args']
        self.items = {}
        self.item_names = []

    def load_meta_data(self):
        processed_data_dir = self.processed_data_dir
        items_list = json.load(open(f"{processed_data_dir}/metadata.json"))
        for r in tqdm(items_list, desc='Loading meta data.'):
            item_name = r['item_name']
            self.items[item_name] = r
            self.item_names.append(item_name)
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)

    @property
    def train_item_names(self):
        range_ = self._convert_range(self.binarization_args['train_range'])
        return self.item_names[range_[0]:range_[1]]

    @property
    def valid_item_names(self):
        range_ = self._convert_range(self.binarization_args['valid_range'])
        return self.item_names[range_[0]:range_[1]]

    @property
    def test_item_names(self):
        range_ = self._convert_range(self.binarization_args['test_range'])
        return self.item_names[range_[0]:range_[1]]

    def _convert_range(self, range_):
        if range_[1] == -1:
            range_[1] = len(self.item_names)
        return range_

    def meta_data(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            yield self.items[item_name]

    # def process_data(self, prefix):
    #     data_dir = hparams['binary_data_dir']
    #     builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
    #     meta_data = list(self.meta_data(prefix))
    #     process_item = partial(self.process_item, binarization_args=self.binarization_args)
    #     ph_lengths = []
    #     mel_lengths = []
    #     total_sec = 0
    #     items = []
    #     args = [{'item': item} for item in meta_data]
    #     for item_id, item in multiprocess_run_tqdm(process_item, args, desc='Processing data'):
    #         if item is not None:
    #             items.append(item)
    #     if self.binarization_args['with_spk_embed']:
    #         args = [{'wav': item['wav']} for item in items]
    #         for item_id, spk_embed in multiprocess_run_tqdm(
    #                 self.get_spk_embed, args,
    #                 init_ctx_func=lambda wid: {'voice_encoder': VoiceEncoder().cuda()}, num_workers=4,
    #                 desc='Extracting spk embed'):
    #             items[item_id]['spk_embed'] = spk_embed

    #     for item in items:
    #         if not self.binarization_args['with_wav'] and 'wav' in item:
    #             del item['wav']
    #         builder.add_item(item)
    #         mel_lengths.append(item['len'])
    #         assert item['len'] > 0, (item['item_name'], item['txt'], item['mel2ph'])
    #         if 'ph_len' in item:
    #             ph_lengths.append(item['ph_len'])
    #         total_sec += item['sec']
    #     builder.finalize()
    #     np.save(f'{data_dir}/{prefix}_lengths.npy', mel_lengths)
    #     if len(ph_lengths) > 0:
    #         np.save(f'{data_dir}/{prefix}_ph_lengths.npy', ph_lengths)
    #     print(f"| {prefix} total duration: {total_sec:.3f}s")

    @classmethod
    def process_item(cls, item, binarization_args):
        item['ph_len'] = len(item['ph_token'])
        item_name = item['item_name']
        wav_fn = item['wav_fn']
        wav, mel = cls.process_audio(wav_fn, item, binarization_args)
        try:
            n_bos_frames, n_eos_frames = 0, 0
            if binarization_args['with_align']:
                tg_fn = f"{hparams['processed_data_dir']}/mfa_outputs/{item_name}.TextGrid"
                item['tg_fn'] = tg_fn
                cls.process_align(tg_fn, item)
                if binarization_args['trim_eos_bos']:
                    n_bos_frames = item['dur'][0]
                    n_eos_frames = item['dur'][-1]
                    T = len(mel)
                    item['mel'] = mel[n_bos_frames:T - n_eos_frames]
                    item['mel2ph'] = item['mel2ph'][n_bos_frames:T - n_eos_frames]
                    item['mel2word'] = item['mel2word'][n_bos_frames:T - n_eos_frames]
                    item['dur'] = item['dur'][1:-1]
                    item['dur_word'] = item['dur_word'][1:-1]
                    item['len'] = item['mel'].shape[0]
                    item['wav'] = wav[n_bos_frames * hparams['hop_size']:len(wav) - n_eos_frames * hparams['hop_size']]
            if binarization_args['with_f0']:
                cls.process_pitch(item, n_bos_frames, n_eos_frames)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        except Exception as e:
            traceback.print_exc()
            print(f"| Skip item. item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return item

    @classmethod
    def process_audio(cls, wav_fn, res, binarization_args):
        wav2spec_dict = librosa_wav2spec(
            wav_fn,
            fft_size=hparams['fft_size'],
            hop_size=hparams['hop_size'],
            win_length=hparams['win_size'],
            num_mels=hparams['audio_num_mel_bins'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            loud_norm=hparams['loud_norm'])
        mel = wav2spec_dict['mel']
        wav = wav2spec_dict['wav'].astype(np.float16)
        # wav = wav2spec_dict['wav']
        if binarization_args['with_linear']:
            res['linear'] = wav2spec_dict['linear']
        res.update({'mel': mel, 'wav': wav, 'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0]})
        return wav, mel

    @staticmethod
    def process_align(tg_fn, item):
        ph = item['ph']
        mel = item['mel']
        ph_token = item['ph_token']
        if tg_fn is not None and os.path.exists(tg_fn):
            mel2ph, dur = get_mel2ph(tg_fn, ph, mel, hparams['hop_size'], hparams['audio_sample_rate'],
                                     hparams['binarization_args']['min_sil_duration'])
        else:
            raise BinarizationError(f"Align not found")
        if np.array(mel2ph).max() - 1 >= len(ph_token):
            raise BinarizationError(
                f"Align does not match: mel2ph.max() - 1: {np.array(mel2ph).max() - 1}, len(phone_encoded): {len(ph_token)}")
        item['mel2ph'] = mel2ph
        item['dur'] = dur

        ph2word = item['ph2word']
        mel2word = [ph2word[p - 1] for p in item['mel2ph']]
        item['mel2word'] = mel2word  # [T_mel]
        dur_word = mel2token_to_dur(mel2word, len(item['word_token']))
        item['dur_word'] = dur_word.tolist()  # [T_word]

    @staticmethod
    def process_pitch(item, n_bos_frames, n_eos_frames):
        wav, mel = item['wav'], item['mel']
        f0 = extract_pitch_simple(item['wav'])
        if sum(f0) == 0:
            raise BinarizationError("Empty f0")
        assert len(mel) == len(f0), (len(mel), len(f0))
        pitch_coarse = f0_to_coarse(f0)
        item['f0'] = f0
        item['pitch'] = pitch_coarse
        if hparams['binarization_args']['with_f0cwt']:
            uv, cont_lf0_lpf = get_cont_lf0(f0)
            logf0s_mean_org, logf0s_std_org = np.mean(cont_lf0_lpf), np.std(cont_lf0_lpf)
            cont_lf0_lpf_norm = (cont_lf0_lpf - logf0s_mean_org) / logf0s_std_org
            cwt_spec, scales = get_lf0_cwt(cont_lf0_lpf_norm)
            item['cwt_spec'] = cwt_spec
            item['cwt_mean'] = logf0s_mean_org
            item['cwt_std'] = logf0s_std_org

    @staticmethod
    def get_spk_embed(wav, ctx):
        return ctx['voice_encoder'].embed_utterance(wav.astype(float))

    @property
    def num_workers(self):
        return int(os.getenv('N_PROC', hparams.get('N_PROC', os.cpu_count())))
    
    def _word_encoder(self):
        fn = f"{hparams['binary_data_dir']}/word_set.json"
        word_set = []
        if self.binarization_args['reset_word_dict']:
            for word_sent in self.item2txt.values():
                word_set += [x for x in word_sent.split(' ') if x != '']
            word_set = Counter(word_set)
            total_words = sum(word_set.values())
            word_set = word_set.most_common(hparams['word_size'])
            num_unk_words = total_words - sum([x[1] for x in word_set])
            word_set = [x[0] for x in word_set]
            json.dump(word_set, open(fn, 'w'))
            print(f"| Build word set. Size: {len(word_set)}, #total words: {total_words},"
                  f" #unk_words: {num_unk_words}, word_set[:10]:, {word_set[:10]}.")
        else:
            word_set = json.load(open(fn, 'r'))
            print("| Load word set. Size: ", len(word_set), word_set[:10])
        return TokenTextEncoder(None, vocab_list=word_set, replace_oov='<UNK>')

class VCBinarizer(BaseBinarizer):
    _spker_map_cache = None
    _spker_map_cache_key = None

    def __init__(self, processed_data_dir=None):
        super().__init__()
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

    def _normalize_control_metadata(self, items_list):
        label_specs = ('emotion', 'style', 'accent')
        string_values = {key: [] for key in label_specs}
        for item in items_list:
            for key in label_specs:
                raw_value, raw_name = self._resolve_optional_label(item, key)
                if isinstance(raw_value, str):
                    string_values[key].append(raw_name)

        label_maps = {}
        for key in label_specs:
            uniq_names = sorted(set(string_values[key]))
            if len(uniq_names) > 0:
                label_maps[key] = {name: idx for idx, name in enumerate(uniq_names)}

        for item in items_list:
            for key in label_specs:
                raw_value, raw_name = self._resolve_optional_label(item, key)
                if raw_value is None:
                    continue
                if isinstance(raw_value, str):
                    item[f'{key}_id'] = label_maps[key][raw_name]
                    item[f'{key}_name'] = raw_name
                    item[key] = raw_name
                else:
                    item[f'{key}_id'] = int(raw_value)
                    item[f'{key}_name'] = raw_name if raw_name is not None else str(raw_value)
                    item[key] = item[f'{key}_name']
                strength_key = f'{key}_strength'
                if strength_key in item and item[strength_key] not in (None, ''):
                    item[strength_key] = float(item[strength_key])
            if 'energy' in item and isinstance(item['energy'], str):
                item['energy'] = [float(x) for x in item['energy'].split()]
        return label_maps

    def split_train_test_set(self, item_names):
        item_names = sorted(deepcopy(item_names))

        def _speaker_token(name):
            return _item_name_speaker_key(name)

        def _normalize_prefixes(prefixes):
            normalized = []
            for prefix in prefixes or []:
                prefix = str(prefix).strip()
                if prefix != "":
                    normalized.append(prefix)
            return normalized

        def _match_by_speaker_prefix(names, prefixes):
            prefix_set = set(_normalize_prefixes(prefixes))
            if len(prefix_set) <= 0:
                return []
            return [name for name in names if _speaker_token(name) in prefix_set]

        test_item_names = _match_by_speaker_prefix(item_names, hparams.get('test_prefixes', []))
        valid_item_names = _match_by_speaker_prefix(item_names, hparams.get('valid_prefixes', []))

        fallback_needed = len(valid_item_names) <= 0 or len(test_item_names) <= 0
        if fallback_needed:
            from collections import defaultdict

            per_speaker = defaultdict(list)
            for name in item_names:
                per_speaker[_speaker_token(name)].append(name)

            fallback_valid = max(1, int(hparams.get('fallback_valid_items_per_speaker', 2)))
            fallback_test = max(1, int(hparams.get('fallback_test_items_per_speaker', 2)))

            reserved_valid = set(valid_item_names)
            reserved_test = set(test_item_names)
            used = reserved_valid | reserved_test

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

            test_item_names = sorted(reserved_test)
            valid_item_names = sorted(reserved_valid - reserved_test)
            has_configured_prefixes = bool(_normalize_prefixes(hparams.get('valid_prefixes', []))) or bool(
                _normalize_prefixes(hparams.get('test_prefixes', []))
            )
            log_fn = logging.warning if has_configured_prefixes else logging.info
            log_fn(
                "Using deterministic per-speaker utterance holdout for valid/test splits. "
                f"valid={len(valid_item_names)}, test={len(test_item_names)}"
            )

        test_set = set(test_item_names)
        valid_set = set(valid_item_names)
        train_item_names = [x for x in item_names if x not in test_set and x not in valid_set]

        logging.info(f"train {len(train_item_names)}")
        logging.info(f"valid {len(valid_item_names)}")
        logging.info(f"test  {len(test_item_names)}")
        return train_item_names, test_item_names, valid_item_names


    def load_meta_data(self):
        processed_data_dir = self.processed_data_dir
        metadata_candidates = [
            f"{processed_data_dir}/metadata_vctk_librittsr_gt.json",
            f"{processed_data_dir}/metadata.json",
        ]
        metadata_path = None
        for candidate in metadata_candidates:
            if os.path.exists(candidate):
                metadata_path = candidate
                break
        if metadata_path is None:
            raise FileNotFoundError(
                f"Cannot find metadata file in {processed_data_dir}. "
                f"Tried: {metadata_candidates}"
            )
        items_list = json.load(open(metadata_path, encoding='utf-8'))
        self.label_maps = self._normalize_control_metadata(items_list)
        for r in tqdm(items_list, desc='Loading meta data.'):
            item_name = r['item_name']
            self.items[item_name] = r
            self.item_names.append(item_name)
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)
        self._write_condition_maps(items_list)
        self._train_item_names, self._test_item_names,self._valid_item_names = self.split_train_test_set(self.item_names)

    def _write_condition_maps(self, items_list):
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        for field, mapping in self.label_maps.items():
            map_path = os.path.join(hparams['binary_data_dir'], f"{field}_map.json")
            json.dump(mapping, open(map_path, "w", encoding='utf-8'), ensure_ascii=False, indent=2)
        for field in CONDITION_FIELDS:
            raw_values = []
            for item in items_list:
                value = item.get(f"{field}_name", item.get(field, None))
                if value is None:
                    continue
                if isinstance(value, str):
                    value = value.strip()
                    if value == "":
                        continue
                raw_values.append(value)
            if not raw_values:
                vocab = ["<UNK>"]
            elif all(isinstance(v, (int, np.integer)) for v in raw_values):
                max_id = int(max(raw_values))
                vocab = [str(i) for i in range(max_id + 1)]
            else:
                normalized = sorted({str(v) for v in raw_values})
                vocab = ["<UNK>"] + normalized
            for target_dir in {self.processed_data_dir, hparams['binary_data_dir']}:
                os.makedirs(target_dir, exist_ok=True)
                json.dump(vocab, open(os.path.join(target_dir, f"{field}_set.json"), "w"), ensure_ascii=False, indent=2)

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
        # ph_set_fn = f"{hparams['binary_data_dir']}/phone_set.json"
        # if not os.path.exists(ph_set_fn):
        #     oldp=os.path.join(hparams["processed_data_dir"], "phone_set.json")
        #     newp=hparams['binary_data_dir']
            # shutil.copy(oldp,newp)
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
        data_dir = hparams['binary_data_dir']
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')

        lengths, spk_ids = [], []
        optional_ids = {field: [] for field in CONDITION_FIELDS}
        total_sec = 0.0
        kept_items = []
        meta_data = list(self.meta_data(prefix))
        process_item = partial(self.process_item,
                            binarization_args=self.binarization_args)

        args = [{'item': it} for it in meta_data]

        if self.binarization_args['with_spk_embed']:
            voice_encoder = VoiceEncoder().cuda()

        if self.num_workers <= 1:
            iterator = (
                (item_id, process_item(item=meta_item))
                for item_id, meta_item in enumerate(tqdm(meta_data, desc=f'Processing {prefix}'))
            )
        else:
            iterator = multiprocess_run_tqdm(process_item, args, desc=f'Processing {prefix}')

        for item_id, item in iterator:
            # item['spk_embed'] = voice_encoder.embed_utterance(item['wav'])             #     if self.binarization_args['with_spk_embed'] else None
            if item is None:
                continue
            builder.add_item(item)          # spk_id is already included in item
            kept_items.append(item)
            lengths.append(item['len'])
            spk_ids.append(item['spk_id'])  # ?? collect spk_id in consistent order
            for field in CONDITION_FIELDS:
                optional_ids[field].append(int(item.get(f'{field}_id', -1)))
            total_sec += item['sec']


        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', np.array(lengths,  np.int32))
        np.save(f'{data_dir}/{prefix}_spk_ids.npy', np.array(spk_ids, np.int32))  # ✅ newly added
        if len(kept_items) > 0:
            np.save(f'{data_dir}/{prefix}_ref_indices.npy', self._build_reference_indices(kept_items))
        for field, values in optional_ids.items():
            if any(v >= 0 for v in values):
                np.save(f'{data_dir}/{prefix}_{field}_ids.npy', np.array(values, np.int32))
        print(f"| {prefix} total duration: {total_sec:.2f}s, #items: {len(lengths)}")


    # @classmethod
    # def process_item(cls, item, binarization_args):
    #     item_name = item['item_name']
    #     wav_fn = item['wav_fn']
    #     wav, mel = cls.process_audio(wav_fn, item, binarization_args)
    #     # item['spk_embed'] = np.load(wav_fn.replace(".wav", "_spk.npy"))
    #     try:
    #         cls.process_pitch(item, 0, 0)
    #         cls.process_align(item["tg_fn"], item)
    #     except BinarizationError as e:
    #         print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
    #         return None
    #     return item
    
class ConanBinarizer(VCBinarizer):
    # ph_encoder = build_token_encoder(os.path.join(hparams["processed_data_dir"], "phone_set.json"))
    _condition_maps = None
    _condition_maps_key = None

    @classmethod
    def _load_condition_maps(cls):
        cache_key = (hparams.get("binary_data_dir"), hparams.get("processed_data_dir"))
        if cls._condition_maps is not None and cls._condition_maps_key == cache_key:
            return cls._condition_maps
        condition_maps = {}
        for field in CONDITION_FIELDS:
            vocab = ["<UNK>"]
            for candidate_dir in [hparams.get("binary_data_dir"), hparams.get("processed_data_dir")]:
                if not candidate_dir:
                    continue
                candidate = os.path.join(candidate_dir, f"{field}_set.json")
                if os.path.exists(candidate):
                    vocab = json.load(open(candidate))
                    break
            condition_maps[field] = {str(v): idx for idx, v in enumerate(vocab)}
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
        return int(condition_maps.get(field, {}).get(str(raw_value), 0))

    @classmethod
    def process_item(cls, item, binarization_args):
        item_name = item['item_name']
        wav_fn = item['wav_fn']
        wav, mel = cls.process_audio(wav_fn, item, binarization_args)
        mel=item['mel']
        wav=item['wav']
        content=[int(float(x)) for x in item['hubert'].split()]

        # item["ph_token"] = cls.ph_encoder.encode(' '.join(item["ph"]))
        spker_map = cls._load_spker_map()
        speaker_key = _item_name_speaker_key(item["item_name"])
        item["spk_id"] = spker_map[speaker_key]
        # item['txt']=" ".join(item['txt'])
        
        # try:
        f0_path = os.path.join(
            os.path.dirname(wav_fn) + "_f0",
            os.path.basename(wav_fn).replace(".wav", "_f0.npy")
        )
        f0 = np.load(f0_path)[:mel.shape[0]]
        min_length = min(len(content), len(mel), len(f0))
        item["f0"] = f0 = f0[:min_length]
        item['mel'] = mel = mel[:min_length]
        item['wav'] = wav = wav[:min_length * hparams['hop_size']]
        item['hubert'] = content = content[:min_length]
        item['len'] = min_length
        frame_energy = np.asarray(item.get("energy", np.abs(mel).mean(axis=-1)), dtype=np.float32)
        if frame_energy.ndim == 0:
            frame_energy = np.full((min_length,), float(frame_energy), dtype=np.float32)
        item["energy"] = frame_energy[:min_length]
        item["emotion_id"] = cls._resolve_condition_id(item, "emotion")
        item["style_id"] = cls._resolve_condition_id(item, "style")
        item["accent_id"] = cls._resolve_condition_id(item, "accent")
        item["emotion"] = str(item.get("emotion", ""))
        item["style"] = str(item.get("style", ""))
        item["accent"] = str(item.get("accent", ""))
        item["arousal"] = _safe_float(item.get("arousal", 0.0), default=0.0)
        item["valence"] = _safe_float(item.get("valence", 0.0), default=0.0)
        # print(f'f0_length: {f0.shape}, mel_length: {mel.shape},wav_length: {wav.shape}, content_length: {content.shape}, item_name: {item_name}')
        # except:
        #     # parselmouth
            # time_step = hparams['hop_size'] / hparams['audio_sample_rate'] * 1000
            # f0_min = 80
            # f0_max = 800
            # if hparams['hop_size'] == 128:
            #     pad_size = 4
            # elif hparams['hop_size'] == 256:
            #     pad_size = 2
            # else:
            #     assert False
            # import parselmouth
            # f0 = parselmouth.Sound(wav, hparams['audio_sample_rate']).to_pitch_ac(
            #     time_step=time_step / 1000, voicing_threshold=0.6,
            #     pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
            # lpad = pad_size * 2
            # rpad = len(mel) - len(f0) - lpad
            # f0 = np.pad(f0, [[lpad, rpad]], mode='constant')
            # delta_l = len(mel) - len(f0)
            # assert np.abs(delta_l) <= 8
            # if delta_l > 0:
            #     f0 = np.concatenate([f0, [f0[-1]] * delta_l], 0)
            # f0 = f0[:len(mel)]
            # item["f0"] = f0
        
        # cls.process_align(item["ph_durs"], mel, item)
            
        return item
    
    @staticmethod
    def process_align(ph_durs, mel, item, hop_size=None, audio_sample_rate=None):
        if hop_size is None:
            hop_size = hparams['hop_size']
        if audio_sample_rate is None:
            audio_sample_rate = hparams['audio_sample_rate']
        mel2ph = np.zeros([mel.shape[0]], int)
        startTime = 0

        for i_ph in range(len(ph_durs)):
            start_frame = int(startTime * audio_sample_rate / hop_size + 0.5)
            end_frame = int((startTime + ph_durs[i_ph]) * audio_sample_rate / hop_size + 0.5)
            mel2ph[start_frame:end_frame] = i_ph + 1
            startTime = startTime + ph_durs[i_ph]

        item['mel2ph'] = mel2ph



class EmformerBinarizer(VCBinarizer):
    # difference between EmformerBinarizer and ConanBinarizer: No f0 information needed
    # ph_encoder = build_token_encoder(os.path.join(hparams["processed_data_dir"], "phone_set.json"))
    @classmethod
    def process_item(cls, item, binarization_args):
        item_name = item['item_name']
        wav_fn = item['wav_fn']
        wav, mel = cls.process_audio(wav_fn, item, binarization_args)
        mel=item['mel']
        wav=item['wav']
        content=[int(float(x)) for x in item['hubert'].split()]

        # item["ph_token"] = cls.ph_encoder.encode(' '.join(item["ph"]))
        spker_map = cls._load_spker_map()
        speaker_key = _item_name_speaker_key(item["item_name"])
        item["spk_id"] = spker_map[speaker_key]
        # item['txt']=" ".join(item['txt'])
        
        min_length = min(len(content), len(mel))
        item['mel'] = mel = mel[:min_length]
        item['wav'] = wav = wav[:min_length * hparams['hop_size']]
        item['hubert'] = content = content[:min_length]
        item['len'] = min_length
        frame_energy = np.asarray(item.get("energy", np.abs(mel).mean(axis=-1)), dtype=np.float32)
        if frame_energy.ndim == 0:
            frame_energy = np.full((min_length,), float(frame_energy), dtype=np.float32)
        item["energy"] = frame_energy[:min_length]
        item["emotion_id"] = ConanBinarizer._resolve_condition_id(item, "emotion")
        item["style_id"] = ConanBinarizer._resolve_condition_id(item, "style")
        item["accent_id"] = ConanBinarizer._resolve_condition_id(item, "accent")
        item["emotion"] = str(item.get("emotion", ""))
        item["style"] = str(item.get("style", ""))
        item["accent"] = str(item.get("accent", ""))
        item["arousal"] = _safe_float(item.get("arousal", 0.0), default=0.0)
        item["valence"] = _safe_float(item.get("valence", 0.0), default=0.0)
        # print(f'f0_length: {f0.shape}, mel_length: {mel.shape},wav_length: {wav.shape}, content_length: {content.shape}, item_name: {item_name}')
        # except:
        #     # parselmouth
            # time_step = hparams['hop_size'] / hparams['audio_sample_rate'] * 1000
            # f0_min = 80
            # f0_max = 800
            # if hparams['hop_size'] == 128:
            #     pad_size = 4
            # elif hparams['hop_size'] == 256:
            #     pad_size = 2
            # else:
            #     assert False
            # import parselmouth
            # f0 = parselmouth.Sound(wav, hparams['audio_sample_rate']).to_pitch_ac(
            #     time_step=time_step / 1000, voicing_threshold=0.6,
            #     pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
            # lpad = pad_size * 2
            # rpad = len(mel) - len(f0) - lpad
            # f0 = np.pad(f0, [[lpad, rpad]], mode='constant')
            # delta_l = len(mel) - len(f0)
            # assert np.abs(delta_l) <= 8
            # if delta_l > 0:
            #     f0 = np.concatenate([f0, [f0[-1]] * delta_l], 0)
            # f0 = f0[:len(mel)]
            # item["f0"] = f0
        
        # cls.process_align(item["ph_durs"], mel, item)
            
        return item
    
    @staticmethod
    def process_align(ph_durs, mel, item, hop_size=None, audio_sample_rate=None):
        if hop_size is None:
            hop_size = hparams['hop_size']
        if audio_sample_rate is None:
            audio_sample_rate = hparams['audio_sample_rate']
        mel2ph = np.zeros([mel.shape[0]], int)
        startTime = 0

        for i_ph in range(len(ph_durs)):
            start_frame = int(startTime * audio_sample_rate / hop_size + 0.5)
            end_frame = int((startTime + ph_durs[i_ph]) * audio_sample_rate / hop_size + 0.5)
            mel2ph[start_frame:end_frame] = i_ph + 1
            startTime = startTime + ph_durs[i_ph]

        item['mel2ph'] = mel2ph
