import argparse
import json
import math
import os
import sys
from pathlib import Path

import librosa.core
import numpy as np
import torch
from tqdm import tqdm
import soundfile as sf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import pyworld as pw
except ImportError:  # pragma: no cover - optional backend
    pw = None

from data_gen.tts.base_binarizer import BaseBinarizer
from utils.commons.dataset_utils import batch_by_size
from utils.commons.hparams import hparams, set_hparams


def _resolve_metadata_paths(processed_data_dir):
    explicit_metafile_path = str(hparams.get('metafile_path', '') or '').strip()
    if explicit_metafile_path:
        metafile_paths = [path.strip() for path in explicit_metafile_path.split(',') if path.strip()]
        missing_paths = [path for path in metafile_paths if not os.path.exists(path)]
        if missing_paths:
            raise FileNotFoundError(f'metafile_path contains missing files: {missing_paths}')
    else:
        default_candidates = [
            os.path.join(processed_data_dir, 'metadata_vctk_librittsr_gt.json'),
            os.path.join(processed_data_dir, 'metadata.json'),
            os.path.join(processed_data_dir, 'metadata_hifigantraining.json'),
        ]
        metafile_paths = [path for path in default_candidates if os.path.exists(path)]
        if len(metafile_paths) <= 0:
            raise FileNotFoundError(
                f'Cannot find metadata file in {processed_data_dir}. Tried: {default_candidates}'
            )
        metafile_paths = [metafile_paths[0]]

    ds_names = str(
        hparams.get('ds_names', ','.join(str(i) for i in range(len(metafile_paths))))
    ).split(',')
    if len(ds_names) < len(metafile_paths):
        ds_names += [str(i) for i in range(len(ds_names), len(metafile_paths))]
    return metafile_paths, ds_names[:len(metafile_paths)]


def _resolve_f0_output_path(wav_fn):
    dir_name = os.path.dirname(wav_fn)
    base_name = os.path.splitext(os.path.basename(wav_fn))[0]
    f0_dir = dir_name + '_f0'
    os.makedirs(f0_dir, exist_ok=True)
    return os.path.join(f0_dir, f'{base_name}_f0.npy')


def _estimate_total_duration(item):
    for key in ('duration', 'sec'):
        value = item.get(key, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    ph_durs = item.get('ph_durs', None)
    if ph_durs is not None:
        if isinstance(ph_durs, str):
            ph_durs = [float(x) for x in ph_durs.replace(',', ' ').split() if x != '']
        return float(np.sum(ph_durs))
    wav_fn = item.get('wav_fn', None)
    if wav_fn and os.path.exists(str(wav_fn)):
        try:
            return float(sf.info(str(wav_fn)).duration)
        except Exception:
            pass
    raise KeyError(
        f"Item '{item.get('item_name', '<unknown>')}' is missing duration/ph_durs metadata needed for batching."
    )


def _resolve_pe_backend():
    return str(hparams.get('pe', 'rmvpe') or 'rmvpe').strip().lower()


def _resolve_rmvpe_ckpt():
    candidate_paths = [
        hparams.get('pe_ckpt', None),
        os.path.join('checkpoints', 'rmvpe.pt'),
        os.path.join('checkpoints', 'RMVPE', 'rmvpe.pt'),
    ]
    for candidate in candidate_paths:
        if candidate and os.path.exists(str(candidate)):
            return str(candidate)
    raise FileNotFoundError(
        'RMVPE checkpoint not found. Set hparams["pe_ckpt"] or pass --pe-ckpt <path>.'
    )


def _build_rmvpe(device=None):
    try:
        from modules.pe.rmvpe import RMVPE
    except Exception as exc:
        raise ImportError(
            'RMVPE could not be imported. Ensure torchaudio/PyTorch binaries are compatible before running offline F0 extraction.'
        ) from exc
    return RMVPE(_resolve_rmvpe_ckpt(), device=device)


class F0Extractor:
    def __init__(self):
        self.processed_data_dir = hparams['processed_data_dir']
        self.binarization_args = hparams['binarization_args']
        self.items = {}
        self.item_names = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_meta_data(self):
        metafile_paths, ds_names = _resolve_metadata_paths(self.processed_data_dir)
        for idx, metafile_path in enumerate(metafile_paths):
            with open(metafile_path, encoding='utf-8') as f:
                items_list = json.load(f)
            for record in tqdm(items_list, desc=f'| Loading meta data for dataset {ds_names[idx]}.'):
                item_name = record['item_name']
                if item_name in self.items:
                    print(f'warning: item name {item_name} duplicated')
                self.items[item_name] = record
                self.item_names.append(item_name)
                self.items[item_name]['ds_name'] = ds_names[idx]

    @staticmethod
    def generate(items, hparams, binarization_args, device=None, task_id=-1):
        saved_f0 = {}
        pe = _resolve_pe_backend()
        rmvpe = None
        if pe == 'rmvpe':
            rmvpe = _build_rmvpe(device=device)
        elif pe == 'pw' and pw is None:
            raise ImportError('pyworld is required when pe=pw. Please install pyworld first.')

        skipped = 0
        saved = 0
        for item_name in tqdm(
            items.keys(),
            total=len(items),
            desc='Extracting' + (f' {task_id}' if task_id >= 0 else ''),
            position=task_id,
        ):
            item = items[item_name]
            wav_fn = item['wav_fn']
            npy_fn = _resolve_f0_output_path(wav_fn)
            if os.path.exists(npy_fn):
                skipped += 1
                continue

            wav, mel = BaseBinarizer.process_audio(wav_fn, item, binarization_args)
            length = int(mel.shape[0])
            if pe == 'rmvpe':
                waveform = item['wav']
                with torch.no_grad():
                    f0, _ = rmvpe.get_pitch(
                        waveform,
                        sample_rate=hparams['audio_sample_rate'],
                        hop_size=hparams['hop_size'],
                        length=length,
                        fmax=hparams['f0_max'],
                        fmin=hparams['f0_min'],
                    )
            elif pe == 'pw':
                f0, _ = pw.harvest(
                    wav.astype(np.double),
                    hparams['audio_sample_rate'],
                    frame_period=hparams['hop_size'] * 1000 / hparams['audio_sample_rate'],
                )
                delta_l = length - len(f0)
                if delta_l < 0:
                    f0 = f0[:length]
                elif delta_l > 0 and len(f0) > 0:
                    f0 = np.concatenate((f0, np.full(delta_l, fill_value=f0[-1])), axis=0)
                elif delta_l > 0:
                    f0 = np.zeros((length,), dtype=np.float32)
            else:
                raise ValueError(f'Unsupported pitch extractor backend: {pe}')

            f0 = np.asarray(f0, dtype=np.float32)
            np.save(npy_fn, f0)
            saved_f0[item_name] = f0
            saved += 1

        print(f'Finished F0 extraction: saved={saved}, skipped={skipped}, total={len(items)}')
        return saved_f0

    @staticmethod
    def generate_batch(items, hparams, binarization_args, device=None, task_id=-1, bsz=1, max_tokens=100000):
        pe = _resolve_pe_backend()
        if pe != 'rmvpe':
            raise ValueError('Batched F0 extraction is only supported for pe=rmvpe. Use --batch-size 1 for pw.')

        rmvpe = _build_rmvpe(device=device)
        skipped = 0
        saved = 0

        item_names = list(items.keys())
        id_and_sizes = []
        for item_name in item_names:
            total_duration = _estimate_total_duration(items[item_name])
            total_frames = math.ceil(total_duration * hparams['audio_sample_rate'] / hparams['hop_size'])
            id_and_sizes.append((item_name, max(1, int(total_frames))))
        batches = batch_by_size(id_and_sizes, lambda pair: pair[1], max_tokens=max_tokens, max_sentences=bsz)
        batches = [[pair[0] for pair in batch] for batch in batches]

        for batch in tqdm(
            batches,
            total=len(batches),
            desc='Extracting' + (f' {task_id}' if task_id >= 0 else ''),
        ):
            wavs, lengths, npy_fns = [], [], []
            for item_name in batch:
                item = items[item_name]
                wav_fn = item['wav_fn']
                npy_fn = _resolve_f0_output_path(wav_fn)
                if os.path.exists(npy_fn):
                    skipped += 1
                    continue
                wav, _ = librosa.core.load(wav_fn, sr=hparams['audio_sample_rate'])
                wavs.append(np.asarray(wav, dtype=np.float32))
                lengths.append((wav.shape[0] + hparams['hop_size'] - 1) // hparams['hop_size'])
                npy_fns.append(npy_fn)
            if len(lengths) <= 0:
                continue

            with torch.no_grad():
                f0s, _ = rmvpe.get_pitch_batch(
                    wavs,
                    sample_rate=hparams['audio_sample_rate'],
                    hop_size=hparams['hop_size'],
                    lengths=lengths,
                    fmax=hparams['f0_max'],
                    fmin=hparams['f0_min'],
                )
            for npy_fn, f0 in zip(npy_fns, f0s):
                np.save(npy_fn, np.asarray(f0, dtype=np.float32))
                saved += 1

        print(f'Finished batched F0 extraction: saved={saved}, skipped={skipped}, total={len(items)}')

    def process(self, args):
        self.load_meta_data()

        if args.num_shards > 1 and args.shard_id >= 0:
            print(f'| Generate shard {args.shard_id}/{args.num_shards}')
            shard_size = math.ceil(len(self.item_names) / args.num_shards)
            shard_items = {
                name: self.items[name]
                for name in self.item_names[args.shard_id * shard_size:(args.shard_id + 1) * shard_size]
            }
            F0Extractor.generate_batch(
                shard_items,
                hparams,
                self.binarization_args,
                self.device,
                task_id=args.shard_id,
                bsz=args.batch_size,
                max_tokens=args.max_tokens,
            )
        elif args.num_shards == 1:
            print('| Generate in 1 shard')
            if args.batch_size == 1:
                F0Extractor.generate(self.items, hparams, self.binarization_args, self.device)
            elif args.batch_size > 1:
                print(f'| Generate in max sentences {args.batch_size}, max tokens {args.max_tokens}')
                F0Extractor.generate_batch(
                    self.items,
                    hparams,
                    self.binarization_args,
                    self.device,
                    bsz=args.batch_size,
                    max_tokens=args.max_tokens,
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--batch-size', '-bsz', required=False, default=1, type=int)
    parser.add_argument('--max-tokens', required=False, default=40000, type=int)
    parser.add_argument('--num-shards', required=False, default=1, type=int)
    parser.add_argument('--shard-id', required=False, default=-1, type=int)
    parser.add_argument(
        '--pe-ckpt',
        required=False,
        default='',
        help='Optional override for the RMVPE checkpoint path.',
    )
    args = parser.parse_args()

    set_hparams(args.config)
    if str(args.pe_ckpt).strip():
        hparams['pe_ckpt'] = str(args.pe_ckpt).strip()
    extractor = F0Extractor()
    extractor.process(args)
