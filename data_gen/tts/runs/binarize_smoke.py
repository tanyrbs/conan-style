import argparse
import os
import numpy as np
import importlib
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.commons.hparams import set_hparams, hparams
from utils.commons.indexed_datasets import IndexedDatasetBuilder


def _copy_aux_files(src_dir, dst_dir):
    if not os.path.isdir(src_dir):
        return
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        if name.endswith('_set.json') or name.endswith('_map.json') or name in {
            'spker_set.json', 'phone_set.json', 'word_set.json', 'meta.json'
        }:
            src = os.path.join(src_dir, name)
            dst = os.path.join(dst_dir, name)
            try:
                with open(src, 'rb') as fsrc:
                    with open(dst, 'wb') as fdst:
                        fdst.write(fsrc.read())
            except Exception:
                continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--hparams', default='')
    parser.add_argument('--processed_data_dir', default=None)
    parser.add_argument('--binary_data_dir', default=None)
    parser.add_argument('--num_items', type=int, default=5)
    args = parser.parse_args()

    set_hparams(config=args.config, hparams_str=args.hparams)
    if args.processed_data_dir:
        hparams['processed_data_dir'] = args.processed_data_dir
    if args.binary_data_dir:
        hparams['binary_data_dir'] = args.binary_data_dir

    binarizer_cls = hparams.get('binarizer_cls', 'data_gen.tts.base_binarizer.BaseBinarizer')
    pkg = ".".join(binarizer_cls.split(".")[:-1])
    cls_name = binarizer_cls.split(".")[-1]
    binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
    binarizer = binarizer_cls(processed_data_dir=hparams['processed_data_dir'])
    binarizer.load_meta_data()

    os.makedirs(hparams['binary_data_dir'], exist_ok=True)
    _copy_aux_files(hparams['processed_data_dir'], hparams['binary_data_dir'])

    item_names = list(getattr(binarizer, 'train_item_names', []))[:args.num_items]
    if len(item_names) <= 0:
        item_names = binarizer.item_names[:args.num_items]
    builder = IndexedDatasetBuilder(f"{hparams['binary_data_dir']}/train")
    lengths, spk_ids = [], []

    kept = 0
    for name in item_names:
        item = binarizer.items[name]
        out = binarizer.process_item(item, binarizer.binarization_args)
        if out is None:
            continue
        builder.add_item(out)
        lengths.append(int(out.get('len', len(out.get('mel', [])))))
        spk_ids.append(int(out.get('spk_id', -1)))
        kept += 1

    builder.finalize()
    np.save(f"{hparams['binary_data_dir']}/train_lengths.npy", np.array(lengths, np.int32))
    np.save(
        f"{hparams['binary_data_dir']}/train_ref_indices.npy",
        np.arange(kept, dtype=np.int32),
    )
    np.save(f"{hparams['binary_data_dir']}/train_spk_ids.npy", np.array(spk_ids, np.int32))

    print(
        f"| Smoke binarize done. kept={kept}, out_dir={hparams['binary_data_dir']}, "
        "train_ref_indices.npy emitted as identity fallback pairs"
    )


if __name__ == '__main__':
    main()
