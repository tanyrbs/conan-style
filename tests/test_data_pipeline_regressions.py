import os
import pickle
import unittest
from tempfile import TemporaryDirectory

import numpy as np

from tasks.tts.dataset_utils import BaseSpeechDataset
from utils.commons.hparams import hparams
from utils.commons.indexed_datasets import IndexedDataset, IndexedDatasetBuilder


class DataPipelineRegressionTests(unittest.TestCase):
    def test_indexed_dataset_builder_writes_numeric_offsets(self):
        with TemporaryDirectory() as tmpdir:
            ds_path = os.path.join(tmpdir, "sample_ds")
            builder = IndexedDatasetBuilder(ds_path)
            builder.add_item({"item_name": "a", "mel": np.ones((2, 3), dtype=np.float32)})
            builder.add_item({"item_name": "b", "mel": np.zeros((1, 3), dtype=np.float32)})
            builder.finalize()
            del builder

            offsets = np.load(f"{ds_path}.idx", allow_pickle=False)
            self.assertEqual(offsets.dtype.kind, "i")
            self.assertEqual(offsets.tolist()[0], 0)

            with IndexedDataset(ds_path) as ds:
                self.assertEqual(ds[0]["item_name"], "a")
                self.assertEqual(ds[1]["item_name"], "b")
                self.assertEqual(tuple(ds[0]["mel"].shape), (2, 3))

    def test_indexed_dataset_reads_legacy_dict_offsets(self):
        with TemporaryDirectory() as tmpdir:
            ds_path = os.path.join(tmpdir, "legacy_ds")
            items = [
                {"item_name": "legacy_a", "value": 1},
                {"item_name": "legacy_b", "value": 2},
            ]
            offsets = [0]
            with open(f"{ds_path}.data", "wb") as data_file:
                for item in items:
                    payload = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
                    data_file.write(payload)
                    offsets.append(offsets[-1] + len(payload))
            with open(f"{ds_path}.idx", "wb") as index_file:
                np.save(index_file, {"offsets": offsets})

            with IndexedDataset(ds_path) as ds:
                self.assertEqual(ds[0]["item_name"], "legacy_a")
                self.assertEqual(ds[1]["value"], 2)

    def test_dataset_bucket_maps_are_seed_deterministic(self):
        original_hparams = dict(hparams)
        try:
            hparams.clear()
            hparams.update(
                {
                    "binary_data_dir": ".",
                    "sort_by_len": False,
                    "test_ids": [],
                    "min_frames": 0,
                    "max_samples_per_spk": 2,
                    "max_samples_per_emotion": 2,
                    "seed": 1234,
                }
            )
            items = [
                {"spk_id": 0, "emotion_id": 0},
                {"spk_id": 0, "emotion_id": 0},
                {"spk_id": 0, "emotion_id": 1},
                {"spk_id": 1, "emotion_id": 1},
                {"spk_id": 1, "emotion_id": 1},
                {"spk_id": 1, "emotion_id": 0},
            ]
            dataset_a = BaseSpeechDataset("train", items=items)
            dataset_b = BaseSpeechDataset("train", items=items)

            dataset_a._build_speaker_map()
            dataset_b._build_speaker_map()
            dataset_a._build_condition_map("emotion")
            dataset_b._build_condition_map("emotion")

            self.assertEqual(dict(dataset_a.spk2indices), dict(dataset_b.spk2indices))
            self.assertEqual(
                dict(dataset_a.cond2indices["emotion"]),
                dict(dataset_b.cond2indices["emotion"]),
            )
        finally:
            hparams.clear()
            hparams.update(original_hparams)


if __name__ == "__main__":
    unittest.main()
