import unittest
import warnings
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import inference.Conan as inference_conan
from modules.Conan.Conan import Conan
from modules.Conan.control.separation_metrics import resolve_dynamic_timbre_frame_weight
from modules.Conan.control.style_success import (
    _topk_farthest_negative_mask,
    resolve_style_success_target_summary,
    resolve_style_success_negative_masks,
    resolve_style_success_rank_source_scale,
)
from modules.Conan.pitch_runtime import ConanPitchGenerationMixin
from modules.Conan.prosody_util import ProsodyAligner
from modules.Conan.prosody_util import VQEmbeddingEMA
from modules.Conan.style_timbre_runtime import _resolve_style_owner_residual
from tasks.Conan.mainline_train_prep import (
    _check_binary_sidecar_consistency,
    _resolve_torchaudio_python_range,
)
from utils.commons.indexed_datasets import (
    _install_numpy_pickle_compat_aliases,
    _load_offsets,
    IndexedDataset,
    IndexedDatasetBuilder,
)
import utils.commons.trainer as trainer_module
from utils.commons.trainer import Trainer


class _DummyUvPredictor(nn.Module):
    def __init__(self, uv_logit: float = 0.0, f0_value: float = 0.0):
        super().__init__()
        self.uv_logit = float(uv_logit)
        self.f0_value = float(f0_value)

    def forward(self, decoder_inp):
        batch, steps, _ = decoder_inp.shape
        uv = torch.full(
            (batch, steps),
            self.uv_logit,
            device=decoder_inp.device,
            dtype=decoder_inp.dtype,
        )
        f0 = torch.full(
            (batch, steps),
            self.f0_value,
            device=decoder_inp.device,
            dtype=decoder_inp.dtype,
        )
        return torch.stack([uv, f0], dim=-1)


class _DummyPitchRuntime(ConanPitchGenerationMixin):
    def __init__(self):
        self.uv_predictor = _DummyUvPredictor()
        self.hparams = {"lambda_f0": 1.0, "silent_token": 0}
        self.content_padding_idx = 999


class ConanMainlineTargetedTests(unittest.TestCase):
    def test_numpy_pickle_compat_aliases_no_longer_trigger_numpy_core_deprecation(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _install_numpy_pickle_compat_aliases()
        self.assertTrue(any(name.startswith("numpy._core") for name in sys.modules))
        self.assertFalse(
            any(
                issubclass(w.category, DeprecationWarning)
                and "numpy.core is deprecated" in str(w.message)
                for w in caught
            )
        )

    def test_indexed_dataset_offsets_loader_accepts_legacy_dict_and_new_array_formats(self):
        with TemporaryDirectory() as tmpdir:
            legacy_path = f"{tmpdir}/legacy"
            with open(f"{legacy_path}.idx", "wb") as f:
                np.save(f, {"offsets": [0, 3, 7]}, allow_pickle=True)
            modern_path = f"{tmpdir}/modern"
            with open(f"{modern_path}.idx", "wb") as f:
                np.save(f, np.asarray([0, 3, 7], dtype=np.int64))
            self.assertTrue(np.array_equal(_load_offsets(legacy_path), np.array([0, 3, 7], dtype=np.int64)))
            self.assertTrue(np.array_equal(_load_offsets(modern_path), np.array([0, 3, 7], dtype=np.int64)))

    def test_indexed_dataset_builder_roundtrip_uses_current_index_format(self):
        with TemporaryDirectory() as tmpdir:
            ds_path = f"{tmpdir}/toy_ds"
            builder = IndexedDatasetBuilder(ds_path)
            builder.add_item({"x": np.asarray([1, 2, 3], dtype=np.int64)})
            builder.add_item({"x": np.asarray([4, 5], dtype=np.int64)})
            builder.finalize()
            ds = IndexedDataset(ds_path, num_cache=0)
            try:
                self.assertEqual(len(ds), 2)
                self.assertTrue(np.array_equal(ds[0]["x"], np.array([1, 2, 3], dtype=np.int64)))
                self.assertTrue(np.array_equal(ds[1]["x"], np.array([4, 5], dtype=np.int64)))
            finally:
                ds.close()

    def test_torchaudio_python_range_tracks_cp312_and_cp313_cutover(self):
        self.assertEqual(_resolve_torchaudio_python_range("2.3.1"), ((3, 8), (3, 12)))
        self.assertEqual(_resolve_torchaudio_python_range("2.4.1"), ((3, 8), (3, 12)))
        self.assertEqual(_resolve_torchaudio_python_range("2.5.1"), ((3, 8), (3, 12)))
        self.assertEqual(_resolve_torchaudio_python_range("2.6.0"), ((3, 9), (3, 13)))

    def test_style_success_proxy_fallback_min_batch_gate(self):
        sample = {
            "mel_lengths": torch.tensor([4, 4, 4], dtype=torch.long),
            "energy": torch.tensor(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.3, 1.3, 1.3, 1.3],
                    [2.0, 2.0, 2.0, 2.0],
                ],
                dtype=torch.float32,
            ),
            "uv": torch.zeros((3, 4), dtype=torch.float32),
            "f0": torch.tensor(
                [
                    [100.0, 100.0, 100.0, 100.0],
                    [140.0, 140.0, 140.0, 140.0],
                    [220.0, 220.0, 220.0, 220.0],
                ],
                dtype=torch.float32,
            ),
        }
        negative_state = resolve_style_success_negative_masks(
            sample,
            batch_size=3,
            device=torch.device("cpu"),
            proxy_threshold=1.25,
            proxy_min_count=2,
            proxy_min_batch=4,
            use_rate_proxy=False,
        )
        self.assertIsNone(negative_state.get("negative_mask"))
        self.assertIsNone(negative_state.get("proxy_negative_mask"))
        self.assertEqual(str(negative_state.get("source", "none")), "none")
        self.assertFalse(bool(negative_state.get("proxy_batch_gate_passed", True)))
        self.assertEqual(
            str(negative_state.get("proxy_disabled_reason", "")),
            "batch_size_below_proxy_min_batch",
        )

    def test_style_success_negative_source_scale_defaults(self):
        self.assertEqual(resolve_style_success_rank_source_scale("label", {}), 1.0)
        self.assertEqual(
            resolve_style_success_rank_source_scale("label_plus_proxy_backfill", {}),
            0.75,
        )
        self.assertEqual(resolve_style_success_rank_source_scale("proxy", {}), 0.5)
        self.assertEqual(
            resolve_style_success_rank_source_scale(
                "proxy",
                {"style_success_proxy_only_rank_row_scale": 0.35},
            ),
            0.35,
        )

    def test_style_owner_residual_mask_backfill_uses_fast_and_slow_masks(self):
        style_decoder_residual = torch.randn(2, 4, 3)
        fast_mask = torch.tensor(
            [
                [False, False, True, True],
                [False, True, True, True],
            ],
            dtype=torch.bool,
        )
        slow_mask = torch.tensor(
            [
                [False, True, False, True],
                [True, False, True, True],
            ],
            dtype=torch.bool,
        )
        payload = {
            "fast_style_decoder_residual": torch.randn(2, 4, 3),
            "slow_style_decoder_residual": torch.randn(2, 4, 3),
            "style_decoder_residual": style_decoder_residual,
            "style_decoder_residual_mask": None,
        }
        ret = {
            "style_trace_mask": fast_mask,
            "slow_style_trace_mask": slow_mask,
        }
        resolved = _resolve_style_owner_residual(payload, ret)
        expected_mask = fast_mask | slow_mask
        self.assertTrue(torch.equal(resolved["style_decoder_residual_mask"], expected_mask))

    def test_style_success_target_summary_rejects_unknown_global_provenance(self):
        summary = torch.randn(2, 3)
        state = resolve_style_success_target_summary(
            {
                "global_style_summary": summary,
                "global_style_summary_source": "none",
                "global_style_summary_runtime_source": "reference_summary",
            }
        )
        self.assertIsNone(state.get("summary"))
        self.assertEqual(state.get("source"), "none")

    def test_temporal_avg_pool_all_masked_returns_finite_zero(self):
        x = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
        mask = torch.ones_like(x, dtype=torch.bool)
        pooled = Conan.temporal_avg_pool(object(), x, mask)
        self.assertTrue(torch.isfinite(pooled).all())
        self.assertTrue(torch.equal(pooled, torch.zeros_like(pooled)))

    def test_prosody_aligner_guided_loss_skips_missing_emotion_mask(self):
        aligner = ProsodyAligner(
            num_layers=1,
            d_model=4,
            nhead=2,
            dropout=0.0,
            dim_feedforward=8,
        )
        src = torch.randn(3, 2, 4)
        local_emotion = torch.randn(5, 2, 4)
        src_mask = torch.tensor(
            [
                [False, False, True],
                [False, True, True],
            ],
            dtype=torch.bool,
        )
        output, guided_loss, attn_emo_list = aligner(
            src,
            local_emotion,
            src_key_padding_mask=src_mask,
            emotion_key_padding_mask=None,
            forcing=False,
        )
        self.assertEqual(tuple(output.shape), tuple(src.shape))
        self.assertEqual(len(attn_emo_list), 1)
        self.assertEqual(guided_loss, 0)

    def test_binary_sidecar_consistency_flags_cross_speaker_refs(self):
        with TemporaryDirectory() as tmpdir:
            np.save(f"{tmpdir}/train_lengths.npy", np.array([10, 12], dtype=np.int32))
            np.save(f"{tmpdir}/train_ref_indices.npy", np.array([1, 0], dtype=np.int32))
            np.save(f"{tmpdir}/train_spk_ids.npy", np.array([0, 1], dtype=np.int32))
            checks = []
            _check_binary_sidecar_consistency(checks, tmpdir, "train")
            self.assertEqual(len(checks), 1)
            self.assertFalse(checks[0]["ok"])
            self.assertFalse(checks[0]["actual"]["same_speaker_refs"])

    def test_add_orig_pitch_all_unvoiced_training_stays_finite(self):
        runtime = _DummyPitchRuntime()
        decoder_inp = torch.zeros((2, 3, 4), dtype=torch.float32)
        f0 = torch.ones((2, 3), dtype=torch.float32)
        uv = torch.ones((2, 3), dtype=torch.float32)
        ret = {}
        runtime.add_orig_pitch(decoder_inp, f0, uv, ret)
        self.assertTrue(torch.isfinite(ret["fdiff"]))
        self.assertEqual(float(ret["fdiff"]), 0.0)

    def test_vq_embedding_all_zero_input_returns_finite_zero_loss(self):
        vq = VQEmbeddingEMA(n_embeddings=4, embedding_dim=3)
        vq.eval()
        vq.data_initialized.fill_(1)
        x = torch.zeros((2, 5, 3), dtype=torch.float32)
        _, loss, indices, perplexity = vq(x)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(tuple(indices.shape), (2, 5))
        self.assertTrue(torch.isfinite(perplexity))

    def test_topk_farthest_negative_mask_backfills_rows_without_loop_regression(self):
        distance = torch.tensor(
            [
                [0.0, 0.9, 0.1],
                [0.9, 0.0, 0.8],
                [0.1, 0.8, 0.0],
            ],
            dtype=torch.float32,
        )
        existing_mask = torch.tensor(
            [
                [False, True, False],
                [True, False, False],
                [False, False, False],
            ],
            dtype=torch.bool,
        )
        backfill = _topk_farthest_negative_mask(
            distance,
            existing_mask=existing_mask,
            min_count=2,
            min_distance=0.0,
        )
        expected = torch.tensor(
            [
                [False, False, True],
                [False, False, True],
                [True, True, False],
            ],
            dtype=torch.bool,
        )
        self.assertTrue(torch.equal(backfill, expected))

    def test_dynamic_timbre_frame_weight_all_masked_rows_remain_finite(self):
        reference = torch.ones((2, 4, 3), dtype=torch.float32)
        sample_energy = torch.ones((2, 4), dtype=torch.float32)
        mask = torch.tensor(
            [
                [True, True, True, True],
                [False, False, False, False],
            ],
            dtype=torch.bool,
        )
        weight = resolve_dynamic_timbre_frame_weight(
            sample_uv=None,
            sample_energy=sample_energy,
            reference=reference,
            mask=mask,
            energy_floor=0.10,
            energy_quantile=0.90,
        )
        self.assertTrue(torch.isfinite(weight).all())
        self.assertTrue(torch.allclose(weight[0], torch.full_like(weight[0], 0.10)))
        self.assertTrue(torch.all(weight[1] >= 0.10))

    def test_streaming_reference_loader_deduplicates_identical_split_reference_paths(self):
        engine = inference_conan.StreamingVoiceConversion.__new__(inference_conan.StreamingVoiceConversion)
        engine.hparams = {
            "allow_split_reference_inputs": True,
            "prompt_ref_fallback_to_style": True,
            "reference_contract_mode": "collapsed_reference",
        }
        engine.device = "cpu"
        calls = []

        def fake_wav_to_mel(path):
            calls.append(path)
            return np.ones((4, 80), dtype=np.float32)

        engine._wav_to_mel = fake_wav_to_mel
        request = {
            "ref_wav": "ref.wav",
            "allow_split_reference_inputs": True,
            "ref_timbre_wav": "ref.wav",
            "ref_style_wav": "style.wav",
            "ref_dynamic_timbre_wav": "style.wav",
            "ref_emotion_wav": "style.wav",
            "ref_accent_wav": "ref.wav",
        }
        with patch.object(
            inference_conan,
            "build_reference_bundle_from_inputs",
            side_effect=lambda **kwargs: kwargs,
        ):
            bundle, meta = engine._load_reference_mels(request)
        self.assertEqual(calls, ["ref.wav", "style.wav"])
        self.assertTrue(meta["split_reference_inputs"])
        self.assertTrue(torch.equal(bundle["ref"], bundle["ref_timbre"]))
        self.assertTrue(torch.equal(bundle["ref_style"], bundle["ref_dynamic_timbre"]))

    def test_trainer_moves_batch_to_cuda_once_per_multi_optimizer_step(self):
        class FakeTask(nn.Module):
            def __init__(self):
                super().__init__()
                self.p0 = nn.Parameter(torch.tensor(1.0))
                self.p1 = nn.Parameter(torch.tensor(1.0))

            def training_step(self, batch, batch_idx, optimizer_idx):
                loss = self.p0 * 0.0 + 1.0 if optimizer_idx == 0 else self.p1 * 0.0 + 1.0
                return {
                    "loss": loss,
                    "progress_bar": {f"opt_{optimizer_idx}": float(batch_idx)},
                    "tb_log": {f"opt_{optimizer_idx}": float(batch_idx)},
                }

            def on_before_optimization(self, opt_idx):
                return None

            def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
                return None

        with TemporaryDirectory() as tmpdir:
            trainer = Trainer(work_dir=tmpdir, val_check_interval=1000, num_sanity_val_steps=0)
            task = FakeTask()
            trainer.task = task
            trainer.on_gpu = True
            trainer.root_gpu = 0
            trainer.optimizers = [
                torch.optim.SGD([task.p0], lr=0.1),
                torch.optim.SGD([task.p1], lr=0.1),
            ]
            move_calls = []

            def fake_move_to_cuda(batch, gpu_id=0):
                move_calls.append((batch, gpu_id))
                return batch

            with patch.object(trainer_module, "move_to_cuda", side_effect=fake_move_to_cuda):
                trainer.run_training_batch(0, {"content": torch.tensor([1])})
            self.assertEqual(len(move_calls), 1)


if __name__ == "__main__":
    unittest.main()
