import unittest
from tempfile import TemporaryDirectory

import numpy as np
import torch

from modules.Conan.Conan import Conan
from modules.Conan.control.style_success import (
    resolve_style_success_target_summary,
    resolve_style_success_negative_masks,
    resolve_style_success_rank_source_scale,
)
from modules.Conan.prosody_util import ProsodyAligner
from modules.Conan.style_timbre_runtime import _resolve_style_owner_residual
from tasks.Conan.mainline_train_prep import (
    _check_binary_sidecar_consistency,
    _resolve_torchaudio_python_range,
)


class ConanMainlineTargetedTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
