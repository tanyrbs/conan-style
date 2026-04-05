from modules.Conan.Conan import Conan, ConanPostnet
from tasks.Conan.base_gen_task import AuxDecoderMIDITask, f0_to_figure
from utils.commons.hparams import hparams
import torch
from tasks.Conan.dataset import ConanDataset
import torch.nn.functional as F
from utils.commons.tensor_utils import tensors_to_scalars
from utils.audio.pitch.utils import denorm_f0
from modules.tts.iclspeech.multi_window_disc import Discriminator
import torch.nn as nn
from tasks.Conan.control_diagnostics import collect_control_diagnostics
from tasks.Conan.control_schedule import resolve_control_regularization_config
from tasks.Conan.style_control_mixin import ConanStyleControlMixin
from tasks.Conan.style_batching_mixin import ConanStyleBatchingMixin
from tasks.Conan.reference_curriculum import sample_training_reference_source
from tasks.Conan.forcing_schedule import sample_forcing_flag
from modules.Conan.reference_bundle import build_reference_bundle_from_inputs
from utils.commons.dataset_utils import _resolve_bool_flag


class ConanEmbTask(AuxDecoderMIDITask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = ConanDataset

    def build_tts_model(self):
        # dict_size = len(self.token_encoder)
        self.model = Conan(0, hparams)
        self.gen_params = [p for p in self.model.parameters() if p.requires_grad]

    def run_model(self, sample):
        with torch.no_grad():
            ref = sample['mels']
            output = self.model.encode_spk_embed(ref.transpose(1, 2)).squeeze(2)
        return {}, {
            "global_timbre_anchor": output,
        }


class ConanTask(ConanStyleBatchingMixin, ConanStyleControlMixin, AuxDecoderMIDITask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = ConanDataset
        self.mse_loss_fn = torch.nn.MSELoss()
        self.build_disc_model()

    def build_tts_model(self):
        # dict_size = len(self.token_encoder)
        self.model = Conan(0, hparams)
        self.gen_params = [p for p in self.model.parameters() if p.requires_grad]

    def build_disc_model(self):
        disc_win_num = hparams['disc_win_num']
        h = hparams['mel_disc_hidden_size']
        self.mel_disc = Discriminator(
            time_lengths=[32, 64, 128][:disc_win_num],
            freq_length=80, hidden_size=h, kernel=(3, 3)
        )
        self.disc_params = list(self.mel_disc.parameters())

    def drop_multi(self, tech, drop_p):
        if torch.rand(1) < drop_p:
            tech = torch.ones_like(tech, dtype=tech.dtype) * 2
        elif torch.rand(1) < drop_p:
            random_tech = torch.rand_like(tech, dtype=torch.float32)
            tech[random_tech < drop_p] = 2
        return tech

    @staticmethod
    def _schedule_end_step(primary_key, fallback_key):
        return int(hparams.get(primary_key, hparams.get(fallback_key, 0)))

    @staticmethod
    def _attach_reference_curriculum_diagnostics(output, state):
        state = dict(state or {})
        use_external_ref = bool(state.get("use_external_ref", False))
        use_self_ref = bool(state.get("use_self_ref", not use_external_ref))
        requested_source = state.get(
            "requested_reference_source",
            "external_ref" if use_external_ref else "self_target",
        )
        effective_source = state.get(
            "effective_reference_source",
            state.get("reference_source", requested_source),
        )
        output["reference_curriculum"] = state
        output["reference_curriculum_mode"] = state.get("mode", "unknown")
        output["reference_curriculum_progress"] = float(state.get("progress", 0.0))
        output["reference_curriculum_external_prob"] = float(state.get("external_prob", 0.0))
        output["reference_curriculum_self_prob"] = float(state.get("self_prob", 1.0))
        output["reference_curriculum_self_ref_floor"] = float(state.get("self_ref_floor", 0.0))
        output["reference_curriculum_sample_mode"] = state.get("sample_mode", "batch")
        output["reference_curriculum_requested_source"] = requested_source
        output["reference_curriculum_use_external_ref"] = use_external_ref
        output["reference_curriculum_use_self_ref"] = use_self_ref
        output["reference_curriculum_gloss_scale"] = float(state.get("gloss_scale", 0.0))
        output["reference_curriculum_source"] = state.get("reference_source", effective_source)
        output["reference_curriculum_effective_source"] = effective_source

    @staticmethod
    def _attach_forcing_schedule_diagnostics(output, state):
        state = dict(state or {})
        output["forcing_schedule"] = state
        output["forcing_schedule_mode"] = state.get("mode", "unknown")
        output["forcing_schedule_progress"] = float(state.get("progress", 0.0))
        output["forcing_prob"] = float(state.get("forcing_prob", 0.0))
        output["forcing_enabled"] = bool(state.get("forcing_enabled", False))

    def _build_runtime_reference_bundle(self, sample, ref):
        allow_split_reference_inputs = bool(
            sample.get(
                "allow_split_reference_inputs",
                hparams.get("allow_split_reference_inputs", False),
            )
        )
        bundle_kwargs = {
            "ref": ref,
            "ref_emotion": sample.get("ref_emotion_mels", sample.get("emotion_ref_mels", None)),
            "ref_accent": sample.get("ref_accent_mels", sample.get("accent_ref_mels", None)),
            "prompt_fallback_to_style": bool(
                sample.get(
                    "prompt_ref_fallback_to_style",
                    hparams.get("prompt_ref_fallback_to_style", False),
                )
            ),
            "reference_contract_mode": sample.get(
                "reference_contract_mode",
                hparams.get("reference_contract_mode", "collapsed_reference"),
            ),
        }
        if allow_split_reference_inputs:
            bundle_kwargs.update(
                {
                    "ref_timbre": sample.get("ref_timbre_mels", sample.get("timbre_ref_mels", None)),
                    "ref_style": sample.get("ref_style_mels", None),
                    "ref_dynamic_timbre": sample.get("ref_dynamic_timbre_mels", None),
                }
            )
        return build_reference_bundle_from_inputs(
            allow_split_reference_inputs=allow_split_reference_inputs,
            **bundle_kwargs,
        )

    def _resolve_reference_inputs(self, sample, target, *, global_step=0, infer=False, test=False):
        has_external_ref = sample.get("ref_mels", None) is not None
        external_ref = sample["ref_mels"] if has_external_ref else target
        curriculum_end = self._schedule_end_step("reference_curriculum_end_steps", "random_speaker_steps")
        if infer or test:
            effective_reference_source = "external_ref" if has_external_ref else "target_fallback"
            state = {
                "mode": "inference_external_only" if has_external_ref else "inference_external_missing_fallback",
                "start_steps": int(
                    hparams.get(
                        "reference_curriculum_start_steps",
                        hparams.get("forcing", 0),
                    )
                ),
                "end_steps": int(curriculum_end),
                "progress": 1.0,
                "external_prob": 1.0,
                "self_prob": 0.0,
                "self_ref_floor": 0.0,
                "sample_mode": "inference_fixed",
                "requested_reference_source": "external_ref",
                "use_external_ref": bool(has_external_ref),
                "use_self_ref": not bool(has_external_ref),
                "gloss_scale": 0.0,
                "reference_source": effective_reference_source,
                "effective_reference_source": effective_reference_source,
            }
            ref = external_ref
        else:
            state = sample_training_reference_source(
                global_step,
                config=hparams,
                device=target.device,
            )
            ref = external_ref if state["use_external_ref"] else target
        reference_bundle = self._build_runtime_reference_bundle(sample, ref)
        return ref, reference_bundle, state

    def _resolve_forcing_schedule_state(self, *, global_step=0, infer=False, test=False, device=None):
        if infer or test:
            return {
                "mode": "inference_disabled",
                "progress": 1.0,
                "forcing_prob": 0.0,
                "legacy_cut": int(hparams.get("forcing", 0)),
                "start_steps": int(hparams.get("forcing_decay_start_steps", hparams.get("forcing", 0))),
                "end_steps": int(hparams.get("forcing_decay_end_steps", hparams.get("forcing", 0))),
                "forcing_enabled": False,
            }
        return sample_forcing_flag(
            global_step,
            config=hparams,
            device=device,
        )

    def run_model(self, sample, infer=False, test=False):
        # txt_tokens = sample["txt_tokens"]
        # mel2ph = sample["mel2ph"]
        # spk_id = sample["spk_ids"]
        content = sample["content"]
        if 'spk_embed' in sample:
            spk_embed = sample["spk_embed"]
        else:
            spk_embed=None
        f0, uv = sample.get("f0", None), sample.get("uv", None)
        # notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]

        target = sample["mels"]
        effective_global_step = int(self.global_step)
        if test:
            effective_global_step = max(
                effective_global_step,
                self._schedule_end_step("reference_curriculum_end_steps", "random_speaker_steps"),
                self._schedule_end_step("forcing_decay_end_steps", "forcing"),
                200000,
            )
        ref, runtime_reference_bundle, reference_curriculum = self._resolve_reference_inputs(
            sample,
            target,
            global_step=effective_global_step,
            infer=infer,
            test=test,
        )
        # assert False, f'content: {content.shape}, target: {target.shape},spk_embed: {spk_embed.shape}'
        # if not infer:
        #     tech_drop = {
        #         'mix': 0.1,
        #         'falsetto': 0.1,
        #         'breathy': 0.1,
        #         'bubble': 0.1,
        #         'strong': 0.1,
        #         'weak': 0.1,
        #         'glissando': 0.1,
        #         'pharyngeal': 0.1,
        #         'vibrato': 0.1,
        #     }
        #     for tech, drop_p in tech_drop.items():
        #         sample[tech] = self.drop_multi(sample[tech], drop_p)
        
        # mix, falsetto, breathy=sample['mix'], sample['falsetto'], sample['breathy']
        # bubble,strong,weak=sample['bubble'],sample['strong'],sample['weak']
        # pharyngeal, vibrato, glissando = sample['pharyngeal'], sample['vibrato'], sample['glissando']
        model_kwargs = self.build_style_model_kwargs(sample, ref)
        model_kwargs["reference_bundle"] = runtime_reference_bundle
        if model_kwargs.get("reference_cache") is not None:
            model_kwargs["reference_cache"] = None
        forcing_schedule_state = self._resolve_forcing_schedule_state(
            global_step=effective_global_step,
            infer=infer,
            test=test,
            device=target.device,
        )
        model_kwargs["forcing_schedule_state"] = forcing_schedule_state
        output = self.model(content,spk_embed=spk_embed, target=target,ref=ref,
                            f0=f0, uv=uv,
                            infer=infer, global_steps=effective_global_step, **model_kwargs)
        self._attach_reference_curriculum_diagnostics(output, reference_curriculum)
        self._attach_forcing_schedule_diagnostics(output, forcing_schedule_state)
        
        losses = {}
        
        if not test:
            self._maybe_attach_output_identity_embeddings(output, ref)
            self.add_mel_loss(output['mel_out'], target, losses)
            # self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
            self.add_pitch_loss(output, sample, losses)
            self.add_control_losses(output, sample, losses)
        
        return losses, output

    def _reference_for_inference(self, sample, *, global_step=None):
        _ = global_step
        if sample.get("ref_mels", None) is not None:
            return sample["ref_mels"]
        return sample["mels"]

    def _run_prefix_online_inference(self, sample, *, tokens_per_chunk=4):
        content_full = sample["content"]
        spk_embed = sample.get("spk_embed", None)
        infer_global_steps = max(
            int(self.global_step),
            self._schedule_end_step("reference_curriculum_end_steps", "random_speaker_steps"),
            self._schedule_end_step("forcing_decay_end_steps", "forcing"),
            200000,
        )
        ref, runtime_reference_bundle, reference_curriculum = self._resolve_reference_inputs(
            sample,
            sample["mels"],
            global_step=infer_global_steps,
            infer=True,
            test=False,
        )
        model_kwargs = self.build_style_model_kwargs(sample, ref)
        model_kwargs["reference_bundle"] = runtime_reference_bundle
        model_kwargs["reference_cache"] = None
        forcing_schedule_state = self._resolve_forcing_schedule_state(
            global_step=infer_global_steps,
            infer=True,
            test=False,
            device=content_full.device,
        )
        model_kwargs["forcing_schedule_state"] = forcing_schedule_state
        with torch.no_grad():
            reference_cache = self.model.prepare_reference_cache(
                reference_bundle=model_kwargs["reference_bundle"],
                spk_embed=spk_embed,
                infer=True,
                global_steps=infer_global_steps,
            )
        model_kwargs["reference_cache"] = reference_cache

        total_tokens = int(content_full.size(1))
        step = max(1, int(tokens_per_chunk))
        prev_mel_len = 0
        prev_full_mel = None
        mel_chunks = []
        last_output = None
        chunk_count = 0
        emitted_chunk_mel_lengths = []
        prefix_rewrite_l1_values = []
        prefix_rewrite_l2_values = []
        prefix_tail_rewrite_l1_values = []
        prefix_overlap_frames = max(0, int(hparams.get("streaming_prefix_overlap_frames", 16)))
        with torch.no_grad():
            for end_idx in range(step, total_tokens + step, step):
                chunk_count += 1
                content_chunk = content_full[:, : min(end_idx, total_tokens)]
                last_output = self.model(
                    content_chunk,
                    spk_embed=spk_embed,
                    target=None,
                    ref=ref,
                    f0=None,
                    uv=None,
                    infer=True,
                    global_steps=infer_global_steps,
                    **model_kwargs,
                )
                mel_out = last_output["mel_out"]
                if isinstance(prev_full_mel, torch.Tensor):
                    prefix_len = min(int(prev_full_mel.size(1)), int(mel_out.size(1)))
                    if prefix_len > 0:
                        prev_prefix = prev_full_mel[:, :prefix_len, :]
                        cur_prefix = mel_out[:, :prefix_len, :]
                        prefix_rewrite_l1_values.append(F.l1_loss(cur_prefix, prev_prefix))
                        prefix_rewrite_l2_values.append(F.mse_loss(cur_prefix, prev_prefix))
                        tail_frames = min(prefix_len, prefix_overlap_frames)
                        if tail_frames > 0:
                            prefix_tail_rewrite_l1_values.append(
                                F.l1_loss(
                                    cur_prefix[:, prefix_len - tail_frames:prefix_len, :],
                                    prev_prefix[:, prefix_len - tail_frames:prefix_len, :],
                                )
                            )
                mel_new = mel_out[:, prev_mel_len:, :]
                if mel_new.size(1) > 0:
                    mel_chunks.append(mel_new)
                    emitted_chunk_mel_lengths.append(int(mel_new.size(1)))
                prev_full_mel = mel_out
                prev_mel_len = int(mel_out.size(1))
        if last_output is None:
            raise RuntimeError("Prefix-online inference produced no decoder output.")
        if mel_chunks:
            last_output["mel_out"] = torch.cat(mel_chunks, dim=1)
        last_output["streaming_eval_mode"] = "prefix_online_content_chunked"
        last_output["streaming_prefix_recompute"] = True
        last_output["streaming_stateful_decoder"] = False
        last_output["streaming_stateful_vocoder"] = False
        last_output["streaming_chunk_tokens"] = int(step)
        last_output["streaming_total_chunks"] = int(chunk_count)
        last_output["streaming_emitted_chunk_mel_lengths"] = emitted_chunk_mel_lengths
        last_output["streaming_prefix_overlap_frames"] = int(prefix_overlap_frames)
        self._attach_reference_curriculum_diagnostics(last_output, reference_curriculum)
        self._attach_forcing_schedule_diagnostics(last_output, forcing_schedule_state)
        if prefix_rewrite_l1_values:
            prefix_rewrite_l1 = torch.stack(prefix_rewrite_l1_values)
            last_output["streaming_prefix_rewrite_l1_mean"] = prefix_rewrite_l1.mean()
            last_output["streaming_prefix_rewrite_l1_max"] = prefix_rewrite_l1.max()
        if prefix_rewrite_l2_values:
            prefix_rewrite_l2 = torch.stack(prefix_rewrite_l2_values)
            last_output["streaming_prefix_rewrite_l2_mean"] = prefix_rewrite_l2.mean()
            last_output["streaming_prefix_rewrite_l2_max"] = prefix_rewrite_l2.max()
        if prefix_tail_rewrite_l1_values:
            prefix_tail_rewrite_l1 = torch.stack(prefix_tail_rewrite_l1_values)
            last_output["streaming_prefix_tail_rewrite_l1_mean"] = prefix_tail_rewrite_l1.mean()
            last_output["streaming_prefix_tail_rewrite_l1_max"] = prefix_tail_rewrite_l1.max()
        return last_output

    @staticmethod
    def _masked_sequence_summary(value, mask=None):
        if not isinstance(value, torch.Tensor):
            return None
        if value.dim() == 2:
            return value
        if value.dim() != 3:
            return None
        if isinstance(mask, torch.Tensor):
            if mask.dim() == 3 and mask.size(-1) == 1:
                mask = mask.squeeze(-1)
            if mask.dim() == 2 and tuple(mask.shape) == tuple(value.shape[:2]):
                valid = (~mask.bool()).unsqueeze(-1).to(value.dtype)
                denom = valid.sum(dim=1).clamp_min(1.0)
                return (value * valid).sum(dim=1) / denom
        return value.mean(dim=1)

    @classmethod
    def _cosine_distance_metric(cls, lhs, rhs, *, lhs_mask=None, rhs_mask=None):
        lhs = cls._masked_sequence_summary(lhs, lhs_mask)
        rhs = cls._masked_sequence_summary(rhs, rhs_mask)
        if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
            return None
        if tuple(lhs.shape) != tuple(rhs.shape):
            return None
        return 1.0 - F.cosine_similarity(lhs, rhs, dim=-1, eps=1e-6).mean()

    def _speaker_summary_from_mel(self, mel):
        if not isinstance(mel, torch.Tensor) or mel.dim() != 3:
            return None
        with torch.no_grad():
            embed = self._encode_identity_embedding(mel, detach_input=True)
        return self._masked_sequence_summary(embed)

    @staticmethod
    def _sum_weighted_grad_losses(loss_dict, loss_weights=None):
        total_loss = None
        loss_weights = loss_weights or {}
        for key, value in loss_dict.items():
            if not isinstance(value, torch.Tensor) or not value.requires_grad:
                continue
            weight = float(loss_weights.get(key, 1.0))
            weighted = value * weight
            total_loss = weighted if total_loss is None else total_loss + weighted
        return total_loss

    def _streaming_parity_metrics(self, offline_output, streaming_output, *, reference_mels=None):
        if not isinstance(offline_output, dict) or not isinstance(streaming_output, dict):
            return {}
        offline_mel = offline_output.get("mel_out")
        streaming_mel = streaming_output.get("mel_out")
        if not isinstance(offline_mel, torch.Tensor) or not isinstance(streaming_mel, torch.Tensor):
            return {}
        if offline_mel.dim() != 3 or streaming_mel.dim() != 3:
            return {}
        min_len = min(int(offline_mel.size(1)), int(streaming_mel.size(1)))
        if min_len <= 0:
            return {}
        offline_aligned = offline_mel[:, :min_len, :]
        streaming_aligned = streaming_mel[:, :min_len, :]
        metrics = {
            "streaming_mel_l1": F.l1_loss(streaming_aligned, offline_aligned),
            "streaming_mel_l2": F.mse_loss(streaming_aligned, offline_aligned),
        }
        tail_frames = min(min_len, int(hparams.get("streaming_parity_tail_frames", 32)))
        if tail_frames > 0:
            metrics["streaming_tail_mel_l1"] = F.l1_loss(
                streaming_aligned[:, -tail_frames:, :],
                offline_aligned[:, -tail_frames:, :],
            )
        chunk_mel_lengths = streaming_output.get("streaming_emitted_chunk_mel_lengths")
        if isinstance(chunk_mel_lengths, (list, tuple)):
            boundary_window = max(1, int(hparams.get("streaming_parity_boundary_window", 8)))
            boundary_errors = []
            boundary_cursor = 0
            for chunk_len in chunk_mel_lengths[:-1]:
                boundary_cursor += int(chunk_len)
                if boundary_cursor <= 0 or boundary_cursor >= min_len:
                    continue
                left = max(0, boundary_cursor - boundary_window)
                right = min(min_len, boundary_cursor + boundary_window)
                if right > left:
                    boundary_errors.append(
                        F.l1_loss(
                            streaming_aligned[:, left:right, :],
                            offline_aligned[:, left:right, :],
                        )
                    )
            if boundary_errors:
                boundary_errors = torch.stack(boundary_errors)
                metrics["streaming_boundary_mel_l1"] = boundary_errors.mean()
                metrics["streaming_boundary_mel_l1_max"] = boundary_errors.max()
        for key in (
            "streaming_prefix_rewrite_l1_mean",
            "streaming_prefix_rewrite_l1_max",
            "streaming_prefix_rewrite_l2_mean",
            "streaming_prefix_rewrite_l2_max",
            "streaming_prefix_tail_rewrite_l1_mean",
            "streaming_prefix_tail_rewrite_l1_max",
        ):
            value = streaming_output.get(key)
            if isinstance(value, torch.Tensor):
                metrics[key] = value

        style_owner_distance = self._cosine_distance_metric(
            offline_output.get("style_decoder_residual"),
            streaming_output.get("style_decoder_residual"),
        )
        if isinstance(style_owner_distance, torch.Tensor):
            metrics["streaming_style_owner_cosine_distance"] = style_owner_distance

        material_distance = self._cosine_distance_metric(
            offline_output.get("dynamic_timbre_decoder_residual"),
            streaming_output.get("dynamic_timbre_decoder_residual"),
            lhs_mask=offline_output.get("dynamic_timbre_mask"),
            rhs_mask=streaming_output.get("dynamic_timbre_mask"),
        )
        if isinstance(material_distance, torch.Tensor):
            metrics["streaming_dynamic_timbre_cosine_distance"] = material_distance

        offline_f0 = offline_output.get("f0_denorm_pred")
        streaming_f0 = streaming_output.get("f0_denorm_pred")
        if isinstance(offline_f0, torch.Tensor) and isinstance(streaming_f0, torch.Tensor):
            min_f0_len = min(int(offline_f0.size(1)), int(streaming_f0.size(1)))
            if min_f0_len > 0:
                metrics["streaming_f0_l1"] = F.l1_loss(
                    streaming_f0[:, :min_f0_len],
                    offline_f0[:, :min_f0_len],
                )

        offline_identity = self._speaker_summary_from_mel(offline_aligned)
        streaming_identity = self._speaker_summary_from_mel(streaming_aligned)
        offline_streaming_identity_distance = self._cosine_distance_metric(
            offline_identity,
            streaming_identity,
        )
        if isinstance(offline_streaming_identity_distance, torch.Tensor):
            metrics["streaming_offline_identity_cosine_distance"] = offline_streaming_identity_distance

        reference_identity = self._speaker_summary_from_mel(reference_mels)
        if isinstance(reference_identity, torch.Tensor):
            offline_reference_identity_distance = self._cosine_distance_metric(
                offline_identity,
                reference_identity,
            )
            streaming_reference_identity_distance = self._cosine_distance_metric(
                streaming_identity,
                reference_identity,
            )
            if isinstance(offline_reference_identity_distance, torch.Tensor):
                metrics["offline_reference_identity_cosine_distance"] = offline_reference_identity_distance
            if isinstance(streaming_reference_identity_distance, torch.Tensor):
                metrics["streaming_reference_identity_cosine_distance"] = streaming_reference_identity_distance
        return metrics

    def add_pitch_loss(self, output, sample, losses):
        # mel2ph = sample['mel2ph']  # [B, T_s]
        content = sample['content']
        uv = sample['uv']
        nonpadding = (content != hparams.get('content_padding_idx', 101)).float()
        if hparams["f0_gen"] == "diff":
            losses["fdiff"] = output["fdiff"]
            uv_denom = nonpadding.sum().clamp_min(1.0)
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / uv_denom * hparams['lambda_uv']
        elif hparams["f0_gen"] == "flow":
            losses["pflow"] = output["pflow"]
            uv_denom = nonpadding.sum().clamp_min(1.0)
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / uv_denom * hparams['lambda_uv']
        elif hparams["f0_gen"] == "gmdiff":
            losses["gdiff"] = output["gdiff"]
            losses["mdiff"] = output["mdiff"]
        elif hparams["f0_gen"] == "orig":
            losses["fdiff"] = output["fdiff"]
            uv_denom = nonpadding.sum().clamp_min(1.0)
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / uv_denom * hparams['lambda_uv']

            
    def _training_step(self, sample, batch_idx, optimizer_idx):
        disc_start = self.global_step >= hparams["disc_start_steps"] and hparams['lambda_mel_adv'] > 0
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            train_loss_output, model_out = self.run_model(sample, infer=False)
            loss_output = dict(train_loss_output)
            loss_weights = {}
            self.model_out_gt = self.model_out = \
                {k: v.detach() for k, v in model_out.items() if isinstance(v, torch.Tensor)}
            if disc_start:
                mel_p = model_out['mel_out']
                if hasattr(self.model, 'out2mel'):
                    mel_p = self.model.out2mel(mel_p)
                o_ = self.mel_disc(mel_p)
                p_, pc_ = o_['y'], o_['y_c']
                if p_ is not None:
                    train_loss_output['a'] = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
                    loss_output['a'] = train_loss_output['a']
                    loss_weights['a'] = hparams['lambda_mel_adv']
                if pc_ is not None:
                    train_loss_output['ac'] = self.mse_loss_fn(pc_, pc_.new_ones(pc_.size()))
                    loss_output['ac'] = train_loss_output['ac']
                    loss_weights['ac'] = hparams['lambda_mel_adv']
            loss_output.update(
                collect_control_diagnostics(
                    model_out,
                    sample,
                    resolve_control_regularization_config(hparams, self.global_step),
                )
            )
            total_loss = self._sum_weighted_grad_losses(train_loss_output, loss_weights)
        else:
            #######################
            #    Discriminator    #
            #######################
            loss_output = {}
            if disc_start and self.global_step % hparams['disc_interval'] == 0:
                model_out = self.model_out_gt
                mel_g = sample['mels']
                mel_p = model_out['mel_out']
                o = self.mel_disc(mel_g)
                p, pc = o['y'], o['y_c']
                o_ = self.mel_disc(mel_p)
                p_, pc_ = o_['y'], o_['y_c']
                if p_ is not None:
                    loss_output["r"] = self.mse_loss_fn(p, p.new_ones(p.size()))
                    loss_output["f"] = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))
                if pc_ is not None:
                    loss_output["rc"] = self.mse_loss_fn(pc, pc.new_ones(pc.size()))
                    loss_output["fc"] = self.mse_loss_fn(pc_, pc_.new_zeros(pc_.size()))
            if len(loss_output) == 0:
                return None
            total_loss = self._sum_weighted_grad_losses(loss_output)
        loss_output['batch_size'] = sample['content'].size()[0]
        return total_loss, loss_output


    def validation_step(self, sample, batch_idx):
        outputs = {}
        valid_losses, offline_out = self.run_model(sample, infer=True)
        outputs['losses'] = dict(valid_losses)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        if batch_idx < hparams['num_valid_plots']:
            plot_out = offline_out
            if bool(hparams.get("valid_use_streaming_prefix_path", True)):
                streaming_out = self._run_prefix_online_inference(
                    sample,
                    tokens_per_chunk=int(hparams.get("streaming_eval_tokens_per_chunk", 4)),
                )
                outputs['losses'].update(
                    self._streaming_parity_metrics(
                        offline_out,
                        streaming_out,
                        reference_mels=self._reference_for_inference(sample),
                    )
                )
                plot_out = streaming_out
            sr = hparams["audio_sample_rate"]
            gt_f0 = denorm_f0(sample['f0'], sample["uv"])
            wav_gt = self.vocoder.spec2wav(sample["mels"][0].cpu().numpy(), f0=gt_f0[0].cpu().numpy())
            self.logger.add_audio(f'wav_gt_{batch_idx}', wav_gt, self.global_step, sr)

            offline_f0_pred = offline_out.get("f0_denorm_pred")
            plot_f0_pred = plot_out.get("f0_denorm_pred")
            if not isinstance(plot_f0_pred, torch.Tensor):
                plot_f0_pred = offline_f0_pred
            plot_f0_np = plot_f0_pred[0].cpu().numpy() if isinstance(plot_f0_pred, torch.Tensor) else None
            wav_pred = self.vocoder.spec2wav(plot_out['mel_out'][0].cpu().numpy(), f0=plot_f0_np)
            if plot_out is not offline_out:
                offline_f0_np = offline_f0_pred[0].cpu().numpy() if isinstance(offline_f0_pred, torch.Tensor) else None
                wav_pred_offline = self.vocoder.spec2wav(offline_out['mel_out'][0].cpu().numpy(), f0=offline_f0_np)
                self.logger.add_audio(f'wav_pred_offline_{batch_idx}', wav_pred_offline, self.global_step, sr)
                self.logger.add_audio(f'wav_pred_streaming_{batch_idx}', wav_pred, self.global_step, sr)
                self.plot_mel(batch_idx, sample['mels'], offline_out['mel_out'][0], f'mel_offline_{batch_idx}')
                self.plot_mel(batch_idx, sample['mels'], plot_out['mel_out'][0], f'mel_streaming_{batch_idx}')
            else:
                self.logger.add_audio(f'wav_pred_{batch_idx}', wav_pred, self.global_step, sr)
                self.plot_mel(batch_idx, sample['mels'], plot_out['mel_out'][0], f'mel_{batch_idx}')
            self.logger.add_figure(
                f'f0_{batch_idx}',
                f0_to_figure(gt_f0[0], None, plot_f0_pred[0] if isinstance(plot_f0_pred, torch.Tensor) else None),
                self.global_step)
        return tensors_to_scalars(outputs)
    
    def test_step(self, sample, batch_idx):
        """
        Prefix-online streaming evaluation:
        1. cache the single reference once
        2. incrementally extend content by 80 ms (=4 tokens) granularity
        3. recompute the acoustic prefix, but only keep newly emitted mel frames
        4. optionally compare the online path against the offline/full-prefix mel
        """
        sample['ref_mels'] = sample.get('ref_mels', sample['mels'])
        tokens_per_chunk = int(hparams.get("streaming_eval_tokens_per_chunk", 4))
        offline_outputs = self.run_model(sample, infer=True, test=True)[1]
        outputs = self._run_prefix_online_inference(sample, tokens_per_chunk=tokens_per_chunk)
        mel_pred = outputs['mel_out'][0]
        item_name = sample['item_name'][0]
        base_fn = f'{item_name.replace(" ", "_")}[P]'
        parity_metrics = self._streaming_parity_metrics(
            offline_outputs,
            outputs,
            reference_mels=sample.get("ref_mels", sample.get("mels")),
        )

        # Pass through vocoder at once
        wav_pred = self.vocoder.spec2wav(mel_pred.cpu().numpy())

        # Optional: save gt (keep consistent with original implementation)
        if hparams.get('save_gt', False):
            mel_gt = sample['mels'][0].cpu().numpy()
            wav_gt = self.vocoder.spec2wav(mel_gt)
            self.saving_result_pool.add_job(
                self.save_result,
                args=[wav_gt, mel_gt,
                    base_fn.replace('[P]', '[G]'),
                    self.gen_dir, None, None,
                    denorm_f0(sample['f0'], sample['uv'])[0].cpu().numpy(),
                    None, None]
            )

        # Save prediction
        self.saving_result_pool.add_job(
            self.save_result,
            args=[wav_pred, mel_pred.cpu().numpy(),
                base_fn, self.gen_dir, None, None,
                denorm_f0(sample['f0'], sample['uv'])[0].cpu().numpy(),
                outputs.get('f0_denorm_pred')[0].cpu().numpy()
                if outputs.get('f0_denorm_pred') is not None else None,
                None]
        )

        scalar_metrics = tensors_to_scalars(parity_metrics)
        scalar_metrics.update(
            {
                "streaming_eval_mode": outputs.get("streaming_eval_mode", "prefix_online_content_chunked"),
                "streaming_prefix_recompute": bool(outputs.get("streaming_prefix_recompute", True)),
                "streaming_stateful_decoder": bool(outputs.get("streaming_stateful_decoder", False)),
                "streaming_stateful_vocoder": bool(outputs.get("streaming_stateful_vocoder", False)),
                "streaming_total_chunks": int(outputs.get("streaming_total_chunks", 0)),
                "streaming_chunk_tokens": int(outputs.get("streaming_chunk_tokens", tokens_per_chunk)),
                "streaming_prefix_overlap_frames": int(outputs.get("streaming_prefix_overlap_frames", 0)),
                "query_anchor_split_applied": bool(outputs.get("query_anchor_split_applied", False)),
                "dynamic_timbre_style_context_owner_safe": bool(
                    outputs.get("dynamic_timbre_style_context_owner_safe", False)
                ),
            }
        )
        return scalar_metrics


    def build_optimizer(self, model):
        
        optimizer_gen = torch.optim.AdamW(
            self.gen_params,
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])

        optimizer_disc = torch.optim.AdamW(
            self.disc_params,
            lr=hparams['disc_lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            **hparams["discriminator_optimizer_params"]) if len(self.disc_params) > 0 else None

        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        disc_scheduler = None
        if optimizer[1] is not None:
            disc_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer[1],
                **hparams["discriminator_scheduler_params"],
            )
        return [
            super().build_scheduler(optimizer[0]),
            disc_scheduler,
        ]

    def on_before_optimization(self, opt_idx):
        if opt_idx == 0:
            nn.utils.clip_grad_norm_(self.gen_params, hparams['clip_grad_norm'])
        else:
            nn.utils.clip_grad_norm_(self.disc_params, hparams["clip_grad_norm"])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        num_updates = self.global_step // hparams['accumulate_grad_batches']

        def _step_scheduler(scheduler):
            if scheduler is None:
                return
            if scheduler.__class__.__module__.startswith('torch.optim.lr_scheduler'):
                scheduler.step()
            else:
                scheduler.step(num_updates)

        if self.scheduler is not None and 0 <= int(optimizer_idx) < len(self.scheduler):
            _step_scheduler(self.scheduler[int(optimizer_idx)])

def self_clone(x):
    if x == None:
        return None
    y = x.clone()
    result = torch.cat((x, y), dim=0)
    return result

class VCPostnetTask(ConanTask):
    def __init__(self):
        super(VCPostnetTask, self).__init__()
        self.drop_prob=hparams['drop_tech_prob']

    def build_model(self):
        self.build_pretrain_model()
        self.model = ConanPostnet(hparams_override=hparams)

    def build_pretrain_model(self):
        dict_size = 0
        self.pretrain = Conan(dict_size, hparams)
        from utils.commons.ckpt_utils import load_ckpt
        load_ckpt(
            self.pretrain,
            hparams['fs2_ckpt_dir'],
            'model',
            strict=_resolve_bool_flag(hparams.get('postnet_pretrain_strict_load', False), default=False),
        )
        for k, v in self.pretrain.named_parameters():
            v.requires_grad = False    
    
    def run_model(self, sample, infer=False, noise=None,test=False):
        content = sample["content"]
        # spk_embed = sample["spk_embed"]
        spk_embed=None
        f0, uv = sample["f0"], sample["uv"]
        # notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        
        target = sample["mels"]
        ref=sample['ref_mels']
        cfg = False
        self.pretrain.eval()
        with torch.no_grad():
            output = self.pretrain(content,spk_embed=spk_embed, target=target,ref=ref,
                    f0=f0, uv=uv,
                    infer=infer)

        self.model(target, infer, output, cfg, cfg_scale=hparams['cfg_scale'],  noise=noise)
        losses = {}
        losses["flow"] = output["flow"]
        return losses, output

    def build_optimizer(self, model):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams['lr'],
            betas=(0.9, 0.98),
            eps=1e-9)
        return self.optimizer

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)

    def on_before_optimization(self, opt_idx):
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_norm)
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            if self.scheduler.__class__.__module__.startswith('torch.optim.lr_scheduler'):
                self.scheduler.step()
            else:
                self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

# class TechSingerTask(RFSingerTask):

#     def build_tts_model(self):
#         dict_size = len(self.token_encoder)
#         self.model = Conan(dict_size, hparams)
#         self.gen_params = [p for p in self.model.parameters() if p.requires_grad]
            
#     def run_model(self, sample, infer=False):
#         txt_tokens = sample["txt_tokens"]
#         mel2ph = sample["mel2ph"]
#         spk_id = sample["spk_ids"]
#         f0, uv = sample["f0"], sample["uv"]
#         notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        
#         target = sample["mels"]
#         cfg = False
        
#         if infer:
#             cfg = True
#             mix, falsetto, breathy = sample['mix'],sample['falsetto'],sample['breathy']
#             bubble,strong,weak=sample['bubble'],sample['strong'],sample['weak']
#             pharyngeal,vibrato,glissando = sample['pharyngeal'],sample['vibrato'],sample['glissando']
#             umix, ufalsetto, ubreathy = torch.ones_like(mix, dtype=mix.dtype) * 2, torch.ones_like(falsetto, dtype=falsetto.dtype) * 2, torch.ones_like(breathy, dtype=breathy.dtype) * 2
#             ububble, ustrong, uweak = torch.ones_like(mix, dtype=mix.dtype) * 2, torch.ones_like(strong, dtype=strong.dtype) * 2, torch.ones_like(weak, dtype=weak.dtype) * 2
#             upharyngeal, uvibrato, uglissando = torch.ones_like(bubble, dtype=bubble.dtype) * 2, torch.ones_like(vibrato, dtype=vibrato.dtype) * 2, torch.ones_like(glissando, dtype=glissando.dtype) * 2
#             mix = torch.cat((mix, umix), dim=0)
#             falsetto = torch.cat((falsetto, ufalsetto), dim=0)
#             breathy = torch.cat((breathy, ubreathy), dim=0)
#             bubble = torch.cat((bubble, ububble), dim=0)
#             strong = torch.cat((strong, ustrong), dim=0)
#             weak = torch.cat((weak, uweak), dim=0)
#             pharyngeal = torch.cat((pharyngeal, upharyngeal), dim=0)
#             vibrato = torch.cat((vibrato, uvibrato), dim=0)
#             glissando = torch.cat((glissando, uglissando), dim=0)
            
#             txt_tokens = self_clone(txt_tokens)
#             mel2ph = self_clone(mel2ph)
#             spk_id = self_clone(spk_id)
#             f0 = self_clone(f0)
#             uv = self_clone(uv)
#             notes = self_clone(notes)
#             note_durs = self_clone(note_durs)
#             note_types = self_clone(note_types)
            
#             output = self.model(txt_tokens, mel2ph=mel2ph, spk_id=spk_id, f0=f0, uv=uv, 
#                                 note=notes, note_dur=note_durs, note_type=note_types,
#                                 mix=mix, falsetto=falsetto, breathy=breathy,
#                                 bubble=bubble, strong=strong, weak=weak,
#                                 pharyngeal=pharyngeal, vibrato=vibrato, glissando=glissando, target=target, cfg=cfg, cfg_scale=1.0,
#                                 infer=infer)
#         else:
#             tech_drop = {
#                 'mix': 0.1,
#                 'falsetto': 0.1,
#                 'breathy': 0.1,
#                 'bubble': 0.1,
#                 'strong': 0.1,
#                 'weak': 0.1,
#                 'glissando': 0.1,
#                 'pharyngeal': 0.1,
#                 'vibrato': 0.1,
#             }
#             for tech, drop_p in tech_drop.items():
#                 sample[tech] = self.drop_multi(sample[tech], drop_p)
#             mix, falsetto, breathy = sample['mix'],sample['falsetto'],sample['breathy']
#             bubble,strong,weak=sample['bubble'],sample['strong'],sample['weak']
#             pharyngeal,vibrato,glissando = sample['pharyngeal'],sample['vibrato'],sample['glissando']
#             output = self.model(txt_tokens, mel2ph=mel2ph, spk_id=spk_id, f0=f0, uv=uv, 
#                                 note=notes, note_dur=note_durs, note_type=note_types,
#                                 mix=mix, falsetto=falsetto, breathy=breathy,
#                                 bubble=bubble, strong=strong, weak=weak,
#                                 pharyngeal=pharyngeal, vibrato=vibrato, glissando=glissando, target=target, cfg=cfg,
#                                 infer=infer)
        
#         losses = {}
#         if not infer:
#             self.add_mel_loss(output['coarse_mel_out'], target, losses)
#             self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
#             self.add_pitch_loss(output, sample, losses)
#         losses["flow"] = output["flow"]
#         return losses, output
