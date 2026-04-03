from contextlib import contextmanager, nullcontext

import torch

from modules.Conan.reference_bundle import (
    build_control_kwargs,
    build_style_runtime_kwargs,
    reference_bundle_to_model_kwargs,
    resolve_reference_bundle,
)
from tasks.Conan.control_schedule import (
    resolve_control_loss_profile,
    resolve_control_regularization_config,
)
from tasks.Conan.reference_curriculum import resolve_reference_curriculum
from tasks.Conan.control_losses import (
    add_classification_losses,
    add_energy_loss,
    add_optional_passthrough_losses,
    add_prompt_regularization_losses,
    add_regression_losses,
    add_style_timbre_regularization_losses,
    add_weighted_output_losses,
)
from utils.commons.hparams import hparams


class ConanStyleControlMixin:
    @staticmethod
    def _control_loss_profile(config=None):
        return resolve_control_loss_profile(
            config if isinstance(config, dict) else hparams,
            default="mainline_minimal",
        )

    @staticmethod
    def _is_minimal_control_profile(name):
        return str(name or "mainline_minimal").strip().lower() in {
            "mainline_minimal",
            "minimal",
            "core",
            "mainline",
        }

    def build_style_model_kwargs(self, sample, ref):
        reference_bundle = resolve_reference_bundle(sample, fallback_ref=ref)
        reference_cache = sample.get("reference_cache", None)
        control_kwargs = build_control_kwargs(sample)
        runtime_kwargs = build_style_runtime_kwargs(sample)
        return reference_bundle_to_model_kwargs(
            reference_bundle,
            reference_cache=reference_cache,
            **control_kwargs,
            **runtime_kwargs,
        )

    @staticmethod
    def _content_nonpadding(sample):
        content = sample.get("content", None)
        if content is None:
            return None
        return (content != hparams.get("content_padding_idx", 101)).float()

    def _identity_encoder(self):
        verifier = getattr(self, "speaker_verifier", None)
        if verifier is not None:
            return verifier
        if getattr(self, "model", None) is None or not hasattr(self.model, "encode_spk_embed"):
            return None
        return self.model.encode_spk_embed

    def _identity_encoder_loss_mode(self):
        if getattr(self, "speaker_verifier", None) is not None:
            return "external_speaker_verifier"
        if bool(hparams.get("freeze_internal_identity_encoder_for_loss", True)):
            return "model_encode_spk_embed_frozen_for_loss"
        return "model_encode_spk_embed_trainable_for_loss"

    def _internal_identity_encoder_modules(self):
        if getattr(self, "model", None) is None:
            return []
        modules = []
        for name in ("global_conv_in", "global_encoder"):
            module = getattr(self.model, name, None)
            if module is not None and hasattr(module, "parameters"):
                modules.append(module)
        return modules

    @contextmanager
    def _freeze_internal_identity_encoder_params(self):
        params = []
        seen = set()
        for module in self._internal_identity_encoder_modules():
            for param in module.parameters():
                key = id(param)
                if key in seen:
                    continue
                seen.add(key)
                params.append(param)
        if not params:
            yield False
            return
        previous_requires_grad = [param.requires_grad for param in params]
        try:
            for param in params:
                param.requires_grad_(False)
            yield True
        finally:
            for param, requires_grad in zip(params, previous_requires_grad):
                param.requires_grad_(requires_grad)

    @staticmethod
    def _normalize_identity_embedding(embedding, mel_btc=None):
        if not isinstance(embedding, torch.Tensor):
            return None
        if embedding.dim() == 2:
            return embedding.unsqueeze(1)
        if embedding.dim() != 3:
            return None
        if embedding.size(1) == 1:
            return embedding
        if embedding.size(-1) == 1:
            return embedding.transpose(1, 2)
        if isinstance(mel_btc, torch.Tensor):
            mel_len = int(mel_btc.size(1))
            if embedding.size(1) == mel_len:
                return embedding.mean(dim=1, keepdim=True)
            if embedding.size(2) == mel_len:
                return embedding.transpose(1, 2).mean(dim=1, keepdim=True)
        return embedding.mean(dim=1, keepdim=True)

    def _encode_identity_embedding(self, mel_btc, *, detach_input=False):
        if not isinstance(mel_btc, torch.Tensor) or mel_btc.dim() != 3:
            return None
        encoder = self._identity_encoder()
        if encoder is None:
            return None
        mel_input = mel_btc.detach() if detach_input else mel_btc
        verifier = getattr(self, "speaker_verifier", None)
        if verifier is not None:
            embedding = encoder(mel_input)
        else:
            freeze_internal_encoder = bool(
                hparams.get("freeze_internal_identity_encoder_for_loss", True)
            )
            freeze_context = (
                self._freeze_internal_identity_encoder_params()
                if freeze_internal_encoder
                else nullcontext(False)
            )
            with freeze_context:
                embedding = encoder(mel_input.transpose(1, 2))
        return self._normalize_identity_embedding(embedding, mel_btc=mel_btc)

    def _maybe_attach_output_identity_embeddings(self, output, reference_mels=None):
        if not isinstance(output, dict):
            return
        mel_out = output.get("mel_out")
        global_timbre_anchor = output.get("global_timbre_anchor")
        if (
            not isinstance(mel_out, torch.Tensor)
            or (
                not isinstance(global_timbre_anchor, torch.Tensor)
                and not isinstance(reference_mels, torch.Tensor)
            )
        ):
            return
        identity_schedule_cfg = hparams.get("control_regularization_schedule", {})
        schedule_identity_enabled = (
            isinstance(identity_schedule_cfg, dict)
            and identity_schedule_cfg.get("lambda_output_identity_cosine") is not None
        )
        needs_identity_features = (
            float(hparams.get("lambda_output_identity_cosine", 0.0)) > 0.0
            or schedule_identity_enabled
            or bool(hparams.get("log_control_diagnostics", True))
        )
        if not needs_identity_features:
            return
        output_identity_embed = self._encode_identity_embedding(
            mel_out,
            detach_input=bool(hparams.get("speaker_verifier_detach_input", False)),
        )
        if not isinstance(output_identity_embed, torch.Tensor):
            return
        identity_encoder_backend = self._identity_encoder_loss_mode()
        output["output_identity_embed"] = output_identity_embed
        output["identity_encoder_backend"] = identity_encoder_backend
        output["identity_encoder_params_frozen_for_loss"] = bool(
            identity_encoder_backend == "model_encode_spk_embed_frozen_for_loss"
        )
        output["identity_encoder_frozen_for_loss"] = bool(
            identity_encoder_backend == "model_encode_spk_embed_frozen_for_loss"
        )
        identity_target_embed = None
        if isinstance(global_timbre_anchor, torch.Tensor):
            detached_global_timbre_anchor = global_timbre_anchor.detach()
            output["output_identity_anchor_target"] = detached_global_timbre_anchor
            identity_target_embed = detached_global_timbre_anchor
            output["output_identity_target_source"] = "global_timbre_anchor"
        if isinstance(reference_mels, torch.Tensor):
            with torch.no_grad():
                reference_identity_embed = self._encode_identity_embedding(
                    reference_mels,
                    detach_input=True,
                )
                if isinstance(reference_identity_embed, torch.Tensor):
                    detached_reference_identity_embed = reference_identity_embed.detach()
                    output["reference_identity_embed"] = detached_reference_identity_embed
                    output["output_identity_reference_target"] = detached_reference_identity_embed
                    identity_target_embed = detached_reference_identity_embed
                    output["output_identity_target_source"] = "reference_identity_embed"
        if isinstance(identity_target_embed, torch.Tensor):
            output["output_identity_target_embed"] = identity_target_embed

    def add_control_losses(self, output, sample, losses):
        regularization_config = resolve_control_regularization_config(hparams, self.global_step)
        control_loss_profile = self._control_loss_profile(regularization_config)
        minimal_profile = self._is_minimal_control_profile(control_loss_profile)

        if not minimal_profile:
            add_classification_losses(
                losses,
                output,
                sample,
                specs=(
                    ("emotion_cls", "emotion_logits", "emotion_ids", regularization_config.get("lambda_emotion_cls", 0.0)),
                    ("accent_cls", "accent_logits", "accent_ids", regularization_config.get("lambda_accent_cls", 0.0)),
                    (
                        "emotion_prompt_cls",
                        "emotion_prompt_logits",
                        "emotion_ids",
                        regularization_config.get("lambda_emotion_prompt_cls", 0.0),
                    ),
                    (
                        "accent_prompt_cls",
                        "accent_prompt_logits",
                        "accent_ids",
                        regularization_config.get("lambda_accent_prompt_cls", 0.0),
                    ),
                ),
            )

            add_regression_losses(
                losses,
                output,
                sample,
                specs=(
                    ("arousal_reg", "arousal_pred", "arousal", regularization_config.get("lambda_arousal_reg", 0.0)),
                    ("valence_reg", "valence_pred", "valence", regularization_config.get("lambda_valence_reg", 0.0)),
                    (
                        "emotion_prompt_arousal_reg",
                        "emotion_prompt_arousal_pred",
                        "arousal",
                        regularization_config.get("lambda_emotion_prompt_arousal", 0.0),
                    ),
                    (
                        "emotion_prompt_valence_reg",
                        "emotion_prompt_valence_pred",
                        "valence",
                        regularization_config.get("lambda_emotion_prompt_valence", 0.0),
                    ),
                ),
            )

        if not minimal_profile:
            add_energy_loss(
                losses,
                output,
                sample,
                lambda_energy=regularization_config.get("lambda_energy", 0.0),
                nonpadding=self._content_nonpadding(sample),
            )
        if not minimal_profile:
            add_prompt_regularization_losses(losses, output, regularization_config)
        add_style_timbre_regularization_losses(losses, output, sample, regularization_config)

        if not minimal_profile:
            add_weighted_output_losses(
                losses,
                output,
                specs=(
                    ("style_trace_smooth", "style_trace_smooth", regularization_config.get("lambda_style_trace_smooth", 0.0)),
                    ("tv_timbre_smooth", "tv_timbre_smooth", regularization_config.get("lambda_tv_timbre_smooth", 0.0)),
                    ("tv_timbre_anchor", "tv_timbre_anchor", regularization_config.get("lambda_tv_timbre_anchor", 0.0)),
                    ("tv_gloss", "tv_gloss", regularization_config.get("lambda_tv_gloss", 0.0)),
                ),
            )

        if output.get("timbre_vq_loss") is not None and self.global_step > hparams.get("tv_timbre_vq_start", hparams["vq_start"]):
            losses["timbre_vq_loss"] = output["timbre_vq_loss"] * regularization_config.get("lambda_timbre_vq", 1.0)

        if hparams.get("style", False):
            gloss_scale = output.get("reference_curriculum_gloss_scale", None)
            if gloss_scale is None:
                curriculum_state = resolve_reference_curriculum(self.global_step, hparams)
                gloss_scale = curriculum_state.get("self_prob", 0.0)
            try:
                gloss_scale = float(gloss_scale)
            except (TypeError, ValueError):
                gloss_scale = 0.0
            if gloss_scale > 0.0 and output.get("gloss") is not None:
                losses["gloss"] = output["gloss"] * gloss_scale
            if self.global_step > hparams["vq_start"]:
                add_optional_passthrough_losses(
                    losses,
                    output,
                    specs=(
                        ("vq_loss", "vq_loss", True),
                        ("ppl", "ppl", True),
                    ),
                )
