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

    def _maybe_attach_output_identity_embeddings(self, output, reference_mels=None):
        if not isinstance(output, dict):
            return
        mel_out = output.get("mel_out")
        global_timbre_anchor = output.get("global_timbre_anchor")
        if (
            not isinstance(mel_out, torch.Tensor)
            or not isinstance(global_timbre_anchor, torch.Tensor)
            or getattr(self, "model", None) is None
            or not hasattr(self.model, "encode_spk_embed")
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
        output["output_identity_embed"] = self.model.encode_spk_embed(
            mel_out.transpose(1, 2)
        ).transpose(1, 2)
        output["output_identity_anchor_target"] = global_timbre_anchor.detach()
        if isinstance(reference_mels, torch.Tensor):
            with torch.no_grad():
                output["reference_identity_embed"] = self.model.encode_spk_embed(
                    reference_mels.transpose(1, 2)
                ).transpose(1, 2)

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
                    ("emotion_cls", "emotion_logits", "emotion_ids", hparams.get("lambda_emotion_cls", 0.0)),
                    ("accent_cls", "accent_logits", "accent_ids", hparams.get("lambda_accent_cls", 0.0)),
                    (
                        "emotion_prompt_cls",
                        "emotion_prompt_logits",
                        "emotion_ids",
                        hparams.get("lambda_emotion_prompt_cls", 0.0),
                    ),
                    (
                        "accent_prompt_cls",
                        "accent_prompt_logits",
                        "accent_ids",
                        hparams.get("lambda_accent_prompt_cls", 0.0),
                    ),
                ),
            )

            add_regression_losses(
                losses,
                output,
                sample,
                specs=(
                    ("arousal_reg", "arousal_pred", "arousal", hparams.get("lambda_arousal_reg", 0.0)),
                    ("valence_reg", "valence_pred", "valence", hparams.get("lambda_valence_reg", 0.0)),
                    (
                        "emotion_prompt_arousal_reg",
                        "emotion_prompt_arousal_pred",
                        "arousal",
                        hparams.get("lambda_emotion_prompt_arousal", 0.0),
                    ),
                    (
                        "emotion_prompt_valence_reg",
                        "emotion_prompt_valence_pred",
                        "valence",
                        hparams.get("lambda_emotion_prompt_valence", 0.0),
                    ),
                ),
            )

        add_energy_loss(
            losses,
            output,
            sample,
            lambda_energy=hparams.get("lambda_energy", 0.0),
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
            if self.global_step > hparams["forcing"] and self.global_step < hparams["random_speaker_steps"]:
                add_optional_passthrough_losses(
                    losses,
                    output,
                    specs=(("gloss", "gloss", True),),
                )
            if self.global_step > hparams["vq_start"]:
                add_optional_passthrough_losses(
                    losses,
                    output,
                    specs=(
                        ("vq_loss", "vq_loss", True),
                        ("ppl", "ppl", True),
                    ),
                )
