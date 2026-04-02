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

    def _identity_encoder(self):
        verifier = getattr(self, "speaker_verifier", None)
        if verifier is not None:
            return verifier
        if getattr(self, "model", None) is None or not hasattr(self.model, "encode_spk_embed"):
            return None
        return self.model.encode_spk_embed

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
            embedding = encoder(mel_input.transpose(1, 2))
        return self._normalize_identity_embedding(embedding, mel_btc=mel_btc)

    def _maybe_attach_output_identity_embeddings(self, output, reference_mels=None):
        if not isinstance(output, dict):
            return
        mel_out = output.get("mel_out")
        global_timbre_anchor = output.get("global_timbre_anchor")
        if (
            not isinstance(mel_out, torch.Tensor)
            or not isinstance(global_timbre_anchor, torch.Tensor)
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
        output["output_identity_embed"] = output_identity_embed
        output["output_identity_anchor_target"] = global_timbre_anchor.detach()
        output["identity_encoder_backend"] = (
            "external_speaker_verifier"
            if getattr(self, "speaker_verifier", None) is not None
            else "model_encode_spk_embed"
        )
        if isinstance(reference_mels, torch.Tensor):
            with torch.no_grad():
                reference_identity_embed = self._encode_identity_embedding(
                    reference_mels,
                    detach_input=True,
                )
                if isinstance(reference_identity_embed, torch.Tensor):
                    output["reference_identity_embed"] = reference_identity_embed

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

        if not minimal_profile:
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
