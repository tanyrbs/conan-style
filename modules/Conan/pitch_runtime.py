"""Conan pitch/runtime mixins extracted from modules.Conan.Conan."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from modules.Conan.common_utils import resolve_content_padding_mask
from modules.Conan.pitch_canvas_utils import (
    normalize_style_to_pitch_residual_mode,
    project_source_sequence_to_pitch_canvas,
    smooth_sequence_2d,
)
from modules.Conan.style_mainline import resolve_style_runtime_value
from utils.audio.pitch.utils import denorm_f0, f0_to_coarse

LOG2_F0_MIN = 6.0
LOG2_F0_MAX = 10.0


def minmax_normalize_log2_f0(
    values: torch.Tensor,
    uv: Optional[torch.Tensor] = None,
    *,
    x_min: float = LOG2_F0_MIN,
    x_max: float = LOG2_F0_MAX,
    strict_upper_bound: bool = False,
):
    if strict_upper_bound and torch.any(values > x_max):
        raise ValueError("check minmax_norm!!")
    normed = (values - x_min) / (x_max - x_min) * 2 - 1
    if uv is not None:
        normed = normed.clone()
        normed[uv > 0] = 0
    return normed


def minmax_denormalize_log2_f0(
    values: torch.Tensor,
    uv: Optional[torch.Tensor] = None,
    *,
    x_min: float = LOG2_F0_MIN,
    x_max: float = LOG2_F0_MAX,
):
    denormed = (values + 1) / 2 * (x_max - x_min) + x_min
    if uv is not None:
        denormed = denormed.clone()
        denormed[uv > 0] = 0
    return denormed


def resolve_midi_pitch_clip_bounds(
    midi_notes: Optional[torch.Tensor],
    *,
    semitone_radius: float = 3.0,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if midi_notes is None:
        return None, None, None
    midi_notes = midi_notes.transpose(-1, -2)
    lower_bound = midi_notes - semitone_radius
    upper_bound = midi_notes + semitone_radius
    upper_norm_f0 = minmax_normalize_log2_f0(
        (2 ** ((upper_bound - 69) / 12) * 440).log2()
    ).clamp(-1.0, 1.0)
    lower_norm_f0 = minmax_normalize_log2_f0(
        (2 ** ((lower_bound - 69) / 12) * 440).log2()
    ).clamp(-1.0, 1.0)
    return midi_notes, lower_norm_f0, upper_norm_f0


class ConanStylePitchRuntimeMixin:
    def _apply_runtime_dynamic_timbre_budget(
        self,
        dynamic_timbre_decoder_residual,
        *,
        style_decoder_residual,
        slow_style_decoder_residual=None,
        content=None,
        kwargs=None,
        ret=None,
    ):
        kwargs = kwargs or {}
        enabled = bool(
            resolve_style_runtime_value(
                "runtime_dynamic_timbre_style_budget_enabled",
                overrides=kwargs,
                hparams=self.hparams,
                default=True,
            )
        )
        if not isinstance(dynamic_timbre_decoder_residual, torch.Tensor):
            return dynamic_timbre_decoder_residual

        if ret is not None:
            ret["runtime_dynamic_timbre_style_budget_enabled"] = enabled
        if not enabled:
            return dynamic_timbre_decoder_residual

        ratio = float(
            resolve_style_runtime_value(
                "runtime_dynamic_timbre_style_budget_ratio",
                overrides=kwargs,
                hparams=self.hparams,
                default=0.40,
            )
        )
        margin = float(
            resolve_style_runtime_value(
                "runtime_dynamic_timbre_style_budget_margin",
                overrides=kwargs,
                hparams=self.hparams,
                default=0.0,
            )
        )
        slow_style_weight = float(
            resolve_style_runtime_value(
                "runtime_dynamic_timbre_style_budget_slow_style_weight",
                overrides=kwargs,
                hparams=self.hparams,
                default=1.0,
            )
        )
        budget_epsilon = float(
            resolve_style_runtime_value(
                "runtime_dynamic_timbre_style_budget_epsilon",
                overrides=kwargs,
                hparams=self.hparams,
                default=1e-6,
            )
        )
        content_padding_mask = None
        if isinstance(content, torch.Tensor) and content.dim() == 2:
            content_padding_mask = content.eq(self.content_padding_idx).unsqueeze(-1)

        bounded, budget_meta = self._apply_dynamic_timbre_runtime_budget(
            dynamic_timbre_decoder_residual,
            style_residual=style_decoder_residual,
            slow_style_residual=slow_style_decoder_residual,
            padding_mask=content_padding_mask,
            budget_ratio=ratio,
            budget_margin=margin,
            slow_style_weight=slow_style_weight,
            budget_epsilon=budget_epsilon,
        )
        if ret is not None:
            ret["runtime_dynamic_timbre_style_budget_ratio"] = float(ratio)
            ret["runtime_dynamic_timbre_style_budget_margin"] = float(margin)
            ret["runtime_dynamic_timbre_style_budget_slow_style_weight"] = float(slow_style_weight)
            ret["runtime_dynamic_timbre_style_budget_epsilon"] = float(budget_epsilon)
            if isinstance(budget_meta, dict):
                if isinstance(budget_meta.get("allowed_energy"), torch.Tensor):
                    ret["runtime_dynamic_timbre_style_budget_cap"] = budget_meta["allowed_energy"]
                if isinstance(budget_meta.get("style_energy"), torch.Tensor):
                    ret["runtime_dynamic_timbre_style_energy"] = budget_meta["style_energy"]
                if isinstance(budget_meta.get("timbre_energy"), torch.Tensor):
                    ret["runtime_dynamic_timbre_dynamic_energy"] = budget_meta["timbre_energy"]
                if isinstance(budget_meta.get("overflow"), torch.Tensor):
                    ret["runtime_dynamic_timbre_style_budget_overflow"] = budget_meta["overflow"]
                if isinstance(budget_meta.get("relative_overflow"), torch.Tensor):
                    ret["runtime_dynamic_timbre_style_budget_relative_overflow"] = budget_meta["relative_overflow"]
                ret["runtime_dynamic_timbre_style_budget_skip_reason"] = budget_meta.get("skip_reason")
                ret["runtime_dynamic_timbre_style_budget_applied"] = budget_meta.get("applied", False)
                clip_frac = budget_meta.get("active_fraction")
                if isinstance(clip_frac, torch.Tensor):
                    ret["runtime_dynamic_timbre_style_budget_clip_frac"] = clip_frac
                else:
                    ret["runtime_dynamic_timbre_style_budget_clip_frac"] = torch.tensor(
                        float(bool(budget_meta.get("applied", False))),
                        device=dynamic_timbre_decoder_residual.device,
                        dtype=dynamic_timbre_decoder_residual.dtype,
                    )
        return bounded

    @staticmethod
    def _resolve_pitch_residual_contexts(
        decoder_inp,
        style_residual,
        timbre_residual,
        *,
        include_timbre: bool = False,
    ):
        if not isinstance(style_residual, torch.Tensor) or tuple(style_residual.shape) != tuple(decoder_inp.shape):
            style_residual = torch.zeros_like(decoder_inp)
        timbre_residual_valid = (
            isinstance(timbre_residual, torch.Tensor)
            and tuple(timbre_residual.shape) == tuple(decoder_inp.shape)
        )
        if not timbre_residual_valid:
            timbre_residual = torch.zeros_like(decoder_inp)
        if not include_timbre:
            timbre_residual = torch.zeros_like(decoder_inp)
        return style_residual, timbre_residual, bool(include_timbre and timbre_residual_valid)

    def _apply_style_to_pitch_residual(
        self,
        f0_out,
        uv_out,
        decoder_inp,
        ret,
        *,
        f0_target=None,
        **kwargs,
    ):
        style_mainline_cfg = ret.get("style_mainline")
        if not isinstance(style_mainline_cfg, dict):
            style_mainline_cfg = {}
        enabled = bool(
            kwargs.get(
                "style_to_pitch_residual",
                style_mainline_cfg.get(
                    "style_to_pitch_residual",
                    self.hparams.get("style_to_pitch_residual", False),
                ),
            )
        )
        residual_mode = normalize_style_to_pitch_residual_mode(
            kwargs.get(
                "style_to_pitch_residual_mode",
                style_mainline_cfg.get(
                    "style_to_pitch_residual_mode",
                    self.hparams.get("style_to_pitch_residual_mode", "auto"),
                ),
            )
        )
        include_timbre = bool(
            kwargs.get(
                "style_to_pitch_residual_include_timbre",
                style_mainline_cfg.get(
                    "style_to_pitch_residual_include_timbre",
                    self.hparams.get("style_to_pitch_residual_include_timbre", False),
                ),
            )
        )
        upper_bound_progress = float(ret.get("expressive_upper_bound_progress", 1.0))
        upper_bound_progress = max(0.0, min(upper_bound_progress, 1.0))
        ret["style_to_pitch_residual_enabled"] = enabled
        ret["style_to_pitch_residual_mode"] = residual_mode
        ret["style_to_pitch_residual_include_timbre"] = include_timbre
        ret["style_to_pitch_residual_applied"] = False
        ret["style_to_pitch_residual_upper_bound_progress"] = upper_bound_progress
        if not enabled or self.style_to_pitch_residual_head is None or not isinstance(f0_out, torch.Tensor):
            return f0_out

        style_residual, timbre_residual, uses_timbre_context = self._resolve_pitch_residual_contexts(
            decoder_inp,
            ret.get("style_decoder_residual"),
            ret.get("dynamic_timbre_decoder_residual"),
            include_timbre=include_timbre,
        )
        ret["style_to_pitch_residual_uses_timbre_context"] = uses_timbre_context

        pitch_residual_hidden = torch.cat([decoder_inp, style_residual, timbre_residual], dim=-1)
        pitch_residual_intent = self.style_to_pitch_residual_head(pitch_residual_hidden).squeeze(-1)
        pitch_residual_scale = float(
            kwargs.get(
                "style_to_pitch_residual_scale",
                style_mainline_cfg.get(
                    "style_to_pitch_residual_scale",
                    self.hparams.get("style_to_pitch_residual_scale", 1.0),
                ),
            )
        )
        max_semitones = float(
            kwargs.get(
                "style_to_pitch_residual_max_semitones",
                style_mainline_cfg.get(
                    "style_to_pitch_residual_max_semitones",
                    self.hparams.get("style_to_pitch_residual_max_semitones", 2.5),
                ),
            )
        )
        max_log2 = (
            max(0.0, max_semitones / 12.0)
            * max(0.0, pitch_residual_scale)
            * upper_bound_progress
        )
        pitch_residual_intent = torch.tanh(pitch_residual_intent) * max_log2

        smooth_factor = float(
            kwargs.get(
                "style_to_pitch_residual_smooth_factor",
                style_mainline_cfg.get(
                    "style_to_pitch_residual_smooth_factor",
                    self.hparams.get("style_to_pitch_residual_smooth_factor", 0.35),
                ),
            )
        )

        content_padding_mask = resolve_content_padding_mask(
            ret.get("content"),
            self.content_padding_idx,
            target=pitch_residual_intent,
        )
        if isinstance(content_padding_mask, torch.Tensor):
            pitch_residual_intent = pitch_residual_intent.masked_fill(content_padding_mask, 0.0)

        target_shape = (
            tuple(uv_out.shape)
            if isinstance(uv_out, torch.Tensor) and uv_out.dim() == 2
            else tuple(f0_out.shape)
        )
        pitch_residual, canvas_meta = project_source_sequence_to_pitch_canvas(
            pitch_residual_intent,
            ret,
            content_padding_idx=self.content_padding_idx,
            target_shape=target_shape,
            mode=residual_mode,
        )
        ret["style_to_pitch_residual_intent"] = pitch_residual_intent

        voiced_mask = None
        if isinstance(uv_out, torch.Tensor) and tuple(uv_out.shape) == tuple(pitch_residual.shape):
            voiced_mask = uv_out == 0
            pitch_residual = pitch_residual * voiced_mask.to(pitch_residual.dtype)

        residual_mask = canvas_meta.get("mask")
        smoothing_valid_mask = canvas_meta.get("valid_mask")
        if isinstance(smoothing_valid_mask, torch.Tensor) and tuple(smoothing_valid_mask.shape) == tuple(pitch_residual.shape):
            smoothing_valid_mask = smoothing_valid_mask.to(
                device=pitch_residual.device,
                dtype=pitch_residual.dtype,
            ).clamp(0.0, 1.0)
        else:
            smoothing_valid_mask = None
        if isinstance(residual_mask, torch.Tensor) and tuple(residual_mask.shape) == tuple(pitch_residual.shape):
            residual_valid = (~residual_mask.bool()).to(
                device=pitch_residual.device,
                dtype=pitch_residual.dtype,
            )
            smoothing_valid_mask = (
                residual_valid
                if smoothing_valid_mask is None
                else (smoothing_valid_mask * residual_valid)
            )
        if isinstance(voiced_mask, torch.Tensor) and tuple(voiced_mask.shape) == tuple(pitch_residual.shape):
            voiced_valid = voiced_mask.to(device=pitch_residual.device, dtype=pitch_residual.dtype)
            smoothing_valid_mask = voiced_valid if smoothing_valid_mask is None else (smoothing_valid_mask * voiced_valid)
        pitch_residual = smooth_sequence_2d(
            pitch_residual,
            smooth_factor=smooth_factor,
            valid_mask=smoothing_valid_mask,
        )
        if isinstance(voiced_mask, torch.Tensor) and tuple(voiced_mask.shape) == tuple(pitch_residual.shape):
            pitch_residual = pitch_residual * voiced_mask.to(pitch_residual.dtype)
        if isinstance(residual_mask, torch.Tensor) and tuple(residual_mask.shape) == tuple(pitch_residual.shape):
            pitch_residual = pitch_residual.masked_fill(residual_mask.bool(), 0.0)

        ret["style_to_pitch_residual"] = pitch_residual
        ret["style_to_pitch_residual_scale"] = pitch_residual_scale
        ret["style_to_pitch_residual_max_log2"] = max_log2
        ret["style_to_pitch_residual_max_semitones"] = max_semitones
        ret["style_to_pitch_residual_smooth_factor"] = smooth_factor
        ret["style_to_pitch_residual_canvas"] = canvas_meta.get("canvas", "source_aligned")
        ret["style_to_pitch_residual_mask"] = canvas_meta.get("mask")
        ret["style_to_pitch_residual_blank_mask"] = canvas_meta.get("blank_mask")
        ret["style_to_pitch_residual_voiced_mask"] = voiced_mask
        ret["style_to_pitch_residual_smoothing_valid_mask"] = smoothing_valid_mask

        base_pred = ret.get("f0_base_pred")
        if isinstance(base_pred, torch.Tensor):
            base_pred_canvas, _ = project_source_sequence_to_pitch_canvas(
                base_pred,
                ret,
                content_padding_idx=self.content_padding_idx,
                target_shape=tuple(pitch_residual.shape),
                mode=residual_mode,
            )
        else:
            base_pred_canvas = None
        if isinstance(base_pred_canvas, torch.Tensor) and tuple(base_pred_canvas.shape) == tuple(pitch_residual.shape):
            ret["style_to_pitch_residual_base_pred"] = base_pred_canvas

        target_canvas = None
        if isinstance(f0_target, torch.Tensor):
            target_canvas, _ = project_source_sequence_to_pitch_canvas(
                f0_target,
                ret,
                content_padding_idx=self.content_padding_idx,
                target_shape=tuple(pitch_residual.shape),
                mode=residual_mode,
            )
        if isinstance(target_canvas, torch.Tensor) and isinstance(base_pred_canvas, torch.Tensor):
            if tuple(target_canvas.shape) == tuple(base_pred_canvas.shape):
                target = (target_canvas - base_pred_canvas.detach()).clamp(-max_log2, max_log2)
                mask = canvas_meta.get("mask")
                if isinstance(mask, torch.Tensor) and tuple(mask.shape) == tuple(target.shape):
                    target = target.masked_fill(mask.bool(), 0.0)
                if isinstance(voiced_mask, torch.Tensor) and tuple(voiced_mask.shape) == tuple(target.shape):
                    target = target * voiced_mask.to(target.dtype)
                target_valid_mask = canvas_meta.get("valid_mask")
                if isinstance(target_valid_mask, torch.Tensor) and tuple(target_valid_mask.shape) == tuple(target.shape):
                    target_valid_mask = target_valid_mask.to(
                        device=target.device,
                        dtype=target.dtype,
                    ).clamp(0.0, 1.0)
                else:
                    target_valid_mask = None
                if isinstance(mask, torch.Tensor) and tuple(mask.shape) == tuple(target.shape):
                    mask_valid = (~mask.bool()).to(device=target.device, dtype=target.dtype)
                    target_valid_mask = mask_valid if target_valid_mask is None else (target_valid_mask * mask_valid)
                if isinstance(voiced_mask, torch.Tensor) and tuple(voiced_mask.shape) == tuple(target.shape):
                    voiced_valid = voiced_mask.to(device=target.device, dtype=target.dtype)
                    target_valid_mask = (
                        voiced_valid
                        if target_valid_mask is None
                        else (target_valid_mask * voiced_valid)
                    )
                target = smooth_sequence_2d(
                    target,
                    smooth_factor=smooth_factor,
                    valid_mask=target_valid_mask,
                )
                if isinstance(voiced_mask, torch.Tensor) and tuple(voiced_mask.shape) == tuple(target.shape):
                    target = target * voiced_mask.to(target.dtype)
                if isinstance(mask, torch.Tensor) and tuple(mask.shape) == tuple(target.shape):
                    target = target.masked_fill(mask.bool(), 0.0)
                ret["style_to_pitch_residual_target"] = target

        apply_during_teacher_forcing = bool(
            kwargs.get(
                "style_to_pitch_residual_apply_during_teacher_forcing",
                self.hparams.get("style_to_pitch_residual_apply_during_teacher_forcing", False),
            )
        )
        should_apply = f0_target is None or apply_during_teacher_forcing
        if should_apply and tuple(f0_out.shape) == tuple(pitch_residual.shape):
            f0_out = f0_out + pitch_residual
            ret["style_to_pitch_residual_applied"] = True
        return f0_out


class ConanPitchGenerationMixin:
    def _apply_content_padding_to_uv(self, uv, ret):
        if not isinstance(uv, torch.Tensor):
            return uv
        if "content" in ret:
            content_padding_mask = (
                (ret["content"] == self.hparams["silent_token"])
                | (ret["content"] == self.content_padding_idx)
            )
            if content_padding_mask.shape == uv.shape:
                uv = uv.clone()
                uv[content_padding_mask] = 1
        return uv

    def add_orig_pitch(self, decoder_inp, f0, uv, ret, encoder_out=None, **kwargs):
        infer = f0 is None
        ret["uv_pred"] = uv_pred = self.uv_predictor(decoder_inp)
        ret["uv_base_pred"] = uv_pred[:, :, 0]
        ret["f0_base_pred"] = uv_pred[:, :, 1]

        if infer:
            uv = self._apply_content_padding_to_uv(uv_pred[:, :, 0] > 0, ret)
            f0 = uv_pred[:, :, 1]
            ret["fdiff"] = 0.0
        else:
            nonpadding = (uv == 0).float()
            f0_pred = uv_pred[:, :, 1]
            ret["fdiff"] = (
                (F.mse_loss(f0_pred, f0, reduction="none") * nonpadding).sum()
                / nonpadding.sum()
                * self.hparams["lambda_f0"]
            )
        return f0, uv

    def add_diff_pitch(
        self, decoder_inp, f0, uv, ret, mel2ph=None, encoder_out=None, **kwargs
    ):
        infer = f0 is None
        ret["uv_pred"] = uv_pred = self.uv_predictor(decoder_inp)
        ret["uv_base_pred"] = uv_pred[:, :, 0]
        ret["f0_base_pred"] = uv_pred[:, :, 1]

        if infer:
            uv = uv_pred[:, :, 0] > 0
            midi_notes, lower_norm_f0, upper_norm_f0 = resolve_midi_pitch_clip_bounds(
                kwargs.get("midi_notes", None)
            )
            if midi_notes is not None:
                uv = uv.clone()
                uv[midi_notes[:, 0, :] == 0] = 1
                f0 = self.f0_gen(
                    decoder_inp.transpose(-1, -2),
                    None,
                    None,
                    ret,
                    infer,
                    dyn_clip=[lower_norm_f0, upper_norm_f0],
                )
            else:
                f0 = self.f0_gen(decoder_inp.transpose(-1, -2), None, None, ret, infer)
            f0 = f0[:, :, 0]
            f0 = minmax_denormalize_log2_f0(f0)
            ret["fdiff"] = 0.0
        else:
            if mel2ph is not None:
                nonpadding = (mel2ph > 0).float()
            else:
                nonpadding = (ret["content"] != self.content_padding_idx).float()
            norm_f0 = minmax_normalize_log2_f0(f0, strict_upper_bound=True)
            ret["fdiff"] = self.f0_gen(
                decoder_inp.transpose(-1, -2),
                norm_f0,
                nonpadding.unsqueeze(dim=1),
                ret,
                infer,
            )
        return f0, uv

    def add_flow_pitch(self, decoder_inp, f0, uv, ret, encoder_out=None, **kwargs):
        infer = f0 is None
        ret["uv_pred"] = uv_pred = self.uv_predictor(decoder_inp)
        ret["uv_base_pred"] = uv_pred[:, :, 0]
        ret["f0_base_pred"] = uv_pred[:, :, 1]
        initial_noise = kwargs.get("initial_noise", None)

        if infer:
            if uv is None:
                if not hasattr(self, "uv_predictor"):
                    raise AttributeError("uv_predictor is not defined in the model.")
                uv = uv_pred[:, :, 0] > 0
                if "content" in ret:
                    content_padding_mask = (
                        (ret["content"] == self.hparams["silent_token"])
                        | (ret["content"] == self.content_padding_idx)
                    )
                    if content_padding_mask.shape == uv.shape:
                        uv = uv.clone()
                        uv[content_padding_mask] = 1
                    else:
                        print(
                            f"Warning: content mask shape {content_padding_mask.shape} doesn't match uv shape {uv.shape}, cannot apply."
                        )
                else:
                    print(
                        "Warning: missing 'content' in ret, cannot apply content padding to UV."
                    )

            f0_pred_norm = self.f0_gen(
                decoder_inp.transpose(1, 2),
                None,
                None,
                ret,
                infer=True,
                initial_noise=initial_noise,
            )
            f0_out = minmax_denormalize_log2_f0(f0_pred_norm, uv)
            ret["pflow"] = 0.0
            uv_out = uv
        else:
            nonpadding = (uv == 0).float()
            norm_f0 = minmax_normalize_log2_f0(f0, uv)
            if norm_f0.ndim == 2:
                norm_f0_unsqueezed = norm_f0.unsqueeze(1).unsqueeze(1)
            else:
                raise ValueError(f"Unexpected norm_f0 shape during training: {norm_f0.shape}")
            ret["pflow"] = self.f0_gen(
                decoder_inp.transpose(1, 2),
                norm_f0_unsqueezed,
                nonpadding.unsqueeze(1),
                ret,
                infer=False,
            )
            f0_out = f0
            uv_out = uv

        return f0_out, uv_out

    def add_gmdiff_pitch(
        self, decoder_inp, f0, uv, ret, mel2ph=None, encoder_out=None, **kwargs
    ):
        infer = f0 is None

        if infer:
            midi_notes, lower_norm_f0, upper_norm_f0 = resolve_midi_pitch_clip_bounds(
                kwargs.get("midi_notes", None)
            )
            if midi_notes is not None:
                pitch_pred = self.f0_gen(
                    decoder_inp.transpose(-1, -2),
                    None,
                    None,
                    None,
                    ret,
                    infer,
                    dyn_clip=[lower_norm_f0, upper_norm_f0],
                )
            else:
                pitch_pred = self.f0_gen(
                    decoder_inp.transpose(-1, -2),
                    None,
                    None,
                    None,
                    ret,
                    infer,
                )
            f0 = pitch_pred[:, :, 0]
            uv = pitch_pred[:, :, 1]
            if midi_notes is not None:
                uv = uv.clone()
                uv[midi_notes[:, 0, :] == 0] = 1
            f0 = minmax_denormalize_log2_f0(f0)
            ret["gdiff"] = 0.0
            ret["mdiff"] = 0.0
        else:
            if mel2ph is not None:
                nonpadding = (mel2ph > 0).float()
            else:
                nonpadding = (ret["content"] != self.content_padding_idx).float()
            norm_f0 = minmax_normalize_log2_f0(f0, strict_upper_bound=True)
            ret["mdiff"], ret["gdiff"], ret["nll"] = self.f0_gen(
                decoder_inp.transpose(-1, -2),
                norm_f0.unsqueeze(dim=1),
                uv,
                nonpadding,
                ret,
                infer,
            )
        return f0, uv

    def forward_pitch(self, decoder_inp, f0, uv, ret, **kwargs):
        pitch_pred_inp = decoder_inp
        if self.hparams["predictor_grad"] != 1:
            pitch_pred_inp = pitch_pred_inp.detach() + self.hparams[
                "predictor_grad"
            ] * (pitch_pred_inp - pitch_pred_inp.detach())

        if self.hparams["f0_gen"] == "diff":
            f0_out, uv_out = self.add_diff_pitch(pitch_pred_inp, f0, uv, ret, **kwargs)
        elif self.hparams["f0_gen"] == "gmdiff":
            f0_out, uv_out = self.add_gmdiff_pitch(
                pitch_pred_inp, f0, uv, ret, **kwargs
            )
        elif self.hparams["f0_gen"] == "flow":
            f0_out, uv_out = self.add_flow_pitch(
                pitch_pred_inp, f0, uv, ret, **kwargs
            )
        elif self.hparams["f0_gen"] == "orig":
            f0_out, uv_out = self.add_orig_pitch(pitch_pred_inp, f0, uv, ret, **kwargs)
        else:
            raise ValueError(f"Unknown f0_gen type: {self.hparams['f0_gen']}")

        f0_out = self._apply_style_to_pitch_residual(
            f0_out,
            uv_out,
            pitch_pred_inp,
            ret,
            f0_target=f0,
            **kwargs,
        )

        f0_denorm = denorm_f0(f0_out, uv_out)
        pitch = f0_to_coarse(f0_denorm)
        ret["f0_denorm_pred"] = f0_denorm
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed


__all__ = [
    "ConanPitchGenerationMixin",
    "ConanStylePitchRuntimeMixin",
    "LOG2_F0_MAX",
    "LOG2_F0_MIN",
    "minmax_denormalize_log2_f0",
    "minmax_normalize_log2_f0",
    "resolve_midi_pitch_clip_bounds",
]
