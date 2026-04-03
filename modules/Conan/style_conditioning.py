import torch
import torch.nn.functional as F

from modules.Conan.common import first_present
from modules.Conan.control.common import (
    lookup_condition_embedding,
    project_scalar_condition,
    resolve_strength,
    squeeze_prompt_vector,
)
from modules.Conan.reference_bundle import resolve_reference_bundle
from modules.Conan.reference_cache import (
    masked_sequence_mean,
    merge_reference_cache,
    pool_reference_memory,
    select_cached_sequence,
    validate_reference_cache,
)
from modules.Conan.dynamic_timbre_control import (
    apply_boundary_suppression_to_gate,
    apply_runtime_budget_to_dynamic_timbre,
    build_dynamic_timbre_boundary_mask,
    recenter_dynamic_timbre_to_anchor,
    resolve_dynamic_timbre_control,
)
from modules.Conan.style_trace_utils import resolve_combined_style_trace
from utils.commons.hparams import hparams


class ConanStyleConditioningMixin:
    def _get_hparam(self, key, default=None):
        local_hparams = getattr(self, "hparams", None)
        if isinstance(local_hparams, dict):
            return local_hparams.get(key, default)
        return hparams.get(key, default)

    def _resolve_strength(self, value, batch_size, device):
        return resolve_strength(value, batch_size, device)

    def _build_ref_upsample(self, ref_mels):
        batch_size, length, _ = ref_mels.shape
        device = ref_mels.device
        group_size = max(1, int(self._get_hparam("ref_group_size", 4)))
        base_ids = torch.arange(length, device=device) // group_size + 1
        return base_ids.unsqueeze(0).expand(batch_size, -1)

    @staticmethod
    def _masked_mean(sequence, mask=None):
        return masked_sequence_mean(sequence, mask, keepdim=False)

    @staticmethod
    def _sequence_smoothness(sequence, mask=None):
        if not isinstance(sequence, torch.Tensor) or sequence.size(1) <= 1:
            return None
        delta = (sequence[:, 1:] - sequence[:, :-1]).abs().mean(dim=-1)
        if mask is None:
            return delta.mean()
        valid_mask = (~mask[:, 1:].bool()).float()
        denom = valid_mask.sum().clamp_min(1.0)
        return (delta * valid_mask).sum() / denom

    @staticmethod
    def _apply_dynamic_timbre_runtime_budget(
        aligned,
        *,
        style_residual=None,
        slow_style_residual=None,
        padding_mask=None,
        budget_ratio=0.50,
        budget_margin=0.0,
        slow_style_weight=1.0,
    ):
        return apply_runtime_budget_to_dynamic_timbre(
            aligned,
            style_residual=style_residual,
            slow_style_residual=slow_style_residual,
            padding_mask=padding_mask,
            budget_ratio=budget_ratio,
            budget_margin=budget_margin,
            slow_style_weight=slow_style_weight,
        )

    def _prepare_dynamic_timbre_style_context(
        self,
        style_context,
        *,
        padding_mask=None,
        stopgrad=True,
    ):
        if not isinstance(style_context, torch.Tensor) or style_context.dim() != 3:
            return None
        prepared = style_context.detach() if bool(stopgrad) else style_context
        norm = getattr(self, "dynamic_timbre_style_context_norm", None)
        if norm is not None:
            prepared = norm(prepared)
        else:
            prepared = F.layer_norm(prepared, (prepared.size(-1),))
        if isinstance(padding_mask, torch.Tensor):
            if padding_mask.dim() == 3 and padding_mask.size(-1) == 1:
                padding_mask = padding_mask.squeeze(-1)
            if padding_mask.dim() == 2 and tuple(padding_mask.shape) == tuple(prepared.shape[:2]):
                prepared = prepared.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return prepared

    def _resolve_pool_size(self, primary_key, default, fallback_key=None):
        value = self._get_hparam(primary_key, None)
        if value is None and fallback_key is not None:
            value = self._get_hparam(fallback_key, None)
        if value is None:
            value = default
        return max(1, int(value))

    @staticmethod
    def _normalize_memory_mode(mode, default="fast"):
        normalized = str(mode or default).strip().lower() or default
        if normalized in {"auto", "default"}:
            return default
        if normalized not in {"fast", "slow"}:
            return default
        return normalized

    def _build_slow_memory(self, memory, mask, *, primary_key, fallback_key=None, default=2):
        pool_size = self._resolve_pool_size(primary_key, default, fallback_key=fallback_key)
        return pool_reference_memory(memory, mask, pool_size=pool_size)

    @staticmethod
    def _normalize_style_embed(style_embed):
        if not isinstance(style_embed, torch.Tensor):
            return style_embed
        if style_embed.dim() == 2:
            return style_embed.unsqueeze(1)
        if style_embed.dim() == 3 and style_embed.size(-1) == 1:
            return style_embed.transpose(1, 2)
        return style_embed

    @staticmethod
    def _summary_vector(value):
        return squeeze_prompt_vector(value)

    def _encode_global_style_summary(self, ref_mels):
        if ref_mels is None:
            return None
        prompt = self._encode_reference_prompt(ref_mels, prompt_type="prosody", axis=None)
        if prompt is None:
            return None
        projector = getattr(self, "global_style_summary_proj", None)
        if projector is not None:
            prompt = projector(prompt)
        return self._normalize_style_embed(prompt)

    def _resolve_global_style_summary_from_cache(self, cache):
        if not isinstance(cache, dict):
            return None, "none"
        summary = cache.get("global_style_summary")
        if isinstance(summary, torch.Tensor):
            return self._normalize_style_embed(summary), str(cache.get("global_style_summary_source", "cache"))
        prosody_memory, prosody_mask, _ = select_cached_sequence(
            cache,
            "prosody_memory",
            "prosody_memory_slow",
            prefer_slow=True,
        )
        summary = self._masked_mean(prosody_memory, prosody_mask)
        if summary is None:
            return None, "none"
        projector = getattr(self, "global_style_summary_proj", None)
        if projector is not None:
            summary = projector(summary)
        return self._normalize_style_embed(summary), "prosody_memory"

    def _encode_reference_prompt(self, ref_mels, prompt_type="prosody", axis=None):
        if ref_mels is None:
            return None
        ref_upsample = self._build_ref_upsample(ref_mels)
        if prompt_type == "prosody":
            if self.prosody_extractor is None:
                return None
            prompt_seq = self.prosody_extractor(ref_mels, ref_upsample, no_vq=True)
        elif prompt_type == "timbre":
            if self.local_timbre_extractor is None:
                return None
            prompt_seq, _, _ = self.local_timbre_extractor(ref_mels, ref_upsample)
        else:
            return None
        prompt_mask = prompt_seq.abs().sum(dim=-1).eq(0)
        pooled = self._masked_mean(prompt_seq, prompt_mask)
        if pooled is None:
            return None
        if getattr(self, "prompt_control", None) is not None and axis in {"emotion", "accent"}:
            return self.prompt_control.project(axis, pooled)
        if getattr(self, "reference_prompt_proj", None) is not None:
            return self.reference_prompt_proj(pooled)
        return pooled

    def _encode_prosody_memory(self, ref_mels, infer=False, global_steps=0):
        if ref_mels is None or self.prosody_extractor is None:
            return {}
        ref_upsample = self._build_ref_upsample(ref_mels)
        if global_steps > self._get_hparam("vq_start", hparams["vq_start"]) or infer:
            prosody_embedding, loss, ppl = self.prosody_extractor(ref_mels, ref_upsample, no_vq=False)
            payload = {"vq_loss": loss, "ppl": ppl}
        else:
            prosody_embedding = self.prosody_extractor(ref_mels, ref_upsample, no_vq=True)
            payload = {}

        positions = self.embed_positions(prosody_embedding[:, :, 0])
        prosody_embedding = self.l1(torch.cat([prosody_embedding, positions], dim=-1))
        prosody_mask = prosody_embedding.abs().sum(dim=-1).eq(0)
        prosody_memory_slow, prosody_mask_slow = self._build_slow_memory(
            prosody_embedding,
            prosody_mask,
            primary_key="reference_prosody_slow_pool",
            fallback_key="style_trace_slow_pool",
            default=max(2, int(self._get_hparam("ref_group_size", 4))),
        )
        payload.update(
            {
                "ref_upsample": ref_upsample,
                "prosody_ref_upsample": ref_upsample,
                "prosody_memory": prosody_embedding,
                "prosody_memory_mask": prosody_mask,
                "prosody_key_padding_mask": prosody_mask,
                "prosody_memory_slow": prosody_memory_slow,
                "prosody_memory_slow_mask": prosody_mask_slow,
                "prosody_key_padding_mask_slow": prosody_mask_slow,
            }
        )
        return payload

    def _encode_timbre_memory(self, ref_mels, infer=False, global_steps=0):
        _ = global_steps
        if ref_mels is None or self.local_timbre_extractor is None:
            return {}
        ref_upsample = self._build_ref_upsample(ref_mels)
        timbre_embedding, timbre_vq_loss, timbre_ppl = self.local_timbre_extractor(ref_mels, ref_upsample)
        timbre_mask = timbre_embedding.abs().sum(dim=-1).eq(0)
        timbre_memory_slow, timbre_mask_slow = self._build_slow_memory(
            timbre_embedding,
            timbre_mask,
            primary_key="reference_timbre_slow_pool",
            fallback_key="style_dual_path_timbre_pool",
            default=2,
        )
        payload = {
            "timbre_ref_upsample": ref_upsample,
            "timbre_memory": timbre_embedding,
            "timbre_memory_mask": timbre_mask,
            "timbre_key_padding_mask": timbre_mask,
            "timbre_memory_slow": timbre_memory_slow,
            "timbre_memory_slow_mask": timbre_mask_slow,
            "timbre_key_padding_mask_slow": timbre_mask_slow,
        }
        if infer or self._get_hparam("tv_timbre_use_vq", False):
            payload["timbre_vq_loss"] = timbre_vq_loss
            payload["timbre_ppl"] = timbre_ppl
        return payload

    def prepare_reference_cache(
        self,
        reference_bundle=None,
        *,
        spk_embed=None,
        infer=False,
        global_steps=0,
        reference_cache=None,
    ):
        resolved_bundle = resolve_reference_bundle(
            reference_bundle,
            fallback_ref=None,
            prompt_fallback_to_style=bool(
                self._get_hparam("prompt_ref_fallback_to_style", False)
            ),
        )
        cache = dict(reference_cache) if isinstance(reference_cache, dict) else dict(reference_cache or {})
        global_timbre_anchor = cache.get("global_timbre_anchor")
        global_timbre_anchor_source = str(cache.get("global_timbre_anchor_source", "reference_cache"))
        if global_timbre_anchor is None and spk_embed is not None:
            global_timbre_anchor = spk_embed
            global_timbre_anchor_source = "spk_embed"
        if global_timbre_anchor is None:
            ref_timbre = resolved_bundle.get("ref_timbre")
            if ref_timbre is None:
                raise ValueError(
                    "prepare_reference_cache requires `spk_embed` or `reference_bundle['ref_timbre']`."
                )
            global_timbre_anchor = self.encode_spk_embed(ref_timbre.transpose(1, 2)).transpose(1, 2)
            global_timbre_anchor_source = "ref_timbre"
        global_timbre_anchor = self._normalize_style_embed(global_timbre_anchor)
        cache["global_timbre_anchor"] = global_timbre_anchor
        cache["global_timbre_anchor_source"] = global_timbre_anchor_source

        summary_source = resolved_bundle.get("summary_source")
        if summary_source is not None:
            cache["summary_source"] = summary_source

        if cache.get("emotion_prompt") is None:
            emotion_prompt = self._encode_reference_prompt(
                resolved_bundle.get("ref_emotion"),
                prompt_type="prosody",
                axis="emotion",
            )
            if emotion_prompt is not None:
                cache["emotion_prompt"] = emotion_prompt

        if cache.get("accent_prompt") is None:
            accent_prompt = self._encode_reference_prompt(
                resolved_bundle.get("ref_accent"),
                prompt_type="timbre",
                axis="accent",
            )
            if accent_prompt is not None:
                cache["accent_prompt"] = accent_prompt

        if cache.get("prosody_memory") is None and self.prosody_extractor is not None:
            cache = merge_reference_cache(
                cache,
                self._encode_prosody_memory(
                    resolved_bundle.get("ref_style"),
                    infer=infer,
                    global_steps=global_steps,
                ),
            )

        if cache.get("timbre_memory") is None and self.local_timbre_extractor is not None:
            cache = merge_reference_cache(
                cache,
                self._encode_timbre_memory(
                    resolved_bundle.get("ref_dynamic_timbre"),
                    infer=infer,
                    global_steps=global_steps,
                ),
            )

        global_style_summary, global_style_summary_source = self._resolve_global_style_summary_from_cache(cache)
        if global_style_summary is None:
            global_style_summary = self._encode_global_style_summary(resolved_bundle.get("ref_style"))
            if global_style_summary is not None:
                global_style_summary_source = "ref_style_prompt"
        if global_style_summary is None and summary_source is not None:
            global_style_summary = self._encode_global_style_summary(summary_source)
            if global_style_summary is not None:
                global_style_summary_source = "summary_source_prompt"
        if global_style_summary is None:
            allow_timbre_fallback = bool(
                self._get_hparam("global_style_summary_fallback_to_timbre", True)
            )
            if allow_timbre_fallback:
                global_style_summary = global_timbre_anchor
                global_style_summary_source = "fallback_timbre_anchor"
            else:
                raise ValueError(
                    "single-reference mode requires a style-backed `global_style_summary`; "
                    "fallback to global timbre anchor is disabled."
                )
        cache["global_style_summary"] = self._normalize_style_embed(global_style_summary)
        cache["global_style_summary_source"] = global_style_summary_source
        cache["global_style_summary_is_fallback"] = bool(
            global_style_summary_source == "fallback_timbre_anchor"
        )

        validate_reference_cache(cache)
        return cache

    def _lookup_condition_embedding(self, ids, table, strength, reference):
        return lookup_condition_embedding(ids, table, strength, reference)

    def _project_scalar_condition(self, value, projector, batch_size, device):
        return project_scalar_condition(value, projector, batch_size, device)

    def get_condition_embed(self, kwargs, reference, ret, ref_emotion=None, ref_accent=None, reference_cache=None):
        batch_size = reference.size(0)
        device = reference.device
        condition_embed = torch.zeros_like(reference)
        enable_emotion = bool(self._get_hparam("enable_emotion_condition", False)) and self.emotion_embed is not None
        enable_accent = bool(self._get_hparam("enable_accent_condition", False)) and self.accent_embed is not None

        emotion_strength = self._resolve_strength(
            kwargs.get("emotion_strength", kwargs.get("emotion_strengths", 1.0)),
            batch_size=batch_size,
            device=device,
        )
        style_strength = self._resolve_strength(
            kwargs.get(
                "style_condition_strength",
                kwargs.get("style_strength", kwargs.get("style_strengths", 1.0)),
            ),
            batch_size=batch_size,
            device=device,
        )
        accent_strength = self._resolve_strength(
            kwargs.get("accent_strength", kwargs.get("accent_strengths", 1.0)),
            batch_size=batch_size,
            device=device,
        )

        if enable_emotion:
            condition_embed = condition_embed + self._lookup_condition_embedding(
                kwargs.get("emotion_id", kwargs.get("emotion_ids")),
                self.emotion_embed,
                emotion_strength,
                reference,
            )
        if self.style_embed_table is not None:
            condition_embed = condition_embed + self._lookup_condition_embedding(
                kwargs.get("style_id", kwargs.get("style_ids")),
                self.style_embed_table,
                style_strength,
                reference,
            )
        if enable_accent:
            condition_embed = condition_embed + self._lookup_condition_embedding(
                kwargs.get("accent_id", kwargs.get("accent_ids")),
                self.accent_embed,
                accent_strength,
                reference,
            )

        emotion_prompt = None
        if enable_emotion:
            emotion_prompt = first_present(reference_cache, "emotion_prompt") if reference_cache is not None else None
            if emotion_prompt is None:
                emotion_prompt = self._encode_reference_prompt(
                    ref_emotion if ref_emotion is not None else kwargs.get("ref_emotion", None),
                    prompt_type="prosody",
                    axis="emotion",
                )
            if emotion_prompt is not None:
                emotion_prompt = self._summary_vector(emotion_prompt)
                ret["emotion_prompt"] = emotion_prompt
                if getattr(self, "prompt_control", None) is not None:
                    emotion_condition, emotion_gate, emotion_prompt = self.prompt_control.fuse(
                        "emotion",
                        reference,
                        emotion_prompt,
                        emotion_strength,
                        projected=True,
                    )
                    ret["emotion_prompt"] = emotion_prompt
                    ret["emotion_condition"] = emotion_condition
                    ret["emotion_gate"] = emotion_gate
                    condition_embed = condition_embed + emotion_condition
                else:
                    condition_embed = condition_embed + emotion_prompt[:, None, :] * emotion_strength

        accent_prompt = None
        if enable_accent:
            accent_prompt = first_present(reference_cache, "accent_prompt") if reference_cache is not None else None
            if accent_prompt is None:
                accent_prompt = self._encode_reference_prompt(
                    ref_accent if ref_accent is not None else kwargs.get("ref_accent", None),
                    prompt_type="timbre",
                    axis="accent",
                )
            if accent_prompt is not None:
                accent_prompt = self._summary_vector(accent_prompt)
                ret["accent_prompt"] = accent_prompt
                if getattr(self, "prompt_control", None) is not None:
                    accent_condition, accent_gate, accent_prompt = self.prompt_control.fuse(
                        "accent",
                        reference,
                        accent_prompt,
                        accent_strength,
                        projected=True,
                    )
                    ret["accent_prompt"] = accent_prompt
                    ret["accent_condition"] = accent_condition
                    ret["accent_gate"] = accent_gate
                    condition_embed = condition_embed + accent_condition
                else:
                    condition_embed = condition_embed + accent_prompt[:, None, :] * accent_strength

        if getattr(self, "prompt_attribute_heads", None) is not None and (enable_emotion or enable_accent):
            ret.update(
                self.prompt_attribute_heads(
                    emotion_prompt=emotion_prompt,
                    accent_prompt=accent_prompt,
                )
            )

        arousal_embed = self._project_scalar_condition(
            kwargs.get("arousal", None), self.arousal_proj, batch_size, device
        )
        if arousal_embed is not None:
            condition_embed = condition_embed + arousal_embed

        valence_embed = self._project_scalar_condition(
            kwargs.get("valence", None), self.valence_proj, batch_size, device
        )
        if valence_embed is not None:
            condition_embed = condition_embed + valence_embed

        ret["condition_embed"] = condition_embed
        return condition_embed

    def _apply_reference_summary_heads(self, ret, summary):
        if getattr(self, "reference_summary_heads", None) is not None:
            ret.update(self.reference_summary_heads(summary))
            return
        ret["reference_summary"] = summary
        if self.emotion_classifier is not None:
            ret["emotion_logits"] = self.emotion_classifier(summary)
        if self.style_classifier is not None:
            ret["style_logits"] = self.style_classifier(summary)
        if self.accent_classifier is not None:
            ret["accent_logits"] = self.accent_classifier(summary)
        if self.arousal_predictor is not None:
            ret["arousal_pred"] = self.arousal_predictor(summary).squeeze(-1)
        if self.valence_predictor is not None:
            ret["valence_pred"] = self.valence_predictor(summary).squeeze(-1)

    def build_reference_summary(self, ret, summary_source=None, reference_cache=None):
        style_global = self._summary_vector(
            first_present(ret, "global_style_summary")
        )
        combined_style_trace, combined_style_trace_mask = resolve_combined_style_trace(ret)
        style_trace = self._masked_mean(combined_style_trace, combined_style_trace_mask)
        dynamic_timbre = self._masked_mean(ret.get("dynamic_timbre"), ret.get("dynamic_timbre_mask"))

        if reference_cache is not None:
            cached_global = self._summary_vector(
                first_present(
                    reference_cache,
                    "global_style_summary",
                )
            )
            if cached_global is not None:
                style_global = cached_global
            cached_trace_memory, cached_trace_mask, cached_trace_source = select_cached_sequence(
                reference_cache,
                "prosody_memory",
                "prosody_memory_slow",
                prefer_slow=True,
            )
            cached_timbre_memory, cached_timbre_mask, cached_timbre_source = select_cached_sequence(
                reference_cache,
                "timbre_memory",
                "timbre_memory_slow",
                prefer_slow=True,
            )
            cached_trace = self._masked_mean(cached_trace_memory, cached_trace_mask)
            cached_timbre = self._masked_mean(cached_timbre_memory, cached_timbre_mask)
            if cached_trace is not None:
                style_trace = cached_trace
                ret["reference_summary_trace_source"] = cached_trace_source
            if cached_timbre is not None:
                dynamic_timbre = cached_timbre
                ret["reference_summary_timbre_source"] = cached_timbre_source

        if style_global is None and summary_source is None:
            return

        if style_global is None and summary_source is not None:
            style_global = self._summary_vector(self._encode_global_style_summary(summary_source))
        if style_trace is None and summary_source is not None and self.prosody_extractor is not None:
            summary_trace = self.prosody_extractor(summary_source, self._build_ref_upsample(summary_source), no_vq=True)
            style_trace = self._masked_mean(summary_trace)
        if dynamic_timbre is None and summary_source is not None and self.use_dynamic_timbre:
            summary_timbre, _, _ = self.local_timbre_extractor(summary_source, self._build_ref_upsample(summary_source))
            dynamic_timbre = self._masked_mean(summary_timbre)

        if style_global is None:
            return
        if style_trace is None:
            style_trace = torch.zeros_like(style_global)
        if dynamic_timbre is None:
            dynamic_timbre = torch.zeros_like(style_global)

        summary = self.reference_summary_proj(torch.cat([style_global, style_trace, dynamic_timbre], dim=-1))
        self._apply_reference_summary_heads(ret, summary)

    def get_prosody(
        self,
        encoder_out,
        ref_mels,
        ret,
        infer=False,
        global_steps=0,
        reference_cache=None,
        memory_mode="fast",
        style_temperature=1.0,
        forcing_schedule_state=None,
    ):
        memory_mode = self._normalize_memory_mode(
            memory_mode,
            default=self._get_hparam("style_reference_memory_mode", "fast"),
        )
        prosody_embedding, prosody_key_padding_mask, _ = select_cached_sequence(
            reference_cache,
            "prosody_memory",
            "prosody_memory_slow",
            prefer_slow=memory_mode == "slow",
        )
        if prosody_embedding is not None:
            ret["ref_upsample"] = first_present(reference_cache, "prosody_ref_upsample", "ref_upsample")
            if reference_cache is not None:
                if reference_cache.get("vq_loss") is not None:
                    ret["vq_loss"] = reference_cache["vq_loss"]
                if reference_cache.get("ppl") is not None:
                    ret["ppl"] = reference_cache["ppl"]
        else:
            if ref_mels is None:
                raise ValueError("get_prosody requires `ref_mels` or cached `prosody_memory`.")
            ret["ref_upsample"] = self._build_ref_upsample(ref_mels)
            if global_steps > self._get_hparam("vq_start", hparams["vq_start"]) or infer:
                prosody_embedding, loss, ppl = self.prosody_extractor(ref_mels, ret["ref_upsample"], no_vq=False)
                ret["vq_loss"] = loss
                ret["ppl"] = ppl
            else:
                prosody_embedding = self.prosody_extractor(ref_mels, ret["ref_upsample"], no_vq=True)
            positions = self.embed_positions(prosody_embedding[:, :, 0])
            prosody_embedding = self.l1(torch.cat([prosody_embedding, positions], dim=-1))
            prosody_key_padding_mask = prosody_embedding.abs().sum(dim=-1).eq(0)
        style_temperature = float(style_temperature)
        if style_temperature != 1.0:
            prosody_embedding = prosody_embedding * style_temperature

        src_key_padding_mask = ret["content"].eq(self.content_padding_idx)
        if isinstance(forcing_schedule_state, dict):
            forcing = bool(forcing_schedule_state.get("forcing_enabled", False))
            ret["prosody_forcing_schedule_mode"] = forcing_schedule_state.get("mode", "unknown")
            ret["prosody_forcing_prob"] = float(forcing_schedule_state.get("forcing_prob", 0.0))
            ret["prosody_forcing_progress"] = float(forcing_schedule_state.get("progress", 0.0))
            ret["prosody_forcing_source"] = "schedule_state"
        else:
            forcing = bool(global_steps < self._get_hparam("forcing", hparams["forcing"]))
            ret["prosody_forcing_schedule_mode"] = "legacy_hard"
            ret["prosody_forcing_prob"] = 1.0 if forcing else 0.0
            ret["prosody_forcing_progress"] = 0.0 if forcing else 1.0
            ret["prosody_forcing_source"] = "legacy_global_step"
        ret["prosody_forcing_enabled"] = bool(forcing)
        output, guided_loss, attn_emo = self.align(
            encoder_out.transpose(0, 1),
            prosody_embedding.transpose(0, 1),
            src_key_padding_mask,
            prosody_key_padding_mask,
            forcing=forcing,
        )
        ret["gloss"] = guided_loss
        ret["attn"] = attn_emo
        ret["style_trace"] = output.transpose(0, 1)
        ret["style_trace_memory"] = prosody_embedding
        ret["style_trace_mask"] = src_key_padding_mask
        ret["style_trace_memory_mask"] = prosody_key_padding_mask
        ret["style_trace_smooth"] = self._sequence_smoothness(ret["style_trace"], src_key_padding_mask)
        return ret["style_trace"]

    def get_dynamic_timbre(
        self,
        encoder_out,
        ref_mels,
        ret,
        infer=False,
        global_steps=0,
        reference_cache=None,
        memory_mode="fast",
        timbre_temperature=1.0,
        style_context=None,
        style_condition_scale=0.5,
        gate_scale=1.0,
        gate_bias=0.0,
        boundary_suppress_strength=0.0,
        boundary_radius=2,
        anchor_preserve_strength=0.0,
        style_context_prepared=False,
        tvt_prior_scale=1.0,
        use_tvt=None,
        upper_bound_progress=1.0,
    ):
        memory_mode = self._normalize_memory_mode(
            memory_mode,
            default=self._get_hparam("dynamic_timbre_reference_memory_mode", "fast"),
        )
        timbre_embedding, timbre_key_padding_mask, _ = select_cached_sequence(
            reference_cache,
            "timbre_memory",
            "timbre_memory_slow",
            prefer_slow=memory_mode == "slow",
        )
        if timbre_embedding is not None:
            if reference_cache is not None:
                if reference_cache.get("timbre_vq_loss") is not None:
                    ret["timbre_vq_loss"] = reference_cache["timbre_vq_loss"]
                if reference_cache.get("timbre_ppl") is not None:
                    ret["timbre_ppl"] = reference_cache["timbre_ppl"]
        else:
            if ref_mels is None:
                raise ValueError(
                    "get_dynamic_timbre requires `ref_mels` or cached `timbre_memory`."
                )
            ref_upsample = self._build_ref_upsample(ref_mels)
            vq_start = self._get_hparam("tv_timbre_vq_start", self._get_hparam("vq_start", hparams["vq_start"]))
            if global_steps > vq_start or infer:
                timbre_embedding, timbre_vq_loss, timbre_ppl = self.local_timbre_extractor(ref_mels, ref_upsample)
                ret["timbre_vq_loss"] = timbre_vq_loss
                ret["timbre_ppl"] = timbre_ppl
            else:
                timbre_embedding, _, _ = self.local_timbre_extractor(ref_mels, ref_upsample)
            timbre_key_padding_mask = timbre_embedding.abs().sum(dim=-1).eq(0)
        timbre_temperature = float(timbre_temperature)
        if timbre_temperature != 1.0:
            timbre_embedding = timbre_embedding * timbre_temperature

        src_key_padding_mask = ret["content"].eq(self.content_padding_idx)
        output, guided_loss, attn = self.timbre_align(
            encoder_out.transpose(0, 1),
            timbre_embedding.transpose(0, 1),
            src_key_padding_mask,
            timbre_key_padding_mask,
            forcing=False,
        )
        aligned = output.transpose(0, 1)
        global_timbre_anchor = first_present(ret, "global_timbre_anchor")
        if isinstance(global_timbre_anchor, torch.Tensor) and global_timbre_anchor.dim() == 3:
            global_timbre_anchor = global_timbre_anchor.expand(-1, aligned.size(1), -1)
        else:
            raise ValueError("global_timbre_anchor is required for dynamic timbre alignment.")
        control = resolve_dynamic_timbre_control(
            {
                "dynamic_timbre_boundary_suppress_strength": boundary_suppress_strength,
                "dynamic_timbre_boundary_radius": boundary_radius,
                "dynamic_timbre_anchor_preserve_strength": anchor_preserve_strength,
            },
            hparams=getattr(self, "hparams", None),
        )
        aligned, anchor_shift = recenter_dynamic_timbre_to_anchor(
            aligned,
            global_anchor=global_timbre_anchor,
            padding_mask=src_key_padding_mask,
            preserve_strength=control.anchor_preserve_strength,
        )
        local_absolute = aligned
        local_delta = local_absolute - global_timbre_anchor
        local_delta = local_delta.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        upper_bound_progress = max(0.0, min(float(upper_bound_progress), 1.0))
        ret["dynamic_timbre_upper_bound_progress"] = float(upper_bound_progress)
        style_context_available = (
            isinstance(style_context, torch.Tensor)
            and tuple(style_context.shape) == tuple(aligned.shape)
        )
        if style_context_available:
            if not bool(style_context_prepared):
                style_context = self._prepare_dynamic_timbre_style_context(
                    style_context,
                    padding_mask=src_key_padding_mask,
                    stopgrad=True,
                )
            else:
                style_context = style_context.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
            style_context_available = (
                isinstance(style_context, torch.Tensor)
                and tuple(style_context.shape) == tuple(aligned.shape)
            )
        if style_context_available:
            encoder_gate_input = encoder_out + float(style_condition_scale) * style_context
        else:
            style_context = None
            encoder_gate_input = encoder_out
        if use_tvt is None:
            use_tvt = bool(getattr(self, "dynamic_timbre_use_tvt", False))
        else:
            use_tvt = bool(use_tvt)
        requested_use_tvt = bool(use_tvt)
        use_tvt = use_tvt and getattr(self, "global_timbre_memory", None) is not None
        use_tvt = use_tvt and getattr(self, "content_sync_timbre_fuser", None) is not None
        if upper_bound_progress <= 0.0:
            use_tvt = False
        ret["dynamic_timbre_tvt_requested"] = bool(requested_use_tvt)
        ret["dynamic_timbre_tvt_deferred_by_curriculum"] = bool(
            requested_use_tvt and upper_bound_progress <= 0.0
        )
        tvt_payload = None
        if use_tvt:
            global_memory = self.global_timbre_memory(first_present(ret, "global_timbre_anchor"))
            tvt_payload = self.content_sync_timbre_fuser(
                query=encoder_gate_input,
                global_anchor=global_timbre_anchor,
                global_memory=global_memory,
                local_absolute=local_absolute,
                style_context=style_context if style_context_available else None,
                prior_scale=float(tvt_prior_scale),
                gate_scale=float(gate_scale),
                gate_bias=float(gate_bias),
            )
            gate = tvt_payload["variation_gate"]
            ret["dynamic_timbre_gate_raw"] = tvt_payload["variation_gate_raw"].squeeze(-1)
            ret["dynamic_timbre_gate_logit_raw"] = tvt_payload["variation_logit"].squeeze(-1)
            ret["dynamic_timbre_gate_calibration"] = "logit_affine"
            ret["dynamic_timbre_tvt_attn"] = tvt_payload["attn"]
            ret["dynamic_timbre_tvt_mix"] = tvt_payload["mix"].squeeze(-1)
            ret["dynamic_timbre_tvt_prior"] = tvt_payload["prior_delta"]
            ret["dynamic_timbre_tvt_prior_absolute"] = tvt_payload["prior_absolute"]
            ret["dynamic_timbre_tvt_candidate_absolute"] = tvt_payload["candidate_absolute"]
            ret["dynamic_timbre_tvt_memory"] = global_memory
            ret["dynamic_timbre_material_router"] = tvt_payload.get("material_router")
            ret["dynamic_timbre_material_logit"] = tvt_payload.get("material_logit")
            ret["dynamic_timbre_tvt_gate_logit"] = tvt_payload["variation_logit"].squeeze(-1)
            ret["dynamic_timbre_tvt_prior_scale"] = float(tvt_prior_scale)
            ret["dynamic_timbre_tvt_enabled"] = True
        else:
            gate_prob = self.timbre_gate(
                torch.cat([encoder_gate_input, local_absolute, global_timbre_anchor], dim=-1)
            )
            ret["dynamic_timbre_gate_raw"] = gate_prob.squeeze(-1)
            gate_logit = torch.logit(gate_prob.clamp(1.0e-5, 1.0 - 1.0e-5))
            ret["dynamic_timbre_gate_logit_raw"] = gate_logit.squeeze(-1)
            ret["dynamic_timbre_gate_calibration"] = "logit_affine"
            gate = torch.sigmoid(gate_logit * float(gate_scale) + float(gate_bias))
            ret["dynamic_timbre_tvt_enabled"] = False
            ret["dynamic_timbre_tvt_prior_scale"] = float(tvt_prior_scale)
        boundary_mask, boundary_meta = build_dynamic_timbre_boundary_mask(
            ret.get("content"),
            padding_mask=src_key_padding_mask,
            padding_idx=self.content_padding_idx,
            silent_token=self._get_hparam("silent_token", None),
            radius=control.boundary_radius,
            return_metadata=True,
        )
        gate, boundary_scale = apply_boundary_suppression_to_gate(
            gate,
            boundary_mask=boundary_mask,
            suppress_strength=control.boundary_suppress_strength,
        )
        if use_tvt and tvt_payload is not None:
            candidate_delta = tvt_payload["candidate_delta"]
            if upper_bound_progress < 1.0:
                candidate_delta = (
                    (1.0 - upper_bound_progress) * local_delta
                    + upper_bound_progress * candidate_delta
                )
            anchor_target_sequence = global_timbre_anchor + candidate_delta
            aligned = candidate_delta * gate
        else:
            candidate_delta = local_delta
            anchor_target_sequence = global_timbre_anchor + candidate_delta
            aligned = candidate_delta * gate
        ret["tv_gloss"] = guided_loss
        ret["dynamic_timbre_attn"] = attn
        ret["dynamic_timbre"] = aligned
        ret["dynamic_timbre_local_absolute"] = local_absolute
        ret["dynamic_timbre_local_delta"] = local_delta
        ret["dynamic_timbre_gate"] = gate.squeeze(-1)
        ret["dynamic_timbre_mask"] = src_key_padding_mask
        ret["dynamic_timbre_memory"] = timbre_embedding
        ret["dynamic_timbre_memory_mask"] = timbre_key_padding_mask
        ret["dynamic_timbre_boundary_mask"] = boundary_mask.squeeze(-1) if isinstance(boundary_mask, torch.Tensor) else None
        ret["dynamic_timbre_boundary_scale"] = (
            boundary_scale.squeeze(-1) if isinstance(boundary_scale, torch.Tensor) else None
        )
        if isinstance(boundary_meta, dict):
            ret["dynamic_timbre_boundary_transition_rate"] = boundary_meta.get("transition_rate")
            ret["dynamic_timbre_boundary_dense_units_detected"] = boundary_meta.get("dense_units_detected")
        ret["dynamic_timbre_anchor_shift"] = anchor_shift.squeeze(1) if isinstance(anchor_shift, torch.Tensor) else None
        ret["dynamic_timbre_style_context"] = style_context if style_context_available else None
        ret["dynamic_timbre_style_condition_scale"] = float(style_condition_scale)
        ret["dynamic_timbre_style_conditioned"] = bool(style_context_available and float(style_condition_scale) != 0.0)
        ret["dynamic_timbre_style_context_owner_safe"] = bool(style_context_available)
        ret["dynamic_timbre_control"] = control.as_dict()
        ret["tv_timbre_smooth"] = self._sequence_smoothness(aligned, src_key_padding_mask)
        if isinstance(anchor_target_sequence, torch.Tensor):
            ret["tv_timbre_anchor"] = F.l1_loss(
                self._masked_mean(anchor_target_sequence, src_key_padding_mask),
                first_present(ret, "global_timbre_anchor").squeeze(1),
            )
        else:
            ret["tv_timbre_anchor"] = F.l1_loss(
                self._masked_mean(aligned, src_key_padding_mask),
                first_present(ret, "global_timbre_anchor").squeeze(1),
            )
        return aligned

    def forward_energy(self, decoder_inp, energy, ret):
        ret["energy_pred"] = energy_pred = self.energy_predictor(decoder_inp)[:, :, 0]
        if energy is None:
            energy_embed_inp = energy_pred
        else:
            if isinstance(energy, (list, tuple)):
                energy = torch.tensor(energy, device=decoder_inp.device, dtype=torch.float32)
            elif not isinstance(energy, torch.Tensor):
                energy = torch.tensor(energy, device=decoder_inp.device, dtype=torch.float32)
            else:
                energy = energy.to(device=decoder_inp.device, dtype=torch.float32)
            if energy.dim() == 0:
                energy = energy.view(1, 1).expand(decoder_inp.size(0), decoder_inp.size(1))
            elif energy.dim() == 1:
                if energy.size(0) == decoder_inp.size(0):
                    energy = energy[:, None].expand(-1, decoder_inp.size(1))
                else:
                    energy = energy[None, :]
            if energy.size(1) != decoder_inp.size(1):
                energy = F.interpolate(
                    energy.unsqueeze(1),
                    size=decoder_inp.size(1),
                    mode="linear",
                    align_corners=False,
                ).squeeze(1)
            energy_embed_inp = energy
        return self.energy_embed_proj(energy_embed_inp.unsqueeze(-1))
