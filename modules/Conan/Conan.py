# midi singer
import torch.nn as nn
from modules.tts.fs import FastSpeech
import math
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, expand_states
from utils.audio.pitch.utils import denorm_f0, f0_to_coarse
import torch
import torch.nn.functional as F
from modules.Conan.diff.gaussian_multinomial_diffusion import (
    GaussianMultinomialDiffusion,
    GaussianMultinomialDiffusionx0,
)
from modules.Conan.diff.net import DiffNet, F0DiffNet, OriDiffNet, CausalConv1d

from utils.commons.hparams import hparams
from modules.Conan.diff.diff_f0 import GaussianDiffusionF0, GaussianDiffusionx0
from modules.Conan.flow.flow_f0 import ReflowF0
from modules.commons.nar_tts_modules import PitchPredictor
from modules.commons.layers import Embedding
from modules.Conan.flow.flow import FlowMel
from modules.commons.conv import ConvBlocks
from modules.commons.conv import TextConvEncoder, ConvBlocks, CausalConvBlocks, CausalFM
from modules.Conan.prosody_util import (
    ProsodyAligner,
    LocalStyleAdaptor,
    LocalTimbreAdaptor,
)
from modules.Conan.control import (
    PromptAttributeHeads,
    PromptControlAdapter,
    ReferenceSummaryHeads,
)
from modules.Conan.reference_bundle import resolve_reference_bundle
from modules.Conan.reference_cache import resolve_reference_cache
from modules.Conan.decoder_style_adapter import ConanDecoderStyleAdapter
from modules.Conan.tvt_memory import ContentSynchronousTimbreFuser, GlobalTimbreMemory
from modules.Conan.decoder_style_bundle import (
    DECODER_STYLE_TIMING_AUTHORITY,
    build_decoder_style_bundle,
    validate_decoder_style_bundle,
)
from modules.Conan.effective_signal import maybe_effective_sequence, maybe_effective_singleton
from modules.Conan.style_mainline import (
    build_style_mainline_memory_payload,
    build_style_mainline_surface_payload,
    resolve_style_mainline_controls,
)
from modules.Conan.style_realization_builder import build_style_realization_payload
from modules.Conan.style_trace_utils import combine_style_traces, resolve_combined_style_trace
from modules.Conan.style_conditioning import ConanStyleConditioningMixin
from modules.commons.transformer import SinusoidalPositionalEmbedding

Flow_DECODERS = {
    "wavenet": lambda hp: DiffNet(hp["audio_num_mel_bins"]),
    "orig": lambda hp: OriDiffNet(hp["audio_num_mel_bins"]),
    "conv": lambda hp: CausalFM(
        hp["hidden_size"],
        hp["hidden_size"],
        hp["dec_dilations"],
        hp["dec_kernel_size"],
        layers_in_block=hp["layers_in_block"],
        norm_type=hp["enc_dec_norm"],
        dropout=hp["dropout"],
        post_net_kernel=hp.get("dec_post_net_kernel", 3),
    ),
}

DEFAULT_MAX_SOURCE_POSITIONS = 2000
DEFAULT_MAX_TARGET_POSITIONS = 2000


class Conan(ConanStyleConditioningMixin, FastSpeech):
    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__(dict_size, hparams, out_dims)

        hidden_size = hparams["hidden_size"]
        kernel_size = hparams["kernel_size"]
        self.content_vocab_size = int(hparams.get("content_vocab_size", hparams.get("content_embedding_dim", 102)))
        self.content_padding_idx = int(hparams.get("content_padding_idx", self.content_vocab_size - 1))
        self.content_embedding = nn.Embedding(
            self.content_vocab_size, hidden_size, padding_idx=self.content_padding_idx
        )
        self.content_proj = nn.Sequential(
            CausalConv1d(hidden_size, hidden_size, kernel_size=kernel_size, dilation=1),
            nn.LeakyReLU(),
        )
        self.global_conv_in = nn.Conv1d(80, hidden_size, 1)
        self.global_encoder = ConvBlocks(
            hidden_size,
            hidden_size,
            None,
            kernel_size=31,
            layers_in_block=2,
            is_BTC=False,
            num_layers=5,
        )

        self.padding_idx = 0
        self.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if hparams["style"]:
            self.padding_idx = 0
            self.prosody_extractor = LocalStyleAdaptor(
                self.hidden_size, hparams["nVQ"], self.padding_idx
            )
            self.local_timbre_extractor = LocalTimbreAdaptor(
                self.hidden_size,
                hparams.get("tv_timbre_nVQ", hparams.get("nVQ", 64)),
                self.padding_idx,
                use_vq=hparams.get("tv_timbre_use_vq", False),
            )
            self.l1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.align = ProsodyAligner(num_layers=2)
            self.timbre_align = ProsodyAligner(num_layers=hparams.get("tv_timbre_layers", 1))
            gate_hidden = int(hparams.get("tv_timbre_gate_hidden", self.hidden_size))
            self.timbre_gate = nn.Sequential(
                nn.Linear(self.hidden_size * 3, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, 1),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.timbre_gate[2].weight)
            nn.init.constant_(
                self.timbre_gate[2].bias,
                float(hparams.get("tv_timbre_gate_bias_init", -1.0)),
            )
            self.dynamic_timbre_use_tvt = bool(hparams.get("dynamic_timbre_use_tvt", True))
            if self.dynamic_timbre_use_tvt:
                self.global_timbre_memory = GlobalTimbreMemory(
                    self.hidden_size,
                    num_facets=hparams.get("dynamic_timbre_num_facets", 8),
                    facet_hidden=hparams.get("dynamic_timbre_facet_hidden", self.hidden_size),
                )
                self.content_sync_timbre_fuser = ContentSynchronousTimbreFuser(
                    self.hidden_size,
                    gate_hidden=gate_hidden,
                    mix_bias=hparams.get("dynamic_timbre_tvt_mix_bias", -0.25),
                    variation_bias=hparams.get("dynamic_timbre_tvt_variation_bias", -1.0),
                    material_bias=hparams.get("dynamic_timbre_material_router_bias", 1.0),
                )
            else:
                self.global_timbre_memory = None
                self.content_sync_timbre_fuser = None
            self.style_query_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )
            self.timbre_query_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )
            self.style_query_norm = nn.LayerNorm(self.hidden_size)
            self.timbre_query_norm = nn.LayerNorm(self.hidden_size)
            self.dynamic_timbre_style_context_norm = nn.LayerNorm(self.hidden_size)
            style_router_hidden = int(hparams.get("style_router_hidden", self.hidden_size))
            self.style_router = nn.Sequential(
                nn.Linear(self.hidden_size * 3, style_router_hidden),
                nn.ReLU(),
                nn.Linear(style_router_hidden, 1),
            )
            nn.init.zeros_(self.style_router[2].weight)
            nn.init.constant_(
                self.style_router[2].bias,
                float(hparams.get("style_router_bias_init", -0.15)),
            )
            pitch_residual_hidden = int(
                hparams.get("style_to_pitch_residual_hidden", self.hidden_size)
            )
            self.style_to_pitch_residual_head = nn.Sequential(
                nn.Linear(self.hidden_size * 3, pitch_residual_hidden),
                nn.ReLU(),
                nn.Linear(pitch_residual_hidden, 1),
            )
            nn.init.zeros_(self.style_to_pitch_residual_head[2].weight)
            nn.init.zeros_(self.style_to_pitch_residual_head[2].bias)

            # build attention layer
            self.embed_positions = SinusoidalPositionalEmbedding(
                self.hidden_size,
                self.padding_idx,
                init_size=self.max_source_positions + self.padding_idx + 1,
            )
        else:
            self.prosody_extractor = None
            self.local_timbre_extractor = None
            self.align = None
            self.timbre_align = None
            self.timbre_gate = None
            self.style_query_proj = None
            self.timbre_query_proj = None
            self.style_query_norm = None
            self.timbre_query_norm = None
            self.dynamic_timbre_style_context_norm = None
            self.style_router = None
            self.style_to_pitch_residual_head = None
            self.dynamic_timbre_use_tvt = False
            self.global_timbre_memory = None
            self.content_sync_timbre_fuser = None
            self.l1 = None
            self.embed_positions = None

        self.num_emotions = int(hparams.get("num_emotions", 0))
        self.num_styles = 0
        self.num_accents = int(hparams.get("num_accents", 0))
        self.emotion_embed = Embedding(max(self.num_emotions, 1), hidden_size, 0) if self.num_emotions > 0 else None
        self.style_embed_table = None
        self.accent_embed = Embedding(max(self.num_accents, 1), hidden_size, 0) if self.num_accents > 0 else None
        self.arousal_proj = nn.Linear(1, hidden_size) if hparams.get("use_arousal", True) else None
        self.valence_proj = nn.Linear(1, hidden_size) if hparams.get("use_valence", True) else None
        self.prompt_control = PromptControlAdapter(
            hidden_size,
            emotion_gate_hidden=hparams.get("emotion_condition_gate_hidden", hidden_size),
            accent_gate_hidden=hparams.get("accent_condition_gate_hidden", hidden_size),
            use_emotion_gate=hparams.get("emotion_condition_use_gate", True),
            use_accent_gate=hparams.get("accent_condition_use_gate", True),
        )
        self.reference_prompt_proj = self.prompt_control.emotion.prompt_proj
        self.global_style_summary_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.reference_summary_proj = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.prompt_attribute_heads = PromptAttributeHeads(
            hidden_size,
            num_emotions=self.num_emotions,
            num_accents=self.num_accents,
            predict_prompt_emotion=hparams.get("predict_prompt_emotion", True),
            predict_prompt_accent=hparams.get("predict_prompt_accent", True),
            predict_prompt_arousal=hparams.get("predict_prompt_arousal", True),
            predict_prompt_valence=hparams.get("predict_prompt_valence", True),
        )
        self.use_dynamic_timbre = bool(hparams.get("use_dynamic_timbre", True)) and self.local_timbre_extractor is not None
        self.reference_summary_heads = ReferenceSummaryHeads(
            hidden_size,
            num_emotions=self.num_emotions,
            num_styles=self.num_styles,
            num_accents=self.num_accents,
            predict_arousal=hparams.get("predict_arousal", True),
            predict_valence=hparams.get("predict_valence", True),
        )
        self.emotion_classifier = self.reference_summary_heads.emotion_classifier
        self.style_classifier = self.reference_summary_heads.style_classifier
        self.accent_classifier = self.reference_summary_heads.accent_classifier
        self.arousal_predictor = self.reference_summary_heads.arousal_predictor
        self.valence_predictor = self.reference_summary_heads.valence_predictor
        self.use_energy_embed = bool(hparams.get("use_energy_embed", False))
        if self.use_energy_embed:
            self.energy_predictor = PitchPredictor(
                self.hidden_size,
                n_chans=int(hparams.get("energy_hidden_size", 128)),
                n_layers=int(hparams.get("energy_predictor_layers", 3)),
                dropout_rate=float(hparams.get("energy_predictor_dropout", 0.1)),
                odim=1,
                kernel_size=int(hparams.get("energy_predictor_kernel", hparams["predictor_kernel"])),
            )
            self.energy_embed_proj = nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.Tanh(),
            )
        else:
            self.energy_predictor = None
            self.energy_embed_proj = None
        self.use_decoder_style_adapter = bool(hparams.get("use_decoder_style_adapter", True))
        if self.use_decoder_style_adapter and isinstance(self.decoder, CausalConvBlocks):
            self.decoder_style_adapter = ConanDecoderStyleAdapter(
                hidden_size,
                gate_hidden=hparams.get("decoder_style_adapter_gate_hidden", hidden_size),
                stage_splits=hparams.get("decoder_style_adapter_stage_splits", None),
                global_timbre_scale_early=hparams.get(
                    "decoder_global_timbre_scale_early",
                    hparams.get("decoder_global_timbre_scale", 0.18),
                ),
                global_timbre_scale_mid=hparams.get("decoder_global_timbre_scale_mid", 0.12),
                global_timbre_scale_late=hparams.get("decoder_global_timbre_scale_late", 0.08),
                global_style_scale=hparams.get("decoder_style_global_scale", 0.2),
                slow_style_scale=hparams.get("decoder_slow_style_trace_scale", 0.70),
                style_trace_scale=hparams.get("decoder_style_trace_scale", 1.00),
                dynamic_timbre_scale=hparams.get("decoder_dynamic_timbre_scale", 0.20),
                dynamic_timbre_scale_mid=hparams.get("decoder_dynamic_timbre_scale_mid", None),
                dynamic_timbre_scale_late=hparams.get("decoder_dynamic_timbre_scale_late", None),
                dynamic_timbre_late_no_style_scale=hparams.get(
                    "decoder_dynamic_timbre_late_no_style_scale",
                    0.0,
                ),
                skip_global_style_when_local_style_present=hparams.get(
                    "decoder_skip_global_style_when_local_style_present",
                    True,
                ),
                effective_signal_epsilon=hparams.get(
                    "decoder_style_adapter_effective_signal_epsilon",
                    1e-8,
                ),
                gate_bias=hparams.get("decoder_style_adapter_gate_bias", -1.0),
            )
        else:
            self.decoder_style_adapter = None
        self.decoder_style_adapter_gate_bias_start = float(
            hparams.get("decoder_style_adapter_gate_bias_start", hparams.get("decoder_style_adapter_gate_bias", -1.0))
        )
        self.decoder_style_adapter_gate_bias_end = float(
            hparams.get("decoder_style_adapter_gate_bias_end", 0.0)
        )
        self.decoder_style_adapter_gate_bias_warmup = int(
            hparams.get("decoder_style_adapter_gate_bias_warmup", 0)
        )

        # self.time_ratio = hparams['sample_rate'] / hparams['hop_size'] / 50.0
        if hparams["f0_gen"] == "flow":
            self.uv_predictor = PitchPredictor(
                self.hidden_size,
                n_chans=128,
                n_layers=5,
                dropout_rate=0.1,
                odim=2,
                kernel_size=hparams["predictor_kernel"],
            )
            self.pitch_flownet = F0DiffNet(in_dims=1)
            self.f0_gen = ReflowF0(
                out_dims=1,
                denoise_fn=self.pitch_flownet,
                timesteps=hparams["f0_timesteps"],
                f0_K_step=hparams["f0_K_step"],
            )
        else:
            self.uv_predictor = PitchPredictor(
                self.hidden_size,
                n_chans=128,
                n_layers=5,
                dropout_rate=0.1,
                odim=2,
                kernel_size=hparams["predictor_kernel"],
            )

    def forward(
        self,
        content,
        spk_embed=None,
        target=None,
        ref=None,
        f0=None,
        uv=None,
        infer=False,
        global_steps=0,
        **kwargs,
    ):
        ret = {}
        ret["content"] = content
        ret["content_padding_idx"] = self.content_padding_idx
        tgt_nonpadding = (content != self.content_padding_idx).float()[:, :, None]
        content_ids = content.clamp(min=0, max=self.content_vocab_size - 1)
        content_embed = self.content_embedding(content_ids)
        content_embed = self.content_proj(content_embed.transpose(1, 2)).transpose(1, 2)
        content_embed = content_embed * tgt_nonpadding
        ret["content_embed_proj"] = content_embed

        reference_bundle = resolve_reference_bundle(
            kwargs,
            fallback_ref=ref,
            prompt_fallback_to_style=bool(
                kwargs.get(
                    "prompt_ref_fallback_to_style",
                    self.hparams.get("prompt_ref_fallback_to_style", False),
                )
            ),
        )
        reference_cache = self.prepare_reference_cache(
            reference_bundle=reference_bundle,
            spk_embed=spk_embed,
            infer=infer,
            global_steps=global_steps,
            reference_cache=resolve_reference_cache(kwargs),
        )
        ret["reference_bundle"] = reference_bundle
        ret["reference_cache"] = reference_cache
        ret["has_reference_cache"] = reference_cache is not None
        ret["reference_contract_mode"] = reference_bundle.get("reference_contract_mode", "collapsed_reference")
        ret["reference_contract"] = reference_bundle.get("reference_contract", {})
        ref_timbre = reference_bundle["ref_timbre"]
        ref_style = reference_bundle["ref_style"]
        ref_dynamic_timbre = reference_bundle["ref_dynamic_timbre"]
        global_timbre_anchor = reference_cache.get("global_timbre_anchor")
        if global_timbre_anchor is None:
            raise ValueError("reference_cache must provide `global_timbre_anchor`.")
        global_style_summary = reference_cache.get("global_style_summary")
        if global_style_summary is None:
            raise ValueError("reference_cache must provide `global_style_summary`.")
        ret["global_timbre_anchor"] = global_timbre_anchor
        ret["global_style_summary"] = global_style_summary
        ret["global_style_summary_source"] = reference_cache.get("global_style_summary_source", None)
        style_mainline = resolve_style_mainline_controls(kwargs, hparams=self.hparams)
        ret["style_mainline"] = style_mainline.as_dict()
        ret["decoder_style_condition_mode"] = style_mainline.mode
        ret["style_trace_mode"] = style_mainline.style_trace_mode
        ret["style_memory_mode"] = style_mainline.style_memory_mode
        ret["dynamic_timbre_memory_mode"] = style_mainline.dynamic_timbre_memory_mode
        ret["style_temperature"] = float(style_mainline.style_temperature)
        ret["global_style_trace_blend_runtime"] = float(style_mainline.global_style_trace_blend)
        ret["dynamic_timbre_temperature"] = float(style_mainline.dynamic_timbre_temperature)
        ret["dynamic_timbre_style_condition_scale_runtime"] = float(
            style_mainline.dynamic_timbre_style_condition_scale
        )
        ret["dynamic_timbre_gate_scale_runtime"] = float(style_mainline.dynamic_timbre_gate_scale)
        ret["dynamic_timbre_gate_bias_runtime"] = float(style_mainline.dynamic_timbre_gate_bias)
        ret["dynamic_timbre_boundary_suppress_strength_runtime"] = float(
            style_mainline.dynamic_timbre_boundary_suppress_strength
        )
        ret["dynamic_timbre_boundary_radius_runtime"] = int(style_mainline.dynamic_timbre_boundary_radius)
        ret["dynamic_timbre_anchor_preserve_strength_runtime"] = float(
            style_mainline.dynamic_timbre_anchor_preserve_strength
        )
        ret["dynamic_timbre_use_tvt_runtime"] = bool(
            kwargs.get(
                "dynamic_timbre_use_tvt",
                self.hparams.get("dynamic_timbre_use_tvt", True),
            )
        )
        ret["dynamic_timbre_tvt_prior_scale_runtime"] = float(
            kwargs.get(
                "dynamic_timbre_tvt_prior_scale",
                self.hparams.get("dynamic_timbre_tvt_prior_scale", 1.0),
            )
        )
        global_style_anchor_strength = self._resolve_strength(
            style_mainline.global_style_anchor_strength,
            batch_size=content.size(0),
            device=content.device,
        )
        ret["global_style_anchor_strength"] = global_style_anchor_strength.squeeze(-1)
        ret["global_style_anchor_applied"] = bool(style_mainline.apply_global_style_anchor)
        ret["global_timbre_to_pitch_enabled"] = bool(style_mainline.global_timbre_to_pitch)
        ret["style_mainline_memory"] = build_style_mainline_memory_payload(reference_cache)

        condition_embed = self.get_condition_embed(
            kwargs=kwargs,
            reference=global_timbre_anchor,
            ret=ret,
            ref_emotion=reference_bundle["ref_emotion"],
            ref_accent=reference_bundle["ref_accent"],
            reference_cache=reference_cache,
        )
        global_timbre_anchor_runtime = global_timbre_anchor * global_style_anchor_strength
        if not style_mainline.apply_global_style_anchor:
            global_timbre_anchor_runtime = torch.zeros_like(global_timbre_anchor_runtime)
        ret["global_timbre_anchor_runtime"] = global_timbre_anchor_runtime
        # Query path is strictly content+condition (no global timbre anchor injection).
        base_condition_inp = content_embed + condition_embed
        ret["query_condition_inp"] = base_condition_inp
        pitch_inp = base_condition_inp
        if style_mainline.apply_global_style_anchor and style_mainline.global_timbre_to_pitch:
            pitch_inp = pitch_inp + global_timbre_anchor_runtime
        ret["pitch_condition_inp"] = pitch_inp
        ret["global_timbre_to_pitch_applied"] = bool(
            style_mainline.apply_global_style_anchor and style_mainline.global_timbre_to_pitch
        )
        global_style_query_prior = self._expand_summary_to_sequence(
            global_style_summary,
            content_embed.size(1),
            device=content_embed.device,
            dtype=content_embed.dtype,
        )
        style_query_global_summary_scale = float(
            kwargs.get(
                "style_query_global_summary_scale",
                self.hparams.get("style_query_global_summary_scale", 0.0),
            )
        )
        style_query_base = base_condition_inp
        if isinstance(global_style_query_prior, torch.Tensor) and style_query_global_summary_scale != 0.0:
            style_query_base = style_query_base + style_query_global_summary_scale * global_style_query_prior
        if self.style_query_norm is not None:
            style_query_base = self.style_query_norm(style_query_base)
        ret["global_style_query_prior"] = global_style_query_prior
        ret["style_query_global_summary_scale"] = style_query_global_summary_scale
        ret["style_query_base"] = style_query_base
        ret["query_anchor_split_applied"] = True
        style_query_inp = (
            self.style_query_proj(style_query_base)
            if self.style_query_proj is not None else style_query_base
        )
        ret["style_query_inp"] = style_query_inp
        fast_style_decoder_residual = None
        slow_style_decoder_residual = None
        M_style_final = None
        M_style_mask = None
        dynamic_timbre_decoder_residual = None

        has_cached_timbre = reference_cache.get("timbre_memory") is not None or reference_cache.get("timbre_memory_slow") is not None
        style_trace_available = False
        dynamic_timbre_available = False
        style_trace_source = "disabled_by_mode" if not style_mainline.apply_style_trace else "missing"
        dynamic_timbre_source = "disabled_by_mode" if not style_mainline.apply_dynamic_timbre else "missing"
        style_strength = self._resolve_strength(
            style_mainline.style_strength,
            batch_size=content.size(0),
            device=content.device,
        )
        dynamic_timbre_strength = self._resolve_strength(
            style_mainline.dynamic_timbre_strength,
            batch_size=content.size(0),
            device=content.device,
        )
        style_payload = build_style_realization_payload(
            self,
            query=style_query_inp,
            ret=ret,
            reference_cache=reference_cache,
            ref_style=ref_style,
            infer=infer,
            global_steps=global_steps,
            controls=style_mainline,
            global_style_summary=global_style_summary,
            style_strength=style_strength,
            forcing_schedule_state=kwargs.get("forcing_schedule_state"),
        )
        style_trace_available = bool(style_payload.get("style_trace_available", False))
        style_trace_source = str(style_payload.get("style_trace_source", style_trace_source))
        fast_style_decoder_residual = style_payload.get("fast_style_decoder_residual")
        slow_style_decoder_residual = style_payload.get("slow_style_decoder_residual")
        M_style_final = style_payload.get("style_decoder_residual")
        M_style_mask = style_payload.get("style_decoder_residual_mask")
        global_style_summary_runtime = style_payload.get("global_style_summary_runtime", global_style_summary)
        global_style_summary_runtime_source = style_payload.get(
            "global_style_summary_runtime_source",
            reference_cache.get("global_style_summary_source", "reference_cache")
            if isinstance(reference_cache, dict)
            else "reference_cache",
        )
        if M_style_final is None:
            M_style_final, M_style_mask = resolve_combined_style_trace(
                {
                    "style_trace": fast_style_decoder_residual,
                    "slow_style_trace": slow_style_decoder_residual,
                    "style_trace_mask": ret.get("style_trace_mask"),
                    "slow_style_trace_mask": ret.get("slow_style_trace_mask"),
                }
            )
        if M_style_final is None:
            M_style_final = combine_style_traces(
                fast_style_decoder_residual,
                slow_style_decoder_residual,
            )
            M_style_mask = ret.get("style_trace_mask", ret.get("slow_style_trace_mask"))
        ret["main_style_owner_residual"] = M_style_final
        ret["style_owner_source"] = style_payload.get("style_owner_source", ret.get("style_owner_source", "missing"))
        dynamic_timbre_coarse_style_context_scale = float(
            kwargs.get(
                "dynamic_timbre_coarse_style_context_scale",
                self.hparams.get("dynamic_timbre_coarse_style_context_scale", 0.0),
            )
        )
        # Dynamic timbre may only condition on the unified local style owner.
        dynamic_timbre_style_context_stopgrad = bool(
            kwargs.get(
                "dynamic_timbre_style_context_stopgrad",
                self.hparams.get("dynamic_timbre_style_context_stopgrad", True),
            )
        )
        content_padding_mask = content.eq(self.content_padding_idx)
        dynamic_timbre_style_context = self._prepare_dynamic_timbre_style_context(
            M_style_final,
            padding_mask=content_padding_mask,
            stopgrad=dynamic_timbre_style_context_stopgrad,
        )
        ret["dynamic_timbre_style_context_raw"] = M_style_final
        ret["dynamic_timbre_style_context"] = dynamic_timbre_style_context
        ret["dynamic_timbre_coarse_style_context_scale"] = dynamic_timbre_coarse_style_context_scale
        ret["dynamic_timbre_coarse_style_context_scale_requested"] = dynamic_timbre_coarse_style_context_scale
        ret["dynamic_timbre_coarse_style_context_applied"] = False
        ret["timbre_query_style_context_applied"] = False
        ret["dynamic_timbre_style_context_stopgrad"] = dynamic_timbre_style_context_stopgrad
        ret["dynamic_timbre_style_context_owner_safe"] = isinstance(dynamic_timbre_style_context, torch.Tensor)
        ret["dynamic_timbre_style_context_bridge"] = (
            "layernorm_stopgrad" if dynamic_timbre_style_context_stopgrad else "layernorm"
        )
        query_style_scale_override = kwargs.get(
            "dynamic_timbre_query_style_condition_scale",
            self.hparams.get("dynamic_timbre_query_style_condition_scale", 0.0),
        )
        if query_style_scale_override is None:
            query_style_scale_override = 0.0
        timbre_query_style_scale = float(query_style_scale_override)
        timbre_query_style_scale_source = (
            "query_style_condition" if timbre_query_style_scale != 0.0 else "disabled"
        )
        ret["timbre_query_style_scale"] = timbre_query_style_scale
        ret["timbre_query_style_scale_source"] = timbre_query_style_scale_source
        timbre_query_base = base_condition_inp
        if isinstance(dynamic_timbre_style_context, torch.Tensor) and timbre_query_style_scale != 0.0:
            timbre_query_base = timbre_query_base + timbre_query_style_scale * dynamic_timbre_style_context
            ret["timbre_query_style_context_applied"] = True
            ret["dynamic_timbre_coarse_style_context_applied"] = bool(
                timbre_query_style_scale_source == "coarse_style_context"
            )
        if self.timbre_query_norm is not None:
            timbre_query_base = self.timbre_query_norm(timbre_query_base)
        timbre_query_inp = (
            self.timbre_query_proj(timbre_query_base)
            if self.timbre_query_proj is not None else timbre_query_base
        )
        ret["timbre_query_base"] = timbre_query_base
        ret["timbre_query_inp"] = timbre_query_inp
        ret["timbre_query_follows_style_owner"] = True
        if style_payload.get("style_trace_skip_reason") is not None:
            ret["style_trace_skip_reason"] = style_payload.get("style_trace_skip_reason")
        if (
            self.use_dynamic_timbre
            and style_mainline.apply_dynamic_timbre
            and (ref_dynamic_timbre is not None or has_cached_timbre)
        ):
            dynamic_timbre = self.get_dynamic_timbre(
                timbre_query_inp,
                ref_dynamic_timbre,
                ret,
                infer=infer,
                global_steps=global_steps,
                reference_cache=reference_cache,
                memory_mode=style_mainline.dynamic_timbre_memory_mode,
                timbre_temperature=style_mainline.dynamic_timbre_temperature,
                style_context=dynamic_timbre_style_context,
                style_condition_scale=style_mainline.dynamic_timbre_style_condition_scale,
                gate_scale=style_mainline.dynamic_timbre_gate_scale,
                gate_bias=style_mainline.dynamic_timbre_gate_bias,
                boundary_suppress_strength=style_mainline.dynamic_timbre_boundary_suppress_strength,
                boundary_radius=style_mainline.dynamic_timbre_boundary_radius,
                anchor_preserve_strength=style_mainline.dynamic_timbre_anchor_preserve_strength,
                style_context_prepared=True,
                tvt_prior_scale=float(
                    kwargs.get(
                        "dynamic_timbre_tvt_prior_scale",
                        self.hparams.get("dynamic_timbre_tvt_prior_scale", 1.0),
                    )
                ),
                use_tvt=bool(style_mainline.dynamic_timbre_use_tvt),
            )
            dynamic_timbre_decoder_residual = dynamic_timbre * dynamic_timbre_strength
            dynamic_timbre_decoder_residual = self._apply_runtime_dynamic_timbre_budget(
                dynamic_timbre_decoder_residual,
                style_decoder_residual=M_style_final,
                slow_style_decoder_residual=None,
                content=content,
                kwargs=kwargs,
                ret=ret,
            )
            dynamic_timbre_available = True
            dynamic_timbre_source = "reference_cache" if has_cached_timbre else "reference_audio"
        if self.use_dynamic_timbre and not style_mainline.apply_dynamic_timbre:
            ret["dynamic_timbre_skip_reason"] = "decoder_style_condition_mode"
        elif self.use_dynamic_timbre and not dynamic_timbre_available and not (ref_dynamic_timbre is not None or has_cached_timbre):
            ret["dynamic_timbre_skip_reason"] = "reference_missing"
        ret["style_trace_applied"] = bool(style_trace_available)
        ret["dynamic_timbre_applied"] = bool(dynamic_timbre_available)
        ret["fast_style_decoder_residual"] = fast_style_decoder_residual
        ret["style_decoder_residual"] = M_style_final
        ret["style_decoder_residual_mask"] = M_style_mask
        ret["slow_style_decoder_residual"] = slow_style_decoder_residual
        ret["dynamic_timbre_decoder_residual"] = dynamic_timbre_decoder_residual
        ret["global_style_summary_runtime"] = global_style_summary_runtime
        ret["global_style_summary_runtime_source"] = global_style_summary_runtime_source
        decoder_signal_eps = float(
            getattr(self.decoder_style_adapter, "effective_signal_epsilon", 1e-8)
            if self.decoder_style_adapter is not None
            else 1e-8
        )
        ret["decoder_style_bundle_effective_signal_epsilon"] = decoder_signal_eps
        decoder_global_timbre_anchor = global_timbre_anchor if style_mainline.apply_global_style_anchor else None
        decoder_global_timbre_anchor_runtime = (
            global_timbre_anchor_runtime if style_mainline.apply_global_style_anchor else None
        )
        ret["decoder_style_bundle"] = build_decoder_style_bundle(
            global_timbre_anchor=maybe_effective_singleton(
                decoder_global_timbre_anchor,
                eps=decoder_signal_eps,
            ),
            global_timbre_anchor_runtime=maybe_effective_singleton(
                decoder_global_timbre_anchor_runtime,
                eps=decoder_signal_eps,
            ),
            global_timbre_source=reference_cache.get("global_timbre_anchor_source", "reference_cache"),
            global_style_summary=maybe_effective_singleton(
                global_style_summary_runtime,
                eps=decoder_signal_eps,
            ),
            global_style_summary_source=global_style_summary_runtime_source,
            slow_style_trace=None,
            slow_style_trace_mask=None,
            slow_style_source="retained_in_logs_only",
            M_style=maybe_effective_sequence(M_style_final, eps=decoder_signal_eps),
            M_style_mask=M_style_mask,
            M_style_source="combined_style_owner",
            M_timbre=maybe_effective_sequence(
                dynamic_timbre_decoder_residual,
                eps=decoder_signal_eps,
            ),
            M_timbre_mask=ret.get("dynamic_timbre_mask"),
            M_timbre_source=dynamic_timbre_source,
            factorization_guaranteed=bool(
                ret.get("reference_contract", {}).get("factorization_guaranteed", False)
            ),
            mainline_owner=style_mainline.mainline_owner,
            bundle_variant=style_mainline.mode,
            bundle_source="decoder_style_mainline",
            timing_authority=DECODER_STYLE_TIMING_AUTHORITY,
            enforce_no_timing_writeback=style_mainline.enforce_decoder_no_timing_writeback,
            effective_signal_epsilon=decoder_signal_eps,
        )
        validate_decoder_style_bundle(ret["decoder_style_bundle"])
        ret["style_mainline_surface"] = build_style_mainline_surface_payload(
            style_mainline,
            style_trace_available=style_trace_available,
            dynamic_timbre_available=dynamic_timbre_available,
            style_trace_source=style_trace_source,
            dynamic_timbre_source=dynamic_timbre_source,
        )
        ret["pitch_embed"] = pitch_inp

        if infer:
            f0, uv = None, None
        self.build_reference_summary(
            ret,
            summary_source=reference_bundle.get("summary_source", None),
            reference_cache=reference_cache,
        )
        pitch_embed_out = self.forward_pitch(pitch_inp, f0, uv, ret, **kwargs)
        energy_embed = self.forward_energy(pitch_inp, kwargs.get("energy", None), ret) \
            if self.use_energy_embed else 0.0
        ret["decoder_style_adapter_enabled"] = bool(self.decoder_style_adapter is not None)
        if self.decoder_style_adapter is not None and self.decoder_style_adapter_gate_bias_warmup > 0:
            ratio = min(1.0, float(global_steps) / float(self.decoder_style_adapter_gate_bias_warmup))
            gate_bias = (
                self.decoder_style_adapter_gate_bias_start
                + (self.decoder_style_adapter_gate_bias_end - self.decoder_style_adapter_gate_bias_start) * ratio
            )
            self.decoder_style_adapter.set_gate_bias(gate_bias)
            ret["decoder_style_adapter_gate_bias_runtime"] = gate_bias
        ret["decoder_inp"] = decoder_inp = pitch_inp + pitch_embed_out + energy_embed
        ret["mel_out"] = self.forward_decoder(decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs)
        ret["tgt_nonpadding"] = tgt_nonpadding
        return ret

    def encode_spk_embed(self, x):
        in_nonpadding = (x.abs().sum(dim=-2) > 0).float()[:, None, :]
        x_global = self.global_conv_in(x) * in_nonpadding
        global_z_e_x = self.global_encoder(x_global, nonpadding=in_nonpadding) * in_nonpadding
        global_z_e_x = self.temporal_avg_pool(x=global_z_e_x, mask=(in_nonpadding == 0))
        return global_z_e_x

    def temporal_avg_pool(self, x, mask=None):
        len_ = (~mask).sum(dim=-1).unsqueeze(-1)
        x = x.masked_fill(mask, 0)
        x = x.sum(dim=-1).unsqueeze(-1)
        out = torch.div(x, len_)
        return out

    @staticmethod
    def _expand_summary_to_sequence(summary, target_len, *, device=None, dtype=None):
        if not isinstance(summary, torch.Tensor):
            return None
        if summary.dim() == 2:
            summary = summary.unsqueeze(1)
        if summary.dim() != 3:
            return None
        if summary.size(1) == target_len:
            expanded = summary
        elif summary.size(1) == 1:
            expanded = summary.expand(-1, target_len, -1)
        elif summary.size(1) > 0:
            expanded = summary.mean(dim=1, keepdim=True).expand(-1, target_len, -1)
        else:
            return None
        return expanded.to(
            device=device if device is not None else expanded.device,
            dtype=dtype if dtype is not None else expanded.dtype,
        )

    @staticmethod
    def _runtime_first_present(mapping, *keys):
        if not isinstance(mapping, dict):
            return None
        for key in keys:
            value = mapping.get(key)
            if value is not None:
                return value
        return None

    @staticmethod
    def _normalize_style_to_pitch_residual_mode(mode) -> str:
        normalized = str(mode or "auto").strip().lower() or "auto"
        alias_map = {
            "source": "source_aligned",
            "sourcealigned": "source_aligned",
            "pre_rhythm": "source_aligned",
            "content": "source_aligned",
            "post": "post_rhythm",
            "after_rhythm": "post_rhythm",
            "joint": "post_rhythm",
            "post_projector": "post_rhythm",
        }
        normalized = alias_map.get(normalized, normalized)
        if normalized not in {"auto", "source_aligned", "post_rhythm"}:
            return "auto"
        return normalized

    @staticmethod
    def _masked_smooth_sequence_1d(sequence, *, valid_mask=None, smooth_factor=0.0, kernel_size=3):
        if not isinstance(sequence, torch.Tensor) or sequence.dim() != 2:
            return sequence
        smooth_factor = float(smooth_factor)
        if smooth_factor <= 0.0 or sequence.size(1) <= 1:
            return sequence
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2
        if isinstance(valid_mask, torch.Tensor):
            if valid_mask.dim() == 3 and valid_mask.size(-1) == 1:
                valid_mask = valid_mask.squeeze(-1)
            if valid_mask.dim() == 2 and tuple(valid_mask.shape) == tuple(sequence.shape):
                valid_mask = valid_mask.to(device=sequence.device, dtype=sequence.dtype).clamp(0.0, 1.0)
            else:
                valid_mask = None
        if valid_mask is None:
            pooled = F.avg_pool1d(
                sequence.unsqueeze(1),
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ).squeeze(1)
            return (1.0 - smooth_factor) * sequence + smooth_factor * pooled

        weighted = sequence * valid_mask
        pooled = F.avg_pool1d(
            weighted.unsqueeze(1),
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        ).squeeze(1)
        denom = F.avg_pool1d(
            valid_mask.unsqueeze(1),
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        ).squeeze(1).clamp_min(1.0e-6)
        pooled = pooled / denom
        smoothed = (1.0 - smooth_factor) * sequence + smooth_factor * pooled
        smoothed = smoothed * valid_mask + sequence * (1.0 - valid_mask)
        return smoothed

    def _project_source_sequence_to_pitch_canvas(self, sequence, ret, *, target_shape=None, mode="auto"):
        meta = {
            "canvas": "source_aligned",
            "mask": None,
            "blank_mask": None,
            "valid_mask": None,
        }
        if not isinstance(sequence, torch.Tensor) or sequence.dim() != 2:
            return sequence, meta

        mode = self._normalize_style_to_pitch_residual_mode(mode)
        if mode == "source_aligned":
            content = ret.get("content")
            if isinstance(content, torch.Tensor) and tuple(content.shape) == tuple(sequence.shape):
                meta["mask"] = content.eq(self.content_padding_idx)
            return sequence, meta

        if target_shape is not None and tuple(sequence.shape) == tuple(target_shape) and mode == "auto":
            content = ret.get("content")
            if isinstance(content, torch.Tensor) and tuple(content.shape) == tuple(sequence.shape):
                meta["mask"] = content.eq(self.content_padding_idx)
            return sequence, meta

        source_frame_index = self._runtime_first_present(
            ret,
            "rhythm_source_frame_index",
            "source_frame_index",
            "frame_source_index",
            "retimed_source_frame_index",
        )
        unit_index = self._runtime_first_present(
            ret,
            "rhythm_frame_unit_index",
            "rhythm_slot_unit_index",
            "frame_unit_index",
            "slot_unit_index",
        )
        frame_valid_mask = self._runtime_first_present(
            ret,
            "rhythm_frame_valid_mask",
            "frame_valid_mask",
            "slot_mask",
        )
        blank_mask = self._runtime_first_present(
            ret,
            "rhythm_frame_blank_mask",
            "frame_blank_mask",
            "slot_is_blank",
        )

        projected = None
        canvas = None
        if isinstance(source_frame_index, torch.Tensor) and source_frame_index.dim() == 2:
            gather_index = source_frame_index.long().clamp(min=0, max=max(0, sequence.size(1) - 1))
            projected = torch.gather(sequence, 1, gather_index)
            canvas = "post_rhythm_source_frame_index"
        elif isinstance(unit_index, torch.Tensor) and unit_index.dim() == 2:
            gather_index = unit_index.long().clamp(min=0, max=max(0, sequence.size(1) - 1))
            projected = torch.gather(sequence, 1, gather_index)
            canvas = "post_rhythm_unit_index"

        if isinstance(projected, torch.Tensor):
            valid_mask = None
            if isinstance(frame_valid_mask, torch.Tensor):
                if frame_valid_mask.dim() == 3 and frame_valid_mask.size(-1) == 1:
                    frame_valid_mask = frame_valid_mask.squeeze(-1)
                if frame_valid_mask.dim() == 2 and tuple(frame_valid_mask.shape) == tuple(projected.shape):
                    valid_mask = frame_valid_mask.to(device=projected.device, dtype=projected.dtype).clamp(0.0, 1.0)
                    projected = projected * valid_mask
            if isinstance(blank_mask, torch.Tensor):
                if blank_mask.dim() == 3 and blank_mask.size(-1) == 1:
                    blank_mask = blank_mask.squeeze(-1)
                if blank_mask.dim() == 2 and tuple(blank_mask.shape) == tuple(projected.shape):
                    projected = projected.masked_fill(blank_mask.bool(), 0.0)
                else:
                    blank_mask = None
            if target_shape is None or tuple(projected.shape) == tuple(target_shape):
                invalid_mask = None
                if isinstance(valid_mask, torch.Tensor):
                    invalid_mask = valid_mask <= 0
                if isinstance(blank_mask, torch.Tensor):
                    invalid_mask = blank_mask.bool() if invalid_mask is None else (invalid_mask | blank_mask.bool())
                meta.update({
                    "canvas": canvas or "post_rhythm",
                    "mask": invalid_mask,
                    "blank_mask": blank_mask.bool() if isinstance(blank_mask, torch.Tensor) else None,
                    "valid_mask": valid_mask,
                })
                return projected, meta

        content = ret.get("content")
        if isinstance(content, torch.Tensor) and tuple(content.shape) == tuple(sequence.shape):
            meta["mask"] = content.eq(self.content_padding_idx)
        return sequence, meta

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
            kwargs.get(
                "runtime_dynamic_timbre_style_budget_enabled",
                self.hparams.get("runtime_dynamic_timbre_style_budget_enabled", True),
            )
        )
        if not isinstance(dynamic_timbre_decoder_residual, torch.Tensor):
            return dynamic_timbre_decoder_residual

        if ret is not None:
            ret["runtime_dynamic_timbre_style_budget_enabled"] = enabled
        if not enabled:
            return dynamic_timbre_decoder_residual

        ratio = float(
            kwargs.get(
                "runtime_dynamic_timbre_style_budget_ratio",
                self.hparams.get("runtime_dynamic_timbre_style_budget_ratio", 0.40),
            )
        )
        margin = float(
            kwargs.get(
                "runtime_dynamic_timbre_style_budget_margin",
                self.hparams.get("runtime_dynamic_timbre_style_budget_margin", 0.0),
            )
        )
        slow_style_weight = float(
            kwargs.get(
                "runtime_dynamic_timbre_style_budget_slow_style_weight",
                self.hparams.get("runtime_dynamic_timbre_style_budget_slow_style_weight", 1.0),
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
        )
        if ret is not None:
            ret["runtime_dynamic_timbre_style_budget_ratio"] = float(ratio)
            ret["runtime_dynamic_timbre_style_budget_margin"] = float(margin)
            if isinstance(budget_meta, dict):
                if isinstance(budget_meta.get("allowed_energy"), torch.Tensor):
                    ret["runtime_dynamic_timbre_style_budget_cap"] = budget_meta["allowed_energy"]
                if isinstance(budget_meta.get("style_energy"), torch.Tensor):
                    ret["runtime_dynamic_timbre_style_energy"] = budget_meta["style_energy"]
                if isinstance(budget_meta.get("timbre_energy"), torch.Tensor):
                    ret["runtime_dynamic_timbre_dynamic_energy"] = budget_meta["timbre_energy"]
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
        enabled = bool(
            kwargs.get(
                "style_to_pitch_residual",
                self.hparams.get("style_to_pitch_residual", False),
            )
        )
        residual_mode = self._normalize_style_to_pitch_residual_mode(
            kwargs.get(
                "style_to_pitch_residual_mode",
                self.hparams.get("style_to_pitch_residual_mode", "auto"),
            )
        )
        ret["style_to_pitch_residual_enabled"] = enabled
        ret["style_to_pitch_residual_mode"] = residual_mode
        ret["style_to_pitch_residual_applied"] = False
        if not enabled or self.style_to_pitch_residual_head is None or not isinstance(f0_out, torch.Tensor):
            return f0_out

        style_residual = ret.get("style_decoder_residual")
        timbre_residual = ret.get("dynamic_timbre_decoder_residual")
        if not isinstance(style_residual, torch.Tensor) or tuple(style_residual.shape) != tuple(decoder_inp.shape):
            style_residual = torch.zeros_like(decoder_inp)
        if not isinstance(timbre_residual, torch.Tensor) or tuple(timbre_residual.shape) != tuple(decoder_inp.shape):
            timbre_residual = torch.zeros_like(decoder_inp)

        pitch_residual_hidden = torch.cat([decoder_inp, style_residual, timbre_residual], dim=-1)
        pitch_residual_intent = self.style_to_pitch_residual_head(pitch_residual_hidden).squeeze(-1)
        pitch_residual_scale = float(
            kwargs.get(
                "style_to_pitch_residual_scale",
                self.hparams.get("style_to_pitch_residual_scale", 1.0),
            )
        )
        max_semitones = float(
            kwargs.get(
                "style_to_pitch_residual_max_semitones",
                self.hparams.get("style_to_pitch_residual_max_semitones", 2.5),
            )
        )
        max_log2 = max(0.0, max_semitones / 12.0) * max(0.0, pitch_residual_scale)
        pitch_residual_intent = torch.tanh(pitch_residual_intent) * max_log2

        smooth_factor = float(
            kwargs.get(
                "style_to_pitch_residual_smooth_factor",
                self.hparams.get("style_to_pitch_residual_smooth_factor", 0.35),
            )
        )

        content = ret.get("content")
        if isinstance(content, torch.Tensor) and tuple(content.shape) == tuple(pitch_residual_intent.shape):
            pitch_residual_intent = pitch_residual_intent.masked_fill(content.eq(self.content_padding_idx), 0.0)

        target_shape = (
            tuple(uv_out.shape)
            if isinstance(uv_out, torch.Tensor) and uv_out.dim() == 2
            else tuple(f0_out.shape)
        )
        pitch_residual, canvas_meta = self._project_source_sequence_to_pitch_canvas(
            pitch_residual_intent,
            ret,
            target_shape=target_shape,
            mode=residual_mode,
        )
        ret["style_to_pitch_residual_intent"] = pitch_residual_intent

        voiced_mask = None
        if isinstance(uv_out, torch.Tensor) and tuple(uv_out.shape) == tuple(pitch_residual.shape):
            voiced_mask = (uv_out == 0)
            pitch_residual = pitch_residual * voiced_mask.to(pitch_residual.dtype)

        residual_mask = canvas_meta.get("mask")
        smoothing_valid_mask = None
        if isinstance(residual_mask, torch.Tensor) and tuple(residual_mask.shape) == tuple(pitch_residual.shape):
            smoothing_valid_mask = (~residual_mask.bool()).to(device=pitch_residual.device, dtype=pitch_residual.dtype)
        if isinstance(voiced_mask, torch.Tensor) and tuple(voiced_mask.shape) == tuple(pitch_residual.shape):
            voiced_valid = voiced_mask.to(device=pitch_residual.device, dtype=pitch_residual.dtype)
            smoothing_valid_mask = voiced_valid if smoothing_valid_mask is None else (smoothing_valid_mask * voiced_valid)
        pitch_residual = self._masked_smooth_sequence_1d(
            pitch_residual,
            valid_mask=smoothing_valid_mask,
            smooth_factor=smooth_factor,
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
            base_pred_canvas, _ = self._project_source_sequence_to_pitch_canvas(
                base_pred,
                ret,
                target_shape=tuple(pitch_residual.shape),
                mode=residual_mode,
            )
        else:
            base_pred_canvas = None
        if isinstance(base_pred_canvas, torch.Tensor) and tuple(base_pred_canvas.shape) == tuple(pitch_residual.shape):
            ret["style_to_pitch_residual_base_pred"] = base_pred_canvas

        target_canvas = None
        if isinstance(f0_target, torch.Tensor):
            target_canvas, _ = self._project_source_sequence_to_pitch_canvas(
                f0_target,
                ret,
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

    def forward_pitch(self, decoder_inp, f0, uv, ret, **kwargs):  # add **kwargs
        pitch_pred_inp = decoder_inp
        # apply predictor_grad to control gradient backprop
        if self.hparams["predictor_grad"] != 1:
            pitch_pred_inp = pitch_pred_inp.detach() + self.hparams[
                "predictor_grad"
            ] * (pitch_pred_inp - pitch_pred_inp.detach())

        # --- select F0 generation method based on config and pass kwargs ---
        if hparams["f0_gen"] == "diff":
            f0_out, uv_out = self.add_diff_pitch(pitch_pred_inp, f0, uv, ret, **kwargs)
        elif hparams["f0_gen"] == "gmdiff":
            f0_out, uv_out = self.add_gmdiff_pitch(
                pitch_pred_inp, f0, uv, ret, **kwargs
            )
        elif hparams["f0_gen"] == "flow":
            f0_out, uv_out = self.add_flow_pitch(
                pitch_pred_inp, f0, uv, ret, **kwargs
            )  # pass kwargs
        elif hparams["f0_gen"] == "orig":
            f0_out, uv_out = self.add_orig_pitch(pitch_pred_inp, f0, uv, ret, **kwargs)
        else:
            raise ValueError(f"Unknown f0_gen type: {hparams['f0_gen']}")

        f0_out = self._apply_style_to_pitch_residual(
            f0_out,
            uv_out,
            pitch_pred_inp,
            ret,
            f0_target=f0,
            **kwargs,
        )

        # --- use f0_out, uv_out returned from add_x_pitch ---
        # f0_out might be log F0 or other forms, denorm_f0 should handle it
        f0_denorm = denorm_f0(f0_out, uv_out)  # use returned f0 and uv for denormalization
        pitch = f0_to_coarse(f0_denorm)  # convert to pitch categories
        ret["f0_denorm_pred"] = f0_denorm  # store final predicted denormalized F0 in ret
        pitch_embed = self.pitch_embed(pitch)  # compute pitch embedding
        return pitch_embed  # return pitch embedding

    # def forward_dur(self, dur_input, mel2ph, txt_tokens, ret):
    #     """

    #     :param dur_input: [B, T_txt, H]
    #     :param mel2ph: [B, T_mel]
    #     :param txt_tokens: [B, T_txt]
    #     :param ret:
    #     :return:
    #     """
    #     src_padding = txt_tokens == 0
    #     if self.hparams['predictor_grad'] != 1:
    #         dur_input = dur_input.detach() + self.hparams['predictor_grad'] * (dur_input - dur_input.detach())
    #     dur = self.dur_predictor(dur_input, src_padding)
    #     ret['dur'] = dur
    #     if mel2ph is None:
    #         dur = (dur.exp() - 1).clamp(min=0)
    #         mel2ph = self.length_regulator(dur, src_padding).detach()
    #     ret['mel2ph'] = mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
    #     return mel2ph

    def add_orig_pitch(self, decoder_inp, f0, uv, ret, encoder_out=None, **kwargs):
        # pitch_padding = mel2ph == 0
        if f0 is None:
            infer = True
        else:
            infer = False
        ret["uv_pred"] = uv_pred = self.uv_predictor(decoder_inp)
        ret["uv_base_pred"] = uv_pred[:, :, 0]
        ret["f0_base_pred"] = uv_pred[:, :, 1]

        if infer:
            uv = uv_pred[:, :, 0] > 0
            # print(f'uv:{uv},content:{ret["content"]}')
            if "content" in ret:
                content_padding_mask = (
                    (ret["content"] == self.hparams['silent_token'])
                    | (ret["content"] == self.content_padding_idx)
                )
                if content_padding_mask.shape == uv.shape:
                    uv[content_padding_mask] = 1  # force padding regions to be unvoiced
            uv = uv
            f0 = uv_pred[:, :, 1]
            ret["fdiff"] = 0.0
        else:
            # nonpadding = (mel2ph > 0).float() * (uv == 0).float()
            nonpadding = (uv == 0).float()
            f0_pred = uv_pred[:, :, 1]
            ret["fdiff"] = (
                (F.mse_loss(f0_pred, f0, reduction="none") * nonpadding).sum()
                / nonpadding.sum()
                * hparams["lambda_f0"]
            )
        return f0, uv

    def add_diff_pitch(
        self, decoder_inp, f0, uv, ret, mel2ph=None, encoder_out=None, **kwargs
    ):
        # pitch_padding = mel2ph == 0
        if f0 is None:
            infer = True
        else:
            infer = False
        ret["uv_pred"] = uv_pred = self.uv_predictor(decoder_inp)
        ret["uv_base_pred"] = uv_pred[:, :, 0]
        ret["f0_base_pred"] = uv_pred[:, :, 1]

        def minmax_norm(x, uv=None):
            x_min = 6
            x_max = 10
            if torch.any(x > x_max):
                raise ValueError("check minmax_norm!!")
            normed_x = (x - x_min) / (x_max - x_min) * 2 - 1
            if uv is not None:
                normed_x[uv > 0] = 0
            return normed_x

        def minmax_denorm(x, uv=None):
            x_min = 6
            x_max = 10
            denormed_x = (x + 1) / 2 * (x_max - x_min) + x_min
            if uv is not None:
                denormed_x[uv > 0] = 0
            return denormed_x

        if infer:
            uv = uv_pred[:, :, 0] > 0
            midi_notes = kwargs.get("midi_notes", None)
            if midi_notes is not None:
                midi_notes = midi_notes.transpose(-1, -2)
                uv[midi_notes[:, 0, :] == 0] = 1
                lower_bound = midi_notes - 3
                upper_bound = midi_notes + 3
                upper_norm_f0 = minmax_norm((2 ** ((upper_bound - 69) / 12) * 440).log2())
                lower_norm_f0 = minmax_norm((2 ** ((lower_bound - 69) / 12) * 440).log2())
                upper_norm_f0[upper_norm_f0 < -1] = -1
                upper_norm_f0[upper_norm_f0 > 1] = 1
                lower_norm_f0[lower_norm_f0 < -1] = -1
                lower_norm_f0[lower_norm_f0 > 1] = 1
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
            f0 = minmax_denorm(f0)
            ret["fdiff"] = 0.0
        else:
            if mel2ph is not None:
                nonpadding = (mel2ph > 0).float()
            else:
                nonpadding = (ret["content"] != self.content_padding_idx).float()
            norm_f0 = minmax_norm(f0)
            ret["fdiff"] = self.f0_gen(
                decoder_inp.transpose(-1, -2),
                norm_f0,
                nonpadding.unsqueeze(dim=1),
                ret,
                infer,
            )
        return f0, uv

    def add_flow_pitch(self, decoder_inp, f0, uv, ret, encoder_out=None, **kwargs):
        # pitch_padding = mel2ph == 0
        if f0 is None:
            infer = True
        else:
            infer = False
        ret["uv_pred"] = uv_pred = self.uv_predictor(decoder_inp)
        ret["uv_base_pred"] = uv_pred[:, :, 0]
        ret["f0_base_pred"] = uv_pred[:, :, 1]

        # define F0 normalization and denormalization functions (minmax_norm, minmax_denorm)
        def minmax_norm(x, uv=None):
            x_min = 6
            x_max = 10
            # if torch.any(x > x_max): # check if there are values exceeding the range (might only check during training)
            #     # print(f"Warning: F0 value > {x_max} found during normalization.")
            #     pass # or can choose to clip
            normed_x = (x - x_min) / (x_max - x_min) * 2 - 1
            if uv is not None:
                normed_x[uv > 0] = 0  # set unvoiced regions to 0
            return normed_x

        def minmax_denorm(x, uv=None):
            x_min = 6
            x_max = 10
            denormed_x = (x + 1) / 2 * (x_max - x_min) + x_min
            if uv is not None:
                denormed_x[uv > 0] = 0  # set unvoiced regions to 0
            return denormed_x

        # --- extract initial_noise from kwargs ---
        initial_noise = kwargs.get("initial_noise", None)

        # determine whether it's inference or training mode
        if f0 is None:  # if no f0 is provided, consider it inference mode
            infer = True
            if uv is None:  # if uv is also not provided, need to get it from predictor
                # ensure uv_predictor is initialized
                if not hasattr(self, "uv_predictor"):
                    raise AttributeError("uv_predictor is not defined in the model.")
                #  uv_pred = self.uv_predictor(decoder_inp) # predict UV, shape [B, T_tok, 2]
                uv = uv_pred[:, :, 0] > 0  # take first dimension as UV flag (True means unvoiced)
                # (optional) apply content padding mask to UV
                if "content" in ret:
                    content_padding_mask = (
                        (ret["content"] == self.hparams['silent_token'])
                        | (ret["content"] == self.content_padding_idx)
                    )
                    if content_padding_mask.shape == uv.shape:
                        uv[content_padding_mask] = 1  # force padding regions to be unvoiced
                    else:
                        print(
                            f"Warning: content mask shape {content_padding_mask.shape} doesn't match uv shape {uv.shape}, cannot apply."
                        )
                else:
                    print(
                        "Warning: missing 'content' in ret, cannot apply content padding to UV."
                    )

            # --- call self.f0_gen (ReflowF0 instance) for F0 prediction, and pass initial_noise ---
            # input cond needs to be [B, C, T] shape
            # decoder_inp is [B, T, C], needs transpose
            # f0_gen output should be normalized F0, shape [B, T]
            f0_pred_norm = self.f0_gen(
                decoder_inp.transpose(1, 2),
                None,
                None,
                ret,
                infer=True,
                initial_noise=initial_noise,
            )
            # use predicted (or provided) uv for denormalization
            f0_out = minmax_denorm(f0_pred_norm, uv)
            ret["pflow"] = 0.0  # no flow loss during inference
            uv_out = uv  # return uv used for denormalization
        else:  # if f0 is provided, consider it training mode
            infer = False
            # compute nonpadding (voiced region mask)
            nonpadding = (uv == 0).float()
            # use provided f0, uv for normalization
            norm_f0 = minmax_norm(f0, uv)
            # call f0_gen to compute flow loss, usually don't pass initial_noise during training
            # f0_gen training input norm_f0 needs to be [B, 1, 1, T] or [B, 1, D, T]
            # add_flow_pitch receives f0 as [B, T], norm_f0 is also [B, T]
            # need to adjust shape before calling f0_gen
            if norm_f0.ndim == 2:
                norm_f0_unsqueezed = norm_f0.unsqueeze(1).unsqueeze(1)  # -> [B, 1, 1, T]
            else:  # if already [B, T, 1] or other shapes, need corresponding adjustment
                raise ValueError(f"Unexpected norm_f0 shape during training: {norm_f0.shape}")
            # nonpadding needs to be [B, 1, T]
            ret["pflow"] = self.f0_gen(
                decoder_inp.transpose(1, 2),
                norm_f0_unsqueezed,
                nonpadding.unsqueeze(1),
                ret,
                infer=False,
            )
            f0_out = f0  # return original f0 during training
            uv_out = uv  # return original uv during training

        return f0_out, uv_out  # return computed or original f0 and uv

    def add_gmdiff_pitch(
        self, decoder_inp, f0, uv, ret, mel2ph=None, encoder_out=None, **kwargs
    ):
        # pitch_padding = mel2ph == 0
        if f0 is None:
            infer = True
        else:
            infer = False

        def minmax_norm(x, uv=None):
            x_min = 6
            x_max = 10
            if torch.any(x > x_max):
                raise ValueError("check minmax_norm!!")
            normed_x = (x - x_min) / (x_max - x_min) * 2 - 1
            if uv is not None:
                normed_x[uv > 0] = 0
            return normed_x

        def minmax_denorm(x, uv=None):
            x_min = 6
            x_max = 10
            denormed_x = (x + 1) / 2 * (x_max - x_min) + x_min
            if uv is not None:
                denormed_x[uv > 0] = 0
            return denormed_x

        if infer:
            midi_notes = kwargs.get("midi_notes", None)
            if midi_notes is not None:
                midi_notes = midi_notes.transpose(-1, -2)
                lower_bound = midi_notes - 3  # 1 for good gtdur F0RMSE
                upper_bound = midi_notes + 3  # 1 for good gtdur F0RMSE
                upper_norm_f0 = minmax_norm((2 ** ((upper_bound - 69) / 12) * 440).log2())
                lower_norm_f0 = minmax_norm((2 ** ((lower_bound - 69) / 12) * 440).log2())
                upper_norm_f0[upper_norm_f0 < -1] = -1
                upper_norm_f0[upper_norm_f0 > 1] = 1
                lower_norm_f0[lower_norm_f0 < -1] = -1
                lower_norm_f0[lower_norm_f0 > 1] = 1
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
                uv[midi_notes[:, 0, :] == 0] = 1
            f0 = minmax_denorm(f0)
            ret["gdiff"] = 0.0
            ret["mdiff"] = 0.0
        else:
            if mel2ph is not None:
                nonpadding = (mel2ph > 0).float()
            else:
                nonpadding = (ret["content"] != self.content_padding_idx).float()
            norm_f0 = minmax_norm(f0)
            ret["mdiff"], ret["gdiff"], ret["nll"] = self.f0_gen(
                decoder_inp.transpose(-1, -2),
                norm_f0.unsqueeze(dim=1),
                uv,
                nonpadding,
                ret,
                infer,
            )
        return f0, uv

    def forward_decoder(self, decoder_inp, tgt_nonpadding, ret, infer, **kwargs):
        x = decoder_inp  # [B, T, H]
        if self.decoder_style_adapter is None or not isinstance(self.decoder, CausalConvBlocks):
            x = self.decoder(x)
            x = self.mel_out(x)
            return x

        decoder_style_bundle = ret.get("decoder_style_bundle")
        if bool(ret.get("style_mainline", {}).get("enforce_decoder_no_timing_writeback", True)):
            validate_decoder_style_bundle(decoder_style_bundle)

        x_bct = x.transpose(1, 2)
        nonpadding = tgt_nonpadding.transpose(1, 2)
        stage_end_indices = self.decoder_style_adapter.resolve_stage_end_indices(len(self.decoder.res_blocks))
        stage_outputs = {}
        for block_idx, block in enumerate(self.decoder.res_blocks):
            x_bct = block(x_bct) * nonpadding
            stage_name = stage_end_indices.get(block_idx)
            if stage_name is None:
                continue
            x_btc = x_bct.transpose(1, 2)
            x_btc, stage_meta = self.decoder_style_adapter.forward_stage(
                stage_name,
                x_btc,
                style_bundle=decoder_style_bundle,
                nonpadding=tgt_nonpadding,
            )
            stage_outputs[stage_name] = stage_meta
            x_bct = x_btc.transpose(1, 2)
        x_bct = self.decoder.last_norm(x_bct) * nonpadding
        ret["decoder_style_adapter_stages"] = stage_outputs
        ret["decoder_hidden"] = x_bct.transpose(1, 2)
        x_bct = self.decoder.post_net1(x_bct) * nonpadding
        x = x_bct.transpose(1, 2)
        x = self.mel_out(x)
        return x


class ConanPostnet(nn.Module):
    def __init__(self):
        super().__init__()
        cond_hs = 80 + hparams["hidden_size"]

        self.ln_proj = nn.Linear(cond_hs, hparams["hidden_size"])
        self.postflow = FlowMel(
            out_dims=80,
            denoise_fn=Flow_DECODERS[hparams["flow_decoder_type"]](hparams),
            timesteps=hparams["timesteps"],
            K_step=hparams["K_step"],
            loss_type=hparams["flow_loss_type"],
            spec_min=hparams["spec_min"],
            spec_max=hparams["spec_max"],
        )

    def forward(self, tgt_mels, infer, ret, cfg=False, cfg_scale=1.0, noise=None):
        g = self.get_condition(ret)
        x_recon = ret["mel_out"]
        ucond = None
        if cfg and infer:
            B = g.shape[0]
            B_f = B // 2
            if tgt_mels != None:
                tgt_mels = tgt_mels[:B_f]
            x_recon = x_recon[:B_f]
            ucond = g[B_f:]
            g = g[:B_f]
        self.postflow(g, tgt_mels, x_recon, ret, infer, ucond, noise, cfg_scale)

    def get_condition(self, ret):
        x_recon = ret["mel_out"]
        decoder_inp = ret["decoder_inp"]
        g = x_recon.detach()
        B, T, _ = g.shape
        g = torch.cat([g, decoder_inp], dim=-1)
        g = self.ln_proj(g)
        return g


# class TechSinger(RFSinger):
    # def __init__(self, dict_size, hparams, out_dims=None):
    #     super().__init__(dict_size, hparams, out_dims)

    #     cond_hs = 80 + hparams["hidden_size"]
    #     self.ln_proj = nn.Linear(cond_hs, hparams["hidden_size"])
    #     self.postflow = FlowMel(
    #         out_dims=80,
    #         denoise_fn=Flow_DECODERS[hparams["flow_decoder_type"]](hparams),
    #         timesteps=hparams["timesteps"],
    #         K_step=hparams["K_step"],
    #         loss_type=hparams["flow_loss_type"],
    #         spec_min=hparams["spec_min"],
    #         spec_max=hparams["spec_max"],
    #     )

    # def forward(
    #     self,
    #     txt_tokens,
    #     mel2ph=None,
    #     spk_id=None,
    #     f0=None,
    #     uv=None,
    #     note=None,
    #     note_dur=None,
    #     note_type=None,
    #     mix=None,
    #     falsetto=None,
    #     breathy=None,
    #     bubble=None,
    #     strong=None,
    #     weak=None,
    #     pharyngeal=None,
    #     vibrato=None,
    #     glissando=None,
    #     target=None,
    #     cfg=False,
    #     cfg_scale=1.0,
    #     infer=False,
    # ):
    #     ret = {}
    #     encoder_out = self.encoder(txt_tokens)  # [B, T, C]
    #     note_out = self.note_encoder(note, note_dur, note_type)
    #     encoder_out = encoder_out + note_out
    #     src_nonpadding = (txt_tokens > 0).float()[:, :, None]
    #     ret["spk_embed"] = style_embed = self.forward_style_embed(None, spk_id)
    #     tech = self.tech_encoder(
    #         mix, falsetto, breathy, bubble, strong, weak, pharyngeal, vibrato, glissando
    #     )
    #     # add dur
    #     dur_inp = (encoder_out + style_embed) * src_nonpadding
    #     mel2ph = self.forward_dur(dur_inp, mel2ph, txt_tokens, ret)
    #     tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
    #     decoder_inp = expand_states(encoder_out, mel2ph)
    #     in_nonpadding = (mel2ph > 0).float()[:, :, None]

    #     ret["tech"] = tech = expand_states(tech, mel2ph) * tgt_nonpadding
    #     ret["mel2ph"] = mel2ph

    #     # add pitch embed
    #     midi_notes = None
    #     pitch_inp = (decoder_inp + style_embed + tech) * tgt_nonpadding
    #     if infer:
    #         f0, uv = None, None
    #         midi_notes = expand_states(note[:, :, None], mel2ph)
    #     decoder_inp = decoder_inp + self.forward_pitch(
    #         pitch_inp, f0, uv, mel2ph, ret, encoder_out, midi_notes=midi_notes
    #     )
    #     # decoder input
    #     ret["decoder_inp"] = decoder_inp = (
    #         decoder_inp + style_embed + tech
    #     ) * tgt_nonpadding
    #     ret["coarse_mel_out"] = self.forward_decoder(
    #         decoder_inp, tgt_nonpadding, ret, infer=infer
    #     )
    #     ret["tgt_nonpadding"] = tgt_nonpadding

    #     self.forward_post(target, infer, ret, cfg=cfg, cfg_scale=cfg_scale)
    #     return ret

    # def forward_post(self, tgt_mels, infer, ret, cfg=False, cfg_scale=1.0, noise=None):
    #     g = self.get_condition(ret)
    #     x_recon = ret["coarse_mel_out"]
    #     ucond = None
    #     if cfg and infer:
    #         B = g.shape[0]
    #         B_f = B // 2
    #         if tgt_mels != None:
    #             tgt_mels = tgt_mels[:B_f]
    #         x_recon = x_recon[:B_f]
    #         ucond = g[B_f:]
    #         g = g[:B_f]
    #     self.postflow(g, tgt_mels, x_recon, ret, infer, ucond, noise, cfg_scale)

    # def get_condition(self, ret):
    #     x_recon = ret["coarse_mel_out"]
    #     decoder_inp = ret["decoder_inp"]
    #     g = x_recon.detach()
    #     B, T, _ = g.shape
    #     g = torch.cat([g, decoder_inp], dim=-1)
    #     g = self.ln_proj(g)
    #     return g
