from __future__ import annotations

from pathlib import Path

import numpy as np


def _safe_float(value, default=None):
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(value) or np.isinf(value):
        return default
    return value


class ResemblyzerSpeakerMetric:
    def __init__(self):
        from resemblyzer import VoiceEncoder

        self.encoder = VoiceEncoder()
        self._cache = {}

    def _embed(self, wav_path):
        wav_path = str(wav_path)
        if wav_path in self._cache:
            return self._cache[wav_path]
        from resemblyzer import preprocess_wav

        wav = preprocess_wav(wav_path)
        embed = self.encoder.embed_utterance(wav)
        self._cache[wav_path] = embed
        return embed

    def similarity(self, wav_a, wav_b):
        embed_a = self._embed(wav_a)
        embed_b = self._embed(wav_b)
        denom = np.linalg.norm(embed_a) * np.linalg.norm(embed_b)
        if denom <= 1e-8:
            return None
        return _safe_float(np.dot(embed_a, embed_b) / denom, default=None)


class TorchaudioContentMetric:
    def __init__(self, bundle_name: str = "WAV2VEC2_BASE"):
        import torch
        import torchaudio

        self.torch = torch
        self.torchaudio = torchaudio
        if not hasattr(torchaudio.pipelines, bundle_name):
            raise ValueError(f"Unknown torchaudio bundle: {bundle_name}")
        self.bundle = getattr(torchaudio.pipelines, bundle_name)
        self.model = self.bundle.get_model().eval()
        self.sample_rate = int(self.bundle.sample_rate)
        self._cache = {}

    def _load(self, wav_path):
        wav_path = str(wav_path)
        if wav_path in self._cache:
            return self._cache[wav_path]
        waveform, sample_rate = self.torchaudio.load(wav_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if int(sample_rate) != self.sample_rate:
            waveform = self.torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        with self.torch.no_grad():
            features, _ = self.model.extract_features(waveform)
        summary = features[-1].mean(dim=1).squeeze(0).cpu()
        self._cache[wav_path] = summary
        return summary

    def similarity(self, wav_a, wav_b):
        a = self._load(wav_a)
        b = self._load(wav_b)
        denom = a.norm().item() * b.norm().item()
        if denom <= 1e-8:
            return None
        return _safe_float(float((a @ b).item() / denom), default=None)


class ExternalMetricSuite:
    def __init__(
        self,
        *,
        enable_speaker: bool = True,
        enable_content: bool = False,
    ):
        self.speaker_metric = None
        self.content_metric = None
        self.init_errors = {}

        if enable_speaker:
            try:
                self.speaker_metric = ResemblyzerSpeakerMetric()
            except Exception as e:
                self.init_errors["speaker"] = str(e)
        if enable_content:
            try:
                self.content_metric = TorchaudioContentMetric()
            except Exception as e:
                self.init_errors["content"] = str(e)

    @property
    def available(self):
        return self.speaker_metric is not None or self.content_metric is not None

    def evaluate(
        self,
        *,
        gen_path,
        src_path=None,
        timbre_ref_path=None,
        style_ref_path=None,
        dynamic_timbre_ref_path=None,
    ):
        metrics = {}
        if self.speaker_metric is not None:
            if timbre_ref_path is not None:
                metrics["ext_speaker_cos_to_timbre_ref"] = self.speaker_metric.similarity(gen_path, timbre_ref_path)
            if src_path is not None:
                metrics["ext_speaker_cos_to_src"] = self.speaker_metric.similarity(gen_path, src_path)
        if self.content_metric is not None and src_path is not None:
            metrics["ext_content_ssl_cos_to_src"] = self.content_metric.similarity(gen_path, src_path)
            if style_ref_path is not None:
                metrics["ext_content_ssl_cos_to_style_ref"] = self.content_metric.similarity(gen_path, style_ref_path)
            if dynamic_timbre_ref_path is not None:
                metrics["ext_content_ssl_cos_to_dynamic_timbre_ref"] = self.content_metric.similarity(
                    gen_path, dynamic_timbre_ref_path
                )
        return {key: value for key, value in metrics.items() if value is not None}


__all__ = [
    "ExternalMetricSuite",
    "ResemblyzerSpeakerMetric",
    "TorchaudioContentMetric",
]
