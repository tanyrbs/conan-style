from tasks.tts.dataset_utils import FastSpeechDataset
import torch
import math
import warnings
from modules.Conan.reference_bundle import canonicalize_reference_bundle, normalize_reference_contract_mode
from modules.Conan.style_mainline import sanitize_mainline_style_strength
from modules.Conan.style_profiles import resolve_style_profile
from utils.commons.dataset_utils import collate_1d_or_2d


def _align_framewise_sample(
    sample,
    *,
    max_trim_frames=None,
    max_trim_ratio=None,
):
    if not isinstance(sample, dict):
        return sample
    mel = sample.get("mel")
    content = sample.get("content")
    if not isinstance(mel, torch.Tensor) or not isinstance(content, torch.Tensor):
        return sample

    frame_lengths = {
        "mel": int(mel.shape[0]),
        "content": int(content.shape[0]),
    }
    for key in ("f0", "uv", "energy"):
        value = sample.get(key)
        if isinstance(value, torch.Tensor):
            frame_lengths[key] = int(value.shape[0])

    common_length = min(frame_lengths.values())
    if common_length <= 0:
        item_name = sample.get("item_name", "<unknown>")
        raise ValueError(
            f"ConanDataset item '{item_name}' resolved a non-positive aligned frame length: {frame_lengths}"
        )

    max_length = max(frame_lengths.values())
    trimmed_frames = int(max_length - common_length)
    trim_ratio = float(trimmed_frames) / float(max_length) if max_length > 0 else 0.0
    item_name = sample.get("item_name", "<unknown>")
    if max_trim_frames is not None and trimmed_frames > int(max_trim_frames):
        raise ValueError(
            f"ConanDataset item '{item_name}' would trim {trimmed_frames} frame(s), exceeding "
            f"dataset_max_frame_trim={int(max_trim_frames)}. frame_lengths={frame_lengths}"
        )
    if max_trim_ratio is not None and trim_ratio > float(max_trim_ratio):
        raise ValueError(
            f"ConanDataset item '{item_name}' would trim {trim_ratio:.3f} of frames, exceeding "
            f"dataset_max_frame_trim_ratio={float(max_trim_ratio):.3f}. frame_lengths={frame_lengths}"
        )

    for key in ("mel", "content", "mel_nonpadding", "f0", "uv", "energy"):
        value = sample.get(key)
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            sample[key] = value[:common_length]
    sample["mel_nonpadding"] = sample["mel"].abs().sum(-1) > 0
    sample["frame_alignment_trimmed"] = bool(trimmed_frames > 0)
    sample["frame_alignment_trimmed_frames"] = int(trimmed_frames)
    sample["frame_alignment_trimmed_ratio"] = float(trim_ratio)
    sample["frame_alignment_lengths"] = dict(frame_lengths)
    return sample


def _scalar_long(value, default=-1):
    if value is None:
        value = default
    return torch.tensor(int(value), dtype=torch.long)


def _scalar_float(value, default=0.0):
    if value is None:
        value = default
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    return torch.tensor(value, dtype=torch.float32)


def _resolve_default_style_strength(hparams):
    try:
        resolved = resolve_style_profile(
            {
                "style_profile": hparams.get("style_profile", "strong_style"),
            },
            preset=hparams.get("style_profile", "strong_style"),
        )
        return float(resolved.get("style_strength", 1.0))
    except Exception:
        return 1.0


class ConanDataset(FastSpeechDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_style_strength = _resolve_default_style_strength(self.hparams)
        self.allow_item_style_strength_override = bool(
            self.hparams.get("allow_item_style_strength_override", False)
        )
        self.dataset_warn_on_frame_trim = bool(
            self.hparams.get("dataset_warn_on_frame_trim", False)
        )
        self.dataset_warn_on_frame_trim_max_events = max(
            0,
            int(self.hparams.get("dataset_warn_on_frame_trim_max_events", 8)),
        )
        self._frame_trim_warned_events = 0

    def _resolve_item_style_strength(self, item):
        style_strength_value = self.default_style_strength
        if self.allow_item_style_strength_override and "style_strength" in item:
            style_strength_value = item.get("style_strength", self.default_style_strength)
        return sanitize_mainline_style_strength(
            style_strength_value,
            default=self.default_style_strength,
        )

    def __getitem__(self, index):
        sample = super(ConanDataset, self).__getitem__(index)
        item = self._get_item(index)

        sample["content"] = torch.as_tensor(item["hubert"], dtype=torch.long).reshape(-1)
        sample["emotion_id"] = _scalar_long(item.get("emotion_id", -1), default=-1)
        sample["accent_id"] = _scalar_long(item.get("accent_id", -1), default=-1)
        sample["arousal"] = _scalar_float(item.get("arousal", 0.0), default=0.0)
        sample["valence"] = _scalar_float(item.get("valence", 0.0), default=0.0)

        if "energy" in item:
            energy = torch.as_tensor(item["energy"], dtype=torch.float32).reshape(-1)
        else:
            energy = sample["mel"].abs().mean(dim=-1)
        sample["energy"] = energy
        sample["style_strength"] = _scalar_float(
            self._resolve_item_style_strength(item),
            default=self.default_style_strength,
        )
        sample["emotion_strength"] = _scalar_float(item.get("emotion_strength", 1.0), default=1.0)
        sample["accent_strength"] = _scalar_float(item.get("accent_strength", 1.0), default=1.0)
        aligned = _align_framewise_sample(
            sample,
            max_trim_frames=self.hparams.get("dataset_max_frame_trim", None),
            max_trim_ratio=self.hparams.get("dataset_max_frame_trim_ratio", None),
        )
        self._maybe_warn_about_frame_trim(aligned)
        return aligned

    def _maybe_warn_about_frame_trim(self, sample):
        if not self.dataset_warn_on_frame_trim:
            return
        if not bool(sample.get("frame_alignment_trimmed", False)):
            return
        if self._frame_trim_warned_events >= self.dataset_warn_on_frame_trim_max_events:
            return
        self._frame_trim_warned_events += 1
        warnings.warn(
            "ConanDataset trimmed misaligned framewise features for "
            f"item '{sample.get('item_name', '<unknown>')}' by "
            f"{int(sample.get('frame_alignment_trimmed_frames', 0))} frame(s) "
            f"({float(sample.get('frame_alignment_trimmed_ratio', 0.0)):.3f}). "
            f"frame_lengths={sample.get('frame_alignment_lengths', {})}",
            stacklevel=2,
        )

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(ConanDataset, self).collater(samples)
        contract_mode = normalize_reference_contract_mode(
            self.hparams.get("reference_contract_mode", "collapsed_reference")
        )
        batch["reference_contract_mode"] = contract_mode
        batch["allow_split_reference_inputs"] = bool(
            self.hparams.get("allow_split_reference_inputs", False)
        )
        batch["ref_timbre_mels"] = batch.get("ref_timbre_mels", batch["ref_mels"])
        raw_reference_bundle = {
            "ref": batch["ref_mels"],
            "ref_emotion": batch.get("emotion_ref_mels", None),
            "ref_accent": batch.get("accent_ref_mels", None),
            "reference_contract_mode": contract_mode,
        }
        normalized_reference_bundle = canonicalize_reference_bundle(
            raw_reference_bundle,
            default_ref=batch["ref_mels"],
            contract_mode=contract_mode,
        )
        batch["reference_contract"] = normalized_reference_bundle.get("reference_contract", {})
        if bool(self.hparams.get("emit_collapsed_reference_aliases", False)):
            batch["style_ref_mels"] = batch["ref_mels"]
            batch["dynamic_timbre_ref_mels"] = batch["ref_mels"]
            batch["ref_style_mels"] = batch["ref_mels"]
            batch["ref_dynamic_timbre_mels"] = batch["ref_mels"]
        if "emotion_ref_mels" in batch:
            batch["ref_emotion_mels"] = batch["emotion_ref_mels"]
        if "accent_ref_mels" in batch:
            batch["ref_accent_mels"] = batch["accent_ref_mels"]
        batch["reference_bundle"] = normalized_reference_bundle
        content_padding_idx = int(self.hparams.get("content_padding_idx", 101))
        batch["content"] = collate_1d_or_2d(
            [sample["content"] for sample in samples], content_padding_idx
        ).long()
        batch["content_lengths"] = torch.tensor(
            [sample["content"].numel() for sample in samples],
            dtype=torch.long,
        )
        batch["emotion_ids"] = torch.stack([sample["emotion_id"] for sample in samples]).long()
        batch["accent_ids"] = torch.stack([sample["accent_id"] for sample in samples]).long()
        batch["arousal"] = torch.stack([sample["arousal"] for sample in samples]).float()
        batch["valence"] = torch.stack([sample["valence"] for sample in samples]).float()
        batch["energy"] = collate_1d_or_2d([sample["energy"] for sample in samples], 0.0).float()
        batch["style_strengths"] = torch.stack([sample["style_strength"] for sample in samples]).float()
        batch["emotion_strengths"] = torch.stack([sample["emotion_strength"] for sample in samples]).float()
        batch["accent_strengths"] = torch.stack([sample["accent_strength"] for sample in samples]).float()
        return batch
