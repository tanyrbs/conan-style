from tasks.tts.dataset_utils import FastSpeechDataset
import torch
from modules.Conan.reference_bundle import canonicalize_reference_bundle, normalize_reference_contract_mode
from utils.commons.dataset_utils import collate_1d_or_2d


def _scalar_long(value, default=-1):
    if value is None:
        value = default
    return torch.tensor(int(value), dtype=torch.long)


def _scalar_float(value, default=0.0):
    if value is None:
        value = default
    return torch.tensor(float(value), dtype=torch.float32)


class ConanDataset(FastSpeechDataset):
    def __getitem__(self, index):
        hparams = self.hparams
        sample = super(ConanDataset, self).__getitem__(index)
        item = self._get_item(index)

        sample["content"] = torch.LongTensor(item["hubert"])
        sample["emotion_id"] = _scalar_long(item.get("emotion_id", -1), default=-1)
        sample["accent_id"] = _scalar_long(item.get("accent_id", -1), default=-1)
        sample["arousal"] = _scalar_float(item.get("arousal", 0.0), default=0.0)
        sample["valence"] = _scalar_float(item.get("valence", 0.0), default=0.0)

        if "energy" in item:
            energy = torch.as_tensor(item["energy"], dtype=torch.float32)
        else:
            energy = sample["mel"].abs().mean(dim=-1)
        sample["energy"] = energy[: sample["mel"].shape[0]]
        sample["style_strength"] = _scalar_float(item.get("style_strength", 1.0), default=1.0)
        sample["emotion_strength"] = _scalar_float(item.get("emotion_strength", 1.0), default=1.0)
        sample["accent_strength"] = _scalar_float(item.get("accent_strength", 1.0), default=1.0)
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(ConanDataset, self).collater(samples)
        contract_mode = normalize_reference_contract_mode(
            self.hparams.get("reference_contract_mode", "collapsed_reference")
        )
        batch["reference_contract_mode"] = contract_mode
        batch["ref_timbre_mels"] = batch.get("ref_timbre_mels", batch["ref_mels"])
        batch["style_ref_mels"] = batch["ref_mels"]
        batch["dynamic_timbre_ref_mels"] = batch["ref_mels"]
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
        batch["emotion_ids"] = torch.stack([sample["emotion_id"] for sample in samples]).long()
        batch["accent_ids"] = torch.stack([sample["accent_id"] for sample in samples]).long()
        batch["arousal"] = torch.stack([sample["arousal"] for sample in samples]).float()
        batch["valence"] = torch.stack([sample["valence"] for sample in samples]).float()
        batch["energy"] = collate_1d_or_2d([sample["energy"] for sample in samples], 0.0).float()
        batch["style_strengths"] = torch.stack([sample["style_strength"] for sample in samples]).float()
        batch["emotion_strengths"] = torch.stack([sample["emotion_strength"] for sample in samples]).float()
        batch["accent_strengths"] = torch.stack([sample["accent_strength"] for sample in samples]).float()
        return batch
