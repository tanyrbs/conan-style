import torch.nn as nn

from modules.Conan.control.common import squeeze_prompt_vector


class PromptAttributeHeads(nn.Module):
    def __init__(
        self,
        hidden_size,
        *,
        num_emotions=0,
        num_accents=0,
        predict_prompt_emotion=True,
        predict_prompt_accent=True,
        predict_prompt_arousal=True,
        predict_prompt_valence=True,
    ):
        super().__init__()
        self.emotion_classifier = (
            nn.Linear(hidden_size, int(num_emotions))
            if int(num_emotions) > 1 and bool(predict_prompt_emotion)
            else None
        )
        self.accent_classifier = (
            nn.Linear(hidden_size, int(num_accents))
            if int(num_accents) > 1 and bool(predict_prompt_accent)
            else None
        )
        self.arousal_predictor = (
            nn.Linear(hidden_size, 1) if bool(predict_prompt_arousal) else None
        )
        self.valence_predictor = (
            nn.Linear(hidden_size, 1) if bool(predict_prompt_valence) else None
        )

    def forward(self, *, emotion_prompt=None, accent_prompt=None):
        outputs = {}

        emotion_prompt = squeeze_prompt_vector(emotion_prompt)
        accent_prompt = squeeze_prompt_vector(accent_prompt)

        if emotion_prompt is not None:
            if self.emotion_classifier is not None:
                outputs["emotion_prompt_logits"] = self.emotion_classifier(emotion_prompt)
            if self.arousal_predictor is not None:
                outputs["emotion_prompt_arousal_pred"] = self.arousal_predictor(emotion_prompt).squeeze(-1)
            if self.valence_predictor is not None:
                outputs["emotion_prompt_valence_pred"] = self.valence_predictor(emotion_prompt).squeeze(-1)

        if accent_prompt is not None and self.accent_classifier is not None:
            outputs["accent_prompt_logits"] = self.accent_classifier(accent_prompt)

        return outputs
