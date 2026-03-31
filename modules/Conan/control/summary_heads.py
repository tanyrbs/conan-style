import torch.nn as nn


class ReferenceSummaryHeads(nn.Module):
    def __init__(
        self,
        hidden_size,
        *,
        num_emotions=0,
        num_styles=0,
        num_accents=0,
        predict_arousal=True,
        predict_valence=True,
    ):
        super().__init__()
        self.emotion_classifier = nn.Linear(hidden_size, int(num_emotions)) if int(num_emotions) > 1 else None
        self.style_classifier = nn.Linear(hidden_size, int(num_styles)) if int(num_styles) > 1 else None
        self.accent_classifier = nn.Linear(hidden_size, int(num_accents)) if int(num_accents) > 1 else None
        self.arousal_predictor = nn.Linear(hidden_size, 1) if bool(predict_arousal) else None
        self.valence_predictor = nn.Linear(hidden_size, 1) if bool(predict_valence) else None

    def forward(self, summary):
        outputs = {"reference_summary": summary}
        if self.emotion_classifier is not None:
            outputs["emotion_logits"] = self.emotion_classifier(summary)
        if self.style_classifier is not None:
            outputs["style_logits"] = self.style_classifier(summary)
        if self.accent_classifier is not None:
            outputs["accent_logits"] = self.accent_classifier(summary)
        if self.arousal_predictor is not None:
            outputs["arousal_pred"] = self.arousal_predictor(summary).squeeze(-1)
        if self.valence_predictor is not None:
            outputs["valence_pred"] = self.valence_predictor(summary).squeeze(-1)
        return outputs
