import torch
import torch.nn as nn


class PromptAxisAdaptor(nn.Module):
    def __init__(self, hidden_size, gate_hidden=None, use_gate=True):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.use_gate = bool(use_gate)
        gate_hidden = int(gate_hidden or hidden_size)
        self.prompt_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        if self.use_gate:
            self.gate = nn.Sequential(
                nn.Linear(hidden_size * 2, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, hidden_size),
                nn.Sigmoid(),
            )
            self._init_gate_bias()
        else:
            self.gate = None

    def _init_gate_bias(self):
        if not self.use_gate:
            return
        last_linear = None
        for layer in reversed(self.gate):
            if isinstance(layer, nn.Linear):
                last_linear = layer
                break
        if last_linear is not None and last_linear.bias is not None:
            nn.init.constant_(last_linear.bias, 2.0)

    def project(self, prompt_vec):
        if prompt_vec is None:
            return None
        return self.prompt_proj(prompt_vec)

    def fuse(self, reference, prompt_vec, strength, *, projected=False):
        if not projected:
            prompt_vec = self.project(prompt_vec)
        if prompt_vec is None:
            return None, None, None
        prompt_expand = prompt_vec[:, None, :].expand(-1, reference.size(1), -1)
        if self.use_gate:
            gate = self.gate(torch.cat([reference, prompt_expand], dim=-1))
        else:
            gate = torch.ones_like(prompt_expand)
        fused = prompt_expand * gate * strength
        return fused, gate, prompt_vec


class PromptControlAdapter(nn.Module):
    def __init__(
        self,
        hidden_size,
        *,
        emotion_gate_hidden=None,
        accent_gate_hidden=None,
        use_emotion_gate=True,
        use_accent_gate=True,
    ):
        super().__init__()
        self.emotion = PromptAxisAdaptor(
            hidden_size,
            gate_hidden=emotion_gate_hidden,
            use_gate=use_emotion_gate,
        )
        self.accent = PromptAxisAdaptor(
            hidden_size,
            gate_hidden=accent_gate_hidden,
            use_gate=use_accent_gate,
        )

    def _axis(self, axis: str):
        if axis == "emotion":
            return self.emotion
        if axis == "accent":
            return self.accent
        raise ValueError(f"Unknown prompt control axis: {axis}")

    def project(self, axis: str, prompt_vec):
        return self._axis(axis).project(prompt_vec)

    def fuse(self, axis: str, reference, prompt_vec, strength, *, projected=False):
        return self._axis(axis).fuse(reference, prompt_vec, strength, projected=projected)
