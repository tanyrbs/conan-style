import torch
from modules.vocoder.hifigan.hifigan_nsf import HifiGanGenerator
from tasks.tts.vocoder_infer.base_vocoder import register_vocoder, BaseVocoder
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams, hparams
from utils.commons.meters import Timer
import numpy as np
import librosa
import json
import glob
import re
import os

def denoise(wav, *, fft_size, hop_size, win_size, v=0.1):
    spec = librosa.stft(y=wav, n_fft=fft_size, hop_length=hop_size,
                        win_length=win_size, pad_mode='constant')
    spec_m = np.abs(spec)
    spec_m = np.clip(spec_m - v, a_min=0, a_max=None)
    spec_a = np.angle(spec)

    return librosa.istft(spec_m * np.exp(1j * spec_a), hop_length=hop_size,
                         win_length=win_size)

def load_model(config_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dict = torch.load(checkpoint_path, map_location="cpu")
    if '.yaml' in config_path:
        config = set_hparams(config_path, global_hparams=False)
        state = ckpt_dict["state_dict"]["model_gen"]
    elif '.json' in config_path:
        config = json.load(open(config_path, 'r'))
        state = ckpt_dict["generator"]

    model = HifiGanGenerator(config)
    model.load_state_dict(state, strict=True)
    model.remove_weight_norm()
    model = model.eval().to(device)
    print(f"| Loaded model parameters from {checkpoint_path}.")
    print(f"| HifiGAN device: {device}.")
    return model, config, device


total_time = 0


def _resolve_vocoder_left_context_frames(config):
    for key in (
        'vocoder_left_context_frames',
        'streaming_vocoder_left_context_frames',
        'vocoder_stream_context',
    ):
        value = config.get(key, None)
        if value is None:
            continue
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 48


@register_vocoder('HifiGAN_NSF')
class HifiGAN(BaseVocoder):
    def __init__(self, runtime_hparams=None, *, local_hparams=None, hparams_override=None):
        super().__init__(
            runtime_hparams=runtime_hparams,
            local_hparams=local_hparams,
            hparams_override=hparams_override,
        )
        base_dir = self._hp('vocoder_ckpt', required=True)
        config_path = f'{base_dir}/config.yaml'
        self.config = {}
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if os.path.exists(config_path):
            ckpt_pattern = rf'{re.escape(base_dir)}/model_ckpt_steps_(\d+)\.ckpt'
            ckpt = sorted(glob.glob(f'{base_dir}/model_ckpt_steps_*.ckpt'), key=
            lambda x: int(re.findall(ckpt_pattern, x)[0]))[-1]
            print('| load HifiGAN: ', ckpt)
            self.model, self.config, self.device = load_model(config_path=config_path, checkpoint_path=ckpt)
        else:
            config_path = f'{base_dir}/config.json'
            ckpt = f'{base_dir}/generator_v1'
            if os.path.exists(config_path):
                self.model, self.config, self.device = load_model(config_path=config_path, checkpoint_path=ckpt)
        if self.model is None:
            raise FileNotFoundError(
                f"HifiGAN_NSF expected config.yaml or config.json under '{base_dir}'."
            )
        context_config = getattr(self, 'local_hparams', None) or hparams
        self.stream_context = _resolve_vocoder_left_context_frames(context_config)
        self.reset_stream()

    def supports_native_streaming(self):
        return False

    def reset_stream(self):
        """Call before starting a new utterance to clear buffer"""
        # The number of mel bins is usually 80
        self.mel_buffer = np.zeros(
            (self.stream_context, self._resolve_num_mels()),
            dtype=np.float32,
        )

    def spec2wav_stream(self, mel, **kwargs):
        raise NotImplementedError(
            "The shipped HifiGAN_NSF wrapper is still exposed as a stateless vocoder path and "
            "does not provide a native streaming contract."
        )

    def spec2wav(self, mel, **kwargs):
        device = self.device
        with torch.no_grad():
            c = self._ensure_mel_tensor(
                mel,
                device,
                num_mels=self._resolve_num_mels(),
            )
            f0 = kwargs.get('f0')
            if f0 is not None and bool(self._hp('use_nsf', False)):
                if not isinstance(f0, torch.Tensor):
                    f0 = torch.as_tensor(f0, dtype=torch.float32, device=device)
                else:
                    f0 = f0.to(device=device, dtype=torch.float32)
                if f0.dim() == 1:
                    f0 = f0.unsqueeze(0)
                y = self.model(c, f0).view(-1)
            else:
                y = self.model(c).view(-1)
        wav_out = y.cpu().numpy()
        denoise_c = float(self._hp('vocoder_denoise_c', 0.0) or 0.0)
        if denoise_c > 0:
            wav_out = denoise(
                wav_out,
                fft_size=self._resolve_fft_size(),
                hop_size=self._resolve_hop_size(),
                win_size=self._resolve_win_size(),
                v=denoise_c,
            )
        return wav_out

    def _resolve_num_mels(self):
        config_num_mels = (
            self.config.get('audio_num_mel_bins', self.config.get('num_mels', 80))
            if isinstance(self.config, dict)
            else 80
        )
        return int(self._hp('audio_num_mel_bins', config_num_mels))

    def _resolve_fft_size(self):
        config_fft_size = self.config.get('fft_size', 1024) if isinstance(self.config, dict) else 1024
        return int(self._hp('fft_size', config_fft_size))

    def _resolve_hop_size(self):
        config_hop_size = self.config.get('hop_size', 256) if isinstance(self.config, dict) else 256
        return int(self._hp('hop_size', config_hop_size))

    def _resolve_win_size(self):
        config_fft_size = self._resolve_fft_size()
        config_win_size = self.config.get('win_size', config_fft_size) if isinstance(self.config, dict) else config_fft_size
        return int(self._hp('win_size', config_win_size))

    
