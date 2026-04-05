from collections.abc import Mapping

import librosa
from utils.audio import librosa_wav2spec
from utils.commons.hparams import hparams
import numpy as np
import torch

REGISTERED_VOCODERS = {}


def register_vocoder(name):
    def _f(cls):
        REGISTERED_VOCODERS[name] = cls
        return cls

    return _f


def get_vocoder_cls(vocoder_name):
    return REGISTERED_VOCODERS.get(vocoder_name)


class BaseVocoder:
    def __init__(self, *, runtime_hparams=None, local_hparams=None, hparams_override=None):
        resolved_local_hparams = None
        for candidate in (runtime_hparams, local_hparams, hparams_override):
            if isinstance(candidate, Mapping):
                resolved_local_hparams = dict(candidate)
                break
        self.local_hparams = resolved_local_hparams

    def _hp(self, key, default=None, *, required=False):
        local_hparams = getattr(self, "local_hparams", None)
        if isinstance(local_hparams, Mapping):
            if key in local_hparams:
                return local_hparams[key]
            if required:
                raise KeyError(
                    f"Missing required hparam '{key}' for {self.__class__.__name__} local runtime config."
                )
            return default
        if key in hparams:
            return hparams[key]
        if required:
            raise KeyError(f"Missing required hparam '{key}' for {self.__class__.__name__}.")
        return default

    def _ensure_mel_tensor(self, mel, device, *, num_mels=None):
        if isinstance(mel, torch.Tensor):
            mel_tensor = mel.to(device=device, dtype=torch.float32)
        else:
            mel_tensor = torch.as_tensor(mel, dtype=torch.float32, device=device)
        if mel_tensor.dim() == 2:
            mel_tensor = mel_tensor.unsqueeze(0)
        if mel_tensor.dim() != 3:
            raise ValueError(f"Expected mel with shape [T, M], [B, T, M], or [B, M, T], got {tuple(mel_tensor.shape)}.")
        resolved_num_mels = int(
            self._hp(
                'audio_num_mel_bins',
                mel_tensor.size(-1) if mel_tensor.size(-1) > 0 else mel_tensor.size(1),
            )
            if num_mels is None
            else num_mels
        )
        if mel_tensor.size(-1) == resolved_num_mels:
            mel_tensor = mel_tensor.transpose(1, 2)
        elif mel_tensor.size(1) != resolved_num_mels:
            raise ValueError(
                f"Could not infer mel axis for tensor shape {tuple(mel_tensor.shape)} with num_mels={resolved_num_mels}."
            )
        return mel_tensor.contiguous()

    def supports_native_streaming(self):
        return False

    def reset_stream(self):
        return None

    def spec2wav_stream(self, mel, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide native streaming vocoder inference."
        )

    def spec2wav(self, mel):
        """

        :param mel: [T, 80]
        :return: wav: [T']
        """

        raise NotImplementedError

    @staticmethod
    def wav2spec(wav_fn):
        """

        :param wav_fn: str
        :return: wav, mel: [T, 80]
        """
        wav_spec_dict = librosa_wav2spec(wav_fn, fft_size=hparams['fft_size'],
                                         hop_size=hparams['hop_size'],
                                         win_length=hparams['win_size'],
                                         num_mels=hparams['audio_num_mel_bins'],
                                         fmin=hparams['fmin'],
                                         fmax=hparams['fmax'],
                                         sample_rate=hparams['audio_sample_rate'],
                                         loud_norm=hparams['loud_norm'])
        wav = wav_spec_dict['wav']
        mel = wav_spec_dict['mel']
        return wav, mel

    @staticmethod
    def wav2mfcc(wav_fn):
        fft_size = hparams['fft_size']
        hop_size = hparams['hop_size']
        win_length = hparams['win_size']
        sample_rate = hparams['audio_sample_rate']
        wav, _ = librosa.core.load(wav_fn, sr=sample_rate)
        mfcc = librosa.feature.mfcc(y=wav, sr=sample_rate, n_mfcc=13,
                                    n_fft=fft_size, hop_length=hop_size,
                                    win_length=win_length, pad_mode="constant", power=1.0)
        mfcc_delta = librosa.feature.delta(mfcc, order=1)
        mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)
        mfcc = np.concatenate([mfcc, mfcc_delta, mfcc_delta_delta]).T
        return mfcc
