import torch
from modules.vocoder.hifigan.hifigan_causal import HifiGanGenerator
from tasks.tts.vocoder_infer.base_vocoder import register_vocoder, BaseVocoder
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams, hparams
from utils.commons.meters import Timer

total_time = 0


@register_vocoder('HifiGAN')
class HifiGAN(BaseVocoder):
    def __init__(self, runtime_hparams=None, *, local_hparams=None, hparams_override=None):
        super().__init__(
            runtime_hparams=runtime_hparams,
            local_hparams=local_hparams,
            hparams_override=hparams_override,
        )
        base_dir = self._hp('vocoder_ckpt', required=True)
        config_path = f'{base_dir}/config.yaml'
        self.config = config = set_hparams(config_path, global_hparams=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HifiGanGenerator(config)
        load_ckpt(self.model, base_dir, 'model_gen')
        self.model.to(self.device)
        self.model.eval()

    def supports_native_streaming(self):
        return False

    def reset_stream(self):
        return None

    def spec2wav_stream(self, mel, **kwargs):
        raise NotImplementedError(
            "The shipped HifiGAN wrapper is stateless and does not expose native streaming inference."
        )

    def spec2wav(self, mel, **kwargs):
        device = self.device
        with torch.no_grad():
            c = self._ensure_mel_tensor(
                mel,
                device,
                num_mels=int(self._hp('audio_num_mel_bins', self.config.get('audio_num_mel_bins', 80))),
            )
            with Timer('hifigan', enable=bool(self._hp('profile_infer', False))):
                y = self.model(c).view(-1)
        wav_out = y.cpu().numpy()
        return wav_out
    
