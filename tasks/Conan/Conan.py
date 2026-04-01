from modules.Conan.Conan import Conan, ConanPostnet
from tasks.Conan.base_gen_task import AuxDecoderMIDITask, f0_to_figure
from utils.commons.hparams import hparams
import torch
from utils.commons.ckpt_utils import load_ckpt
from tasks.Conan.dataset import ConanDataset
import torch.nn.functional as F
from utils.commons.tensor_utils import tensors_to_scalars
from utils.audio.pitch.utils import denorm_f0
import math
from modules.tts.iclspeech.multi_window_disc import Discriminator
import torch.nn as nn
import random
from tasks.Conan.control_diagnostics import collect_control_diagnostics
from tasks.Conan.control_schedule import resolve_control_regularization_config
from tasks.Conan.style_control_mixin import ConanStyleControlMixin
from tasks.Conan.style_batching_mixin import ConanStyleBatchingMixin


class ConanEmbTask(AuxDecoderMIDITask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = ConanDataset

    def build_tts_model(self):
        # dict_size = len(self.token_encoder)
        self.model = Conan(0, hparams)
        self.gen_params = [p for p in self.model.parameters() if p.requires_grad]

    def run_model(self, sample):
        with torch.no_grad():
            ref = sample['mels']
            output = self.model.encode_spk_embed(ref.transpose(1, 2)).squeeze(2)
        return {}, {
            "global_timbre_anchor": output,
        }


class ConanTask(ConanStyleBatchingMixin, ConanStyleControlMixin, AuxDecoderMIDITask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = ConanDataset
        self.mse_loss_fn = torch.nn.MSELoss()
        self.build_disc_model()

    def build_tts_model(self):
        # dict_size = len(self.token_encoder)
        self.model = Conan(0, hparams)
        self.gen_params = [p for p in self.model.parameters() if p.requires_grad]

    def build_disc_model(self):
        disc_win_num = hparams['disc_win_num']
        h = hparams['mel_disc_hidden_size']
        self.mel_disc = Discriminator(
            time_lengths=[32, 64, 128][:disc_win_num],
            freq_length=80, hidden_size=h, kernel=(3, 3)
        )
        self.disc_params = list(self.mel_disc.parameters())

    def drop_multi(self, tech, drop_p):
        if torch.rand(1) < drop_p:
            tech = torch.ones_like(tech, dtype=tech.dtype) * 2
        elif torch.rand(1) < drop_p:
            random_tech = torch.rand_like(tech, dtype=torch.float32)
            tech[random_tech < drop_p] = 2
        return tech
            
    def run_model(self, sample, infer=False, test=False):
        # txt_tokens = sample["txt_tokens"]
        # mel2ph = sample["mel2ph"]
        # spk_id = sample["spk_ids"]
        content = sample["content"]
        if 'spk_embed' in sample:
            spk_embed = sample["spk_embed"]
        else:
            spk_embed=None
        f0, uv = sample.get("f0", None), sample.get("uv", None)
        # notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]

        target = sample["mels"]
        effective_global_step = int(self.global_step)
        if test:
            effective_global_step = max(
                effective_global_step,
                int(hparams.get("random_speaker_steps", 0)),
                200000,
            )
        if effective_global_step >= hparams["random_speaker_steps"]:
            ref = sample.get('ref_mels', target)
        else:
            ref = target
        # assert False, f'content: {content.shape}, target: {target.shape},spk_embed: {spk_embed.shape}'
        # if not infer:
        #     tech_drop = {
        #         'mix': 0.1,
        #         'falsetto': 0.1,
        #         'breathy': 0.1,
        #         'bubble': 0.1,
        #         'strong': 0.1,
        #         'weak': 0.1,
        #         'glissando': 0.1,
        #         'pharyngeal': 0.1,
        #         'vibrato': 0.1,
        #     }
        #     for tech, drop_p in tech_drop.items():
        #         sample[tech] = self.drop_multi(sample[tech], drop_p)
        
        # mix, falsetto, breathy=sample['mix'], sample['falsetto'], sample['breathy']
        # bubble,strong,weak=sample['bubble'],sample['strong'],sample['weak']
        # pharyngeal, vibrato, glissando = sample['pharyngeal'], sample['vibrato'], sample['glissando']
        model_kwargs = self.build_style_model_kwargs(sample, ref)
        output = self.model(content,spk_embed=spk_embed, target=target,ref=ref,
                            f0=f0, uv=uv,
                            infer=infer, global_steps=effective_global_step, **model_kwargs)
        
        losses = {}
        
        if not test:
            self._maybe_attach_output_identity_embeddings(output, ref)
            self.add_mel_loss(output['mel_out'], target, losses)
            # self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
            self.add_pitch_loss(output, sample, losses)
            self.add_control_losses(output, sample, losses)
        
        return losses, output

    def _reference_for_inference(self, sample, *, global_step=None):
        effective_global_step = int(self.global_step if global_step is None else global_step)
        if effective_global_step >= hparams["random_speaker_steps"]:
            return sample.get("ref_mels", sample["mels"])
        return sample["mels"]

    def _run_prefix_online_inference(self, sample, *, tokens_per_chunk=4):
        content_full = sample["content"]
        spk_embed = sample.get("spk_embed", None)
        infer_global_steps = max(int(self.global_step), int(hparams.get("random_speaker_steps", 0)), 200000)
        ref = self._reference_for_inference(sample, global_step=infer_global_steps)
        model_kwargs = self.build_style_model_kwargs(sample, ref)
        with torch.no_grad():
            reference_cache = self.model.prepare_reference_cache(
                reference_bundle=model_kwargs["reference_bundle"],
                spk_embed=spk_embed,
                infer=True,
                global_steps=infer_global_steps,
            )
        model_kwargs["reference_cache"] = reference_cache

        total_tokens = int(content_full.size(1))
        step = max(1, int(tokens_per_chunk))
        prev_mel_len = 0
        mel_chunks = []
        last_output = None
        chunk_count = 0
        with torch.no_grad():
            for end_idx in range(step, total_tokens + step, step):
                chunk_count += 1
                content_chunk = content_full[:, : min(end_idx, total_tokens)]
                last_output = self.model(
                    content_chunk,
                    spk_embed=spk_embed,
                    target=None,
                    ref=ref,
                    f0=None,
                    uv=None,
                    infer=True,
                    global_steps=infer_global_steps,
                    **model_kwargs,
                )
                mel_out = last_output["mel_out"]
                mel_new = mel_out[:, prev_mel_len:, :]
                if mel_new.size(1) > 0:
                    mel_chunks.append(mel_new)
                prev_mel_len = int(mel_out.size(1))
        if last_output is None:
            raise RuntimeError("Prefix-online inference produced no decoder output.")
        if mel_chunks:
            last_output["mel_out"] = torch.cat(mel_chunks, dim=1)
        last_output["streaming_eval_mode"] = "prefix_online_content_chunked"
        last_output["streaming_chunk_tokens"] = int(step)
        last_output["streaming_total_chunks"] = int(chunk_count)
        return last_output

    @staticmethod
    def _streaming_parity_metrics(offline_output, streaming_output):
        if not isinstance(offline_output, dict) or not isinstance(streaming_output, dict):
            return {}
        offline_mel = offline_output.get("mel_out")
        streaming_mel = streaming_output.get("mel_out")
        if not isinstance(offline_mel, torch.Tensor) or not isinstance(streaming_mel, torch.Tensor):
            return {}
        if offline_mel.dim() != 3 or streaming_mel.dim() != 3:
            return {}
        min_len = min(int(offline_mel.size(1)), int(streaming_mel.size(1)))
        if min_len <= 0:
            return {}
        offline_aligned = offline_mel[:, :min_len, :]
        streaming_aligned = streaming_mel[:, :min_len, :]
        metrics = {
            "streaming_mel_l1": F.l1_loss(streaming_aligned, offline_aligned),
            "streaming_mel_l2": F.mse_loss(streaming_aligned, offline_aligned),
        }
        tail_frames = min(min_len, int(hparams.get("streaming_parity_tail_frames", 32)))
        if tail_frames > 0:
            metrics["streaming_tail_mel_l1"] = F.l1_loss(
                streaming_aligned[:, -tail_frames:, :],
                offline_aligned[:, -tail_frames:, :],
            )
        return metrics

    def add_pitch_loss(self, output, sample, losses):
        # mel2ph = sample['mel2ph']  # [B, T_s]
        content = sample['content']
        f0 = sample['f0']
        uv = sample['uv']
        nonpadding = (content != hparams.get('content_padding_idx', 101)).float()
        if hparams["f0_gen"] == "diff":
            losses["fdiff"] = output["fdiff"]
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / nonpadding.sum() * hparams['lambda_uv']
        elif hparams["f0_gen"] == "flow":
            losses["pflow"] = output["pflow"]
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / nonpadding.sum() * hparams['lambda_uv']
        elif hparams["f0_gen"] == "gmdiff":
            losses["gdiff"] = output["gdiff"]
            losses["mdiff"] = output["mdiff"]
        elif hparams["f0_gen"] == "orig":
            losses["fdiff"] = output["fdiff"]
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / nonpadding.sum() * hparams['lambda_uv']

            
    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output = {}
        loss_weights = {}
        disc_start = self.global_step >= hparams["disc_start_steps"] and hparams['lambda_mel_adv'] > 0
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            loss_output, model_out = self.run_model(sample, infer=False)
            self.model_out_gt = self.model_out = \
                {k: v.detach() for k, v in model_out.items() if isinstance(v, torch.Tensor)}
            if disc_start:
                mel_p = model_out['mel_out']
                if hasattr(self.model, 'out2mel'):
                    mel_p = self.model.out2mel(mel_p)
                o_ = self.mel_disc(mel_p)
                p_, pc_ = o_['y'], o_['y_c']
                if p_ is not None:
                    loss_output['a'] = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
                    loss_weights['a'] = hparams['lambda_mel_adv']
                if pc_ is not None:
                    loss_output['ac'] = self.mse_loss_fn(pc_, pc_.new_ones(pc_.size()))
                    loss_weights['ac'] = hparams['lambda_mel_adv']
            loss_output.update(
                collect_control_diagnostics(
                    model_out,
                    sample,
                    resolve_control_regularization_config(hparams, self.global_step),
                )
            )
        else:
            #######################
            #    Discriminator    #
            #######################
            if disc_start and self.global_step % hparams['disc_interval'] == 0:
                model_out = self.model_out_gt
                mel_g = sample['mels']
                mel_p = model_out['mel_out']
                o = self.mel_disc(mel_g)
                p, pc = o['y'], o['y_c']
                o_ = self.mel_disc(mel_p)
                p_, pc_ = o_['y'], o_['y_c']
                if p_ is not None:
                    loss_output["r"] = self.mse_loss_fn(p, p.new_ones(p.size()))
                    loss_output["f"] = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))
                if pc_ is not None:
                    loss_output["rc"] = self.mse_loss_fn(pc, pc.new_ones(pc.size()))
                    loss_output["fc"] = self.mse_loss_fn(pc_, pc_.new_zeros(pc_.size()))
            if len(loss_output) == 0:
                return None
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['content'].size()[0]
        return total_loss, loss_output


    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {"flow": 0}
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            outputs['losses'], offline_out = self.run_model(sample, infer=True)
            plot_out = offline_out
            if bool(hparams.get("valid_use_streaming_prefix_path", True)):
                streaming_out = self._run_prefix_online_inference(
                    sample,
                    tokens_per_chunk=int(hparams.get("streaming_eval_tokens_per_chunk", 4)),
                )
                outputs['losses'].update(self._streaming_parity_metrics(offline_out, streaming_out))
                plot_out = streaming_out
            outputs['total_loss'] = sum(outputs['losses'].values())
            sr = hparams["audio_sample_rate"]
            gt_f0 = denorm_f0(sample['f0'], sample["uv"])
            wav_gt = self.vocoder.spec2wav(sample["mels"][0].cpu().numpy(), f0=gt_f0[0].cpu().numpy())
            self.logger.add_audio(f'wav_gt_{batch_idx}', wav_gt, self.global_step, sr)

            offline_f0_pred = offline_out.get("f0_denorm_pred")
            plot_f0_pred = plot_out.get("f0_denorm_pred")
            if not isinstance(plot_f0_pred, torch.Tensor):
                plot_f0_pred = offline_f0_pred
            plot_f0_np = plot_f0_pred[0].cpu().numpy() if isinstance(plot_f0_pred, torch.Tensor) else None
            wav_pred = self.vocoder.spec2wav(plot_out['mel_out'][0].cpu().numpy(), f0=plot_f0_np)
            if plot_out is not offline_out:
                offline_f0_np = offline_f0_pred[0].cpu().numpy() if isinstance(offline_f0_pred, torch.Tensor) else None
                wav_pred_offline = self.vocoder.spec2wav(offline_out['mel_out'][0].cpu().numpy(), f0=offline_f0_np)
                self.logger.add_audio(f'wav_pred_offline_{batch_idx}', wav_pred_offline, self.global_step, sr)
                self.logger.add_audio(f'wav_pred_streaming_{batch_idx}', wav_pred, self.global_step, sr)
                self.plot_mel(batch_idx, sample['mels'], offline_out['mel_out'][0], f'mel_offline_{batch_idx}')
                self.plot_mel(batch_idx, sample['mels'], plot_out['mel_out'][0], f'mel_streaming_{batch_idx}')
            else:
                self.logger.add_audio(f'wav_pred_{batch_idx}', wav_pred, self.global_step, sr)
                self.plot_mel(batch_idx, sample['mels'], plot_out['mel_out'][0], f'mel_{batch_idx}')
            self.logger.add_figure(
                f'f0_{batch_idx}',
                f0_to_figure(gt_f0[0], None, plot_f0_pred[0] if isinstance(plot_f0_pred, torch.Tensor) else None),
                self.global_step)
        return tensors_to_scalars(outputs)
    
    def test_step(self, sample, batch_idx):
        """
        Prefix-online streaming evaluation:
        1. cache the single reference once
        2. incrementally extend content by 80 ms (=4 tokens) granularity
        3. recompute the acoustic prefix, but only keep newly emitted mel frames
        4. optionally compare the online path against the offline/full-prefix mel
        """
        sample['ref_mels'] = sample.get('ref_mels', sample['mels'])
        tokens_per_chunk = int(hparams.get("streaming_eval_tokens_per_chunk", 4))
        offline_outputs = self.run_model(sample, infer=True, test=True)[1]
        outputs = self._run_prefix_online_inference(sample, tokens_per_chunk=tokens_per_chunk)
        mel_pred = outputs['mel_out'][0]
        item_name = sample['item_name'][0]
        base_fn = f'{item_name.replace(" ", "_")}[P]'
        parity_metrics = self._streaming_parity_metrics(offline_outputs, outputs)

        # Pass through vocoder at once
        wav_pred = self.vocoder.spec2wav(mel_pred.cpu().numpy())

        # Optional: save gt (keep consistent with original implementation)
        if hparams.get('save_gt', False):
            mel_gt = sample['mels'][0].cpu().numpy()
            wav_gt = self.vocoder.spec2wav(mel_gt)
            self.saving_result_pool.add_job(
                self.save_result,
                args=[wav_gt, mel_gt,
                    base_fn.replace('[P]', '[G]'),
                    self.gen_dir, None, None,
                    denorm_f0(sample['f0'], sample['uv'])[0].cpu().numpy(),
                    None, None]
            )

        # Save prediction
        self.saving_result_pool.add_job(
            self.save_result,
            args=[wav_pred, mel_pred.cpu().numpy(),
                base_fn, self.gen_dir, None, None,
                denorm_f0(sample['f0'], sample['uv'])[0].cpu().numpy(),
                outputs.get('f0_denorm_pred')[0].cpu().numpy()
                if outputs.get('f0_denorm_pred') is not None else None,
                None]
        )

        scalar_metrics = tensors_to_scalars(parity_metrics)
        scalar_metrics.update(
            {
                "streaming_eval_mode": outputs.get("streaming_eval_mode", "prefix_online_content_chunked"),
                "streaming_total_chunks": int(outputs.get("streaming_total_chunks", 0)),
                "streaming_chunk_tokens": int(outputs.get("streaming_chunk_tokens", tokens_per_chunk)),
                "query_anchor_split_applied": bool(outputs.get("query_anchor_split_applied", False)),
                "dynamic_timbre_style_context_owner_safe": bool(
                    outputs.get("dynamic_timbre_style_context_owner_safe", False)
                ),
            }
        )
        return scalar_metrics


    def build_optimizer(self, model):
        
        optimizer_gen = torch.optim.AdamW(
            self.gen_params,
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])

        optimizer_disc = torch.optim.AdamW(
            self.disc_params,
            lr=hparams['disc_lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            **hparams["discriminator_optimizer_params"]) if len(self.disc_params) > 0 else None

        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        return [
            super().build_scheduler( optimizer[0]), # Generator Scheduler
            torch.optim.lr_scheduler.StepLR(optimizer=optimizer[1], # Discriminator Scheduler
                **hparams["discriminator_scheduler_params"]),
        ]

    def on_before_optimization(self, opt_idx):
        if opt_idx == 0:
            nn.utils.clip_grad_norm_(self.gen_params, hparams['clip_grad_norm'])
        else:
            nn.utils.clip_grad_norm_(self.disc_params, hparams["clip_grad_norm"])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler[0].step(self.global_step // hparams['accumulate_grad_batches'])
            self.scheduler[1].step(self.global_step // hparams['accumulate_grad_batches'])

def self_clone(x):
    if x == None:
        return None
    y = x.clone()
    result = torch.cat((x, y), dim=0)
    return result

class VCPostnetTask(ConanTask):
    def __init__(self):
        super(VCPostnetTask, self).__init__()
        self.drop_prob=hparams['drop_tech_prob']

    def build_model(self):
        self.build_pretrain_model()
        self.model = ConanPostnet()

    def build_pretrain_model(self):
        dict_size = 0
        self.pretrain = Conan(dict_size, hparams)
        from utils.commons.ckpt_utils import load_ckpt
        load_ckpt(self.pretrain, hparams['fs2_ckpt_dir'], 'model', strict=True) 
        for k, v in self.pretrain.named_parameters():
            v.requires_grad = False    
    
    def run_model(self, sample, infer=False, noise=None,test=False):
        content = sample["content"]
        # spk_embed = sample["spk_embed"]
        spk_embed=None
        f0, uv = sample["f0"], sample["uv"]
        # notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        
        target = sample["mels"]
        ref=sample['ref_mels']
        cfg = False
        output = self.pretrain(content,spk_embed=spk_embed, target=target,ref=ref,
                f0=f0, uv=uv,
                infer=infer)

        self.model(target, infer, output, cfg, cfg_scale=hparams['cfg_scale'],  noise=noise)
        losses = {}
        losses["flow"] = output["flow"]
        return losses, output

    def build_optimizer(self, model):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams['lr'],
            betas=(0.9, 0.98),
            eps=1e-9)
        return self.optimizer

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)

    def on_before_optimization(self, opt_idx):
        if self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_norm)
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_value_(self.parameters(), self.gradient_clip_val)

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if self.scheduler is not None:
            self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

# class TechSingerTask(RFSingerTask):

#     def build_tts_model(self):
#         dict_size = len(self.token_encoder)
#         self.model = Conan(dict_size, hparams)
#         self.gen_params = [p for p in self.model.parameters() if p.requires_grad]
            
#     def run_model(self, sample, infer=False):
#         txt_tokens = sample["txt_tokens"]
#         mel2ph = sample["mel2ph"]
#         spk_id = sample["spk_ids"]
#         f0, uv = sample["f0"], sample["uv"]
#         notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        
#         target = sample["mels"]
#         cfg = False
        
#         if infer:
#             cfg = True
#             mix, falsetto, breathy = sample['mix'],sample['falsetto'],sample['breathy']
#             bubble,strong,weak=sample['bubble'],sample['strong'],sample['weak']
#             pharyngeal,vibrato,glissando = sample['pharyngeal'],sample['vibrato'],sample['glissando']
#             umix, ufalsetto, ubreathy = torch.ones_like(mix, dtype=mix.dtype) * 2, torch.ones_like(falsetto, dtype=falsetto.dtype) * 2, torch.ones_like(breathy, dtype=breathy.dtype) * 2
#             ububble, ustrong, uweak = torch.ones_like(mix, dtype=mix.dtype) * 2, torch.ones_like(strong, dtype=strong.dtype) * 2, torch.ones_like(weak, dtype=weak.dtype) * 2
#             upharyngeal, uvibrato, uglissando = torch.ones_like(bubble, dtype=bubble.dtype) * 2, torch.ones_like(vibrato, dtype=vibrato.dtype) * 2, torch.ones_like(glissando, dtype=glissando.dtype) * 2
#             mix = torch.cat((mix, umix), dim=0)
#             falsetto = torch.cat((falsetto, ufalsetto), dim=0)
#             breathy = torch.cat((breathy, ubreathy), dim=0)
#             bubble = torch.cat((bubble, ububble), dim=0)
#             strong = torch.cat((strong, ustrong), dim=0)
#             weak = torch.cat((weak, uweak), dim=0)
#             pharyngeal = torch.cat((pharyngeal, upharyngeal), dim=0)
#             vibrato = torch.cat((vibrato, uvibrato), dim=0)
#             glissando = torch.cat((glissando, uglissando), dim=0)
            
#             txt_tokens = self_clone(txt_tokens)
#             mel2ph = self_clone(mel2ph)
#             spk_id = self_clone(spk_id)
#             f0 = self_clone(f0)
#             uv = self_clone(uv)
#             notes = self_clone(notes)
#             note_durs = self_clone(note_durs)
#             note_types = self_clone(note_types)
            
#             output = self.model(txt_tokens, mel2ph=mel2ph, spk_id=spk_id, f0=f0, uv=uv, 
#                                 note=notes, note_dur=note_durs, note_type=note_types,
#                                 mix=mix, falsetto=falsetto, breathy=breathy,
#                                 bubble=bubble, strong=strong, weak=weak,
#                                 pharyngeal=pharyngeal, vibrato=vibrato, glissando=glissando, target=target, cfg=cfg, cfg_scale=1.0,
#                                 infer=infer)
#         else:
#             tech_drop = {
#                 'mix': 0.1,
#                 'falsetto': 0.1,
#                 'breathy': 0.1,
#                 'bubble': 0.1,
#                 'strong': 0.1,
#                 'weak': 0.1,
#                 'glissando': 0.1,
#                 'pharyngeal': 0.1,
#                 'vibrato': 0.1,
#             }
#             for tech, drop_p in tech_drop.items():
#                 sample[tech] = self.drop_multi(sample[tech], drop_p)
#             mix, falsetto, breathy = sample['mix'],sample['falsetto'],sample['breathy']
#             bubble,strong,weak=sample['bubble'],sample['strong'],sample['weak']
#             pharyngeal,vibrato,glissando = sample['pharyngeal'],sample['vibrato'],sample['glissando']
#             output = self.model(txt_tokens, mel2ph=mel2ph, spk_id=spk_id, f0=f0, uv=uv, 
#                                 note=notes, note_dur=note_durs, note_type=note_types,
#                                 mix=mix, falsetto=falsetto, breathy=breathy,
#                                 bubble=bubble, strong=strong, weak=weak,
#                                 pharyngeal=pharyngeal, vibrato=vibrato, glissando=glissando, target=target, cfg=cfg,
#                                 infer=infer)
        
#         losses = {}
#         if not infer:
#             self.add_mel_loss(output['coarse_mel_out'], target, losses)
#             self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
#             self.add_pitch_loss(output, sample, losses)
#         losses["flow"] = output["flow"]
#         return losses, output
