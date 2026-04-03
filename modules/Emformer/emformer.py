import importlib
import time

import torch
import torch.nn as nn


def _load_torchaudio():
    try:
        return importlib.import_module("torchaudio")
    except Exception as exc:
        raise RuntimeError(
            "EmformerDistillModel requires a working torchaudio installation. "
            "Check that torch and torchaudio are a compatible pair for the active Python version."
        ) from exc

class EmformerDistillModel(nn.Module):
    def __init__(self, hparams, input_dim=None, output_dim=None):
        super().__init__()

        if input_dim is None:
            input_dim = int(hparams.get('emformer_input_dim', 80))
        if output_dim is None:
            output_dim = int(hparams.get('emformer_output_dim', 768))

        torchaudio = _load_torchaudio()
        segment_length = max(1, int(hparams['chunk_size']) // 20)
        self.emformer = torchaudio.models.Emformer(
            input_dim=int(input_dim),
            num_heads=8,
            ffn_dim=2048,
            num_layers=hparams['emformer_layers'],
            segment_length=segment_length,
            left_context_length=50,
            right_context_length=hparams['right_context'],
        )
        self.segment_length = segment_length
        # If output dimension differs from HuBERT, add projection layer
        self.proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.right_context_len = hparams['right_context']
        self.mode = hparams.get('emformer_mode', hparams.get('mode', None))
        if self.mode == 'both':
            self.proj1 = nn.Linear(input_dim, 100)
            self.proj2 = nn.Linear(input_dim, 768)
    def forward(self, mel_input, lengths):
        """
        mel_input: [B, T, mel_dim]
        """
        # Emformer requires input shape to be [T, B, F]
        # x = mel_input.transpose(0, 1)
        # lengths = torch.full((x.size(1),), x.size(0), device=x.device)
        
        # Get Emformer output
        output, lengths = self.emformer(mel_input, lengths)
        if self.mode == 'both':
            output1 = self.proj1(output)
            output2 = self.proj2(output)
            return output1, output2, lengths
        output = self.proj(output)
        
        return output, lengths
    @torch.inference_mode()
    def inference(self, mel_input: torch.Tensor):
        """
        Streaming inference that honours arbitrary right-context length.

        Args
        ----
        mel_input : (B, T, F)
            T = number of frames, F = mel-bin count (e.g., 80).

        Returns
        -------
        • proj(mel-level features) shaped (B, T, out_dim) or the two
          heads when `self.mode == 'both'`.
        """
        B, T, F = mel_input.shape
        seg, rc  = self.segment_length, self.right_context_len
        device   = mel_input.device

        pos, state, out_chunks = 0, None, []

        while pos < T:
            # 1) How many NEW frames do we want to emit this step?
            emit = min(seg, T - pos)

            # 2) How much genuine look-ahead is still available?
            look = min(rc, T - (pos + emit))

            # 3) Build the real chunk (emit + look) … then pad
            real_len = emit + look
            chunk = mel_input[:, pos:pos + real_len, :]             # (B, real_len, F)

            # Pad so that len(chunk) == seg + rc, as Emformer expects
            need_pad = (seg + rc) - real_len
            if need_pad > 0:
                pad = chunk[:, -1:, :].expand(B, need_pad, F)       # repeat last frame
                chunk = torch.cat([chunk, pad], dim=1)              # (B, seg+rc, F)

            # 4) Run one streaming step  (length **includes** the right context) 
            lengths = torch.full((B,), chunk.size(1), dtype=torch.long, device=device)
            chunk_out, _, state = self.emformer.infer(chunk, lengths, state)

            # Emformer pads the missing tail context on the last chunk.
            # Those padded positions should not be emitted into the streamed sequence,
            # otherwise the streamed path can become longer than the offline path.
            tail_context_deficit = max(0, rc - look)
            effective_emit = max(0, emit - tail_context_deficit)
            effective_emit = min(int(chunk_out.size(1)), int(effective_emit))
            if effective_emit > 0:
                out_chunks.append(chunk_out[:, :effective_emit, :])
            pos += emit

        # ------------------------------------------------------------------ #
        if len(out_chunks) <= 0:
            streamed_out = mel_input.new_zeros((B, 0, mel_input.size(-1)))
        else:
            streamed_out = torch.cat(out_chunks, dim=1)     # (B, T, F')
        if self.mode == 'both':
            return self.proj1(streamed_out), self.proj2(streamed_out)
        return self.proj(streamed_out)
    @torch.inference_mode()
    def inference_rtf(self, mel_input: torch.Tensor):
        """
        Streaming inference that honours arbitrary right-context length.

        Args
        ----
        mel_input : (B, T, F)
            T = number of frames, F = mel-bin count (e.g., 80).

        Returns
        -------
        • proj(mel-level features) shaped (B, T, out_dim) or the two
          heads when `self.mode == 'both'`.
        """
        B, T, F = mel_input.shape
        seg, rc  = self.segment_length, self.right_context_len
        device   = mel_input.device

        pos, state, out_chunks = 0, None, []
        latency_list = []
        rtf_list = []

        while pos < T:
            # 1) How many NEW frames do we want to emit this step?
            emit = min(seg, T - pos)

            # 2) How much genuine look-ahead is still available?
            look = min(rc, T - (pos + emit))

            # 3) Build the real chunk (emit + look) … then pad
            real_len = emit + look
            chunk = mel_input[:, pos:pos + real_len, :]             # (B, real_len, F)

            # Pad so that len(chunk) == seg + rc, as Emformer expects
            need_pad = (seg + rc) - real_len
            if need_pad > 0:
                pad = chunk[:, -1:, :].expand(B, need_pad, F)       # repeat last frame
                chunk = torch.cat([chunk, pad], dim=1)              # (B, seg+rc, F)

            # 4) Run one streaming step  (length **includes** the right context) 
            
            lengths = torch.full((B,), chunk.size(1), dtype=torch.long, device=device)
            start_time = time.time()
            chunk_out, _, state = self.emformer.infer(chunk, lengths, state)
            latency = time.time() - start_time
            latency_list.append(latency)
            rtf_list.append(latency / (seg * 0.02))  # Assuming 50Hz sampling rate (20ms per frame)
            print('latency: {:.4f}, rtf: {:.4f}'.format(latency, rtf_list[-1]))
            tail_context_deficit = max(0, rc - look)
            effective_emit = max(0, emit - tail_context_deficit)
            effective_emit = min(int(chunk_out.size(1)), int(effective_emit))
            if effective_emit > 0:
                out_chunks.append(chunk_out[:, :effective_emit, :])
            pos += emit

        # ------------------------------------------------------------------ #
        if len(out_chunks) <= 0:
            streamed_out = mel_input.new_zeros((B, 0, mel_input.size(-1)))
        else:
            streamed_out = torch.cat(out_chunks, dim=1)     # (B, T, F')
        if self.mode == 'both':
            return self.proj1(streamed_out), self.proj2(streamed_out), latency_list
        return self.proj(streamed_out), latency_list, rtf_list
