""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder, Bert_encoder, prompt_encoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prompt_encoder = prompt_encoder
        self.Bert_encoder = Bert_encoder

    def forward(self, src, tgt, lengths, bptt=False, latent_input=None, segment_input=None,
                adj=None, prompt_embeds=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        # prompt_embeds = self.prompt_encoder(src.shape[1])    # src.shape[1] = batch_size
        enc_state, memory_bank, lengths = self.encoder(src, lengths, prompt_embeds=prompt_embeds, adj=adj)
        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        # dec_out, attns = self.decoder(
        #     tgt, memory_bank, memory_lengths=lengths, latent_input=latent_input,
        #     segment_input=segment_input)
        dec_out, attns = self.decoder(
            tgt, memory_bank, memory_lengths=lengths, latent_input=latent_input,
            segment_input=segment_input)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)


# class PretrainModel(nn.Module):
#     '''
#     fused embedding
#     '''
#     def __init__(self, GNN, LM):
#         super(PretrainModel, self).__init__()
#         self.GNN = GNN
#         self.LM = LM
#
#     def forward(self):

