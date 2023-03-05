"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn
import torch

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout,
                 max_relative_positions=0):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask, prompt_embeds=None):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``
            prompt_embeds (FloatTensor): ``(n_block, batch_size, n_head, prompt_len, head_dim)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, attn_type="self", prompt_embeds=prompt_embeds)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

    def update_dropout(self, dropout):
        self.self_attn.update_dropout(dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings,
                 max_relative_positions):
        super(TransformerEncoder, self).__init__()

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings,
            opt.max_relative_positions)

    def forward(self, src, lengths=None, prompt_embeds=None, adj=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        # emb = self.embeddings(src, None, lengths)
        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        # out = emb
        if isinstance(adj, torch.Tensor):
            mask = ~sequence_mask(lengths).unsqueeze(1) + ~adj.bool()
        elif isinstance(adj, list):
            mask = list()
            for ad in adj:
                mask.append(~sequence_mask(lengths).unsqueeze(1) + ~ad.bool())
        elif adj is None:
            mask = ~sequence_mask(lengths).unsqueeze(1)

        # Run the forward pass of every layer of the tranformer.
        for i, layer in enumerate(self.transformer):
            out = layer(out, mask, prompt_embeds=prompt_embeds[i])
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), lengths

    def update_dropout(self, dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout)


class Prompt(nn.Module):
    # def __init__(
    #         self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
    #         n_entity, num_relations, num_bases, edge_index, edge_type,
    #         n_prefix_rec=None, n_prefix_conv=None
    # ):
    def __init__(
            self, opt, n_block=2, n_prefix_conv=None
    ):
        super(Prompt, self).__init__()
        hidden_size = opt.enc_rnn_size
        self.n_layer = opt.enc_layers
        self.n_block = n_block
        self.n_head = opt.heads
        self.head_dim = hidden_size // opt.heads
        self.n_prefix_conv = n_prefix_conv

        self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
        nn.init.normal_(self.conv_prefix_embeds)
        self.conv_prefix_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, opt.enc_layers * n_block * hidden_size)

    def forward(self, batch_size, atom_ids=None, token_embeds=None, output_atom=False, use_rec_prefix=False,
                use_conv_prefix=False):
        prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
        # prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
        # prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
        # prompt_len += self.n_prefix_conv
        prompt_embeds = prefix_embeds.expand(batch_size, -1, -1)
        # print(prompt_embeds.shape)
        # print(token_embeds.shape)
        prompt_embeds = torch.cat([prompt_embeds, token_embeds], dim=1)
        prompt_len = token_embeds.shape[1]
        prompt_len += self.n_prefix_conv
        # prompt_len = self.n_prefix_conv

        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5)  # (n_layer, n_block, batch_size, n_head, prompt_len, head_dim)

        return prompt_embeds

