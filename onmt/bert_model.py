# import math
#
# import torch
# import torch.nn as nn
#
#
# def gelu(x):
#     return 0.5 * x * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.))))
#
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads, dropout):
#         super(MultiHeadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.d_model = d_model
#         self.max_relative_positions = 4
#
#         assert d_model % self.num_heads == 0
#
#         self.depth = d_model // self.num_heads
#
#         self.wq = nn.Linear(d_model, d_model, device='cuda')
#         self.wk = nn.Linear(d_model, d_model, device='cuda')
#         self.wv = nn.Linear(d_model, d_model, device='cuda')
#
#         self.dense = nn.Linear(d_model, d_model, device='cuda')
#
#         self.softmax = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(dropout)
#
#         if self.max_relative_positions > 0:
#             vocab_size = self.max_relative_positions * 2 + 1
#             self.relative_positions_embeddings = nn.Embedding(
#                 vocab_size, self.depth, device='cuda')
#
#     # def split_heads(self, x, batch_size):
#     #     """Split the last dimension into (num_heads, depth).
#     #     Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
#     #     """
#     #     x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
#     #     return tf.transpose(x, perm=[0, 2, 1, 3])
#     def scaled_dot_product_attention(self, q, k, v, mask, adjoin_matrix):
#         """Calculate the attention weights.
#         q, k, v must have matching leading dimensions.
#         k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
#         The mask has different shapes depending on its type(padding or look ahead)
#         but it must be broadcastable for addition.
#
#         Args:
#           q: query shape == (..., seq_len_q, depth)
#           k: key shape == (..., seq_len_k, depth)
#           v: value shape == (..., seq_len_v, depth_v)
#           mask: Float tensor with shape broadcastable
#                 to (..., seq_len_q, seq_len_k). Defaults to None.
#
#         Returns:
#           output, attention_weights
#         """
#
#         matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)
#
#         # scale matmul_qk
#         # dk = tf.cast(tf.shape(k)[-1], tf.float32)
#         scaled_attention_logits = matmul_qk / math.sqrt(k.size()[-1])
#
#         if self.max_relative_positions > 0:
#             key_len = k.size(2)
#             # 1 or key_len x key_len
#             relative_positions_matrix = generate_relative_positions_matrix(
#                 key_len, self.max_relative_positions)
#             #  1 or key_len x key_len x dim_per_head
#             relations_keys = self.relative_positions_embeddings(
#                 relative_positions_matrix.to(k.device))
#             #  1 or key_len x key_len x dim_per_head
#             relations_values = self.relative_positions_embeddings(
#                 relative_positions_matrix.to(k.device))
#
#         # add the mask to the scaled tensor.
#         if mask is not None:
#             scaled_attention_logits += (mask * -1e9)
#         if adjoin_matrix is not None:
#             scaled_attention_logits += adjoin_matrix
#
#         if self.max_relative_positions > 0:
#             scores = scaled_attention_logits + relative_matmul(q, relations_keys, True)
#         else:
#             scores = scaled_attention_logits
#
#         # softmax is normalized on the last axis (seq_len_k) so that the scores
#         # add up to 1.
#         attn = self.softmax(scores).to(q.dtype)
#         drop_attn = self.dropout(attn)
#
#         output = torch.matmul(drop_attn, v)
#
#         if self.max_relative_positions > 0:
#             output = output + relative_matmul(drop_attn, relations_values, False)
#
#         # attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
#         # output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
#
#         return output, drop_attn
#
#     def forward(self, v, k, q, mask, adjoin_matrix):
#         batch_size = q.size(0)
#         # print(q.device)
#         # print(self.wq.device)
#         q = self.wq(q)  # (batch_size, seq_len, d_model)
#         k = self.wk(k)  # (batch_size, seq_len, d_model)
#         v = self.wv(v)  # (batch_size, seq_len, d_model)
#
#         dim_per_head = self.depth
#         head_count = self.num_heads
#
#         def shape(x):
#             """Projection."""
#             return x.view(batch_size, -1, head_count, dim_per_head) \
#                 .transpose(1, 2)
#
#         q = shape(q)
#         k = shape(k)
#         v = shape(v)
#         # q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
#         # k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
#         # v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
#
#         # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
#         # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
#         scaled_attention, attention_weights = self.scaled_dot_product_attention(
#             q, k, v, mask, adjoin_matrix)
#
#         scaled_attention = scaled_attention.transpose(1, 2).contiguous()  # (batch_size, seq_len_q, num_heads, depth)
#
#         concat_attention = scaled_attention.view(batch_size, -1, self.d_model)  # (batch_size, seq_len_q, d_model)
#
#         output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
#
#         return output, attention_weights
#
#
# class PositionwiseFeedForward(nn.Module):
#     """ A two-layer Feed-Forward-Network with residual layer norm.
#
#     Args:
#         d_model (int): the size of input for the first-layer of the FFN.
#         d_ff (int): the hidden layer size of the second-layer
#             of the FNN.
#         dropout (float): dropout probability in :math:`[0, 1)`.
#     """
#
#     def __init__(self, d_model, d_ff, dropout=0.1):
#         super(PositionwiseFeedForward, self).__init__()
#         self.w_1 = nn.Linear(d_model, d_ff, device='cuda')
#         self.w_2 = nn.Linear(d_ff, d_model, device='cuda')
#         self.layer_norm = nn.LayerNorm(d_model, eps=1e-6, device='cuda')
#         self.dropout_1 = nn.Dropout(dropout)
#         self.relu = nn.ReLU()
#         self.dropout_2 = nn.Dropout(dropout)
#
#     def forward(self, x):
#         """Layer definition.
#
#         Args:
#             x: ``(batch_size, input_len, model_dim)``
#
#         Returns:
#             (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
#         """
#
#         inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
#         output = self.dropout_2(self.w_2(inter))
#         return output + x
#
#
# class EncoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, dff, dropout=0.1):
#         super(EncoderLayer, self).__init__()
#
#         self.mha = MultiHeadAttention(d_model, num_heads, dropout)
#         self.ffn = PositionwiseFeedForward(d_model, dff)
#
#         self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6, device='cuda')
#         self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6, device='cuda')
#
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#
#     def forward(self, x, training, mask, adjoin_matrix):
#         attn_output, attention_weights = self.mha(x, x, x, mask, adjoin_matrix)  # (batch_size, input_seq_len, d_model)
#         attn_output = self.dropout1(attn_output)
#         # out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
#         out1 = x + attn_output
#
#         ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
#         # ffn_output = self.dropout2(ffn_output, training=training)
#         out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
#
#         return out2, attention_weights
#
#
# class Encoder(nn.Module):
#     def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
#                  maximum_position_encoding, dropout=0.1):
#         super(Encoder, self).__init__()
#
#         self.d_model = d_model
#         self.num_layers = num_layers
#
#         self.embedding = nn.Embedding(input_vocab_size, d_model, device='cuda')
#         # self.pos_encoding = positional_encoding(maximum_position_encoding,
#         #                                         self.d_model)
#
#         self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout)
#                            for _ in range(num_layers)]
#
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, training, mask, adjoin_matrix):
#         seq_len = x.shape[1]
#         # adjoin_matrix.unsqueeze(1)
#         # adjoin_matrix = adjoin_matrix[:, tf.newaxis, :, :]
#         # adding embedding and position encoding.
#         # print(x.device)
#         # x = x.squeeze(2)
#
#         x = self.embedding(x)   # (batch_size, input_seq_len, d_model)
#         x *= math.sqrt(self.d_model)
#         x = x.squeeze(2)
#
#         x = self.dropout(x)
#
#         for i in range(self.num_layers):
#             x, attention_weights = self.enc_layers[i](x, training, mask, adjoin_matrix)
#         return x  # (batch_size, input_seq_len, d_model)
#
#
# class BertModel(nn.Module):
#     def __init__(self, num_layers=3, d_model=256, dff=512, num_heads=4, vocab_size=17, dropout_rate=0.1):
#         super(BertModel, self).__init__()
#         self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
#                                num_heads=num_heads, dff=dff, input_vocab_size=vocab_size, maximum_position_encoding=200,
#                                dropout=dropout_rate)
#         self.fc1 = nn.Linear(d_model, d_model)
#         self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
#         # self.fc2 = nn.Linear(d_model, vocab_size)
#
#     def forward(self, x, adjoin_matrix, mask, training=False):
#         x = self.encoder(x, training=training, mask=mask, adjoin_matrix=adjoin_matrix)
#         x = self.fc1(x)
#         x = self.layer_norm(x)
#         # x = self.fc2(x)
#         return x
#
#
# class MaskLM(nn.Module):
#     """BERT的掩蔽语言模型任务"""
#     def __init__(self, vocab_size, num_hiddens, num_inputs=768, max_length=256, **kwargs):
#         super(MaskLM, self).__init__(**kwargs)
#         self.max_length = max_length
#         self.num_inputs = num_inputs
#
#         self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens, device='cuda'),
#                                  nn.ReLU(),
#                                  nn.LayerNorm(num_hiddens, device='cuda'),
#                                  nn.Linear(num_hiddens, vocab_size, device='cuda'))
#
#     def forward(self, X, pred_positions):
#         # num_pred_positions = pred_positions.shape[1]
#         # pred_positions = pred_positions.reshape(-1)
#         # batch_size = X.shape[0]
#         # batch_idx = torch.arange(0, batch_size)
#         # # 假设batch_size=2，num_pred_positions=3
#         # # 那么batch_idx是np.array（[0,0,0,1,1,1]）
#         # batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
#         # masked_X = X[batch_idx, pred_positions]
#         # masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
#         y = torch.nonzero(pred_positions).to('cuda')
#         x = X.reshape(-1, self.num_inputs)
#         masked_X = [x[idx[0]*self.max_length+idx[1]] for idx in y]
#         masked_X = torch.tensor([item.cpu().detach().numpy() for item in masked_X]).to('cuda')
#         mlm_Y_hat = self.mlp(masked_X)
#         return mlm_Y_hat
#
#
# def generate_relative_positions_matrix(length, max_relative_positions,
#                                        cache=False):
#     """Generate the clipped relative positions matrix
#        for a given length and maximum relative positions"""
#     if cache:
#         distance_mat = torch.arange(-length+1, 1, 1).unsqueeze(0)
#     else:
#         range_vec = torch.arange(length)
#         range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
#         distance_mat = range_mat - range_mat.transpose(0, 1)
#     distance_mat_clipped = torch.clamp(distance_mat,
#                                        min=-max_relative_positions,
#                                        max=max_relative_positions)
#     # Shift values to be >= 0
#     final_mat = distance_mat_clipped + max_relative_positions
#     return final_mat
#
#
# def relative_matmul(x, z, transpose):
#     """Helper function for relative positions attention."""
#     batch_size = x.shape[0]
#     heads = x.shape[1]
#     length = x.shape[2]
#     x_t = x.permute(2, 0, 1, 3)
#     x_t_r = x_t.reshape(length, heads * batch_size, -1)
#     if transpose:
#         z_t = z.transpose(1, 2)
#         x_tz_matmul = torch.matmul(x_t_r, z_t)
#     else:
#         x_tz_matmul = torch.matmul(x_t_r, z)
#     x_tz_matmul_r = x_tz_matmul.reshape(length, batch_size, heads, -1)
#     x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
#     return x_tz_matmul_r_t

import torch
from torch import nn
from d2l import torch as d2l


#@save
class BERTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        # self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))
        # self.last = nn.Linear(num_hiddens, vocab_size)

    def forward(self, tokens, segments, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        # X = self.token_embedding(tokens) + self.segment_embedding(segments)
        # print(tokens.shape)
        # print(valid_lens.shape)
        X = self.token_embedding(tokens)
        X = X.transpose(0, 1).contiguous()
        # print(X.shape)
        # print(self.pos_embedding.data.shape)
        X = X + self.pos_embedding.data[:, :int(valid_lens.max()+1), :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        # X = self.last(X)
        return X

