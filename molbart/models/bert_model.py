import torch
from torch import nn
from d2l import torch as d2l


# @save
# class BERTEncoder(nn.Module):
#     """BERT编码器"""
#     def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
#                  ffn_num_hiddens, num_heads, num_layers, dropout,
#                  max_len=1000, key_size=768, query_size=768, value_size=768,
#                  **kwargs):
#         super(BERTEncoder, self).__init__(**kwargs)
#         self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
#         # self.segment_embedding = nn.Embedding(2, num_hiddens)
#         self.blks = nn.Sequential()
#         for i in range(num_layers):
#             self.blks.add_module(f"{i}", d2l.EncoderBlock(
#                 key_size, query_size, value_size, num_hiddens, norm_shape,
#                 ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
#         # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
#         self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
#                                                       num_hiddens))
#         # self.last = nn.Linear(num_hiddens, vocab_size)
#
#     def forward(self, tokens, segments, valid_lens):
#         # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
#         # X = self.token_embedding(tokens) + self.segment_embedding(segments)
#         # print(tokens.shape)
#         # print(valid_lens.shape)
#         X = self.token_embedding(tokens)
#         X = X.transpose(0, 1).contiguous()
#         # print(X.shape)
#         # print(self.pos_embedding.data.shape)
#         X = X + self.pos_embedding.data[:, :int(valid_lens.max()), :]
#         for blk in self.blks:
#             X = blk(X, valid_lens)
#         # X = self.last(X)
#         return X

class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, num_hiddens, max_len=1000,
                 **kwargs):
        super(BertEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, valid_lens):
        X = self.token_embedding(tokens)
        X = X + self.pos_embedding.data[:, :int(valid_lens.max()), :]
        return X


#@save
class BERTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self, vocab_size: object, num_hiddens: object, norm_shape: object, ffn_num_input: object,
                 ffn_num_hiddens: object, num_heads: object, num_layers: object, dropout: object,
                 max_len: object = 1000, key_size: object = 768, query_size: object = 768, value_size: object = 768,
                 **kwargs: object) -> object:
        super(BERTEncoder, self).__init__(**kwargs)
        # self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        # self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        # self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
        #                                               num_hiddens))
        # self.last = nn.Linear(num_hiddens, vocab_size)

    def forward(self, X, segments, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        # X = self.token_embedding(tokens) + self.segment_embedding(segments)
        # X = self.token_embedding(tokens)
        # X = X + self.pos_embedding.data
        for blk in self.blks:
            X = blk(X, valid_lens)
        # X = self.last(X)
        return X


class LastLine(nn.Module):
    def __init__(self, num_hiddens, vocab_size):
        super(LastLine, self).__init__()
        self.last = nn.Linear(num_hiddens, vocab_size)

    def forward(self, X):
        return self.last(X)


class BertModel(nn.Module):
    def __init__(self, emb, bert, cls):
        super(BertModel, self).__init__()
        self.emb = emb
        self.bert = bert
        self.cls = cls

    def forward(self, X, segments, valid_lens):
        x = self.emb(X, valid_lens)
        outputs = self.bert(x, None, valid_lens)
        # outputs = self.cls(outputs)
        return outputs

