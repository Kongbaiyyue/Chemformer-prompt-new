import math
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

from molbart.models.graph_transformer_pytorch import GraphTransformer
from molbart.models.bert_model import BERTEncoder, BertEmbedding, LastLine, BertModel


class TPrompt(nn.Module):
    def __init__(
            self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
            Bert_model=None, GNN_model=None, cross_model=None, n_prefix_rec=None, n_prefix_conv=None
    ):
        super(TPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv

        entity_hidden_size = hidden_size // 2

        # src_emb = BertModel(vocab_size=vocab_size)
        # Bert_encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, max_len=256, key_size=256, query_size=256, value_size=256)

        # emb = BertEmbedding(vocab_size, num_hiddens, max_len=256)
        # model_bert = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
        #                          dropout, max_len=256, key_size=256, query_size=256, value_size=256)
        # cls = LastLine(num_hiddens, vocab_size)
        # Bert_encoder = BertModel(emb, model_bert, cls)
        # self.gcn_model = GraphTransformer(
        self.graph_model = GraphTransformer(
            input_dim=9,
            h_dim=512,
            depth=3,
            edge_input_dim=9,
            edge_h_dim=512,
            # optional - if left out, edge dimensions is assumed to be the same as the node dimensions above
            with_feedforwards=True,
            # whether to add a feedforward after each attention layer, suggested by literature to be needed
            gated_residual=True,  # to use the gated residual to prevent over-smoothing
            rel_pos_emb=True  # set to True if the nodes are ordered, default to False
        )

        # cross_model = TransformerCross(num_layers, num_hiddens, num_heads, ffn_num_input, dropout)
        # self.fuse_model = PretrainFuseModel(Bert_model, GNN_model, cross_model)

        self.token_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.node_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        
        # self.graph_proj = nn.Linear(512, hidden_size)

        # self.token_proj1 = nn.Sequential(
        #     nn.Linear(token_hidden_size, token_hidden_size // 2),
        #     nn.ReLU(),
        #     nn.Linear(token_hidden_size // 2, token_hidden_size),
        # )
        # self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)

        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, n_layer * n_block * hidden_size)

        # conv_prompt
        self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
        nn.init.normal_(self.conv_prefix_embeds)
        self.conv_prefix_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )

    def get_entity_embeds(self, nodes, edges, n_lengths=None, n_adj=None):
        nodes_embeds, edges_embeds = self.graph_model(nodes, edges, lengths=n_lengths, adj=n_adj)
        return nodes_embeds

    def forward(self, nodes, edges, n_lengths=None, n_adj=None, src=None, src_lengths=None, token_embeds=None, output_entity=False, use_conv_prefix=False):
        # batch_size, entity_embeds, entity_len, token_len = None, None, None, None
        batch_size, entity_len = nodes.shape[:2]
        # if nodes is not None:
        #     batch_size, entity_len = nodes.shape[:2]
        #     entity_embeds = self.get_entity_embeds(nodes, edges, n_lengths=n_lengths, n_adj=n_adj)   # (batch_size, entity_len, hidden_size)
        #     entity_embeds = self.node_proj1(entity_embeds)
        
        
        
            # entity_embeds = entity_embeds[entity_ids]  # (batch_size, entity_len, hidden_size)
        # if token_embeds is not None:
        #     batch_size, token_len = token_embeds.shape[:2]
        #     token_embeds = self.token_proj1(token_embeds) + token_embeds  # (batch_size, token_len, hidden_size)
        #     token_embeds = self.token_proj2(token_embeds)
        #
        # attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(0, 2,
        #                                                                      1)  # (batch_size, token_len, entity_len)
        # attn_weights /= self.hidden_size
        # node_emb, edge_emb = self.fuse_model.graph(nodes, edges,
        #                                     lengths=n_lengths, adj=n_adj)
        # text_emb = self.fuse_model.bert(src.squeeze(-1).transpose(0, 1).contiguous(), None, src_lengths)
        # prompt_embeds = self.fuse_model.cross(text_emb, node_emb, memory_lengths=n_lengths)[0]
        # prompt_embeds = self.fuse_model.cross(node_emb, text_emb, memory_lengths=src_lengths)[0]

        # entity_weights = F.softmax(attn_weights, dim=2)
        # prompt_embeds = entity_weights @ entity_embeds + token_embeds
        # prompt_len = token_len

        # token_weights = F.softmax(attn_weights, dim=1).permute(0, 2, 1)
        # prompt_embeds = token_weights @ token_embeds + entity_embeds
        # batch_size, prompt_len = prompt_embeds.shape[:2]

        # prompt_embeds = entity_embeds
        # prompt_len = entity_len

        # add learnable prompt
        prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
        # prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
        prefix_embeds = prefix_embeds.expand(batch_size, -1, -1)
        prompt_embeds = prefix_embeds
        # prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
        
        # prompt_len += self.n_prefix_conv
        prompt_len = self.n_prefix_conv

        # prompt_embeds = entity_embeds
        # prompt_len = entity_len

        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.hidden_size
        ).permute(2, 3, 1, 0, 4)  # (n_layer, n_block, batch_size, n_head, prompt_len, head_dim)
        return prompt_embeds


class PretrainFuseModel(nn.Module):
    def __init__(self, bert, graph, cross):
        super(PretrainFuseModel, self).__init__()
        self.bert = bert
        self.graph = graph
        self.cross = cross

    def forward(self, tokens, valid_lens):
        X = self.token_embedding(tokens)
        X = X + self.pos_embedding.data[:, :int(valid_lens.max())+1, :]
        return X




if __name__ == '__main__':
    pass