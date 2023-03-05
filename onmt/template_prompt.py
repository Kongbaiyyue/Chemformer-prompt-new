import math
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

from onmt.graph_transformer_pytorch import GraphTransformer
from onmt.rgcn_model import RGCNConv2


class TPrompt(nn.Module):
    def __init__(
            self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
            num_relations, entity_fea_size, num_bases=None, n_prefix_rec=None, n_prefix_conv=None
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
        # self.kg_encoder = RGCNConv2(entity_fea_size, entity_hidden_size, num_relations=num_relations,
        #                            num_bases=num_bases)
        self.gcn_model = GraphTransformer(
            input_dim=9,
            h_dim=hidden_size,
            depth=6,
            edge_input_dim=9,
            edge_h_dim=hidden_size,
            # optional - if left out, edge dimensions is assumed to be the same as the node dimensions above
            with_feedforwards=True,
            # whether to add a feedforward after each attention layer, suggested by literature to be needed
            gated_residual=True,  # to use the gated residual to prevent over-smoothing
            rel_pos_emb=True  # set to True if the nodes are ordered, default to False
        )
        # self.node_embeds = nn.Parameter(torch.empty(n_max_entity, entity_hidden_size))
        # stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        # self.node_embeds.data.uniform_(-stdv, stdv)
        # self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        # self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)

        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)

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
        # node_embeds = self.node_embeds   # (n_max_entity, entity_hidden_size)
        # node_embeds = entity_input
        # entity_embeds = self.kg_encoder(node_embeds, edge_index, self.edge_type) + node_embeds
        # # entity_embeds = self.kg_encoder(node_embeds, edge_index, edge_type)
        # with torch.no_grad():
        nodes_embeds, edges_embeds = self.gcn_model(nodes, edges, lengths=n_lengths, adj=n_adj)
        # entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        # entity_embeds = self.entity_proj2(entity_embeds)
        return nodes_embeds

    def forward(self, nodes, edges, n_lengths=None, n_adj=None, token_embeds=None, output_entity=False, use_conv_prefix=False):
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None
        if nodes is not None:
            batch_size, entity_len = nodes.shape[:2]
            entity_embeds = self.get_entity_embeds(nodes, edges, n_lengths=n_lengths, n_adj=n_adj)   # (batch_size, entity_len, hidden_size)
            # entity_embeds = entity_embeds[entity_ids]  # (batch_size, entity_len, hidden_size)
        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(token_embeds) + token_embeds  # (batch_size, token_len, hidden_size)
            token_embeds = self.token_proj2(token_embeds)

        attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(0, 2,
                                                                             1)  # (batch_size, token_len, entity_len)
        attn_weights /= self.hidden_size

        entity_weights = F.softmax(attn_weights, dim=2)
        prompt_embeds = entity_weights @ entity_embeds + token_embeds
        prompt_len = token_len
        # prompt_embeds = token_embeds
        # prompt_len = token_len

        # add learnable prompt
        prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
        prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
        prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
        prompt_len += self.n_prefix_conv

        # prompt_embeds = entity_embeds
        # prompt_len = entity_len

        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5)  # (n_layer, n_block, batch_size, n_head, prompt_len, head_dim)

        return prompt_embeds


if __name__ == '__main__':
    pass