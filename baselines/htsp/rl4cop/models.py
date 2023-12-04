import math
import numpy as np
import torch
import torch.nn.functional as F

import baselines.htsp.rl4cop.cop_utils as utils


class MHAEncoderLayer(torch.nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super().__init__()

        self.n_heads = n_heads
        self.Wq = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.multi_head_combine = torch.nn.Linear(embedding_dim, embedding_dim)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim * 4, embedding_dim))
        self.norm1 = torch.nn.BatchNorm1d(embedding_dim)
        self.norm2 = torch.nn.BatchNorm1d(embedding_dim)

    def forward(self, x, mask=None):
        q = utils.make_heads(self.Wq(x), self.n_heads)
        k = utils.make_heads(self.Wk(x), self.n_heads)
        v = utils.make_heads(self.Wv(x), self.n_heads)
        x = x + self.multi_head_combine(utils.multi_head_attention(q, k, v, mask))
        x = self.norm1(x.view(-1, x.size(-1))).view(*x.size())
        x = x + self.feed_forward(x)
        x = self.norm2(x.view(-1, x.size(-1))).view(*x.size())
        return x


class MHAEncoder(torch.nn.Module):
    def __init__(self, n_layers, n_heads, embedding_dim, input_dim, add_init_projection=True):
        super().__init__()
        if add_init_projection or input_dim != embedding_dim:
            self.init_projection_layer = torch.nn.Linear(input_dim, embedding_dim)
        self.attn_layers = torch.nn.ModuleList([MHAEncoderLayer(embedding_dim=embedding_dim, n_heads=n_heads)
                                                for _ in range(n_layers)])

    def forward(self, x, mask=None):
        if hasattr(self, "init_projection_layer"):
            x = self.init_projection_layer(x)
        for layer in self.attn_layers:
            x = layer(x, mask)
        return x


class PathDecoder(torch.nn.Module):
    def __init__(self, embedding_dim, n_heads=8, tanh_clipping=10.0, n_decoding_neighbors=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping
        self.n_decoding_neighbors = n_decoding_neighbors

        self.Wq_graph = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wq_source = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wq_target = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wq_first = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wq_last = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.multi_head_combine = torch.nn.Linear(embedding_dim, embedding_dim)

        self.q_graph = None  # saved q1, for multi-head attention
        self.q_source = None  # saved q2, for multi-head attention
        self.q_target = None  # saved q3, for multi-head attention
        self.q_first = None  # saved q4, for multi-head attention
        self.glimpse_k = None  # saved key, for multi-head attention
        self.glimpse_v = None  # saved value, for multi-head_attention
        self.logit_k = None  # saved, for single-head attention
        self.group_ninf_mask = None  # reference to ninf_mask owned by state

    def reset(self, coordinates, embeddings, group_ninf_mask, source_node, target_node, first_node):
        # embeddings.shape = [B, N, H]
        # graph_embedding.shape = [B, 1, H]
        B, N, H = embeddings.shape
        G = group_ninf_mask.size(1)
        self.coordinates = coordinates
        self.embeddings = embeddings
        graph_embedding = self.embeddings.mean(dim=1, keepdim=True)
        # q_graph.hape = [B, n_heads, 1, key_dim]
        self.q_graph = utils.make_heads(self.Wq_graph(graph_embedding), self.n_heads)
        # q_source.hape = [B, n_heads, 1, key_dim] - [B, n_heads, G, key_dim]
        source_node_index = source_node.view(B, G, 1).expand(B, G, H)
        source_node_embedding = self.embeddings.gather(1, source_node_index)
        self.q_source = utils.make_heads(self.Wq_source(source_node_embedding), self.n_heads)
        # q_target.hape = [B, n_heads, 1, key_dim]
        target_node_index = target_node.view(B, G, 1).expand(B, G, H)
        target_node_embedding = self.embeddings.gather(1, target_node_index)
        self.q_target = utils.make_heads(self.Wq_target(target_node_embedding), self.n_heads)
        # q_first.hape = [B, n_heads, 1, key_dim]
        first_node_index = first_node.view(B, G, 1).expand(B, G, H)
        first_node_embedding = self.embeddings.gather(1, first_node_index)
        self.q_first = utils.make_heads(self.Wq_first(first_node_embedding), self.n_heads)
        # glimpse_k.shape = glimpse_v.shape =[B, n_heads, N, key_dim]
        # logit_k.shape = [B, H, N]
        # group_ninf_mask.shape = [B, G, N]
        self.glimpse_k = utils.make_heads(self.Wk(embeddings), self.n_heads)
        self.glimpse_v = utils.make_heads(self.Wv(embeddings), self.n_heads)
        self.logit_k = embeddings.transpose(1, 2)
        self.group_ninf_mask = group_ninf_mask

    def forward(self, last_node):
        B, N, H = self.embeddings.shape
        G = self.group_ninf_mask.size(1)

        # q_last.shape = q_last.shape = [B, n_heads, G, key_dim]
        last_node_index = last_node.view(B, G, 1).expand(-1, -1, H)
        last_node_embedding = self.embeddings.gather(1, last_node_index)
        q_last = utils.make_heads(self.Wq_last(last_node_embedding), self.n_heads)
        # glimpse_q.shape = [B, n_heads, G, key_dim]
        glimpse_q = self.q_graph + self.q_source + self.q_target + self.q_first + q_last

        if self.n_decoding_neighbors is not None:
            D = self.coordinates.size(-1)
            K = torch.count_nonzero(self.group_ninf_mask[0, 0] == 0.0).item()
            K = min(self.n_decoding_neighbors, K)
            last_node_coordinate = self.coordinates.gather(dim=1, index=last_node.unsqueeze(-1).expand(B, G, D))
            distances = torch.cdist(last_node_coordinate, self.coordinates)
            distances[self.group_ninf_mask == -np.inf] = np.inf
            indices = distances.topk(k=K, dim=-1, largest=False).indices
            glimpse_mask = torch.ones_like(self.group_ninf_mask) * (-np.inf)
            glimpse_mask.scatter_(dim=-1, index=indices, src=torch.zeros_like(glimpse_mask))
        else:
            glimpse_mask = self.group_ninf_mask
        attn_out = utils.multi_head_attention(q=glimpse_q, k=self.glimpse_k, v=self.glimpse_v, mask=glimpse_mask)

        # mha_out.shape = [B, G, H]
        # score.shape = [B, G, N]
        final_q = self.multi_head_combine(attn_out)
        score = torch.matmul(final_q, self.logit_k) / math.sqrt(H)
        score_clipped = self.tanh_clipping * torch.tanh(score)
        score_masked = score_clipped + self.group_ninf_mask

        probs = F.softmax(score_masked.float(), dim=2, dtype=torch.float32).type_as(score_masked)
        assert (probs == probs).all(), "Probs should not contain any nans!"
        return probs

