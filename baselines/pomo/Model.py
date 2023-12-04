import torch
import torch.nn as nn
import torch.nn.functional as F


class POMO_Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = POMO_Encoder()
        self.decoder = POMO_Decoder()
        self.encoded_nodes = None  # [bsz,gsz,embedding]

    def pre_forward(self, x):
        self.encoded_nodes = self.encoder(x)  # [bsz,gsz,embedding]
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, x, pomo_size, enable_aug, device='cuda:0'):
        batch_size, graph_size, _ = x.size()
        assert graph_size >= pomo_size

        # initialize variables
        batch_idx = torch.arange(batch_size).unsqueeze(1).expand(batch_size, pomo_size).to(device)  # [bsz,pomo]
        pomo_idx = torch.arange(pomo_size).unsqueeze(0).expand(batch_size, pomo_size).to(device)  # [bsz,pomo]
        tours = torch.zeros(batch_size, pomo_size, 0).long().to(device)  # [bsz,pomo,0-gsz]
        ninf_mask = torch.zeros(batch_size, pomo_size, graph_size).to(device)  # [bsz,pomo,gsz]

        self.pre_forward(x)  # encode node and set decoder

        # select the first node
        selected = torch.randperm(graph_size)[:pomo_size].unsqueeze(0).expand(batch_size, pomo_size).to(device)
        encoded_first_node = _get_encoding(self.encoded_nodes, selected)  # [bsz,pomo,embedding]
        self.decoder.set_q1(encoded_first_node)
        tours = torch.cat((tours, selected.unsqueeze(2)), dim=2)  # [bsz,pomo,0-gsz]
        ninf_mask[batch_idx, pomo_idx, selected] = float('-inf')  # [bsz,pomo,gsz]
        selected_count = 1

        while selected_count < graph_size:  # select the remaining nodes
            encoded_last_node = _get_encoding(self.encoded_nodes, selected)  # [bsz,pomo,embedding]
            probs = self.decoder(encoded_last_node, ninf_mask=ninf_mask)  # [bsz,pomo,gsz]

            # select the next node
            selected = probs.argmax(dim=2)  # [bsz,pomo], eval_type == 'argmax'
            tours = torch.cat((tours, selected.unsqueeze(2)), dim=2)  # [bsz,pomo,0-gsz]
            ninf_mask[batch_idx, pomo_idx, selected] = float('-inf')  # [bsz,pomo,gsz]
            selected_count += 1

        # select the minimum from multiple choices
        lengths = _get_travel_distance(x, tours, batch_size, graph_size, pomo_size)  # [bsz,pomo]
        if enable_aug:
            lengths = lengths.reshape(8, -1, pomo_size)  # [aug_factor,raw_bsz,pomo]
            tours = tours.reshape(8, -1, pomo_size, graph_size)
            bsz_indices = torch.arange(lengths.size(1))
            min_pomo_lengths, pomo_indices = lengths.min(dim=2)
            min_lengths, aug_indices = min_pomo_lengths.min(dim=0)
            min_tours = tours[aug_indices, bsz_indices, pomo_indices[aug_indices, bsz_indices], :]
        else:
            bsz_indices = torch.arange(lengths.size(0))
            min_lengths, pomo_indices = lengths.min(dim=1)  # [bsz,]
            min_tours = tours[bsz_indices, pomo_indices[bsz_indices], :]

        return min_tours, min_lengths

        # # note the minus sign!
        # reward = -_get_travel_distance(x, selected_node_list, batch_size, graph_size, pomo_size)  # [bsz,pomo]
        # aug_reward = reward.reshape(aug_factor, -1, pomo_size)  # [af,ibsz,pomo]
        # max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo, [af,ibsz]
        # no_aug_score = -max_pomo_reward[0, :].mean()  # negative sign to make positive value
        # max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation, [bsz,]
        # aug_score = -max_aug_pomo_reward.mean()  # negative sign to make positive value
        # return no_aug_score.item(), aug_score.item()


def _get_encoding(encoded_nodes, node_index_to_pick):  # [bsz,gsz,embedding] / [bsz,pomo]

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index) # [bsz,pomo,embedding]

    return picked_nodes


class POMO_Encoder(nn.Module):
    def __init__(self, embedding_dim=128, encoder_layer_num=6, qkv_dim=16, head_num=8):
        super().__init__()
        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim, head_num, qkv_dim) for _ in range(encoder_layer_num)])

    def forward(self, x):  # [bsz,gsz,2]
        out = self.embedding(x)  # [bsz,gsz,embedding)
        for layer in self.layers:
            out = layer(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim=128, head_num=8, qkv_dim=16, ff_hidden_dim=512):
        super().__init__()
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(embedding_dim)
        self.feedForward = Feed_Forward_Module(embedding_dim, ff_hidden_dim)
        self.addAndNormalization2 = Add_And_Normalization_Module(embedding_dim)

    def forward(self, input1, head_num=8):  # [bsz, gsz, embedding]
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)  # [bsz, head, gsz, KEY_DIM]
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)  # [bsz, gsz, head*KEY_DIM]
        multi_head_out = self.multi_head_combine(out_concat)  # [bsz, gsz, embedding]

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)  # [bsz, gsz, embedding]

        return out3


class POMO_Decoder(nn.Module):
    def __init__(self, embedding_dim=128, head_num=8, qkv_dim=16):
        super().__init__()
        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention

    def set_kv(self, encoded_nodes, head_num=8):  # [bsz,gsz,embedding]
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)  # [bsz,head,gsz,qkv_dim]
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)  # [bsz,head,gsz,qkv_dim]
        self.single_head_key = encoded_nodes.transpose(1, 2)  # [bsz,embedding,gsz]


    def set_q1(self, encoded_q1, head_num = 8):  # [bsz, n, embedding], n can be 1 or pomo
        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)  # [bsz, head, n, qkv_dim]

    def forward(self, encoded_last_node, ninf_mask, head_num=8,
                sqrt_embedding_dim=128 ** 0.5, logit_clipping=10): # [bsz, pomo, embedding] / [bsz, pomo, gsz]
        #  Multi-Head Attention
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)  # [bsz, head, pomo, qkv_dim]
        q = self.q_first + q_last  # [bsz, head, pomo, qkv_dim]
        out_concat = multi_head_attention(q, self.k, self.v, rank_ninf_mask=ninf_mask)  # [bsz, pomo, head*qkv_dim]
        mh_attn_out = self.multi_head_combine(out_concat)  # [bsz, pomo, embedding]

        #  Single-Head Attention, for probability calculation
        score = torch.matmul(mh_attn_out, self.single_head_key)  # [bsz, pomo, gsz]
        score_scaled = score / sqrt_embedding_dim  # [bsz, pomo, gsz]
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)  # [bsz, pomo, gsz]

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):  # [bsz,n,head*key_dim], n can be either 1 or gsz
    q_reshaped = qkv.reshape(qkv.size(0), qkv.size(1), head_num, -1)  # [bsz,n,head,key_dim]
    return q_reshaped.transpose(1, 2)  # [bsz,head,n,key_dim]


def multi_head_attention(q, k, v, rank_ninf_mask=None):
    bsz, head_num, n, key_dim = q.size()  # n ~ 1 to gsz
    gsz = k.size(2)
    scale_factor = torch.tensor(key_dim).sqrt()

    if n >= 1000:
        out = torch.tensor([]).float().to(q.device)
        k = k.transpose(2, 3)
        for idx in range(n):
            q_row = q[:, :, [idx], :]
            score_row = torch.matmul(q_row, k) / scale_factor
            if rank_ninf_mask is not None:
                score_row = score_row + rank_ninf_mask[:, None, [idx], :].expand(bsz, head_num, 1, gsz)
            weights_row = score_row.softmax(dim=3)
            out_row = torch.matmul(weights_row, v)
            out = torch.cat((out, out_row), dim=2)
    else:
        score = torch.matmul(q, k.transpose(2, 3)) / scale_factor  # [bsz,head,n,gsz]
        if rank_ninf_mask is not None:
            score = score + rank_ninf_mask[:, None, :, :].expand(bsz, head_num, n, gsz)
        # weights = nn.Softmax(dim=3)(score)  # [bsz,head,n,gsz]
        weights = score.softmax(dim=3)
        out = torch.matmul(weights, v)  # [bsz,head,n,key_dim]
    out = out.transpose(1, 2).reshape(bsz, n, head_num * key_dim)  # [bsz,n,head*key_dim]

    return out


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):  # [bsz,gsz,embedding]
        # added = input1 + input2
        # transposed = added.transpose(1, 2)  # [bsz,embedding,gsz]
        # normalized = self.norm(transposed)  # [bsz,embedding,gsz]
        # back_trans = normalized.transpose(1, 2)  # [bsz,gsz,embedding]
        # return back_trans
        return self.norm((input1 + input2).transpose(1, 2)).transpose(1, 2)  # [bsz,gsz,embedding]


class Feed_Forward_Module(nn.Module):
    def __init__(self, embedding_dim=128, ff_hidden_dim=512):
        super().__init__()
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):  # [bsz,gsz,embedding]
        return self.W2(F.relu(self.W1(input1)))


def _get_travel_distance(x, selected_node_list, bsz, gsz, pomo):
    gathering_index = selected_node_list.unsqueeze(3).expand(bsz, -1, gsz, 2)  # [bsz,pomo,gsz,2]
    seq_expanded = x.unsqueeze(1).expand(bsz, pomo, gsz, 2)
    ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)  # [bsz,pomo,gsz,2]
    rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
    segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()  # [bsz,pomo,gsz]
    travel_distances = segment_lengths.sum(2)  # [bsz,pomo]
    return travel_distances