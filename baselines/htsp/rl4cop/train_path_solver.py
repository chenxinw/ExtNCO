import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from baselines.htsp.rl4cop.models import MHAEncoder, PathDecoder


class GroupState:
    def __init__(self, group_size, x, source, target):
        # x.shape = [B, N, 2]
        self.batch_size = x.size(0)
        self.graph_size = x.size(1)
        self.node_dim = x.size(2)
        self.group_size = group_size
        self.device = x.device
        # source.shape = target.shape = [B, G]
        self.source = source
        self.target = target

        self.selected_count = 0
        # current_node.shape = [B, G]
        self.current_node = None
        # selected_node_list.shape = [B, G, selected_count]
        self.selected_node_list = torch.zeros(
            x.size(0), group_size, 0, device=x.device
        ).long()
        # ninf_mask.shape = [B, G, N]
        self.ninf_mask = torch.zeros(x.size(0), group_size, x.size(1), device=x.device)

    def move_to(self, selected_idx_mat):
        # selected_idx_mat.shape = [B, G]
        self.selected_count += 1
        self.__move_to(selected_idx_mat)
        next_selected_idx_mat = self.__connect_source_target_city(selected_idx_mat)
        if (selected_idx_mat != next_selected_idx_mat).any():
            self.__move_to(next_selected_idx_mat)

    def __move_to(self, selected_idx_mat):
        self.current_node = selected_idx_mat
        self.selected_node_list = torch.cat(
            (self.selected_node_list, selected_idx_mat[:, :, None]), dim=2
        )
        self.mask(selected_idx_mat)

    def __connect_source_target_city(self, selected_idx_mat):
        source_idx = torch.where(selected_idx_mat == self.source)
        target_idx = torch.where(selected_idx_mat == self.target)
        next_selected_idx_mat = selected_idx_mat.clone()
        next_selected_idx_mat[source_idx] = self.target[source_idx]
        next_selected_idx_mat[target_idx] = self.source[target_idx]
        return next_selected_idx_mat

    def mask(self, selected_idx_mat):
        # selected_idx_mat.shape = [B, G]
        self.ninf_mask.scatter_(
            dim=-1, index=selected_idx_mat[:, :, None], value=-torch.inf
        )


class Env:
    def __init__(self, x):
        self.x = x
        self.batch_size = self.B = x.size(0)
        self.graph_size = self.N = x.size(1)
        self.node_dim = self.C = x.size(2)
        self.group_size = self.G = None
        self.group_state = None

    def reset(self, group_size, source, target):
        self.group_size = group_size
        self.group_state = GroupState(group_size=group_size, x=self.x, source=source, target=target)
        self.fixed_edge_length = self._get_edge_length(source, target)
        reward = None
        done = False
        return self.group_state, reward, done

    def step(self, selected_idx_mat):
        # move state
        self.group_state.move_to(selected_idx_mat)

        # returning values
        done = self.group_state.selected_count == (self.graph_size - 1)
        if done:
            reward = -self._get_path_distance()  # note the minus sign!
        else:
            reward = None
        return self.group_state, reward, done

    def _get_edge_length(self, source, target):
        idx_shp = (self.batch_size, self.group_size, 1, self.node_dim)
        coord_shp = (self.batch_size, self.group_size, self.graph_size, self.node_dim)
        source_idx = source[..., None, None].expand(*idx_shp)
        target_idx = target[..., None, None].expand(*idx_shp)
        fixed_edge_idx = torch.cat([source_idx, target_idx], dim=2)
        seq_expanded = self.x[:, None, :, :].expand(*coord_shp)
        ordered_seq = seq_expanded.gather(dim=2, index=fixed_edge_idx)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        delta = (ordered_seq - rolled_seq)[:, :, :-1, :]
        edge_length = (delta**2).sum(3).sqrt().sum(2)
        return edge_length

    def _get_path_distance(self) -> torch.Tensor:
        # selected_node_list.shape = [B, G, selected_count]
        interval = (
            torch.tensor([-1], device=self.x.device)
            .long()
            .expand(self.B, self.group_size)
        )
        selected_node_list = torch.cat(
            (self.group_state.selected_node_list, interval[:, :, None]),
            dim=2,
        ).flatten()
        unique_selected_node_list = selected_node_list.unique_consecutive()
        assert unique_selected_node_list.shape[0] == (
            self.B * self.group_size * (self.N + 1)
        ), unique_selected_node_list.shape
        unique_selected_node_list = unique_selected_node_list.view(
            [self.B, self.group_size, -1]
        )[..., :-1]
        shp = (self.B, self.group_size, self.N, self.C)
        gathering_index = unique_selected_node_list.unsqueeze(3).expand(*shp)
        seq_expanded = self.x[:, None, :, :].expand(*shp)
        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        delta = ordered_seq - rolled_seq
        tour_distances = (delta**2).sum(3).sqrt().sum(2)
        # minus the length of the fixed edge
        path_distances = tour_distances - self.fixed_edge_length
        return path_distances


class PathSolver(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        if cfg.node_dim > 2:
            assert "noAug" in cfg.val_type, "High-dimension TSP doesn't support augmentation"
        if cfg.encoder_type == "mha":
            self.encoder = MHAEncoder(n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                                             embedding_dim=cfg.embedding_dim, input_dim=cfg.node_dim,
                                             add_init_projection=cfg.add_init_projection)
        else:
            raise NotImplementedError('only support MHA encoder')
        self.decoder = PathDecoder(embedding_dim=cfg.embedding_dim, n_heads=cfg.n_heads,
                                          tanh_clipping=cfg.tanh_clipping,
                                          n_decoding_neighbors=cfg.n_decoding_neighbors)
        self.cfg = cfg
        self.save_hyperparameters(cfg)


    def forward(self, batch, source_nodes, target_nodes, val_type="x8Aug_2Traj",
                return_pi=False, group_size=2, greedy=True):
        val_type = val_type or self.cfg.val_type
        if "x8Aug" in val_type:
            from baselines.htsp.rl4cop.cop_utils import augment_xy_data_by_8_fold
            batch = augment_xy_data_by_8_fold(batch)
            source_nodes = torch.repeat_interleave(source_nodes, 8, dim=0)
            target_nodes = torch.repeat_interleave(target_nodes, 8, dim=0)

        B, N, _ = batch.shape
        # only support 1Traj or 2Traj
        # G = int(val_type[-5])
        # support larger Traj by sampling
        G = group_size
        assert G <= self.cfg.graph_size
        # batch_idx_range = torch.arange(B)[:, None].expand(B, G)
        # group_idx_range = torch.arange(G)[None, :].expand(B, G)

        source_action = source_nodes.view(B, 1).expand(B, G)
        target_action = target_nodes.view(B, 1).expand(B, G)

        env = Env(batch)
        s, r, d = env.reset(group_size=G, source=source_action, target=target_action)
        embeddings = self.encoder(batch)

        first_action = torch.randperm(N, device=self.device)[None, :G].expand(B, G)
        s, r, d = env.step(first_action)
        self.decoder.reset(batch, embeddings, s.ninf_mask, source_action, target_action, first_action)
        for _ in range(N - 2):
            action_probs = self.decoder(s.current_node)
            if greedy:
                action = action_probs.argmax(dim=2)
            else:
                action = action_probs.reshape(B * G, -1).multinomial(1).squeeze(dim=1).reshape(B, G)
            #     # Check if sampling went OK, can go wrong due to bug on GPU
            #     # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            #     while self.decoder.group_ninf_mask[batch_idx_range, group_idx_range, action].bool().any():
            #         action = action_probs.reshape(B * G, -1).multinomial(1).squeeze(dim=1).reshape(B, G)
            s, r, d = env.step(action)
        interval = torch.tensor([-1], device=self.device).long().expand(B, G)
        selected_node_list = torch.cat((s.selected_node_list, interval[:, :, None]), dim=2).flatten()
        unique_selected_node_list = selected_node_list.unique_consecutive()
        assert unique_selected_node_list.shape[0] == B * G * (N + 1), unique_selected_node_list.shape
        pi = unique_selected_node_list.view([B, G, -1])[..., :-1]

        if val_type == "noAug_1Traj":
            max_reward = r
            best_pi = pi
        elif val_type == "noAug_nTraj":
            max_reward, idx_dim_1 = r.max(dim=1)
            idx_dim_1 = idx_dim_1.reshape(B, 1, 1)
            best_pi = pi.gather(1, idx_dim_1.repeat(1, 1, N))
        else:
            B = round(B / 8)
            reward = r.reshape(8, B, G)
            max_reward, idx_dim_2 = reward.max(dim=2)
            max_reward, idx_dim_0 = max_reward.max(dim=0)
            pi = pi.reshape(8, B, G, N)
            idx_dim_0 = idx_dim_0.reshape(1, B, 1, 1)
            idx_dim_2 = idx_dim_2.reshape(8, B, 1, 1).gather(0, idx_dim_0)
            best_pi = pi.gather(0, idx_dim_0.repeat(1, 1, G, N))
            best_pi = best_pi.gather(2, idx_dim_2.repeat(1, 1, 1, N))

        if return_pi:
            best_pi = best_pi.squeeze()
            return -max_reward, best_pi
        return -max_reward
