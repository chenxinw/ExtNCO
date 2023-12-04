import time
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from abc import ABC, abstractmethod
from hydra import initialize
from typing import Callable, List, Optional, Tuple, Union
from omegaconf import OmegaConf
from contextlib import contextmanager
from torch.utils.data import DataLoader

from baselines.htsp.rl4cop.train_path_solver import PathSolver
from baselines.htsp.rl_models import IMPALAEncoder, ActorPPO, CriticPPO


def load_htsp(upper='tsp10000', device='cuda:0'):  # load pretrained model
    ckpt = torch.load('pretrained/htsp/upper-level/{}/best.ckpt'.format(upper))
    with initialize(config_path="", version_base="1.1"):
        cfg = OmegaConf.create(ckpt["hyper_parameters"])
    cfg.low_level_load_path = 'pretrained/htsp/lower-level/lower200/best.ckpt'
    cfg.encoder_type = 'pixel'  # 'cnn' is not support in code
    htsp_model = HTSP_PPO(cfg).to(device)
    htsp_model.load_state_dict(ckpt["state_dict"])
    htsp_env = VecEnv(k=40, frag_len=200, max_new_nodes=190, max_improvement_step=0)  # lower solver

    return htsp_model, htsp_env


def htsp_solve(coords, htsp_model, htsp_env):  # solve instance with htsp pretrained model
    st = time.time()
    infos = {'iter_time': [], 'up_time': [], 'low_time': [], 'new': [], 'frag': [], 'ratio': []}
    s = htsp_env.reset(coords)
    infos['reset_time'] = time.time() - st
    while not htsp_env.done:
        step_time = time.time()
        a = htsp_model(s).detach()
        up_time = time.time() - step_time
        s, r, d, info = htsp_env.step(a, htsp_model.low_level_solver, frag_buffer=htsp_model.val_frag_buffer)
        iter_time = time.time() - step_time
        infos['iter_time'] += [iter_time]
        infos['up_time'] += [up_time]
        infos['low_time'] += [iter_time - up_time]
        infos['new'] += [len(info['new_cities'][0])]
        infos['frag'] += [len(info['fragments'][0])]
        infos['ratio'] += [len(info['new_cities'][0]) / len(info['fragments'][0])]

    assert len(htsp_env.envs) == 1
    tour = htsp_env.envs[0].state.current_tour
    assert len(np.unique(tour)) == coords.size(1)
    return torch.tensor(tour), time.time() - st, infos

    # batch mode
    # tours = [env.state.current_tour for env in htsp_env.envs]
    # return tours, time.time() - st


class LargeState:
    def __init__(self, x, k, init_tour, dist_matrix=None, knn_neighbor=None):
        # for now only support single graph
        assert x.ndim == 2
        assert isinstance(x, torch.Tensor)
        self.x = x
        self.device = x.device
        self.graph_size = x.shape[0]
        self.k = k  # used for k-Nearest-Neighbor

        if dist_matrix is None:
            self.dist_matrix = torch.cdist(
                x.type(torch.float64) * 1000,
                x.type(torch.float64) * 1000,
            ).type(torch.float32)
        else:
            assert isinstance(dist_matrix, torch.Tensor)
            self.dist_matrix = dist_matrix

        if knn_neighbor is None:
            self.knn_neighbor = self.dist_matrix.topk(k=self.k + 1, largest=False).indices[:, 1:]
        else:
            assert isinstance(knn_neighbor, torch.Tensor)
            self.knn_neighbor = knn_neighbor

        self.numpy_knn_neighbor = self.knn_neighbor.cpu().numpy()
        assert len(init_tour) == 2 or len(init_tour) == 0
        self.selected_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=self.device)
        self.available_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=self.device)
        self.neighbor_coord = torch.zeros((x.shape[0], 4), dtype=torch.float32, device=self.device)

        # start with a 2-city-tour or empty tour
        self.current_tour = init_tour.copy()
        self.current_num_cities = len(self.current_tour)
        # to make final return equals to tour length
        self.current_tour_len = get_tour_distance(init_tour, self.x) / 1000
        update_neighbor_coord_(self.neighbor_coord, self.current_tour, self.x)
        self.mask(self.current_tour)

    def move_to(self, new_path: List[int]) -> None:
        """state transition given current state (partial tour) and action (new path)"""
        if self.current_num_cities == 0:
            self.current_tour = new_path
        else:
            start_idx = self.current_tour.index(new_path[0])
            end_idx = self.current_tour.index(new_path[-1])
            assert start_idx != end_idx, new_path
            # assuming we always choose a fragment from left to right
            # replace the old part in current tour with the new path, there is a city-overlap between them
            if end_idx > start_idx:
                self.current_tour = self.current_tour[:start_idx]+ new_path+ self.current_tour[end_idx + 1 :]
            else:
                self.current_tour = self.current_tour[end_idx + 1 : start_idx] + new_path
        # make sure no duplicate cities
        assert np.unique(self.current_tour).shape[0] == len(self.current_tour), f"{self.current_tour}"
        # update info of current tour
        if len(self.current_tour) < self.graph_size:
            assert (
                len(self.current_tour) > self.current_num_cities
            ), f"{len(self.current_tour)}-{self.current_num_cities}-{len(new_path)}"
        else:
            assert len(self.current_tour) == self.graph_size, f"{len(self.current_tour)}-{self.graph_size}"

        self.current_num_cities = len(self.current_tour)
        self.current_tour_len = get_tour_distance(self.current_tour, self.x)
        update_neighbor_coord_(self.neighbor_coord, self.current_tour, self.x)
        # update two masks
        self.mask(new_path)

    def mask(self, new_path: List[int]) -> None:
        """update mask status w.r.t new path"""
        self.selected_mask[new_path] = True
        # update mask of available starting cities for k-NN process
        self.available_mask[new_path] = True
        avaliable_cities = torch.where(self.available_mask == True)
        available_status = torch.logical_not(self.selected_mask[self.knn_neighbor[avaliable_cities]].all(dim=1))
        self.available_mask[avaliable_cities] = available_status

    def get_nearest_cluster_city_idx(self, old_cities: List[int], new_cities: List[int]) -> int:
        """find the index of the old city nearest to given new cities"""
        assert old_cities
        assert new_cities
        city_idx = get_nearest_cluster_city_idx(self.dist_matrix, old_cities, new_cities)
        return city_idx

    def get_nearest_old_city_idx(self, predict_coord: torch.Tensor) -> int:
        """find the index of the new city nearest to given coordinates"""
        assert predict_coord.shape == (2,)
        city_idx = get_nearest_city_idx(self.x, predict_coord, self.selected_mask)

        return city_idx

    def get_nearest_new_city_idx(self, predict_coord: torch.Tensor) -> int:
        """find the index of the new city nearest to given coordinates"""
        assert predict_coord.shape == (2,)
        city_idx = get_nearest_city_idx(self.x, predict_coord, ~self.selected_mask)

        return city_idx

    def get_nearest_new_city_coord(self, predict_coord: torch.Tensor) -> int:
        """find the coordinate of the available city nearest to given coordinates"""

        city_idx = self.get_nearest_new_city_idx(predict_coord)
        city_coord = self.x[city_idx]
        assert city_coord.shape == (2,)

        return city_coord

    def get_nearest_avail_city_idx(self, predict_coord: torch.Tensor) -> int:
        """find the index of the available city nearest to given coordinates"""
        assert predict_coord.shape == (2,)
        city_idx = get_nearest_city_idx(self.x, predict_coord, self.available_mask)

        return city_idx

    def get_nearest_avail_city_coord(self, predict_coord: torch.Tensor) -> int:
        """find the coordinate of the available city nearest to given coordinates"""

        city_idx = self.get_nearest_avail_city_idx(predict_coord)
        city_coord = self.x[city_idx]
        assert city_coord.shape == (2,)

        return city_coord

    def to_tensor(self) -> torch.Tensor:  # concatenate `x`, `selected_mask` to produce an state
        available_mask = self.available_mask
        return torch.cat([self.x, self.selected_mask[:, None], available_mask[:, None], self.neighbor_coord, ], dim=1)

    def to_device_(self, device=None):
        self.x = self.x.to(device)
        self.device = self.x.device
        self.dist_matrix = self.dist_matrix.to(device)
        self.knn_neighbor = self.knn_neighbor.to(device)
        self.selected_mask = self.selected_mask.to(device)
        self.available_mask = self.available_mask.to(device)
        self.neighbor_coord = self.neighbor_coord.to(device)

    def __repr__(self) -> str:
        return (
            f"Large Scale TSP state, num_cities: {self.x.shape[0]},"
            f"current_tour: {len(self.current_tour)} - {self.current_tour_len}"
        )


def get_tour_distance(tour, coords):
    cur_tour = coords[tour]
    prev_tour = coords[torch.tensor(tour[-1:] + tour[:-1])]
    return (((cur_tour - prev_tour) ** 2).sum(dim=1) ** 0.5).sum(dim=0)


@torch.jit.script
def update_neighbor_coord_(neighbor_coord: torch.Tensor, current_tour: List[int], nodes_coord: torch.Tensor):
    assert nodes_coord.ndim == 2, nodes_coord.shape
    num_nodes = nodes_coord.shape[0]
    node_dim = nodes_coord.shape[1]
    num_edges = len(current_tour)
    tour = torch.tensor(current_tour, dtype=torch.int64, device=nodes_coord.device)
    curr_pre_next = torch.stack([tour, torch.roll(tour, -1), torch.roll(tour, 1)], dim=1)
    assert curr_pre_next.shape[1] == 3, curr_pre_next.shape
    pre_coord = nodes_coord.gather(index=curr_pre_next[:, 1][:, None].expand(-1, node_dim), dim=0)
    next_coord = nodes_coord.gather(index=curr_pre_next[:, 2][:, None].expand(-1, node_dim), dim=0)
    pre_next_coord = torch.cat([pre_coord, next_coord], dim=-1)
    neighbor_coord.scatter_(index=curr_pre_next[:, 0][:, None].expand(-1, 2 * node_dim), src=pre_next_coord, dim=0, )


@torch.jit.script
def get_nearest_city_idx(x, predict_coord, mask):
    """find the city nearest to given coordinates, return its coordinates"""
    assert x.ndim == 2
    assert predict_coord.shape == (2,)
    assert mask.ndim == 1
    masked_cities = torch.where(mask == torch.tensor(True))[0]
    dist_matrix = torch.cdist(
        x[masked_cities].type(torch.float64), predict_coord[None, :].type(torch.float64)
    ).type(x.dtype)  # shape=[len(available_idx), 1]
    nearest_city = masked_cities[dist_matrix.argmin()]

    return nearest_city.item()


class Env:
    def __init__(self, k, frag_len=200, max_new_nodes=160, max_improvement_step=5):
        self.k = k
        self.frag_len = frag_len  # length of the fragment sent to low-level agent
        self.max_new_nodes = max_new_nodes
        self.max_improvement_step = max_improvement_step

    def reset(self, x: torch.Tensor, no_depot=False) -> Tuple[LargeState, float, bool]:
        self.x = x.type(torch.float32)
        self.device = x.device
        self.graph_size = self.N = x.shape[0]
        self.node_dim = self.C = x.shape[1]
        # for numerical stability
        # dist_matrix = torch.cdist(x.type(torch.float64) * 1000,
        #                           x.type(torch.float64) * 1000).type(torch.float32)
        # dist_matrix.fill_diagonal_(float("inf"))  # [N,N] -> [N,k]
        # knn_neighbor = dist_matrix.topk(k=self.k, largest=False).indices  # [N,k]

        # modified dist_matrix variable
        dist_matrix = torch.zeros(self.N, self.k).to(x.device)
        knn_neighbor = torch.zeros(self.N, self.k).long().to(x.device)
        grid_k = round((self.N / (4 * self.k)) ** 0.5)
        grids = [[[] for _ in range(grid_k)] for _ in range(grid_k)]
        for idx, (xi, yi) in enumerate(torch.div(x, 1 / grid_k, rounding_mode='floor').long()):
            if xi == grid_k:
                xi -= 1
            if yi == grid_k:
                yi -= 1
            grids[xi][yi] += [idx]
        for xi in range(grid_k):
            for yi in range(grid_k):
                nodes = torch.tensor(grids[xi][yi])
                if len(nodes) == 0:
                    continue
                step = 1
                while True:  # collect nbs
                    nbs = []
                    for nxi in range(max(0, xi - step), min(grid_k, xi + step + 1)):
                        for nyi in range(max(0, yi - step), min(grid_k, yi + step + 1)):
                            nbs += grids[nxi][nyi]
                    if len(nbs) > 2 * self.k:
                        break
                    else:
                        step += 1
                nbs = torch.tensor(nbs).to(x.device)
                nb_dists = ((x[nodes].unsqueeze(1) - x[nbs]) ** 2).sum(dim=2) ** 0.5
                vals, indices = nb_dists.topk(self.k + 1, largest=False)
                dist_matrix[nodes] = vals[:, 1:]
                knn_neighbor[nodes] = nbs[indices][:, 1:]
        dist_matrix *= 1000

        if no_depot:
            init_tour = []
        else:
            depot = 0
            init_tour = [depot, knn_neighbor[depot][0].item()]
        self.state = LargeState(x=self.x, k=self.k, init_tour=init_tour,
                                dist_matrix=dist_matrix, knn_neighbor=knn_neighbor)
        self._improvement_steps = 0

        return self.state.to_tensor()

    def append_selected(self, neighbor, selected_set, selected_mask):
        result = []
        for n in neighbor:
            if n not in selected_set and not selected_mask[n]:
                result.append(n)
        return result

    def get_fragment_knn(self, predict_coord):
        if self.done:  # early return if already done
            return []
        assert predict_coord.ndim == 1
        assert predict_coord.min() >= 0
        assert predict_coord.min() <= 1
        if not self._construction_done:
            # choose starting city based on the model predicted coordinate
            nearest_new_city = self.state.get_nearest_new_city_idx(predict_coord)
            if self.state.current_num_cities != 0:
                start_city = self.state.get_nearest_old_city_idx(self.x[nearest_new_city])
                new_city_start = self.state.get_nearest_new_city_idx(self.x[start_city])
            else:
                start_city = None
                new_city_start = nearest_new_city
            # search for new cities with k-NN heuristic
            max_cities = max(self.max_new_nodes, self.frag_len - self.state.current_num_cities)
            new_cities = []
            knn_deque = []
            selected_set = set()
            knn_deque.append(new_city_start)
            selected_set.add(new_city_start)
            selected_mask = self.state.selected_mask.cpu().numpy()
            while len(knn_deque) != 0 and len(new_cities) < max_cities:
                node = knn_deque.pop(0)
                new_cities.append(node)
                res = self.append_selected(self.state.numpy_knn_neighbor[node], selected_set, selected_mask)
                knn_deque.extend(res)
                selected_set.update(res)
        else:
            # refine a old fragment
            assert not self._improvement_done
            nearest_old_city = self.state.get_nearest_old_city_idx(predict_coord)
            new_cities = []
        # extend fragment to `frag_len` with some old cities
        if self.state.current_num_cities != 0:
            fragment = self._extend_fragment(start_city, new_cities)
        else:
            fragment = new_cities

        for i in fragment:
            if i in new_cities:
                assert not self.state.selected_mask[i]
            else:
                assert self.state.selected_mask[i]

        return (torch.tensor(fragment, device=self.device),
                torch.tensor(new_cities, device=self.device),
                self.unscale_action(self.x[start_city]))

    def step(self, predict_coord, solver, greedy_reward=False, average_reward=False, float_available_status=False):
        # raise exception if already done
        if self.done:
            raise RuntimeError("Environment has terminated!")

        # get fragment
        predict_coord = self.scale_action(predict_coord)
        fragment, _ = self.get_fragment_knn(predict_coord)
        # solve subproblem
        if isinstance(solver, RLSolver):
            new_paths, _ = solver.solve(self.x[None, ...], fragment[None, ...])
            new_path = new_paths[0]
        else:
            new_path = solver.solve(self.x, fragment)
        return self._step(new_path, greedy_reward, average_reward)

    def _step(self, new_path, greedy_reward=False, average_reward=False):
        """step function for VecEnv"""
        # raise exception if already done
        if self.done:
            raise RuntimeError("Environment has terminated!")
        if self._construction_done and not self._improvement_done:
            self._improvement_steps += 1
        # record old states
        old_len = self.state.current_tour_len
        old_num_cities = self.state.current_num_cities
        # update state
        self.state.move_to(new_path)
        self.state.current_tour_len = get_tour_distance(self.state.current_tour, self.state.x) / 1000

        if greedy_reward:
            # len_rl = (
            #     get_tour_distance(new_path, self.state.x)
            #     - self.state.dist_matrix[new_path[0], new_path[-1]]
            # ) / 1000
            td = get_tour_distance(new_path, self.state.x)
            nbs = self.state.knn_neighbor[new_path[0]].tolist()
            if new_path[-1] in nbs:
                idx = nbs.index(new_path[-1])
                md = self.state.dist_matrix[new_path[0], idx]
            else:
                md = 1000 * (((self.state.x[new_path[0]] - self.state.x[new_path[-1]]) ** 2).sum() ** 0.5).item()
            len_rl = (td - md) / 1000
            _, len_greedy = GreedySolver().solve(self.x, new_path)
            reward = len_greedy - len_rl
        else:
            reward = old_len - self.state.current_tour_len
        if average_reward:
            added_num_cities = self.state.current_num_cities - old_num_cities
            reward /= added_num_cities

        return self.state.to_tensor(), reward, self.done

    @property
    def _construction_done(self):
        return self.state.current_num_cities == self.graph_size

    @property
    def _improvement_done(self):
        return self._improvement_steps >= self.max_improvement_step

    @property
    def done(self):
        return self._construction_done and self._improvement_done

    @staticmethod
    def scale_action(a):  # scale action from [-1, 1] to [0, 1]
        return a * 0.5 + 0.5

    @staticmethod
    def unscale_action(a):  # unscale action from [0, 1] to [-1, 1]
        return a * 2 - 1

    def _extend_fragment(self, nearest_city: int, new_cities: List[int]):
        nearest_idx = self.state.current_tour.index(nearest_city)
        total_extend_len = self.frag_len - len(new_cities)
        assert total_extend_len > 0, total_extend_len
        offset = nearest_idx - (total_extend_len // 2)
        reorder_tour = (
            self.state.current_tour[offset:] + self.state.current_tour[:offset]
        )
        assert len(reorder_tour) == len(self.state.current_tour)

        fragment = (
            reorder_tour[: (total_extend_len // 2)]
            + new_cities
            + reorder_tour[(total_extend_len // 2) : total_extend_len]
        )

        assert len(fragment) == total_extend_len + len(
            new_cities
        ), f"{len(fragment)}-{total_extend_len}-{len(new_cities)}"

        assert len(fragment) == self.frag_len, len(fragment)
        assert np.unique(fragment).shape[0] == len(fragment), np.unique(fragment).shape

        return fragment

    def to_device_(self, device: Optional[Union[torch.device, str]] = None):
        self.x = self.x.to(device)
        self.device = self.x.device
        self.state.to_device_(device)


class VecEnv:
    def __init__(self, k, frag_len=200, max_new_nodes=160, max_improvement_step=5,
                 auto_reset=False, no_depot=False):
        self.k = k  # num of nearest neighbor
        self.frag_len = frag_len
        self.max_new_nodes = max_new_nodes
        self.max_improvement_step = max_improvement_step
        self.auto_reset = auto_reset
        self.no_depot = no_depot

    def reset(self, x: torch.Tensor):
        self.x = x
        self.device = x.device
        self.batch_size = self.B = x.shape[0]
        self.graph_size = self.N = x.shape[1]
        self.node_dim = self.C = x.shape[2]
        self.envs = []  # batch envs
        for _ in range(self.batch_size):
            self.envs.append(Env(self.k, self.frag_len, self.max_new_nodes, self.max_improvement_step))
        states = [self.envs[i].reset(x[i], self.no_depot) for i in range(self.B)]
        self.states = states

        return torch.stack(states).type(torch.float32)  # [16, 10000, 8]

    @property
    def done(self):
        return all(e.done for e in self.envs)

    def step(self, predict_coords, solver, greedy_reward= False, average_reward= False,
             float_available_status=False, tsp_data: Callable = None,
             frag_buffer = None, log: Callable = None):
        if self.done:
            raise RuntimeError("Environment has terminated!")
        assert predict_coords.ndim == 2
        assert predict_coords.shape[0] == self.batch_size
        actions = [Env.scale_action(coord) for coord in predict_coords]

        # get active indecies
        if not self.auto_reset:
            active_idx = [i for i in range(self.B) if not self.envs[i].done]
            active_envs = [self.envs[i] for i in active_idx]
            active_acts = [actions[i] for i in active_idx]
        else:
            active_idx = list(range(self.B))
            active_envs = self.envs
            active_acts = actions

        # get fragment of cities with model predicted coordinate
        fragments = []
        new_cities = []
        start_cities = []
        for env, a in zip(active_envs, active_acts):
            frag, new_city, start_city = env.get_fragment_knn(a)
            fragments.append(frag)
            new_cities.append(new_city)
            start_cities.append(start_city)

        # solve the fragment of cities to get a new path
        if isinstance(solver, RLSolver):
            new_paths, _ = solver.solve(self.x[active_idx], torch.stack(fragments), frag_buffer)
        else:
            new_paths = []
            for env, frag in zip(self.envs, fragments):
                res = solver.solve(env.x, frag)
                new_paths.append(res)
        # env update states and rewards given the new path
        outputs = []
        for env, p in zip(active_envs, new_paths):
            res = env._step(p, greedy_reward, average_reward)
            outputs.append(res)

        if not self.auto_reset:
            states = [None] * self.B
            rewards = [0.0] * self.B
            dones = [False] * self.B
            count = 0
            for i in range(self.B):
                if i in active_idx:
                    states[i] = outputs[count][0]
                    rewards[i] = outputs[count][1]
                    dones[i] = outputs[count][2]
                    count += 1
            assert count == len(active_idx), f"{count}!={len(active_idx)}"
        else:
            states = [output[0] for output in outputs]
            rewards = [output[1] for output in outputs]
            dones = [output[2] for output in outputs]
            for i in range(self.B):
                env = self.envs[i]
                if env.done:
                    log("explore/tour_length", env.state.current_tour_len.item(), on_step=True)
                    data = tsp_data().squeeze().to(self.device)
                    state = env.reset(data, self.no_depot)
                    self.x[i] = data
                    states[i] = state

        for i in active_idx:
            self.states[i] = states[i]

        return (torch.stack(self.states).type(torch.float32),
                torch.tensor(rewards, dtype=torch.float32, device=self.device),
                torch.tensor(dones, dtype=torch.float32, device=self.device),
                {"fragments" : fragments,
                 "new_cities": new_cities,
                 "start_city": torch.stack(start_cities), })

    def to_device_(self, device: Optional[Union[torch.device, str]] = None):
        for env in self.envs:
            env.to_device_(device)


class LowLevelSolver(ABC):
    @abstractmethod
    def solve(self, x, fragment, frag_buffer):
        pass


@contextmanager
def evaluating(net):  # Temporarily switch to evaluation mode
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


class RLSolver(LowLevelSolver):
    def __init__(self, low_level_model, sample_size=200):
        self._solver_model = low_level_model
        self._sample_size = max(sample_size, 2)

    def solve(self, data, fragment, frag_buffer):
        node_pos = torch.gather(input=data, index=fragment[..., None].expand(-1, -1, data.shape[-1]), dim=1)
        assert node_pos.ndim == 3
        device = self._solver_model.device
        B = node_pos.shape[0]
        N = node_pos.shape[1]
        x = node_pos.to(device=device)
        x1_min = x[..., 0].min(dim=-1, keepdim=True)[0]
        x2_min = x[..., 1].min(dim=-1, keepdim=True)[0]
        x1_max = x[..., 0].max(dim=-1, keepdim=True)[0]
        x2_max = x[..., 1].max(dim=-1, keepdim=True)[0]
        s = 0.9 / torch.maximum(x1_max - x1_min, x2_max - x2_min)
        x_new = torch.empty_like(x)
        x_new[..., 0] = s * (x[..., 0] - x1_min) + 0.05
        x_new[..., 1] = s * (x[..., 1] - x2_min) + 0.05
        frag_buffer.update_buffer(x_new.cpu())
        source_nodes = torch.tensor([[0]], device=device).expand(B, 1)
        target_nodes = torch.tensor([[N - 1]], device=device).expand(B, 1)
        with amp.autocast(enabled=False) as a, torch.no_grad() as b, evaluating(self._solver_model) as solver:
            lengths, paths = solver(x_new.float(), source_nodes=source_nodes, target_nodes=target_nodes,
                                    val_type="x8Aug_2Traj", return_pi=True, group_size=self._sample_size)
        lengths = (lengths / s.squeeze()).detach().cpu().numpy()

        # Flip and construct tour on original graph
        out_paths = []
        paths = paths.detach().cpu().numpy()
        fragment = fragment.cpu().numpy()
        if B == 1:
            paths = [paths]
        for i, path in enumerate(paths):
            # make sure path start with `0` and end with `N-1`
            zero_idx = np.nonzero(path == 0)[0][0]
            path = np.roll(path, -zero_idx)
            assert path[0] == 0, f"{zero_idx=}, {path=}"
            if path[-1] != N - 1:
                path = np.roll(path, -1)
                path = np.flip(path)
            assert path[-1] == N - 1, f"{path=}"
            assert np.unique(path).shape[0] == path.shape[0] == N
            assert path.shape[0] == fragment[i].shape[0] == N
            new_path = [fragment[i][j] for j in path]
            out_paths.append(new_path)

        return out_paths, lengths


class HTSP_PPO(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.low_level_model = PathSolver.load_from_checkpoint(cfg.low_level_load_path)
        self.low_level_solver = RLSolver(self.low_level_model, cfg.low_level_sample_size)

        if cfg.encoder_type == "pixel":
            self.encoder = IMPALAEncoder(input_dim=cfg.input_dim, embedding_dim=cfg.embedding_dim)
            self.encoder_target = IMPALAEncoder(input_dim=cfg.input_dim, embedding_dim=cfg.embedding_dim)
        else:
            raise TypeError(f"Encoder type {cfg.encoder_type} not supported!")
        self.actor = ActorPPO(state_dim=cfg.embedding_dim,mid_dim=cfg.acotr_mid_dim,
                                     action_dim=cfg.nb_actions,init_a_std_log=cfg.init_a_std_log)
        self.critic = CriticPPO(state_dim=cfg.embedding_dim, mid_dim=cfg.hidden_dim,
                                       _action_dim=cfg.nb_actions)
        self.critic_target = CriticPPO(state_dim=cfg.embedding_dim, mid_dim=cfg.hidden_dim,
                                              _action_dim=cfg.nb_actions)
        hard_update(self.critic_target, self.critic)
        hard_update(self.encoder_target, self.encoder)

        self.mse_loss = nn.MSELoss()
        self.criterion = nn.MSELoss()
        self.lambda_entropy = cfg.lambda_entropy

        self.cfg = cfg
        self.save_hyperparameters(cfg)
        # Memory
        self.memory = []
        self.traj_list = [[[] for _ in range(cfg.experience_items)] for _ in range(cfg.env_num)]  # [16,6]
        self.frag_buffer = FragmentBuffer(cfg.low_level_buffer_size, cfg.frag_len, cfg.node_dim)
        self.val_frag_buffer = FragmentBuffer(cfg.low_level_buffer_size, cfg.frag_len, cfg.node_dim)
        # Environment
        self.env = VecEnv(k=cfg.k, frag_len=cfg.frag_len, max_new_nodes=cfg.max_new_nodes, auto_reset=True,
                          max_improvement_step=cfg.max_improvement_step, no_depot=self.cfg.no_depot)
        self.states = None
        self.__tsp_data_epoch = None
        self.__tsp_used_times = 0
        # Optimizer
        self.automatic_optimization = False

    def select_action(self, state):
        """
        Select an action given a state.

        :param state: a state in a shape (state_dim, ).
        :return: a action in a shape (action_dim, ) where each action is clipped into range(-1, 1).
        """
        with torch.no_grad():
            s_tensor = torch.as_tensor(state[np.newaxis], device=self.device)
            graph_feat = self.encoder(s_tensor)
            a_tensor = self.actor(graph_feat)
        action = a_tensor.detach().cpu().numpy()
        return action.tanh()

    def select_actions(self, state):
        """
        Select actions given an array of states.

        :param state: an array of states in a shape (batch_size, state_dim, ).
        :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
        """
        with torch.no_grad():
            state = state.to(self.device)
            graph_feat = self.encoder(state)
            action, noise = self.actor.get_action(graph_feat)
        return action.detach(), noise.detach()

    @torch.no_grad()
    def explore_vec_env(self, target_steps):
        traj_list = []
        experience = []
        # initialize env
        if self.states is None:
            tsp_data = torch.cat([self.tsp_data() for _ in range(self.cfg.env_num)], dim=0).to(self.device)
            self.states = self.env.reset(tsp_data)

        self.env.to_device_(self.device)
        step = 0
        ten_s = self.states
        last_done = torch.zeros(self.cfg.env_num, dtype=torch.int, device=self.device)
        while step < target_steps:
            ten_a, ten_n = self.select_actions(ten_s)
            ten_s_next, ten_rewards, ten_dones, info = self.env.step(
                ten_a.tanh(),
                self.low_level_solver,
                self.cfg.greedy_reward,
                self.cfg.average_reward,
                self.cfg.float_available_status,
                self.tsp_data,
                self.frag_buffer,
                self.log,
            )
            traj_list.append((ten_s.clone(), ten_rewards.clone(), ten_dones.clone(), ten_a, ten_n, info["start_city"]))
            ten_s = ten_s_next

            step += 1
            last_done[torch.where(ten_dones)[0]] = step  # behind `step+=1`

        self.states = ten_s

        buf_srdan = list(map(list, zip(*traj_list)))
        assert len(buf_srdan) == self.cfg.experience_items
        del traj_list[:]

        buf_srdan[0] = torch.stack(buf_srdan[0])
        buf_srdan[1] = (torch.stack(buf_srdan[1]) * self.cfg.reward_scale).unsqueeze(2)
        buf_srdan[2] = ((1 - torch.stack(buf_srdan[2])) * self.cfg.gamma).unsqueeze(2)
        buf_srdan[3] = torch.stack(buf_srdan[3])
        buf_srdan[4] = torch.stack(buf_srdan[4])
        buf_srdan[5] = torch.stack(buf_srdan[5])

        experience.append(self.splice_trajectory(buf_srdan, last_done)[0])
        self.env.to_device_("cpu")
        return experience

    def get_reward_sum_gae(self, buf_len, ten_reward, ten_mask, ten_value):
        """
        Calculate the **reward-to-go** and **advantage estimation** using GAE.

        :param buf_len: the length of the ``ReplayBuffer``.
        :param buf_reward: a list of rewards for the state-action pairs.
        :param buf_mask: a list of masks computed by the product of done signal and discount factor.
        :param buf_value: a list of state values estimiated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        buf_r_sum = torch.empty(
            buf_len, dtype=torch.float32, device=ten_reward.device
        )  # old policy value
        buf_adv_v = torch.empty(
            buf_len, dtype=torch.float32, device=ten_reward.device
        )  # advantage value

        pre_r_sum = 0
        pre_adv_v = 0  # advantage value of previous step
        ten_bool = torch.not_equal(ten_mask, 0).float()
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
            buf_adv_v[i] = ten_reward[i] + ten_bool[i] * (pre_adv_v - ten_value[i])
            pre_adv_v = ten_value[i] + buf_adv_v[i] * self.cfg.lambda_gae_adv
        return buf_r_sum, buf_adv_v

    def splice_trajectory(self, buf_srdan, last_done):
        out_srdan = []
        for j in range(self.cfg.experience_items):
            cur_items = []
            buf_items = buf_srdan.pop(0)  # buf_srdan[j]

            for env_i in range(self.cfg.env_num):
                last_step = last_done[env_i]

                pre_item = self.traj_list[env_i][j]
                if len(pre_item):
                    cur_items.append(pre_item)

                cur_items.append(buf_items[:last_step, env_i])
                self.traj_list[env_i][j] = buf_items[last_step:, env_i]
                if j == 2:
                    assert buf_items[last_step - 1, env_i] == 0.0, (
                        buf_items[last_step - 1, env_i],
                        last_step,
                        env_i,
                    )

            out_srdan.append(torch.vstack(cur_items).detach().cpu())

        del buf_srdan
        return [
            out_srdan,
        ]

    def _tsp_dataloader(self, tsp_batch_size):
        dataset = TSPDataset(size=self.cfg.graph_size, node_dim=self.cfg.node_dim, num_samples=self.cfg.epoch_size,
                             data_distribution=self.cfg.data_distribution)
        return DataLoader(dataset, batch_size=tsp_batch_size, num_workers=0, pin_memory=True)

    def tsp_data(self):
        # get one tsp graph at epoch start
        from baselines.htsp.rl4cop.cop_utils import augment_xy_data_by_8_fold
        if self.__tsp_data_epoch is None or self.__tsp_data_epoch != self.current_epoch:
            self._stored_data = torch.rand(1, self.cfg.graph_size, self.cfg.node_dim)
            self._stored_data = augment_xy_data_by_8_fold(self._stored_data)
            self.__tsp_data_epoch = self.current_epoch
            self.__tsp_used_times = 0

        # iterate over the 8 augmentations
        return_data = self._stored_data[
            self.__tsp_used_times : self.__tsp_used_times + 1
        ]
        self.__tsp_used_times += 1

        if self.__tsp_used_times >= 8:
            self.__tsp_used_times = 0

        return return_data.detach().clone()

    def forward(self, states):
        graph_feat = self.encoder(states)
        action = self.actor(graph_feat)
        return action


class FragmentBuffer:
    def __init__(self, max_len, frag_len, node_dim=2):
        self.max_len = max_len
        self.frag_len = frag_len
        self.node_dim = node_dim
        self.frag_buffer = torch.empty((max_len, frag_len, node_dim))
        self.if_full = False
        self.now_len = 0
        self.next_idx = 0

    def update_buffer(self, fragments: torch.Tensor) -> None:
        size = fragments.shape[0]
        assert fragments.shape[1] == self.frag_len
        assert fragments.shape[2] == self.node_dim
        next_idx = self.next_idx + size
        if next_idx > self.max_len:
            self.frag_buffer[self.next_idx : self.max_len] = fragments[
                : self.max_len - self.next_idx
            ]
            self.if_full = True
            next_idx = next_idx - self.max_len
            self.frag_buffer[0:next_idx] = fragments[-next_idx:]
        else:
            self.frag_buffer[self.next_idx : next_idx] = fragments
        self.next_idx = next_idx
        self.update_now_len()

    def update_now_len(self):
        self.now_len = self.max_len if self.if_full else self.next_idx


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
