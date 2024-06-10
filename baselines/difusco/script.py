import torch
import numpy as np
import scipy.sparse
import scipy.spatial
from argparse import ArgumentParser
from sklearn.neighbors import KDTree
from torch_geometric.data import Data as GraphData
from baselines.difusco.diffusion_schedulers import InferenceSchedule
from baselines.difusco.tsp_utils import merge_tours


def arg_parser():
    parser = ArgumentParser(description='Train a Pytorch-Lightning diffusion model on a TSP dataset.')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_scheduler', type=str, default='constant')

    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--use_activation_checkpoint', action='store_true')

    parser.add_argument('--diffusion_type', type=str, default='gaussian')
    parser.add_argument('--diffusion_schedule', type=str, default='linear')
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--inference_diffusion_steps', type=int, default=1000)
    parser.add_argument('--inference_schedule', type=str, default='linear')
    parser.add_argument('--inference_trick', type=str, default="ddim")
    parser.add_argument('--sequential_sampling', type=int, default=1)
    parser.add_argument('--parallel_sampling', type=int, default=1)

    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--sparse_factor', type=int, default=-1)
    parser.add_argument('--aggregation', type=str, default='sum')
    parser.add_argument('--two_opt_iterations', type=int, default=1000)

    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--resume_weight_only', action='store_true')

    args = parser.parse_args()
    return args


def parse_data(idx, coords, sparse_factor=-1):
    assert len(coords.shape) == 2
    node_num = len(coords)
    # tour = np.concatenate((np.arange(node_num), np.array([0]))).astype(int)
    tour = torch.arange(node_num + 1).long()
    tour[-1] = 0

    if sparse_factor <= 0:  # Return a densely connected graph
        adj_matrix = torch.zeros(node_num, node_num).float()
        for i in range(node_num):
            adj_matrix[tour[i], tour[i + 1]] = 1
        return (coords, adj_matrix, tour)
    else:  # Return a sparse graph where each node is connected to its k nearest neighbors
        kdt = KDTree(coords.numpy(), leaf_size=30, metric='euclidean')
        dis_knn, idx_knn = kdt.query(coords, k=sparse_factor, return_distance=True)

        edge_index_0 = torch.arange(node_num).unsqueeze(1).repeat(1, sparse_factor).flatten()
        edge_index_1 = torch.from_numpy(idx_knn.flatten())
        edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

        tour_edges = np.zeros(node_num, dtype=np.int64)
        tour_edges[tour[:-1]] = tour[1:]
        tour_edges = torch.from_numpy(tour_edges)
        tour_edges = tour_edges.unsqueeze(1).repeat(1, sparse_factor).flatten()
        tour_edges = torch.eq(edge_index_1, tour_edges).unsqueeze(1)
        graph_data = GraphData(x=coords, edge_index=edge_index, edge_attr=tour_edges)

        point_indicator = torch.tensor([idx]).long()
        edge_indicator = torch.tensor(edge_index.shape[1]).long()

        return (graph_data, point_indicator, edge_indicator, tour)


def inference(model, batch, device):
    edge_index = None
    np_edge_index = None

    # parse input data
    if not model.sparse:
        points, adj_matrix, gt_tour = batch
        np_points = points.cpu().numpy()
    else:
        graph_data, point_indicator, edge_indicator, gt_tour = batch
        route_edge_flags = graph_data.edge_attr
        points = graph_data.x
        edge_index = graph_data.edge_index
        num_edges = edge_index.shape[1]
        batch_size = point_indicator.shape[0]
        adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
        assert len(points.shape) == 2 and points.shape[1] == 2
        assert len(edge_index.shape) == 2 and edge_index.shape[0] == 2
        np_points = points.cpu().numpy()
        np_edge_index = edge_index.cpu().numpy()

    if model.args.parallel_sampling > 1:
        if not model.sparse:
            points = points.repeat(model.args.parallel_sampling, 1, 1)
        else:
            points = points.repeat(model.args.parallel_sampling, 1)
            edge_index = model.duplicate_edge_index(edge_index, np_points.shape[0], device)

    greedy_tours = []
    for _ in range(model.args.sequential_sampling):
        xt = torch.randn_like(adj_matrix.float()).to(model.device)
        if model.args.parallel_sampling > 1:
            if not model.sparse:
                xt = xt.repeat(model.args.parallel_sampling, 1, 1)
            else:
                xt = xt.repeat(model.args.parallel_sampling, 1)
            xt = torch.randn_like(xt)

        if model.diffusion_type == 'categorical':
            xt = (xt > 0).long()
        else:
            assert model.diffusion_type == 'gaussian'
            xt.requires_grad = True

        if model.sparse:
            xt = xt.reshape(-1)

        # Diffusion iterations
        time_schedule = InferenceSchedule(inference_schedule=model.args.inference_schedule,
                                          T=model.diffusion.T, inference_T=model.args.inference_diffusion_steps)
        for i in range(model.args.inference_diffusion_steps):
            t1, t2 = time_schedule(i)
            t1 = np.array([t1]).astype(int)
            t2 = np.array([t2]).astype(int)

            if model.diffusion_type == 'categorical':
                xt = model.categorical_denoise_step(points, xt, t1, device, edge_index, target_t=t2)
            else:
                assert model.diffusion_type == 'gaussian'
                xt = model.gaussian_denoise_step(points, xt, t1, device, edge_index, target_t=t2)

        if model.diffusion_type == 'categorical':
            adj_mat = xt.float().cpu().detach().numpy() + 1e-6
        else:
            assert model.diffusion_type == 'gaussian'
            adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
        torch.cuda.empty_cache()

        tours, merge_iterations = merge_tours(adj_mat, np_points, np_edge_index, sparse_graph=model.sparse,
                                              parallel_sampling=model.args.parallel_sampling)
        assert len(tours) == model.args.parallel_sampling == 1
        greedy_tours += [tours[0]]

    assert len(greedy_tours) == model.args.sequential_sampling
    return torch.tensor(greedy_tours).long()[:, :-1]
