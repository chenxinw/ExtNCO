import warnings
import time
import torch
import numpy as np
import scipy.sparse
import scipy.spatial
from tqdm import tqdm
from multiprocessing import Pool
from baselines.difusco.cython_merge.cython_merge import merge_cython


def cuda_2_opt(points, tour, thr=0, iter_num=None, hide_bar=True):
    run_time = time.time()
    assert len(points) == len(tour)
    tour = torch.tensor(tour.tolist() + [tour[0].item()]).long().to(points.device)
    with torch.inference_mode():
        count = 0
        gain = -1.0
        total_gain = 0
        pbar = tqdm(desc='[2-opt searching]', unit='round', disable=hide_bar)
        while gain < 0.0:
            points_i = points[tour[:-1]].reshape(-1, 1, 2)
            points_j = points[tour[:-1]].reshape(1, -1, 2)
            points_i_plus_1 = points[tour[1:]].reshape(-1, 1, 2)
            points_j_plus_1 = points[tour[1:]].reshape(1, -1, 2)

            gains = torch.sqrt(torch.sum((points_i - points_j) ** 2, dim=-1))
            gains += torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, dim=-1))
            gains -= torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, dim=-1))
            gains -= torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, dim=-1))
            gains = torch.triu(gains, diagonal=2)

            gain = torch.min(gains)
            min_idx = torch.argmin(gains.flatten())
            min_i = torch.div(min_idx, len(points), rounding_mode='floor')
            min_j = torch.remainder(min_idx, len(points))

            if gain < thr:
                tour[min_i + 1:min_j + 1] = torch.flip(tour[min_i + 1:min_j + 1], dims=(0,))
                total_gain += gain
                count += 1
                pbar.set_postfix(gain=gain.item(), total_gain=total_gain.item())
                pbar.update(1)
            else:
                break
            if iter_num is None:
                continue
            if count == iter_num:
                pbar.close()
                break
    assert len(tour) == len(points) + 1
    assert tour[0] == tour[-1]
    return tour[:-1].cpu(), time.time() - run_time, count


# def cuda_2_opt(points, tour, max_iterations=1000, device='cpu'):
#     # tour = tour.copy()
#     with torch.inference_mode():
#         # cuda_points = torch.from_numpy(points).to(device)
#         # cuda_tour = torch.from_numpy(tour).to(device)
#         batch_size = tour.shape[0]
#         iterator = 0
#         min_change = -1.0
#         while min_change < 0.0:
#             points_i = points[tour[:, :-1].reshape(-1)].reshape((batch_size, -1, 1, 2))
#             points_j = points[tour[:, :-1].reshape(-1)].reshape((batch_size, 1, -1, 2))
#             points_i_plus_1 = points[tour[:, 1:].reshape(-1)].reshape((batch_size, -1, 1, 2))
#             points_j_plus_1 = points[tour[:, 1:].reshape(-1)].reshape((batch_size, 1, -1, 2))
#
#             valid_change = torch.sqrt(torch.sum((points_i - points_j) ** 2, dim=-1))
#             valid_change += torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, dim=-1))
#             valid_change -= torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, dim=-1))
#             valid_change -= torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, dim=-1))
#             valid_change = torch.triu(valid_change, diagonal=2)
#
#             min_change = torch.min(valid_change)
#             flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
#             min_i = torch.div(flatten_argmin_index, len(points), rounding_mode='floor')
#             min_j = torch.remainder(flatten_argmin_index, len(points))
#
#             if min_change < -1e-6:
#                 for i in range(batch_size):
#                     tour[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(tour[i, min_i[i] + 1:min_j[i] + 1], dims=(0,))
#                 iterator += 1
#             else:
#                 break
#
#             if iterator >= max_iterations:
#                 break
#         tour = tour.cpu()
#     assert batch_size == 1 and len(tour[0]) == len(points) + 1
#     return tour[0, :-1], iterator


def numpy_merge(points, adj_mat):
    dists = np.linalg.norm(points[:, None] - points, axis=-1)

    components = np.zeros((adj_mat.shape[0], 2)).astype(int)
    components[:] = np.arange(adj_mat.shape[0])[..., None]
    real_adj_mat = np.zeros_like(adj_mat)
    merge_iterations = 0
    for edge in (-adj_mat / dists).flatten().argsort():
        merge_iterations += 1
        a, b = edge // adj_mat.shape[0], edge % adj_mat.shape[0]
        if not (a in components and b in components):
            continue
        ca = np.nonzero((components == a).sum(1))[0][0]
        cb = np.nonzero((components == b).sum(1))[0][0]
        if ca == cb:
            continue
        cca = sorted(components[ca], key=lambda x: x == a)
        ccb = sorted(components[cb], key=lambda x: x == b)
        newc = np.array([[cca[0], ccb[0]]])
        m, M = min(ca, cb), max(ca, cb)
        real_adj_mat[a, b] = 1
        components = np.concatenate([components[:m], components[m + 1:M], components[M + 1:], newc], 0)
        if len(components) == 1:
            break
    real_adj_mat[components[0, 1], components[0, 0]] = 1
    real_adj_mat += real_adj_mat.T
    return real_adj_mat, merge_iterations


def cython_merge(points, adj_mat):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        real_adj_mat, merge_iterations = merge_cython(points.astype("double"), adj_mat.astype("double"))
        real_adj_mat = np.asarray(real_adj_mat)
    return real_adj_mat, merge_iterations


def merge_tours(adj_mat, np_points, edge_index_np, sparse_graph=False, parallel_sampling=1):
# """
# To extract a tour from the inferred adjacency matrix A, we used the following greedy edge insertion
# procedure.
# • Initialize extracted tour with an empty graph with N vertices.
# • Sort all the possible edges (i, j) in decreasing order of Aij/kvi − vjk (i.e., the inverse edge weight,
# multiplied by inferred likelihood). Call the resulting edge list (i1, j1),(i2, j2), . . . .
# • For each edge (i, j) in the list:
#   – If inserting (i, j) into the graph results in a complete tour, insert (i, j) and terminate.
#   – If inserting (i, j) results in a graph with cycles (of length < N), continue.
#   – Otherwise, insert (i, j) into the tour.
# • Return the extracted tour.
# """
    batch_adj_mat = np.split(adj_mat, parallel_sampling, axis=0)

    if not sparse_graph:
        batch_adj_mat = [adj_mat[0] + adj_mat[0].T for adj_mat in batch_adj_mat]
    else:
      batch_adj_mat = [
          scipy.sparse.coo_matrix(
              (adj_mat, (edge_index_np[0], edge_index_np[1])),
          ).toarray() + scipy.sparse.coo_matrix(
              (adj_mat, (edge_index_np[1], edge_index_np[0])),
          ).toarray() for adj_mat in batch_adj_mat
      ]

    batch_points = [np_points for _ in range(parallel_sampling)]

    if np_points.shape[0] > 1000 and parallel_sampling > 1:
        with Pool(parallel_sampling) as p:
          results = p.starmap(cython_merge, zip(batch_points, batch_adj_mat),)
    else:
        results = [cython_merge(_np_points, _adj_mat) for _np_points, _adj_mat in zip(batch_points, batch_adj_mat)]

    real_adj_mat, batch_merge_iterations = zip(*results)

    tours = []
    for i in range(parallel_sampling):
        tour = [0]
        while len(tour) < batch_adj_mat[i].shape[0] + 1:
            n = np.nonzero(real_adj_mat[i][tour[-1]])[0]
            if len(tour) > 1:
                n = n[n != tour[-2]]
            tour.append(n.max())
        tours.append(tour)

    merge_iterations = np.mean(batch_merge_iterations)
    return tours, merge_iterations


class TSPEvaluator(object):
    def __init__(self, points):
        self.dist_mat = scipy.spatial.distance_matrix(points, points)

    def evaluate(self, route):
        total_cost = 0
        for i in range(len(route) - 1):
            total_cost += self.dist_mat[route[i], route[i + 1]]
        return total_cost
