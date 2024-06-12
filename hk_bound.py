import math
import torch
from tqdm import tqdm
from mst import *
from graph import delaunay_neighbor


def hk_bound_np(coords, M=1000, nb_num=10):
	node_num = len(coords)
	nb_mat = np.array(delaunay_neighbor(coords, nb_num, expand=True))
	nb_dist = ((coords[nb_mat] - coords.reshape(node_num, 1, 2)) ** 2).sum(axis=-1) ** 0.5
	pis = np.zeros(node_num)
	best_hk_value = 0
	step_1 = 0

	for m in range(1, M + 1):  # iterative step to find better hk_value
		cost_mat = nb_dist + pis[nb_mat] + pis.reshape(node_num, 1).repeat(nb_num, axis=1)
		mst = kruskal(nb_mat, cost_mat, hide_bar=True)
		m1t, degrees = min_1_tree_np(mst, nb_mat, cost_mat)
		hk_value, tree_cost = tree_weight_np(m1t, degrees, nb_dist, pis)
		if hk_value > best_hk_value:
			best_hk_value = hk_value
			step_1 = tree_cost / (2 * node_num)
		scale = ((m - 1) * (2 * M - 5) / (2 * M - 2) - (m - 2) + (m - 1) * (m - 2) / (2 * (M - 1) * (M - 2)))
		step = scale * step_1
		pis = pis + step * (degrees - 2)

	return best_hk_value


def min_1_tree_np(mst, nb_mat, cost_mat):
	nb_num = nb_mat.shape[1]
	node_degrees = stat_degree_np(mst, nb_mat)
	leaf_indices = np.argwhere(node_degrees == 1).squeeze()
	costs = cost_mat[leaf_indices].flatten()
	sorted_indices = np.argsort(costs)

	cur_idx = 0
	while True:
		cur_pos = sorted_indices[cur_idx]
		src = cur_pos // nb_num
		src = leaf_indices[src]  # forgot in torch implementation!!!
		dest_idx = cur_pos % nb_num
		dest = nb_mat[src][dest_idx]
		src_idx = np.where(nb_mat[dest] == src)[0]
		if len(src_idx) == 0:  # src not in nb_mat[dest]
			if not mst[src][dest_idx]:
				mst[src][dest_idx] = True
				node_degrees[src] += 1
				node_degrees[dest] += 1
				break
		else:
			assert len(src_idx) == 1
			src_idx = src_idx.item()
		if not mst[src][dest_idx] and not mst[dest][src_idx]:
			mst[src][dest_idx] = True
			node_degrees[src] += 1
			node_degrees[dest] += 1
			break
		cur_idx += 1

	assert node_degrees.sum() == 2 * len(mst)
	return mst, node_degrees


def stat_degree_np(mst, nb_mat):
	node_num = len(mst)
	# calculate in-degree for each node
	in_degree = np.zeros(node_num, dtype=int)
	pos = np.argwhere(mst).T
	indices = nb_mat[pos[0], pos[1]]
	idx, count = np.unique(indices, return_counts=True)
	in_degree[idx] += count
	# calculate out-degree for each node and add it with in-degree
	out_degree = mst.sum(axis=1)
	return in_degree + out_degree


def tree_weight_np(m1t, degree, nb_dist, pis):
	pos = np.argwhere(m1t).T
	weights = nb_dist[pos[0], pos[1]].sum()
	value = weights + (pis * (degree - 2)).sum()

	return value, weights


def hk_bound(coords, M=500, device='cuda:0', hide_bar=False):
	nb_num = 20
	nb_mat = knn_graph(coords, k=nb_num, candidate_thr=150, device=device)
	nb_dist = ((coords[nb_mat] - coords.unsqueeze(1)) ** 2).sum(dim=-1).sqrt()
	pis = torch.zeros(coords.size(0)).to(device)
	best_hk_value = 0
	# step = 0
	step_1 = 0

	# iterative step to find better hk_value
	pbar = tqdm(total=M, desc='[HK Bound]', unit='iter', disable=hide_bar)
	for m in range(1, M + 1):
		cost_mat = nb_dist + pis[nb_mat] + pis.reshape(-1,1).repeat_interleave(nb_num, dim=1)
		mst = kruskal(nb_mat, cost_mat, k=nb_num, hide_bar=True, device=device)
		m1t, degrees = min_1_tree(mst, nb_mat, cost_mat)
		hk_value, tree_cost = tree_weight(m1t, degrees, nb_dist, pis)
		if hk_value > best_hk_value:
			best_hk_value = hk_value
			step_1 = tree_cost / (2 * coords.size(0))
		# print('round: {}, max_degree: {}, tree_weight: {:.2f}, cur hk: {:.2f}, best hk: {:.2f}'.format(
		# 		m, degrees.max().item(), tree_cost, hk_value, best_hk_value))
		scale = ((m - 1) * (2 * M - 5) / (2 * M - 2) - (m - 2) + (m - 1) * (m - 2) / (2 * (M - 1) * (M - 2)))
		step = scale * step_1
		pis = pis + step * (degrees - 2)
		pbar.set_postfix(cur_val=hk_value, best_val=best_hk_value)
		pbar.update(1)
	pbar.close()
	return best_hk_value


def min_1_tree(mst, nb_mat, cost_mat, nb_num):
	node_degrees = stat_degree(mst, nb_mat)
	leaf_indices = torch.argwhere(node_degrees == 1).squeeze()
	costs = cost_mat[leaf_indices].reshape(-1)
	sorted_indices = torch.argsort(costs)
	cur_idx = 0
	while True:
		cur_pos = sorted_indices[cur_idx].item()
		src = cur_pos // nb_num
		dest_idx = cur_pos % nb_num
		dest = nb_mat[src][dest_idx].item()
		src_idx = torch.argwhere(nb_mat[dest] == src).squeeze(0)
		if len(src_idx) == 0:  # src not in nb_mat[dest]
			if not mst[src][dest_idx]:
				mst[src][dest_idx] = True
				node_degrees[src] += 1
				node_degrees[dest] += 1
				break
		else:
			src_idx = src_idx.item()
		if not mst[src][dest_idx] and not mst[dest][src_idx]:
			mst[src][dest_idx] = True
			node_degrees[src] += 1
			node_degrees[dest] += 1
			break
		cur_idx += 1

	assert node_degrees.sum() == 2 * mst.size(0)
	return mst, node_degrees


def stat_degree(mst, nb_mat):  # torch implementation
	# calculate in-degree for each node
	pos = torch.argwhere(mst)
	indices = nb_mat[pos[:, 0], pos[:, 1]]
	idx, count = indices.unique(return_counts=True)
	in_degree = torch.zeros(mst.size(0)).long().to(mst.device)
	in_degree[idx] += count
	# calculate out-degree for each node and add it with in-degree
	out_degree = mst.long().sum(dim=1)
	degree = in_degree + out_degree

	return degree


def tree_weight(m1t, degree, nb_dist, pis):
	pos = torch.argwhere(m1t)
	weights = nb_dist[pos[:, 0], pos[:, 1]].sum()
	value = weights + (pis * (degree - 2)).sum()

	return value.item(), weights.item()


def knn_graph(coords, k=20, candidate_thr=150, device='cuda:0'):
# grid-based neighbor search
	nb_mat = torch.zeros(coords.size(0), k).long().to(device)
	grids = grid_allocation(coords, length=1, grid_size=50)
	grid_num = len(grids)

	for xi in range(grid_num):
		for yi in range(grid_num):
			if len(grids[xi][yi]) == 0:
				continue
			node_indices = torch.tensor(grids[xi][yi])
			step = 0
			while True:
				step += 1
				nb_indices = []
				for cxi in range(max(0, xi - step), min(grid_num, xi + step + 1)):
					for cyi in range(max(0, yi - step), min(grid_num, yi + step + 1)):
						nb_indices += grids[cxi][cyi]
				if len(nb_indices) >= candidate_thr:
					break
			nb_indices = torch.tensor(nb_indices).to(device)
			node_coords = coords[node_indices].unsqueeze(1)
			nb_coords = coords[nb_indices]
			node_nb_dist = ((node_coords - nb_coords) ** 2).sum(dim=-1)
			sorted_indices = torch.argsort(node_nb_dist, dim=-1)[:, 1:1 + k]
			nb_mat[node_indices] = nb_indices[sorted_indices]

	return nb_mat


def grid_allocation(coords, length=1, grid_size=100):
# allocate nodes to grids
	node_num = coords.size(0)
	k = math.ceil((node_num / grid_size) ** 0.5)
	grids = [[[] for _ in range(k)] for _ in range(k)]

	# vertical divide
	x_coords = coords[:, 0]
	vertical_strips = [None for _ in range(k)]
	strip_width = length / k
	node_to_strips = torch.div(x_coords, strip_width, rounding_mode='floor')
	indices = torch.argwhere(node_to_strips == k).squeeze(1)
	if indices.size(0) != 0:
		node_to_strips[indices] -= 1
	for xi in range(k):  # assign node to vertical strips
		node_indices = torch.argwhere(node_to_strips == xi).squeeze(1)
		vertical_strips[xi] = node_indices
	# horizontal divide
	for xi in range(k):
		node_indices = vertical_strips[xi]
		y_coords = coords[node_indices, 1]
		node_to_grids = torch.div(y_coords, strip_width, rounding_mode='floor')
		indices = torch.argwhere(node_to_grids == k).squeeze(1)
		if indices.size(0) != 0:
			node_to_grids[indices] -= 1
		for yi in range(k):  # assign node to vertical grids
			y_indices = torch.argwhere(node_to_grids == yi).squeeze(1)
			grids[xi][yi] += node_indices[y_indices].tolist()

	return grids
