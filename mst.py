import torch
import numpy as np
from tqdm import tqdm


def kruskal(nb_mat, cost_mat, rand=False, use_np=True, hide_bar=False):  # build MST using Kruskal's Algorithm
	node_num = nb_mat.shape[0]
	nb_num = nb_mat.shape[1]
	assert cost_mat.shape[0] == node_num
	assert cost_mat.shape[1] == nb_num
	mst = np.zeros([node_num, nb_num], dtype=bool)
	trees = np.arange(node_num).reshape(node_num, 1).tolist()  # node indices of connected component
	tree_ref = np.arange(node_num)  # return idx of the connected component (tree)
	visited = np.zeros(node_num, dtype=bool)
	cost_val = cost_mat.flatten()
	if rand == False:
		if use_np:
			cost_pos = np.argsort(cost_val)
		else:
			cost_pos = torch.argsort(cost_val).tolist()
			nb_mat = nb_mat.numpy()
	else:
		# cost_pos = torch.argsort(cost_val).tolist()
		cost_pos = torch.randperm(len(cost_val)).tolist()
		nb_mat = nb_mat.numpy()

	cur_idx = 0
	edge_num = 0
	pbar = tqdm(total=node_num - 1, desc='[building mst]', unit='edge', disable=hide_bar)
	while edge_num < node_num - 1:
		min_idx = cost_pos[cur_idx]
		src = min_idx // nb_num
		dest_idx = min_idx % nb_num
		dest = nb_mat[src][dest_idx]
		src_visited = visited[src]
		dest_visited = visited[dest]
		src_tree_idx = tree_ref[src]
		dest_tree_idx = tree_ref[dest]

		if src_visited and dest_visited:
			if src_tree_idx != dest_tree_idx:
				if len(trees[dest_tree_idx]) > len(trees[src_tree_idx]):
					tmp = src_tree_idx
					src_tree_idx = dest_tree_idx
					dest_tree_idx = tmp
				trees[src_tree_idx] += trees[dest_tree_idx]
				node_indices = np.array(trees[dest_tree_idx])
				tree_ref[node_indices] = src_tree_idx
				trees[dest_tree_idx] = []
				mst[src][dest_idx] = True
				edge_num += 1
				pbar.update(1)
		else:
			if dest_visited:
				trees[dest_tree_idx] += trees[src_tree_idx]
				trees[src_tree_idx] = []
				tree_ref[src] = dest_tree_idx
			else:
				trees[src_tree_idx] += trees[dest_tree_idx]
				trees[dest_tree_idx] = []
				tree_ref[dest] = src_tree_idx
			visited[src] = True
			visited[dest] = True
			mst[src][dest_idx] = True
			edge_num += 1
			pbar.update(1)
		cur_idx += 1
	pbar.close()
	assert mst.sum() == node_num - 1

	return mst
