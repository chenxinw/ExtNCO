import copy
import torch
import numpy as np
from scipy.spatial import Delaunay


def delaunay_graph(coords, sorted=False):
	node_num = len(coords)
	graph = [set() for _ in range(node_num)]
	tris = Delaunay(coords).simplices.tolist()
	for tri in tris:  # tri: [n1, n2, n3]
		for idx, node in enumerate(tri):
			if idx != 2:
				graph[node].add(tri[idx + 1])
			else:
				graph[node].add(tri[0])
			graph[node].add(tri[idx - 1])
	for idx in range(node_num):
		graph[idx] = np.sort(list(graph[idx])).tolist() if sorted else list(graph[idx])
	return graph


def delaunay_neighbor(coords, nb_num, expand=True):
	node_num = len(coords)
	graph = delaunay_graph(coords)
	nb_nums = [len(nbs) for nbs in graph]
	min_nb_num = min(nb_nums)
	if min_nb_num < nb_num:
		if not expand:
			nb_num = min_nb_num
		else:
			new_graph = copy.deepcopy(graph)
			for idx in range(node_num):
				while len(new_graph[idx]) < nb_num:
					nbs = list(new_graph[idx])
					new_graph[idx] = set(new_graph[idx])
					for nb_idx in nbs:
						new_graph[idx] = new_graph[idx] | set(graph[nb_idx])
					new_graph[idx] = list(new_graph[idx])
					new_graph[idx].remove(idx)
				assert len(new_graph[idx]) >= nb_num
			graph = new_graph
	for idx in range(node_num):
		nb_indices = np.array(graph[idx])
		nb_dists = ((coords[nb_indices] - coords[idx]) ** 2).sum(axis=1)
		sorted = np.argsort(nb_dists)[:nb_num]
		graph[idx] = nb_indices[sorted].tolist()
	return graph
