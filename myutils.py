import os
import math
import pickle
import tsplib95
import torch
import numpy as np


# def load_trans_model(model_name, device, rename_key):
# 	from nets.tsp_transformer import load_tt_model
# 	load_model = load_tt_model('pretrained/tsp_transformer/{}.pkl'.format(model_name), device, rename_key)
# 	load_model.to(device)
# 	load_model.eval()
# 	print('load pretrained tsp transformer for {} running on {}'.format(model_name, device))
# 	return load_model


def save_data(data, filepath, overwrite=False):
	if not overwrite:
		assert not os.path.exists(filepath), print('Error: filepath {} already exist!'.format(filepath))
	else:
		if os.path.exists(filepath):
			os.remove(filepath)
	save_file = open(filepath, 'wb')
	pickle.dump(data, save_file)
	save_file.close()


def load_data(filepath):
	assert os.path.exists(filepath), print('Error: filepath {} not exist!'.format(filepath))
	load_file = open(filepath, 'rb')
	load_data = pickle.load(load_file)
	load_file.close()
	return load_data


def load_tour(filename):
	file = open(filename, 'r')
	tour = []
	flag = False
	for line in file.readlines():
		if not flag:
			if line == 'TOUR_SECTION\n':
				flag = True
			continue
		node = int(line)
		if node == -1:
			break
		else:
			tour += [node]
	file.close()
	return torch.tensor(tour).long() - 1


def load_instance_np(filename, scale=True):
	problem = tsplib95.load(filename)
	node_indices = list(problem.get_nodes())
	data = [problem.node_coords[node_idx] for node_idx in node_indices]
	data = np.array(data)
	if scale:
		d_min = data.min(axis=0)
		factor = max(data.max(axis=0) - d_min)
		data = (data - d_min) / factor
	return data


def load_instance(filename, scale=True):
	problem = tsplib95.load(filename)
	node_indices = list(problem.get_nodes())
	data = [problem.node_coords[node_idx] for node_idx in node_indices]
	data = torch.tensor(data)
	if scale:
		d_min = data.min(dim=0)[0]
		factor = max(data.max(dim=0)[0] - d_min)
		data = (data - d_min) / factor
	return data


def scale_coords(data):  # scale coords to unit square
	d_min = data.min(dim=0)[0]
	factor = max(data.max(dim=0)[0] - d_min)
	data = (data - d_min) / factor
	return data


def generate_dataset(tsp_size, samples=100, device='cuda:0', save=False, save_path=None):
	dataset = torch.Tensor([]).to(device)
	for sample_idx in range(samples):
		while True:  # regenerate sample if contain nodes with same coordinate
			tmp_data = torch.FloatTensor(tsp_size, 2).uniform_(0, 1).to(device)
			if torch.unique(tmp_data, dim=0).size(0) == tsp_size:
				data = tmp_data.unsqueeze(0)
				dataset = torch.cat((dataset, data))
				break
	if save:
		save_data(dataset.cpu(), filepath=save_path)
	return dataset


def instance_name(tsp_size, scale='Small'):
	# TSP of scale [1e6, inf]
	if scale == 'large':
		assert tsp_size >= 1e6, print('tsp_size: {} mismatch scale: {}'.format(tsp_size, scale))
		name = 'TSP-{}M'.format(int(tsp_size / 1e6))
	# TSP of scale [5e4,1e6]
	elif scale == 'Medium':
		assert tsp_size >= 5e3 and tsp_size <= 1e6, print('tsp_size: {} mismatch scale: {}'.format(tsp_size, scale))
		name = 'TSP-{}K'.format(int(tsp_size / 1e3))
	# TSP of scale [1,5e4]
	else:
		assert tsp_size <= 5e4, print('tsp_size: {} mismatch scale: {}'.format(tsp_size, scale))
		name = 'TSP-{}'.format(tsp_size)
	return name


def cal_dist(c1, c2):  # calculate distance between nodes with numpy
	return ((c1 - c2) ** 2).sum() ** 0.5


def compute_tour_length(tour, coords, batch=False):
	if batch:
		assert tour.size(1) != 0, print('invalid empty tour!')
		prev_tour = torch.cat((tour[:, -1:], tour[:, :-1]), dim=1)
	else:
		assert tour.size(0) != 0, print('invalid empty tour!')
		prev_tour = torch.cat((tour[-1:], tour[:-1]))
	prev_tour = coords[prev_tour]
	cur_tour = coords[tour]
	if batch:  # return tensor of batch tour length
		return (((cur_tour - prev_tour) ** 2).sum(dim=2) ** 0.5).sum(dim=1)
	else:  # return tour length
		return (((cur_tour - prev_tour) ** 2).sum(dim=1) ** 0.5).sum(dim=0).item()


def compute_tour_length_np(coords, tour):
	assert len(tour) != 0, print('invalid empty tour!')
	cur_tour = coords[tour]
	prev_tour = coords[np.concatenate((tour[-1:], tour[:-1]))]
	return (((cur_tour - prev_tour) ** 2).sum(axis=1) ** 0.5).sum(axis=0)


def find_sub_regions(region, name=False):
	sub_regions = region.sub_regs
	while True:
		region_lists = []
		for cur_reg in sub_regions:
			region_lists += cur_reg.sub_regs if len(cur_reg.sub_regs) != 0 else [cur_reg]
		if len(region_lists) != len(sub_regions):
			sub_regions = region_lists
		else:
			if name:
				for idx, reg in enumerate(region_lists):
					reg.region_index = idx
			return region_lists


def find_neighbor_regions(regions, coords, type='grid', nb_num=8, thr_num=2000, device=None):
	update_region_centroid(regions, coords)
	centroids = torch.tensor([reg.centroid.tolist() for reg in regions]).to(device)

	if type == 'dist':
		if len(regions) < nb_num + 1:
			reg_indices = [reg.region_index for reg in regions]
			for reg_idx, reg in enumerate(regions):
				reg.nb_regs = reg_indices[:reg_idx] + reg_indices[reg_idx + 1:]
		else:
			if len(regions) <= thr_num:
				dists = ((centroids.unsqueeze(1) - centroids) ** 2).sum(dim=2)
				sorted = torch.argsort(dists, dim=1)
				for reg_idx, reg in enumerate(regions):
					reg.nb_regs = sorted[reg_idx][1:1 + nb_num]
					reg.nb_regs = torch.sort(reg.nb_regs)[0].tolist()
			else:  # allocate regions to grid and then calculate dist
				K = math.floor((len(regions) / 200) ** 0.5)
				# minx, miny = coords.min(dim=0)[0].tolist()
				# maxx, maxy = coords.max(dim=0)[0].tolist()
				# assert minx >= 0 and miny>=0 and maxx <= 1 and maxy <= 1
				grids = [[[] for _ in range(K)] for _ in range(K)]
				width = 1 / K
				to_grids = centroids.div(width, rounding_mode='floor').long()
				for xi in range(K):
					indices = torch.argwhere(to_grids[:, 0] == xi).squeeze(1)
					if len(indices) == 0:
						continue
					for yi in range(K):
						sub_indices = torch.argwhere(to_grids[indices, 1] == yi).squeeze(1)
						if len(sub_indices) == 0:
							continue
						grids[xi][yi] += indices[sub_indices].tolist()
				# for idx, (gx, gy) in enumerate(to_grids):
				# 	grids[gx][gy] += [idx]
				for xi in range(K):
					for yi in range(K):
						if len(grids[xi][yi]) == 0:
							continue
						reg_indices = torch.tensor(grids[xi][yi])
						step =2
						# while True:
						nbs = []
						for nxi in range(max(0, xi - step), min(K, xi + step + 1)):
							for nyi in range(max(0, yi - step), min(K, yi + step + 1)):
								nbs += grids[nxi][nyi]
						# 	if len(nbs) >= 2 * len(reg_indices):
						# 		break
						# 	else:
						# 		step += 1
						nb_indices = torch.tensor(nbs)
						reg_centroids = centroids[reg_indices]
						nb_centroids = centroids[nb_indices]
						dists = ((reg_centroids.unsqueeze(1) - nb_centroids) ** 2).sum(dim=2)
						sorted = torch.argsort(dists, dim=1).cpu()
						for idx, reg_idx in enumerate(reg_indices):
							cur_reg = regions[reg_idx]
							cur_reg.nb_regs = nb_indices[sorted[idx][1:1 + nb_num]]
							cur_reg.nb_regs = torch.sort(cur_reg.nb_regs)[0].tolist()
	elif type == 'grid':
		K = int(len(regions) ** 0.5)
		for reg_idx, reg in enumerate(regions):
			assert reg_idx == reg.region_index
			reg_col = reg_idx // K
			reg_row = reg_idx % K
			cur_centroid = centroids[reg_idx]

			nb_indices = []
			step = 2
			for nb_col in range(max(0, reg_col - step), min(K, reg_col + step + 1)):
				for nb_row in range(max(0, reg_row - step), min(K, reg_row + step + 1)):
					nb_idx = nb_col * K + nb_row
					nb_indices += [nb_idx]
			nb_indices = torch.tensor(nb_indices)
			nb_centroids = centroids[nb_indices]
			nb_dists = ((nb_centroids - cur_centroid) ** 2).sum(dim=1)
			reg.nb_regs = nb_indices[torch.argsort(nb_dists)[1:nb_num + 1]].tolist()


def check_nbs_connected(regions):
	reg_num = len(regions)
	visited = np.zeros(reg_num).astype(bool)
	cons = []
	con = set()
	con.add(0)
	while True:
		con_num = len(con)
		for idx in list(con):
			if visited[idx]:
				continue
			visited[idx] = True
			con = con | set(regions[idx].nb_regs)
		if visited.sum() == reg_num:
			cons += [con]
			break
		if len(con) == con_num:
			find = False
			indices = np.where(visited == False)[0]  # traverse un-visited nodes
			for idx in indices:
				if len(con & set(regions[idx].nb_regs)) != 0:
					find = True
					con.add(idx)
					break
			if not find:
				cons += [con]
				con = set()
				con.add(indices[0])
	con_nums = [len(con) for con in cons]
	assert sum(con_nums) == len(regions)
	if len(cons) == 1:
		return True
	else:
		assert len(cons) >= 2
		return False


def remove_tiny_regions(regions, coords, min_thr):
	removed = []
	valid_indices = []
	valid_centroids = []
	for reg_idx, reg in enumerate(regions):
		assert reg_idx == reg.region_index
		if reg.node_num < min_thr:
			removed += [reg_idx]
		else:
			valid_indices += [reg_idx]
			valid_centroids += [reg.centroid.tolist()]
	if len(removed) == 0:
		return regions, False
	valid_centroids = torch.tensor(valid_centroids).to(coords.device)

	for reg_idx in removed:  # traverse tiny regions and allocate its nodes to its neighbor regions
		reg = regions[reg_idx]
		nbs = reg.nb_regs
		for idx in removed:
			if idx in nbs:
				nbs.remove(idx)  # remove tiny neighbor regions from current region
		if len(nbs) != 0:
			nb_centroids = torch.tensor([regions[nb_idx].centroid.tolist() for nb_idx in nbs]).to(coords.device)
			dists = ((coords[reg.node_index].unsqueeze(1) - nb_centroids) ** 2).sum(dim=2)
			allocate = dists.argmin(dim=1).tolist()
			for idx, to_idx in enumerate(allocate):
				node = reg.node_index[idx].unsqueeze(0)
				nb_reg = regions[nbs[to_idx]]
				nb_reg.node_index = torch.cat((nb_reg.node_index, node))
				nb_reg.node_num += 1
		else:
			dists = ((coords[reg.node_index].unsqueeze(1) - valid_centroids) ** 2).sum(dim=2)
			allocate = dists.argmin(dim=1).tolist()
			for idx, to_idx in enumerate(allocate):
				node = reg.node_index[idx].unsqueeze(0)
				to_reg = regions[valid_indices[to_idx]]
				to_reg.node_index = torch.cat((to_reg.node_index, node))
				to_reg.node_num += 1
	new_regions = []
	for reg_idx, reg in enumerate(regions):  # re-index new regions
		if reg_idx not in removed:
			reg.region_index = len(new_regions)
			new_regions += [reg]
	update_region_centroid(new_regions, coords)
	return new_regions, True


def update_region_centroid(regions, coords):
	for reg in regions:
		if reg.node_num != 0:
			reg.centroid = coords[reg.node_index].mean(dim=0)
		else:
			reg.centroid = torch.tensor([-float('inf'), -float('inf')]).to(coords.device)


def stat_sub_scale(scales, target=50, tol=0.5):
	min_scale = target * (1 - tol)
	max_scale = target * (1 + tol)
	indices = np.where(scales <= max_scale)[0]
	selected = scales[indices]
	indices = np.where(selected >= min_scale)[0]
	selected = scales[indices]
	return 100 * len(selected) / len(scales), 100 * sum(selected) / sum(scales)


def stat_sub_gap(sub_gaps, scales, val_1=3, val_2=10):
	node_num = sum(scales)
	indices = np.where(sub_gaps <= val_1)[0]
	per_1 = 100 * len(indices) / len(sub_gaps)
	num_1 = 100 * sum(scales[indices]) / node_num
	indices = np.where(sub_gaps >= val_2)[0]
	per_2 = 100 * len(indices) / len(sub_gaps)
	num_2 = 100 * sum(scales[indices]) / node_num
	return per_1, num_1, per_2, num_2


def stat_region_scale(regions):
	reg_scales = [reg.node_num for reg in regions]
	scales, counts = torch.tensor(reg_scales).unique(sorted=True, return_counts=True)
	out = ['{}/{}'.format(scale, counts[idx]) for idx, scale in enumerate(scales)]
	# for idx, scale in enumerate(scales):
	# 	print('scale: {}, count: {}'.format(scale, counts[idx]))
	print('stat: {}'.format(out))
	print('total {} nodes, {} sub_regions'.format(sum(reg_scales), len(regions)))


def slice_list(input, srt, end, forward=True):
	length = len(input)
	if srt == end:
		return [input[srt % length]]
	if forward:
		assert -1 * length < srt and srt < end and end < 2 * length
		assert end - srt < length
		if srt >= length and end >= length:
			srt -= length
			end -= length
		elif srt < 0 and end < 0:
			srt += length
			end += length
		if srt < 0:
			return input[srt:] + input[:end + 1]
		if end >= length:
			return input[srt:] + input[:end - length + 1]
		return input[srt:end + 1]
	else:
		return None  # TODO: to be done


def slice_array(input, srt, end, forward=True):
	length = len(input)
	if srt == end:
		return input[None, srt % length]
	if forward:
		assert -1 * length < srt and srt < end and end < 2 * length
		assert end - srt < length
		if srt >= length and end >= length:
			srt -= length
			end -= length
		elif srt < 0 and end < 0:
			srt += length
			end += length
		if srt < 0:
			return np.concatenate((input[srt:], input[:end + 1]))
		if end >= length:
			return np.concatenate((input[srt:], input[:end - length + 1]))
		return input[srt:end + 1]
	else:
		return None  # TODO: to be done


def locate_node(nodes):
	node_pos = np.zeros(len(nodes)).astype(int)
	for idx, node in enumerate(nodes):
		node_pos[node] = idx
	return node_pos
