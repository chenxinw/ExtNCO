import time
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpire import WorkerPool
from mst import kruskal
from hk_bound import stat_degree
from graph import *
from myutils import *


def merge(region, coords, params, device='cuda:0', hide_bar=False):
	# if params['type'] == 'mst' or params['type'] == 'random' or params['type'] == 'tsp':
	# 	update_region_centroid(region.sub_regs, coords)  # update sub_region's centroid
	# 	tour, time_dict = mst_merge(params['type'], region, coords, params['opt'], device, hide_bar=hide_bar)
	# 	region.traverse_time = time_dict
	# 	region.tour = torch.tensor(tour)
	# elif params['type'] == 'rec':  # used by GalaxyTSP
	# 	recursive_merge(region, coords)
	# else:
	# 	print('unknown merge type!')
	update_region_centroid(region.sub_regs, coords)  # update sub_region's centroid
	tour, time_dict = mst_merge(region, coords, params['comb'], params['var'], params['ord'], device, hide_bar)
	region.time_dict = time_dict
	region.tour = torch.tensor(tour)


def mst_merge(region, coords, comb, var, orders, device='cuda:0', hide_bar=True):
	regions = region.sub_regs
	regions_num = len(regions)
	nb_num = len(regions[0].nb_regs)

	# step 0: initialization
	init_time = time.time()
	reg_infos = [{'tour'       : reg.tour.tolist(),
				  'centroid'   : reg.centroid.cpu().numpy(),
				  'node_coords': coords[reg.tour].cpu().numpy(), } for reg in regions]
	nb_pairs = []  # neighbor relationship

	if comb == 'mst':
		for reg_idx, reg in enumerate(regions):
			assert reg.region_index == reg_idx
			for idx, nb_idx in enumerate(reg.nb_regs):
				if nb_idx > reg_idx:
					nb_pairs += [(reg_idx, nb_idx, idx)]
				elif reg_idx not in regions[nb_idx].nb_regs:
					nb_pairs += [(reg_idx, nb_idx, idx)]
	elif comb == 'random':
		for reg_idx, reg in enumerate(regions):
			for idx, nb_idx in enumerate(reg.nb_regs):
				nb_pairs += [(reg_idx, nb_idx, idx)]
	# print('init time: {:.2f}'.format(time.time() - init_time))
	region.misc_time += time.time() - init_time

	if comb == 'tsp':  # skip step 1 when comb == 'tsp'
		mst = np.zeros([regions_num, nb_num], dtype=bool)
		new_coords = np.array([reg.centroid.tolist() for reg in regions])
		from solve import lkh_solve
		res = lkh_solve(new_coords)
		tour = res['tour']
		for idx, reg_idx in enumerate(tour[:-1]):
			cur_reg = regions[reg_idx]
			next_idx = tour[idx + 1]
			if next_idx in cur_reg.nb_regs:
				idx_1 = cur_reg.nb_regs.index(next_idx)
			else:
				cur_reg.nb_regs[0] = next_idx
				idx_1 = 0
			mst[reg_idx][idx_1] = True
		nb_mat = torch.tensor([reg.nb_regs for reg in regions])
		for reg_idx, reg in enumerate(regions):
			for idx, nb_idx in enumerate(reg.nb_regs):
				nb_pairs += [(reg_idx, nb_idx, idx)]
		params = [{'cur_reg': reg_infos[reg_idx], 'nb_reg': reg_infos[nb_idx], } for reg_idx, nb_idx, _ in nb_pairs]
		with WorkerPool(n_jobs=10) as pool:
			parl_res = pool.map(nn_pairs, params, progress_bar=not hide_bar,
								progress_bar_options={'desc': 'nn-pairs evaluating'})
			merge_res = [{'cost': res['cost'], 'pairs': res['pairs'], } for res in parl_res]
			pool.terminate()
	else:  # step 1: parallel evaluation of merging cost
		eval_time = time.time()
		params = [{'cur_reg': reg_infos[cur_idx], 'nb_reg': reg_infos[nb_idx], } for cur_idx, nb_idx, _ in nb_pairs]
		with WorkerPool(n_jobs=10) as pool:
			parl_res = pool.map(nn_pairs, params, progress_bar=not hide_bar,
								progress_bar_options={'desc': 'nnp eval cost', 'unit': 'pairs', })
			merge_res = [{'pairs': res['pairs'], 'cost': res['cost']} for res in parl_res]
			pool.terminate()
		# print('nn-pairs eval time: {:.2f}'.format(time.time() - eval_time))

		if var == 'qlt':
			kopt_time = time.time()
			# for idx, param in tqdm(enumerate(params), desc='kopt evaluating'):
			# 	ni, nj = merge_res[idx]['pairs']
			# 	res = k_opt_merge(param['cur_reg'], param['nb_reg'], ni, nj, None)
			# 	merge_res[idx]['cost'] = res['cost']

			for idx, param in enumerate(params):
				param['ni'], param['nj'] = merge_res[idx]['pairs']
				# param['idx'] = idx
			with WorkerPool(n_jobs=10) as pool:
				parl_res = pool.map(k_opt_merge, params, progress_bar=not hide_bar,
									progress_bar_options={'desc': 'kopt eval cost', 'unit': 'pairs', })
				# insights = pool.get_insights()
				pool.terminate()
			for idx, res in enumerate(parl_res):
				reg_idx, nb_idx, _ = nb_pairs[idx]
				tour = res['tour']  # list
				merge_res[idx]['tour'] = tour
				merge_res[idx]['cost'] = res['cost']
					# merge_res[idx]['cost'] = compute_tour_length(torch.tensor(tour).to(device), coords) \
					# 						 - regions[reg_idx].length - regions[nb_idx].length
			# print('k-opt eval time: {:.2f}'.format(time.time() - kopt_time))
		region.eval_time = time.time() - eval_time

		# step 2: build Minimum Spanning Tree using Kruskal Algorithm
		mst_time = time.time()
		nb_mat = torch.tensor([reg.nb_regs for reg in regions])
		cost_mat = torch.full([regions_num, nb_num], np.inf).to(device)
		for ri, (reg_idx, _, idx) in enumerate(nb_pairs):
			cost_mat[reg_idx][idx] = merge_res[ri]['cost']
		mst = kruskal(nb_mat, cost_mat, rand=(comb == 'random'), use_np=False, hide_bar=hide_bar)
		# print('build mst time: {:.2f}'.format(time.time() - mst_time))
		region.misc_time += time.time() - mst_time

	# step 3: convert mst to binary tree
	bt_time = time.time()
	degrees = stat_degree(torch.tensor(mst), nb_mat)
	linked_nodes = [[] for _ in range(regions_num)]
	edge_ref = [set() for _ in range(regions_num)]
	edge_infos = []
	for res_idx, (reg_idx, nb_idx, idx) in enumerate(nb_pairs):
		if mst[reg_idx][idx]:
			linked_nodes[reg_idx] += [nb_idx]
			linked_nodes[nb_idx] += [reg_idx]
			edge_idx = len(edge_infos)
			edge_ref[reg_idx].add(edge_idx)
			edge_ref[nb_idx].add(edge_idx)
			edge_infos += [(reg_idx, nb_idx, idx, res_idx)]
	node_pairs = traverse_mst(copy.deepcopy(linked_nodes), copy.deepcopy(degrees.numpy()), regions_num)
	# print('convert mst time: {:.2f}'.format(time.time() - bt_time))

	time_dict = dict()
	tours = []
	coords = coords.cpu().numpy()
	op = 'nn-pairs' if var == 'eff' else 'k-opt'
	region.misc_time += time.time() - bt_time
	# step 4: traverse mst to merge
	for ord_idx, ord_key in enumerate(orders):
		cur_edge_ref = edge_ref if ord_idx == len(orders) - 1 else copy.deepcopy(edge_ref)
		time_dict[ord_key] = time.time()
		if ord_key == 'btp':
			tours += [merge_mst(regions, coords, node_pairs, op, merge_res, True, cur_edge_ref, edge_infos, hide_bar)]
		elif ord_key == 'bts':
			tours += [merge_mst(regions, coords, node_pairs, op, merge_res, False, cur_edge_ref, edge_infos, hide_bar)]
		else:
			assert ord_key == 'bfs'
			tours += [merge_mst_BFS(regions, coords, op, merge_res, cur_edge_ref, edge_infos)]
		# length = compute_tour_length_np(coords, tours[-1])
		time_dict[ord_key] = time.time() - time_dict[ord_key]
		# print('{}: length: {:.2f}, time: {:.2f}'.format(ord_key, length, time_dict[ord_key] ))
	region.trav_time = time_dict[orders[0]]

	return tours[0], time_dict


class Vertex(object):  # vertex in cost graph, used for parallel merging
	def __init__(self, sub_regs, tour, vertex_index):
		self.sub_regs = sub_regs    # list
		self.tour = tour            # ndArray -> list!
		self.index = vertex_index   # int


def merge_mst(regions, coords, node_pairs, operator, merge_res, parallel, edge_ref, edge_infos, hide_bar=True):
	vertices = [Vertex([reg_idx], reg.tour.tolist(), reg_idx) for reg_idx, reg in enumerate(regions)]
	pbar = tqdm(desc='[merge sub-tours]', total=len(regions) - 1, unit='pairs', disable=hide_bar)
	for round, pairs in enumerate(node_pairs):
		# rt = time.time()
		if parallel:  # parallel merging
			# step 1: collect parameters for parallel process
			parl_params = []
			edge_indices = []
			for idx_1, idx_2 in pairs:
				edge_idx = (edge_ref[idx_1] & edge_ref[idx_2]).pop()
				edge_indices += [edge_idx]
				reg_idx, nb_idx, _, res_idx = edge_infos[edge_idx]
				if reg_idx in vertices[idx_1].sub_regs:
					assert nb_idx in vertices[idx_2].sub_regs
					cur_tour = vertices[idx_1].tour
					nb_tour = vertices[idx_2].tour
				else:
					assert nb_idx in vertices[idx_1].sub_regs
					assert reg_idx in vertices[idx_2].sub_regs
					cur_tour = vertices[idx_2].tour
					nb_tour = vertices[idx_1].tour
				cur_coords = coords[cur_tour]
				nb_coords = coords[nb_tour]
				ni, nj = merge_res[res_idx]['pairs']
				parl_params += [{'cur_reg': {'tour': cur_tour, 'node_coords': cur_coords, }, 'ni': ni,
								 'nb_reg' : {'tour': nb_tour,  'node_coords': nb_coords,  }, 'nj': nj, }]
			# step 2: parallel merging execution and update information
			if operator == 'k-opt':
				with WorkerPool(n_jobs=min(20, len(parl_params))) as pool:
					parl_res = pool.map(k_opt_merge, parl_params)
					pool.terminate()
			else:
				assert operator == 'nn-pairs'
				with WorkerPool(n_jobs=min(5, len(parl_params))) as pool:
					parl_res = pool.map(merge_vertex_nnpairs, parl_params)
					pool.terminate()
			for idx, res in enumerate(parl_res):
				idx_1, idx_2 = pairs[idx]
				vertices[idx_2].sub_regs += vertices[idx_1].sub_regs
				vertices[idx_2].tour = res['tour']
				vertices[idx_1] = None
				edge_idx = edge_indices[idx]
				edge_ref[idx_1].remove(edge_idx)
				edge_ref[idx_2].remove(edge_idx)
				edge_ref[idx_2] |= edge_ref[idx_1]
				edge_ref[idx_1] = None
		else:  # serial merging
			for idx_1, idx_2 in pairs:
				edge_idx = (edge_ref[idx_1] & edge_ref[idx_2]).pop()
				reg_idx, nb_idx, _, res_idx = edge_infos[edge_idx]
				if reg_idx in vertices[idx_1].sub_regs:
					assert nb_idx in vertices[idx_2].sub_regs
					cur_tour = vertices[idx_1].tour
					nb_tour = vertices[idx_2].tour
				else:
					assert nb_idx in vertices[idx_1].sub_regs
					assert reg_idx in vertices[idx_2].sub_regs
					cur_tour = vertices[idx_2].tour
					nb_tour = vertices[idx_1].tour
				cur_coords = coords[cur_tour]
				nb_coords = coords[nb_tour]
				cur_reg = {'tour': cur_tour, 'node_coords': cur_coords, }
				nb_reg = {'tour': nb_tour,  'node_coords': nb_coords,  }
				ni, nj = merge_res[res_idx]['pairs']
				if operator == 'k-opt':
					tour = k_opt_merge(cur_reg, nb_reg, ni, nj)['tour']
				else:
					assert operator == 'nn-pairs'
					tour = merge_vertex_nnpairs(cur_reg, nb_reg, ni, nj)['tour']
				vertices[idx_2].sub_regs += vertices[idx_1].sub_regs
				vertices[idx_2].tour = tour
				vertices[idx_1] = None
				edge_ref[idx_1].remove(edge_idx)
				edge_ref[idx_2].remove(edge_idx)
				edge_ref[idx_2] |= edge_ref[idx_1]
				edge_ref[idx_1] = None
		pbar.update(len(pairs))
	pbar.close()
	vtx_idx = node_pairs[-1][0][1]
	return vertices[vtx_idx].tour


def merge_mst_BFS(regions, coords, operator, merge_res, edge_ref, edge_infos):
	node_num = len(regions)
	node_visited = np.zeros(node_num).astype(bool)
	edge_visited = np.zeros(node_num - 1).astype(bool)

	# start from edge_0
	reg_idx, nb_idx, _, res_idx = edge_infos[0]
	edge_visited[0] = True
	node_visited[reg_idx] = True
	node_visited[nb_idx] = True
	cur_tour = regions[reg_idx].tour.tolist()
	nb_tour = regions[nb_idx].tour.tolist()
	cur_reg = {'tour': cur_tour, 'node_coords': coords[cur_tour], }
	nb_reg = {'tour': nb_tour, 'node_coords': coords[nb_tour], }
	ni, nj = merge_res[res_idx]['pairs']
	if operator == 'k-opt':
		merged_tour = k_opt_merge(cur_reg, nb_reg, ni, nj)['tour']
	else:
		assert operator == 'nn-pairs'
		merged_tour = merge_vertex_nnpairs(cur_reg, nb_reg, ni, nj)['tour']
	edges2go = []
	for edge_idx in edge_ref[reg_idx]:
		if not edge_visited[edge_idx]:
			edges2go += [edge_idx]
	for edge_idx in edge_ref[nb_idx]:
		if not edge_visited[edge_idx]:
			edges2go += [edge_idx]

	# traverse the remaining edges
	while sum(node_visited) < node_num:
		idx = np.random.randint(len(edges2go))  # random select an edge to merge
		edge_idx = edges2go[idx]
		del edges2go[idx]
		if edge_visited[edge_idx]:
			continue
		reg_idx, nb_idx, _, res_idx = edge_infos[edge_idx]
		edge_visited[edge_idx] = True
		if node_visited[reg_idx]:
			assert not node_visited[nb_idx]
			node_visited[nb_idx] = True
			cur_tour = merged_tour
			nb_tour = regions[nb_idx].tour.tolist()
		else:
			assert node_visited[nb_idx]
			node_visited[reg_idx] = True
			cur_tour = regions[reg_idx].tour.tolist()
			nb_tour = merged_tour
		cur_reg = {'tour': cur_tour, 'node_coords': coords[cur_tour], }
		nb_reg = {'tour': nb_tour, 'node_coords': coords[nb_tour], }
		ni, nj = merge_res[res_idx]['pairs']
		if operator == 'k-opt':
			merged_tour = k_opt_merge(cur_reg, nb_reg, ni, nj)['tour']
		else:
			assert operator == 'nn-pairs'
			merged_tour = merge_vertex_nnpairs(cur_reg, nb_reg, ni, nj)['tour']
		for edge_idx in edge_ref[reg_idx]:
			if not edge_visited[edge_idx]:
				edges2go += [edge_idx]
		for edge_idx in edge_ref[nb_idx]:
			if not edge_visited[edge_idx]:
				edges2go += [edge_idx]

	return merged_tour


def nn_pairs(cur_reg, nb_reg, k=5):  # merge tours with near nodes pair, ni -> ni_pred -> nj -> nj_pred
	cur_tour = cur_reg['tour']
	cur_coords = cur_reg['node_coords']
	nb_tour = nb_reg['tour']
	nb_cent = nb_reg['centroid']
	nb_coords = nb_reg['node_coords']

	costs = []
	infos = []
	sorted_i = np.argsort(((cur_coords - nb_cent) ** 2).sum(axis=1))
	for i in sorted_i[:k]:
		ci = cur_coords[i]
		ca = cur_coords[i - 1]  # ni_pred
		sorted_j = np.argsort(((nb_coords - ci) ** 2).sum(axis=1))
		dist_ia = cal_dist(ci, ca)
		for j in sorted_j[:k]:
			cj = nb_coords[j]
			cb = nb_coords[j - 1]  # nj_pred
			# costs += [cal_dist(ca, cj) + cal_dist(cb, ci) - cal_dist(ci, ca) - cal_dist(cj, cb)]
			costs += [cal_dist(ca, cj) + cal_dist(cb, ci) - dist_ia - cal_dist(cj, cb)]
			infos += [(cur_tour[i], nb_tour[j])]  # [(cur_tour[i], cur_tour[i - 1], nb_tour[j], nb_tour[j - 1])]
	min_idx = np.argmin(costs)
	return {'pairs': infos[min_idx], 'cost': costs[min_idx], }


def best_nn_pairs(cur_tour, cur_coords, nb_tour, nb_coords, ni, nj, step=2):
	c_num = len(cur_tour)
	n_num = len(nb_tour)

	costs = []
	pairs = []
	ni_idx = cur_tour.index(ni)
	nj_idx = nb_tour.index(nj)
	for i_pos in range(ni_idx - step, ni_idx + step + 1):
		pi = i_pos % c_num
		ci = cur_coords[pi]
		ca = cur_coords[pi - 1]
		for j_pos in range(nj_idx - step, nj_idx + step + 1):
			pj = j_pos % n_num
			cj = nb_coords[pj]
			cb = nb_coords[pj - 1]
			costs += [cal_dist(ca, cj) + cal_dist(cb, ci) - cal_dist(ci, ca) - cal_dist(cj, cb)]
			pairs += [(pi, pj)]
	min_idx = np.argmin(costs)
	min_cost = np.min(costs)
	return pairs[min_idx], min_cost


def k_opt_merge(cur_reg, nb_reg, ni, nj, step=10):
	cur_tour = cur_reg['tour']
	cur_coords = cur_reg['node_coords']
	nb_tour = nb_reg['tour']
	nb_coords = nb_reg['node_coords']
	c_num = len(cur_tour)
	n_num = len(nb_tour)
	node_num = c_num + n_num

	(pi, pj), cost = best_nn_pairs(cur_tour, cur_coords, nb_tour, nb_coords, ni, nj)
	# assert 2 * step + 3 <= c_num and 2 * step + 3 <= n_num

	# concatenate candidate nodes and special nodes into a tour (list)
	# sp_nodes = [((pi + step + 1) % c_num, (pi - step - 1) % c_num),
	# 			((pj + step + 1) % n_num, (pj - step - 1) % n_num)]
	sp_tour = slice_list(nb_tour, pj - step, pj - 1) + slice_list(cur_tour, pi, pi + step) + [(0, True)] + \
			  slice_list(cur_tour, pi - step, pi - 1) + slice_list(nb_tour, pj, pj + step) + [(1, False)]
	sp_coords = np.concatenate((
			slice_array(nb_coords, pj - step, pj - 1), slice_array(cur_coords, pi, pi + step), [[-np.inf, -np.inf]],
			slice_array(cur_coords, pi - step, pi - 1), slice_array(nb_coords, pj, pj + step), [[-np.inf, -np.inf]]))

	# run k-opt
	new_tour, gain = simp_k_opt(sp_tour, sp_coords, step)

	# recover tour from simp_tour info
	tour = []
	idx_1 = 2 * step + 1
	idx_2 = len(new_tour) - 1
	for idx in new_tour:
		if abs(idx) == idx_1:
			srt = (pi + step + 1) % c_num
			end = (pi - step - 1) % c_num
			ref = cur_tour
			# srt, end, flag = simp_tour[abs(idx)]
			# ref = cur_tour if flag else nb_tour
			if srt < end:
				tour += ref[srt:end + 1] if idx > 0 else ref[end:srt:-1] + [ref[srt]]
			else:
				tour += ref[srt:] + ref[:end + 1] if idx > 0 else ref[end::-1] + ref[:srt:-1] + [ref[srt]]
		elif abs(idx) == idx_2:
			srt = (pj + step + 1) % n_num
			end = (pj - step - 1) % n_num
			ref = nb_tour
			if srt < end:
				tour += ref[srt:end + 1] if idx > 0 else ref[end:srt:-1] + [ref[srt]]
			else:
				tour += ref[srt:] + ref[:end + 1] if idx > 0 else ref[end::-1] + ref[:srt:-1] + [ref[srt]]
		else:
			tour += [sp_tour[idx]]
	assert len(tour) == node_num, print('len_tour: {}, node_num: {}'.format(len(tour), node_num))

	return {'tour' : tour, 'cost': cost - gain, }


def simp_k_opt(tour, coords, step, nb_num=5, max_k=5):
	node_num = len(tour)
	new_tour = np.arange(node_num).tolist()  # represent nodes in tour by [0,N-1] sequence
	sp_indices = {2 * step + 1, node_num - 1}
	cd_indices = np.array(new_tour[:2 * step + 1] + new_tour[2 * step + 2:-1]).astype(int)
	node_nbs = delaunay_neighbor(coords[cd_indices], nb_num, expand=True)  # correspond to cand_nodes

	total_gain = 0
	while True:
		update_tour = None
		gain = None
		for n1 in cd_indices:
			t1 = new_tour.index(n1)
			for t2 in [(t1 + 1) % node_num, (t1 - 1) % node_num]:
				n2 = new_tour[t2]
				if abs(n2) in sp_indices:
					continue
				g1 = cal_dist(coords[n1], coords[n2])
				update_tour, gain = simp_k_opt_rec(coords, step, cd_indices, sp_indices, node_nbs,
				                                   new_tour, [t1, t2], [g1], 2, max_k)
				if gain is None:  # search for n2
					continue
				else:
					break
			if gain is None:  # search for n1
				continue
			else:
				break
		if gain is None:  # no improvement can be found
			return new_tour, total_gain
		else:  # update new_tour
			assert gain > 0
			total_gain += gain
			new_tour = update_tour


def simp_k_opt_rec(coords, step, cands, simps, nbs, tour, t_seq, gains, cur_k, max_k):
	t1 = t_seq[0]
	t2 = t_seq[-1]
	n1 = tour[t1]
	n2 = tour[t2]
	# p2 = np.argwhere(cands == n2).item()
	p2 = n2 if n2 <= 2 * step else n2 - 1
	for n3_idx in nbs[p2]:
		n3 = cands[n3_idx]
		if abs(n3) in simps:
			continue
		t3 = tour.index(n3)
		if t3 == (t2 + 1) % len(tour) or t3 == (t2 - 1) % len(tour):
			continue  # t3 can't be t2_suc or t2_pre
		g1 = -cal_dist(coords[n2], coords[n3])
		G1 = round(sum(gains) + g1, 8)
		if G1 <= 0:
			continue
		for t4 in [(t3 + 1) % len(tour), (t3 - 1) % len(tour)]:
			n4 = tour[t4]
			if abs(n4) in simps:
				continue
			g2 = cal_dist(coords[n3], coords[n4])
			g3 = -cal_dist(coords[n4], coords[n1])
			G2 = round(G1 + g2 + g3, 8)
			if G2 > 0 and check_kopt_valid(t_seq + [t3, t4]):
				new_tour = simp_kopt_move(simps, tour, t_seq + [t3, t4])
				return new_tour, G2
			if cur_k < max_k:
				new_tour, gain = simp_k_opt_rec(coords, step, cands, simps, nbs, tour, t_seq + [t3, t4],
				                                gains + [g1, g2], cur_k + 1, max_k)
				if gain == None:
					continue
				return new_tour, gain
	return tour, None


def simp_kopt_move(simps, tour, t_seq):
	len_t = len(t_seq)
	incl = [len_t - 1]
	for i in range(2, len_t, 2):
		incl += [i, i - 1]
	incl += [0]
	p = np.argsort(t_seq)
	q = np.argsort(p)

	new_tour = []
	cur_t = p[0]
	while True:
		incl_t = incl[cur_t]
		incl_p = q[incl_t]
		suc = (incl_p + 1) % len_t
		pre = (incl_p - 1) % len_t
		if incl_t % 2 == 0:  # skip out-edge and choose another side
			clockwise = False if p[suc] == incl_t + 1 else True
		else:  # next = pre if p[suc] == incl_t - 1 else suc
			clockwise = False if p[suc] == incl_t - 1 else True
		next = suc if clockwise else pre
		cur_t = p[next]
		srt = t_seq[incl_t]
		end = t_seq[cur_t]
		if clockwise:
			if srt < end:
				new_tour += tour[srt:end] + [tour[end]]
			else:
				new_tour += tour[srt:] + tour[:end] + [tour[end]]
		else:
			if srt > end:
				tmp_slice = tour[srt:end:-1] + [tour[end]]
			else:
				tmp_slice = tour[srt::-1] + tour[:end - len_t:-1] + [tour[end]]
			for node in list(simps):
				if node in tmp_slice:
					idx = tmp_slice.index(node)
					tmp_slice[idx] = -node
				elif -node in tmp_slice:
					idx = tmp_slice.index(-node)
					tmp_slice[idx] = node
			new_tour += tmp_slice
		if cur_t == p[0]:
			break
	assert len(new_tour) == len(tour)
	return new_tour


def check_kopt_valid(t_seq):
	if len(np.unique(t_seq)) != len(t_seq):
		return False

	# slightly different python implementation of FeasibleKOptMove algorithm described in paper below
	# An Effective Implementation of K-opt Moves for the Lin-Kernighan TSP Heuristic
	len_t = len(t_seq)
	k = int(len_t / 2)
	incl = [len_t - 1]
	for i in range(2, len_t, 2):
		incl += [i, i - 1]
	incl += [0]
	p = np.argsort(t_seq)
	q = np.argsort(p)
	visit = [False for _ in range(len_t)]

	count = 0
	cur_t = p[0]
	while True:
		incl_t = incl[cur_t]
		visit[cur_t] = True
		visit[incl_t] = True
		incl_t_pos = q[incl_t]
		suc = (incl_t_pos + 1) % len_t
		pre = (incl_t_pos - 1) % len_t
		if incl_t % 2 == 0:  # skip out-edge and choose another side
			next = pre if p[suc] == incl_t + 1 else suc
		else:
			next = pre if p[suc] == incl_t - 1 else suc
		cur_t = p[next]
		count += 1
		if visit[cur_t]:
			break
	return True if count == k else False


def execute_kopt_move(tour, t_seq):
	len_t = len(t_seq)
	incl = [len_t - 1]
	for i in range(2, len_t, 2):
		incl += [i, i - 1]
	incl += [0]
	p = np.argsort(t_seq)
	q = np.argsort(p)

	new_tour = []
	cur_t = p[0]
	while True:
		incl_t = incl[cur_t]
		incl_p = q[incl_t]
		suc = (incl_p + 1) % len_t
		pre = (incl_p - 1) % len_t
		if incl_t % 2 == 0:  # skip out-edge and choose another side
			clockwise = False if p[suc] == incl_t + 1 else True
		else:  # next = pre if p[suc] == incl_t - 1 else suc
			clockwise = False if p[suc] == incl_t - 1 else True
		next = suc if clockwise else pre
		cur_t = p[next]
		srt = t_seq[incl_t]
		end = t_seq[cur_t]
		if clockwise:
			if srt < end:
				new_tour += tour[srt:end] + [tour[end]]
			else:
				new_tour += tour[srt:] + tour[:end] + [tour[end]]
		else:
			if srt > end:
				new_tour += tour[srt:end:-1] + [tour[end]]
			else:
				new_tour += tour[srt::-1] + tour[:end - len_t:-1] + [tour[end]]
		if cur_t == p[0]:
			break
	assert len(new_tour) == len(tour)
	return new_tour


def traverse_mst(linked_nodes, degrees, node_num):
	# tst = time.time()
	visited = np.zeros(node_num).astype(bool)
	node_pairs = []
	# round = 1
	while True:
		# print('traverse mst...round: {}'.format(round))
		# round += 1
		pairs = []
		cands = set()
		# step 1: select all nodes whose degree = 1
		indices = np.where(degrees==1)[0]
		for cur_node in indices:
			if visited[cur_node]:
				continue
			nb_node = linked_nodes[cur_node][0]
			if not visited[nb_node]:
				pairs += [(cur_node, nb_node)]
				linked_nodes[cur_node].remove(nb_node)
				linked_nodes[nb_node].remove(cur_node)
				degrees[cur_node] -= 1
				degrees[nb_node] -= 1
				visited[cur_node] = True
				visited[nb_node] = True
				cands = cands | set(linked_nodes[nb_node])  # some candidate nodes may be visited

		# step 2: traverse edges from candidate nodes
		cands = list(cands)
		while len(cands) != 0:
			cur_node = cands[0]
			if visited[cur_node]:
				cands.remove(cur_node)
				continue
			for nb_node in linked_nodes[cur_node]:
				if visited[nb_node]:
					continue
				pairs += [(cur_node, nb_node)]  # use nb_node to represent the two merged nodes
				linked_nodes[cur_node].remove(nb_node)
				linked_nodes[nb_node].remove(cur_node)
				for node in linked_nodes[cur_node] + linked_nodes[nb_node]:
					if not visited[node]:
						cands += [node]
				for nnb_node in linked_nodes[cur_node]:
					linked_nodes[nnb_node].remove(cur_node)
					linked_nodes[nnb_node] += [nb_node]
				linked_nodes[nb_node] += linked_nodes[cur_node]
				linked_nodes[cur_node] = []
				degrees[nb_node] += degrees[cur_node] - 2
				degrees[cur_node] = 0
				visited[nb_node] = True
				break
			visited[cur_node] = True
			cands.remove(cur_node)

		# step 3: traverse remaining unvisited nodes (if exists)
		if sum(visited) < node_num:
			indices = np.where(visited == False)[0]
			for cur_node in indices:
				for nb_node in linked_nodes[cur_node]:
					if visited[nb_node]:
						continue
					pairs += [(cur_node, nb_node)]
					linked_nodes[cur_node].remove(nb_node)
					linked_nodes[nb_node].remove(cur_node)
					for nnb_node in linked_nodes[cur_node]:
						linked_nodes[nnb_node].remove(cur_node)
						linked_nodes[nnb_node] += [nb_node]
					linked_nodes[nb_node] += linked_nodes[cur_node]
					linked_nodes[cur_node] = []
					degrees[nb_node] += degrees[cur_node] - 2
					degrees[cur_node] = 0
					visited[nb_node] = True
					break
				visited[cur_node] = True

		# step 4: determine pairs of current round and update information for next round
		node_pairs += [pairs]
		assert sum(visited) == node_num  # all nodes are visited
		assert len(np.where(degrees < 0)[0]) == 0
		indices = np.where(degrees > 0)[0]
		if len(indices) > 0:
			visited[indices] = False
		else:
			break
	pair_nums = [len(pair) for pair in node_pairs]
	assert sum(pair_nums) == node_num - 1
	# print('traverse MST ({} edges) using {:.2f}s'.format(node_num - 1, time.time() - tst))
	return node_pairs


def traverse_mst_random(linked_nodes, degrees, node_num):
	visited = np.zeros(node_num).astype(bool)
	node_pairs = []
	while True:
		pairs = []
		for cur_idx in torch.randperm(node_num).tolist():
			if visited[cur_idx]:
				continue
			nb_indices = linked_nodes[cur_idx]
			for nb_idx in nb_indices:
				if visited[nb_idx]:
					continue
				pairs += [(cur_idx, nb_idx)]
				linked_nodes[cur_idx].remove(nb_idx)
				linked_nodes[nb_idx].remove(cur_idx)
				degrees[cur_idx] -= 1
				degrees[nb_idx] -= 1
				visited[cur_idx] = True
				visited[nb_idx] = True
				if degrees[cur_idx] > 0:
					for nb_node in linked_nodes[cur_idx]:
						linked_nodes[nb_node].remove(cur_idx)
						linked_nodes[nb_node] += [nb_idx]
					linked_nodes[nb_idx] += linked_nodes[cur_idx]
					linked_nodes[cur_idx] = []
					degrees[nb_idx] += degrees[cur_idx]
					degrees[cur_idx] = 0
				break
		node_pairs += [pairs]
		assert len(np.where(degrees < 0)[0]) == 0
		indices = np.where(degrees > 0)[0]
		if len(indices) > 0:
			visited[indices] = False
		else:
			break
	pair_nums = [len(pair) for pair in node_pairs]
	assert sum(pair_nums) == node_num - 1
	return node_pairs


def merge_vertex_nnpairs(cur_reg, nb_reg, ni, nj):
	cur_tour = cur_reg['tour']
	nb_tour = nb_reg['tour']
	cur_coords = cur_reg['node_coords']
	nb_coords = nb_reg['node_coords']
	(pi, pj), _ = best_nn_pairs(cur_tour, cur_coords, nb_tour, nb_coords, ni, nj)
	merged_tour = cur_tour[pi:] + cur_tour[:pi] + nb_tour[pj:] + nb_tour[:pj]
	return {'tour': merged_tour, }


def plot_tours(coord, m_tour, tour_1, tour_2, n_tour, num_tour, e_nodes, d_nodes, p_nodes):
	plt.figure(dpi=150)
	coords  = coord * 1000
	nodes = torch.cat((coords[tour_1[-1]].unsqueeze(0), coords[tour_1]), dim=0)
	x_coords, y_coords = torch.chunk(nodes, 2, dim=1)
	x_coords = torch.squeeze(x_coords).tolist()
	y_coords = torch.squeeze(y_coords).tolist()
	plt.plot(x_coords, y_coords, 'g-', linewidth=2)

	nodes = torch.cat((coords[tour_2[-1]].unsqueeze(0), coords[tour_2]), dim=0)
	x_coords, y_coords = torch.chunk(nodes, 2, dim=1)
	x_coords = torch.squeeze(x_coords).tolist()
	y_coords = torch.squeeze(y_coords).tolist()
	plt.plot(x_coords, y_coords, 'y-', linewidth=2)

	nodes = torch.cat((coords[m_tour[-1]].unsqueeze(0), coords[m_tour]), dim=0)
	x_coords, y_coords = torch.chunk(nodes, 2, dim=1)
	x_coords = torch.squeeze(x_coords).tolist()
	y_coords = torch.squeeze(y_coords).tolist()
	plt.plot(x_coords, y_coords, 'k--', linewidth=1)

	nodes = torch.cat((coords[n_tour[-1]].unsqueeze(0), coords[n_tour]), dim=0)
	x_coords, y_coords = torch.chunk(nodes, 2, dim=1)
	x_coords = torch.squeeze(x_coords).tolist()
	y_coords = torch.squeeze(y_coords).tolist()
	plt.plot(x_coords, y_coords, 'r-', linewidth=1)

	# nodes = coords[num_tour]
	# x_coords, y_coords = torch.chunk(nodes, 2, dim=1)
	# x_coords = torch.squeeze(x_coords).tolist()
	# y_coords = torch.squeeze(y_coords).tolist()
	# for i, n in enumerate(num_tour):
	# 	plt.text(x_coords[i], y_coords[i], str(i), fontdict={'fontsize': 'xx-large'})

	x_coords, y_coords = coords[e_nodes[0]].tolist()
	plt.plot([x_coords], [y_coords], 'ro', markersize=5)
	x_coords, y_coords = coords[e_nodes[1]].tolist()
	plt.plot([x_coords], [y_coords], 'rx', markersize=5)
	x_coords, y_coords = coords[e_nodes[2]].tolist()
	plt.plot([x_coords], [y_coords], 'ro', markersize=5)
	x_coords, y_coords = coords[e_nodes[3]].tolist()
	plt.plot([x_coords], [y_coords], 'rx', markersize=5)

	nodes = coords[p_nodes]
	x_coords, y_coords = torch.chunk(nodes, 2, dim=1)
	x_coords = torch.squeeze(x_coords).tolist()
	y_coords = torch.squeeze(y_coords).tolist()
	plt.plot(x_coords, y_coords, 'bo', markersize=3)

	nodes = coords[d_nodes]
	x_coords, y_coords = torch.chunk(nodes, 2, dim=1)
	x_coords = torch.squeeze(x_coords).tolist()
	y_coords = torch.squeeze(y_coords).tolist()
	plt.plot(x_coords, y_coords, 'bx', markersize=3)

	plt.show()
