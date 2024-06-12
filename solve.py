import time
import lkh
from tqdm import tqdm
from mpire import WorkerPool
from myutils import *
from baselines.am.am import am_solve
from baselines.pomo.pomo import pomo_solve


def batch_model_solve(region, coords, batch_size=256, model=None, find_sub=False, hide_bar=False):
	region.solve_time = time.time()
	regions = region.sub_regs if not find_sub else find_sub_regions(region)
	region_scale = torch.Tensor([reg.node_num for reg in regions]).long()
	scale_indices = torch.argsort(region_scale)
	scales, counts = torch.unique(region_scale, sorted=True, return_counts=True)
	scales = scales.tolist()
	counts = counts.tolist()

	pbar = tqdm(total=len(regions), desc='[batch model solving]', unit='region', disable=hide_bar)
	cur_idx = 0
	for scale_idx, scale in enumerate(scales):
		if scale == 0:
			index_step = counts[scale_idx]
			for idx in range(index_step):
				cur_reg = regions[scale_indices[cur_idx + idx]]
				cur_reg.length = 0
				cur_reg.tour = None
				cur_reg.solved = True
			cur_idx += index_step
			pbar.set_postfix(cur_scale=scale)
			pbar.update(index_step)
			continue

		while counts[scale_idx] != 0:
			index_step = min(counts[scale_idx], batch_size)
			batch_node = regions[scale_indices[cur_idx]].node_index.unsqueeze(0)
			for idx in range(1, index_step):
				cur_node = regions[scale_indices[cur_idx + idx]].node_index.unsqueeze(0)
				batch_node = torch.cat((batch_node, cur_node), dim=0)
			batch_node = batch_node.to(coords.device)
			batch_coords = coords[batch_node]

			# scale node coordinates into [0,1]
			b_x, b_y = batch_coords.chunk(2, dim=2)  # [bsz,gsz,1]
			b_x_min, b_y_min = torch.min(batch_coords, dim=1)[0].chunk(2, dim=1)  # [bsz,1]
			b_x_max, b_y_max = torch.max(batch_coords, dim=1)[0].chunk(2, dim=1)  # [bsz,1]
			b_n_scale = torch.cat(((b_x_max - b_x_min).max(dim=1)[0].unsqueeze(1),  # [bsz,1]
			                       (b_y_max - b_y_min).max(dim=1)[0].unsqueeze(1)), dim=1).max(dim=1)[0].unsqueeze(1)
			b_x = (b_x - b_x_min.unsqueeze(2)) / b_n_scale.unsqueeze(2)
			b_y = (b_y - b_y_min.unsqueeze(2)) / b_n_scale.unsqueeze(2)
			b_n_coords = torch.cat((b_x, b_y), dim=2)
			batch_tours, _ = model(b_n_coords, decode_type='greedy')
			for bi in range(index_step):
				batch_tours[bi] = batch_node[bi, batch_tours[bi]]
			batch_length = compute_tour_length(batch_tours, coords, batch=True).tolist()

			for idx in range(index_step):
				cur_reg = regions[scale_indices[cur_idx + idx]]
				cur_reg.tour = batch_tours[idx].cpu()
				cur_reg.length = batch_length[idx]
				cur_reg.solved = True
			cur_idx += index_step
			counts[scale_idx] -= index_step
			pbar.set_postfix(cur_scale=scale)
			pbar.update(index_step)
			torch.cuda.empty_cache()
	pbar.close()

	region.solve_time = time.time() - region.solve_time
	region.single_solve_time = region.solve_time / len(regions)
	region.total_time += region.solve_time


def batch_am_solve(region, coords, model, batch_size=256, find_sub=False, hide_bar=False):
	region.solve_time = time.time()
	regions = region.sub_regs if not find_sub else find_sub_regions(region)
	region_scale = torch.Tensor([reg.node_num for reg in regions]).long().to(coords.device)
	scale_indices = torch.argsort(region_scale)
	scales, counts = torch.unique(region_scale, sorted=True, return_counts=True)
	del region_scale
	scales = scales.tolist()
	counts = counts.tolist()

	pbar = tqdm(total=len(regions), desc='[batch am solving]', unit='region', disable=hide_bar)
	cur_idx = 0
	for scale_idx, scale in enumerate(scales):
		if scale == 0:
			index_step = counts[scale_idx]
			for idx in range(index_step):
				cur_reg = regions[scale_indices[cur_idx + idx]]
				cur_reg.length = 0
				cur_reg.tour = None
				cur_reg.solved = True
			cur_idx += index_step
			pbar.set_postfix(cur_scale=scale)
			pbar.update(index_step)
			continue

		while counts[scale_idx] != 0:
			index_step = min(counts[scale_idx], batch_size)
			batch_node = regions[scale_indices[cur_idx]].node_index.unsqueeze(0)
			for idx in range(1, index_step):
				cur_node = regions[scale_indices[cur_idx + idx]].node_index.unsqueeze(0)
				batch_node = torch.cat((batch_node, cur_node), dim=0)
			batch_node = batch_node.to(coords.device)
			batch_coords = coords[batch_node]

			# scale node coordinates into [0,1]
			b_x, b_y = batch_coords.chunk(2, dim=2)  # [bsz,gsz,1]
			b_x_min, b_y_min = torch.min(batch_coords, dim=1)[0].chunk(2, dim=1)  # [bsz,1]
			b_x_max, b_y_max = torch.max(batch_coords, dim=1)[0].chunk(2, dim=1)  # [bsz,1]
			b_n_scale = torch.cat(((b_x_max - b_x_min).max(dim=1)[0].unsqueeze(1),  # [bsz,1]
								   (b_y_max - b_y_min).max(dim=1)[0].unsqueeze(1)), dim=1).max(dim=1)[0].unsqueeze(1)
			b_x = (b_x - b_x_min.unsqueeze(2)) / b_n_scale.unsqueeze(2)
			b_y = (b_y - b_y_min.unsqueeze(2)) / b_n_scale.unsqueeze(2)
			b_n_coords = torch.cat((b_x, b_y), dim=2)
			_, batch_tours, _ = am_solve(model, b_n_coords, batch_size, 'greedy', coords.device)
			for bi in range(index_step):
				batch_tours[bi] = batch_node[bi, batch_tours[bi]]
			batch_length = compute_tour_length(batch_tours, coords, batch=True).tolist()

			for idx in range(index_step):
				cur_reg = regions[scale_indices[cur_idx + idx]]
				cur_reg.tour = batch_tours[idx].cpu()
				cur_reg.length = batch_length[idx]
				cur_reg.solved = True
			cur_idx += index_step
			counts[scale_idx] -= index_step
			pbar.set_postfix(cur_scale=scale)
			pbar.update(index_step)
			torch.cuda.empty_cache()
	pbar.close()

	region.solve_time = time.time() - region.solve_time
	region.single_solve_time = region.solve_time / len(regions)
	region.total_time += region.solve_time


def batch_pomo_solve(region, coords, model, pomo_size=5, enable_aug=True, find_sub=False, hide_bar=False):
	region.solve_time = time.time()
	regions = region.sub_regs if not find_sub else find_sub_regions(region)
	region_scale = torch.Tensor([reg.node_num for reg in regions]).long().to(coords.device)
	scale_indices = torch.argsort(region_scale)
	scales, counts = torch.unique(region_scale, sorted=True, return_counts=True)
	del region_scale
	scales = scales.tolist()
	counts = counts.tolist()

	pbar = tqdm(total=len(regions), desc='[batch model solving]', unit='region', disable=hide_bar)
	cur_idx = 0
	assert scales[0] >= 10, print('scale {} too large'.format(scales[0]))
	for scale_idx, scale in enumerate(scales):
		batch_size = 512
		# if scale <= 125:
		# 	batch_size = 512
		# elif scale <= 170:
		# 	batch_size = 256
		# else:
		# 	batch_size = 128

		while counts[scale_idx] != 0:
			step_size = min(counts[scale_idx], batch_size)
			batch_node = torch.tensor([]).long()
			for idx in range(step_size):
				cur_node = regions[scale_indices[cur_idx + idx]].node_index.unsqueeze(0).cpu()
				batch_node = torch.cat((batch_node, cur_node), dim=0)
			batch_coords = coords[batch_node]

			# scale node coordinates into [0,1]
			b_x, b_y = batch_coords.chunk(2, dim=2)  # [bsz,gsz,1]
			b_x_min, b_y_min = torch.min(batch_coords, dim=1)[0].chunk(2, dim=1)  # [bsz,1]
			b_x_max, b_y_max = torch.max(batch_coords, dim=1)[0].chunk(2, dim=1)  # [bsz,1]
			bn_scale = torch.cat(((b_x_max - b_x_min).max(dim=1)[0].unsqueeze(1),  # [bsz,1]
			                       (b_y_max - b_y_min).max(dim=1)[0].unsqueeze(1)), dim=1).max(dim=1)[0].unsqueeze(1)
			b_x = (b_x - b_x_min.unsqueeze(2)) / bn_scale.unsqueeze(2)
			b_y = (b_y - b_y_min.unsqueeze(2)) / bn_scale.unsqueeze(2)
			bn_coords = torch.cat((b_x, b_y), dim=2)
			del b_x, b_y

			# batch solve
			batch_tours, batch_lengths, _ = pomo_solve(coords=bn_coords, model=model, pomo_size=pomo_size,
													   enable_aug=enable_aug, device=coords.device)
			for bi in range(step_size):
				batch_tours[bi] = batch_node[bi, batch_tours[bi]]
			# batch_length = compute_tour_length(batch_tours, coords, batch=True).tolist()
			# checking
			# for idx, cal_len in enumerate(batch_length):
			# 	assert round(cal_len / bn_scale[idx].item() - batch_lengths[idx], 4) == 0, \
			# 		print('cal_len: {:.6f} ret_len : {:.6f}'.format(cal_len, batch_lengths[idx]))

			for idx in range(step_size):
				cur_reg = regions[scale_indices[cur_idx + idx]]
				cur_reg.tour = batch_tours[idx]
				cur_reg.length = batch_lengths[idx] * bn_scale[idx].item()
				cur_reg.solved = True
			cur_idx += step_size
			counts[scale_idx] -= step_size
			pbar.set_postfix(cur_scale=scale)
			pbar.update(step_size)
			torch.cuda.empty_cache()
	pbar.close()

	region.solve_time = time.time() - region.solve_time
	region.single_solve_time = region.solve_time / len(regions)
	region.total_time += region.solve_time


def parl_lkh_solve(region, coords, thread_num, mode, hide_bar=False):
	region.solve_time = time.time()
	regions = find_sub_regions(region)

	params = [{'coords': coords[reg.node_index].cpu().numpy(), 'mode': mode, } for reg in regions]
	with WorkerPool(n_jobs=thread_num) as pool:
		parl_res = pool.map(lkh_solve, params, progress_bar=not hide_bar,
		                    progress_bar_options={'desc': 'default lkh solving'})
		for idx, res in enumerate(parl_res):
			cur_reg = regions[idx]
			cur_reg.tour = cur_reg.node_index[res['tour']]
			cur_reg.length = res['length']
			cur_reg.solved = True

	region.solve_time = time.time() - region.solve_time
	region.single_solve_time = region.solve_time / len(regions)
	region.total_time += region.solve_time


def lkh_solve(coords, mode='default', scale=True):
	node_num = len(coords)
	if scale:
		scale_factor = 10000
		while True:
			node_coords = (coords * scale_factor).round().astype(int)
			if len(np.unique(node_coords, axis=0)) == node_num:
				break
			else:
				scale_factor *= 100
	else:
		node_coords = coords.astype(int)

	# problem_str = 'NAME: sample\nCOMMENT: single sample\nTYPE: TSP\nDIMENSION: ' \
	#               + str(len(coords)) + '\nEDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n'
	# for idx, (x, y) in enumerate(node_coords):
	# 	problem_str += '{} {} {}\n'.format(idx + 1, x, y)
	# problem = tsplib95.parse(problem_str)
	st = time.time()
	problem = tsplib95.models.StandardProblem()
	problem.name = 'TSP'
	problem.type = 'TSP'
	problem.dimension = node_num
	problem.edge_weight_type = 'EUC_2D'
	problem.node_coords = {idx + 1: node_coords[idx] for idx in range(node_num)}
	if mode == 'default':
		solution = lkh.solve('./LKH-3.0.7/LKH', problem=problem)[0]
	else:
		assert mode == 'fast'
		solution = lkh.solve('./LKH-3.0.7/LKH', problem=problem,
							 initial_tour_algorithm='greedy', initial_period=100, max_swaps=1000,
							 candidate_set_type='delaunay', extra_candidate_set_type='quadrant')[0]
	tour = np.array(solution) - 1
	return {
			'tour'  : tour,
			'time'  : time.time() - st,
			'length': compute_tour_length_np(coords, tour),
	}
