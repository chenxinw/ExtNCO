import time


class Region(object):
	def __init__(self, node_index, region_index=None, xs=None, ys=None, xl=None, yl=None):
		self.node_index = node_index        # torch.Tensor
		self.node_num = len(node_index)     # int
		self.region_index = region_index    # int / None
		self.area = {'xs': xs, 'ys': ys, 'xl': xl, 'yl': yl, }

		self.sub_regs = []          # region objects, list
		self.nb_regs = []           # indices of neighbor regions, list
		self.centroid = None        # torch.Tensor / None
		self.tour = None            # torch.Tensor / None
		self.length = 0

		self.divided = False
		self.solved = False
		self.merged = False

		self.total_time = 0
		self.divide_time = 0
		self.solve_time = 0
		self.single_solve_time = 0
		self.merge_time = 0
		self.eval_time = 0
		self.trav_time = 0
		self.misc_time = 0
		self.time_dict = None
		# self.time = dict()

	# def divide(self, coords, params, device='cuda:0', hide_bar=False):
	# 	self.divide_time = time.time()
	# 	from prev_divide import divide
	# 	divide(self, coords, params, nb_num=10, device=device, hide_bar=hide_bar)
	# 	self.divide_time = time.time() - self.divide_time
	# 	self.total_time += self.divide_time

	def merge(self, coords, params, device='cuda:0', hide_bar=False):
		self.merge_time = time.time()
		from merge import merge
		merge(self, coords, params, device, hide_bar)
		self.merge_time = time.time() - self.merge_time
		# print('wrong merge_time: {:.2f}'.format(self.merge_time))
		orders = params['ord']
		for key in orders[1:]:
			self.merge_time -= self.time_dict[key]
		self.total_time += self.merge_time

	# def fine_tune(self, coords, params, device='cuda:0', hide_bar=False):
	# 	self.ft_time = time.time()
	# 	from finetune import fine_tune
	# 	fine_tune(self, coords, params['step'], params['round'], device, hide_bar)
	# 	self.ft_time = time.time() - self.ft_time

	def check_valid(self):
		if len(self.tour) != self.node_num:
			print('Error: len(tour) {} mismatch instance scale: {}'.format(len(self.tour), self.node_num))
			return False
		if len(self.tour.unique()) != self.node_num:
			print('Error: tour.unique(): {} mismatch instance scale: {}'.format(len(self.tour.unique()), self.node_num))
			return False
		if self.tour.min().item() != 0 or self.tour.max().item() != self.node_num - 1:
			print('Error: min: {}, max: {} mismatch instance scale: {}'.format(self.tour.min().item(), self.tour.max().item(), self.node_num))
			return False
		return True


	def detail_time(self):
		print('divide_time:{:.4f}, solve_time: {:.4f}/{:.4f}, merge_time: {:.4f}'.format(
				self.divide_time, self.solve_time, self.single_solve_time, self.merge_time))