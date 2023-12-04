import time
import numpy as np
from tqdm import tqdm
from myutils import load_data
from baselines.am.am import load_am, am_solve


if __name__ == "__main__":
    device = 'cuda:0'
    dataset = load_data('Dataset/rei/tiny/TSP-100_dataset')
    baseline = load_data('Dataset/rei/tiny/TSP-100_baseline')
    model = load_am('tsp-100', 'greedy', device)

    gaps = []
    st = time.time()
    for idx, data in tqdm(enumerate(dataset), desc='serial solving...', total=len(dataset)):
        costs, tours, durations = am_solve(model, data, 1, 'greedy', device)
        gap = (costs[0] / baseline['length'][idx] - 1) * 100
        gaps += [gap]
        # print('sample {}, gap: {:.2f}%'.format(idx, gap))
    print('avg_gap: {:.2f}%, total_time: {:.2f}'.format(np.mean(gaps), time.time() - st))

    print('parallel solving...')
    gaps = []
    pt = time.time()
    costs, tours, durations = am_solve(model, dataset, 100, 'greedy', device)
    for idx, cost in enumerate(costs):
        gap = (cost / baseline['length'][idx] - 1) * 100
        gaps += [gap]
        # print('sample {}, gap: {:.2f}%'.format(idx, gap))
    print('avg_gap: {:.2f}%, total_time: {:.2f}'.format(np.mean(gaps), time.time() - pt))
