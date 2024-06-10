import time
import torch
from myutils import load_data

from baselines.deepaco.net import Net
from baselines.deepaco.aco import ACO
from baselines.deepaco.utils import load_my_dataset
from tqdm import tqdm


EPS = 1e-10

@torch.no_grad()
def infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, k_sparse=None):
    model.eval()
    heu_vec = model(pyg_data)
    heu_mat = model.reshape(pyg_data, heu_vec) + EPS
    # print('model calculation done!')

    aco = ACO(n_ants=n_ants, heuristic=heu_mat.cpu(), distances=distances.cpu(), device='cpu', local_search='nls', )

    costs = torch.zeros(size=(len(t_aco_diff),))
    paths = []
    for i, t in enumerate(t_aco_diff):
        tt = time.time()
        # print('cur_t_aco: {}'.format(i + 1))
        cost, path = aco.run(t, inference=True)
        costs[i] = cost
        paths += [path]
        # print('take {:.2f}s'.format(time.time() - tt))
    best_idx = costs.argmin()
    return costs[best_idx], paths[best_idx]


@torch.no_grad()
def test(dataset, model, n_ants, iterations, k_sparse=None):
    t_aco = list(range(1, 1 + iterations))
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i + 1] - _t_aco[i] for i in range(len(_t_aco) - 1)]
    costs = []
    paths = []
    times = []
    for pyg_data, distances in dataset:
        start_time = time.time()
        best_cost, best_path = infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, k_sparse)
        costs += [best_cost]
        paths += [best_path]
        times  += [time.time() - start_time]

    return costs, paths, times


# def main(n_node, model_file, k_sparse=None, n_ants=48):
def main(n_node, model_file, k_sparse=None, n_ants=15):
    # coords = torch.Tensor([]).float()
    # for sample_idx in range(1):
    #     while True:  # regenerate sample if contain nodes with same coordinate
    #         tmp_data = torch.FloatTensor(n_node, 2).uniform_(0, 1)
    #         if torch.unique(tmp_data, dim=0).size(0) == n_node:
    #             data = tmp_data.unsqueeze(0)
    #             coords = torch.cat((coords, data))
    #             break

    coords = load_data('Dataset/rei/large/{}_data'.format(n_node)).float()[:1]
    dataset = load_my_dataset(coords, k_sparse, device, start_node=0)
    baseline = load_data('Dataset/rei/large/{}_base'.format(n_node))

    t_aco = list(range(1, 3))
    model = Net().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    costs, paths, times = test(dataset, model, n_ants, t_aco, k_sparse, hk_val=baseline['hk'][0])

    gaps = []
    for idx in range(len(coords)):
        hk = prev_baseline['hk'][idx]
        gaps += [(costs[idx] / hk - 1) * 100]
    print('avg length: {:.2f}'.format(np.mean(costs)))
    print('avg gap: {:.2f}'.format(np.mean(gaps)))
    print('avg time: {:.2f}'.format(np.mean(times)))


if __name__ == "__main__":

    print('TSP-100 pretrained model')
    main(n_node='10K', model_file='pretrained/deepaco/deepaco/tsp100.pt', k_sparse=50)
    print('\nTSP-500 pretrained model')
    main(n_node='10K', model_file='pretrained/deepaco/deepaco/tsp500.pt', k_sparse=50)
    print('\nTSP-1000 pretrained model')
    main(n_node='10K', model_file='pretrained/deepaco/deepaco/tsp1000.pt', k_sparse=50)
