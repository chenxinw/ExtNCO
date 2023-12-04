import os
import time
import torch
import numpy as np
import scipy.sparse
import scipy.spatial
from tqdm import tqdm
from myutils import load_data, save_data, compute_tour_length, load_instance

from argparse import ArgumentParser
from sklearn.neighbors import KDTree
from torch_geometric.data import Data as GraphData
from baselines.difusco.pl_tsp_model import TSPModel
from baselines.difusco.diffusion_schedulers import InferenceSchedule
from baselines.difusco.tsp_utils import cuda_2_opt, merge_tours


def arg_parser():
    parser = ArgumentParser(description='Train a Pytorch-Lightning diffusion model on a TSP dataset.')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_scheduler', type=str, default='constant')

    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--use_activation_checkpoint', action='store_true')

    parser.add_argument('--diffusion_type', type=str, default='gaussian')
    parser.add_argument('--diffusion_schedule', type=str, default='linear')
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--inference_diffusion_steps', type=int, default=1000)
    parser.add_argument('--inference_schedule', type=str, default='linear')
    parser.add_argument('--inference_trick', type=str, default="ddim")
    parser.add_argument('--sequential_sampling', type=int, default=1)
    parser.add_argument('--parallel_sampling', type=int, default=1)

    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--sparse_factor', type=int, default=-1)
    parser.add_argument('--aggregation', type=str, default='sum')
    parser.add_argument('--two_opt_iterations', type=int, default=1000)

    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--resume_weight_only', action='store_true')

    args = parser.parse_args()
    return args


def parse_data(idx, coords, sparse_factor=-1):
    assert len(coords.shape) == 2
    node_num = len(coords)
    # tour = np.concatenate((np.arange(node_num), np.array([0]))).astype(int)
    tour = torch.arange(node_num + 1).long()
    tour[-1] = 0

    if sparse_factor <= 0:  # Return a densely connected graph
        adj_matrix = torch.zeros(node_num, node_num).float()
        for i in range(node_num):
            adj_matrix[tour[i], tour[i + 1]] = 1
        return (coords, adj_matrix, tour)
    else:  # Return a sparse graph where each node is connected to its k nearest neighbors
        kdt = KDTree(coords.numpy(), leaf_size=30, metric='euclidean')
        dis_knn, idx_knn = kdt.query(coords, k=sparse_factor, return_distance=True)

        edge_index_0 = torch.arange(node_num).unsqueeze(1).repeat(1, sparse_factor).flatten()
        edge_index_1 = torch.from_numpy(idx_knn.flatten())
        edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

        tour_edges = np.zeros(node_num, dtype=np.int64)
        tour_edges[tour[:-1]] = tour[1:]
        tour_edges = torch.from_numpy(tour_edges)
        tour_edges = tour_edges.unsqueeze(1).repeat(1, sparse_factor).flatten()
        tour_edges = torch.eq(edge_index_1, tour_edges).unsqueeze(1)
        graph_data = GraphData(x=coords, edge_index=edge_index, edge_attr=tour_edges)

        point_indicator = torch.tensor([idx]).long()
        edge_indicator = torch.tensor(edge_index.shape[1]).long()

        return (graph_data, point_indicator, edge_indicator, tour)


def save_heatmap(adj_mat, edge_index, sparse=False, filepath=None):
    adj_mat = np.split(adj_mat, 1, axis=0)

    if not sparse:
        adj_mat = [data[0] + data[0].T for data in adj_mat]
    else:
        adj_mat = [
                scipy.sparse.coo_matrix(
                        (data, (edge_index[0], edge_index[1])),
                ).toarray() + scipy.sparse.coo_matrix(
                        (data, (edge_index[1], edge_index[0])),
                ).toarray() for data in adj_mat
        ]

    fp = open(filepath, 'a+')
    node_num = int(len(adj_mat[0]))
    print('{}'.format(node_num), file=fp)
    for node_idx in range(node_num):
        outStr = '{:.8f}'.format(adj_mat[0][node_idx, 0])
        for val in adj_mat[0][node_idx, 1:]:
            outStr += ' {:.8f}'.format(val)
        print(outStr, file=fp)
    fp.close()


def inference(model, batch, device, gen_heatmap=False, filepath=None):
    edge_index = None
    np_edge_index = None

    # parse input data
    if not model.sparse:
        points, adj_matrix, gt_tour = batch
        np_points = points.cpu().numpy()
    else:
        graph_data, point_indicator, edge_indicator, gt_tour = batch
        route_edge_flags = graph_data.edge_attr
        points = graph_data.x
        edge_index = graph_data.edge_index
        num_edges = edge_index.shape[1]
        batch_size = point_indicator.shape[0]
        adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
        assert len(points.shape) == 2 and points.shape[1] == 2
        assert len(edge_index.shape) == 2 and edge_index.shape[0] == 2
        np_points = points.cpu().numpy()
        np_edge_index = edge_index.cpu().numpy()

    if model.args.parallel_sampling > 1:
        if not model.sparse:
            points = points.repeat(model.args.parallel_sampling, 1, 1)
        else:
            points = points.repeat(model.args.parallel_sampling, 1)
            edge_index = model.duplicate_edge_index(edge_index, np_points.shape[0], device)

    greedy_tours = []
    for _ in range(model.args.sequential_sampling):
        xt = torch.randn_like(adj_matrix.float()).to(model.device)
        if model.args.parallel_sampling > 1:
            if not model.sparse:
                xt = xt.repeat(model.args.parallel_sampling, 1, 1)
            else:
                xt = xt.repeat(model.args.parallel_sampling, 1)
            xt = torch.randn_like(xt)

        if model.diffusion_type == 'categorical':
            xt = (xt > 0).long()
        else:
            assert model.diffusion_type == 'gaussian'
            xt.requires_grad = True

        if model.sparse:
            xt = xt.reshape(-1)

        # Diffusion iterations
        time_schedule = InferenceSchedule(inference_schedule=model.args.inference_schedule,
                                          T=model.diffusion.T, inference_T=model.args.inference_diffusion_steps)
        for i in tqdm(range(model.args.inference_diffusion_steps), desc='denoise steps'):
            t1, t2 = time_schedule(i)
            t1 = np.array([t1]).astype(int)
            t2 = np.array([t2]).astype(int)

            if model.diffusion_type == 'categorical':
                xt = model.categorical_denoise_step(points, xt, t1, device, edge_index, target_t=t2)
            else:
                assert model.diffusion_type == 'gaussian'
                xt = model.gaussian_denoise_step(points, xt, t1, device, edge_index, target_t=t2)

        if model.diffusion_type == 'categorical':
            adj_mat = xt.float().cpu().detach().numpy() + 1e-6
        else:
            assert model.diffusion_type == 'gaussian'
            adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
        torch.cuda.empty_cache()

        if gen_heatmap == True:
            assert filepath is not None
            save_heatmap(adj_mat, np_edge_index, sparse=model.sparse, filepath=filepath)
            print('save heatmap to {}'.format(filepath))

        tours, merge_iterations = merge_tours(adj_mat, np_points, np_edge_index, sparse_graph=model.sparse,
                                              parallel_sampling=model.args.parallel_sampling)
        assert len(tours) == model.args.parallel_sampling == 1
        greedy_tours += [tours[0]]

    assert len(greedy_tours) == model.args.sequential_sampling
    return torch.tensor(greedy_tours).long()[:, :-1]


def main(args):
    print('sparse factor: {}, diffusion step: {}, 2-opt step: {}'.format(
            args.sparse_factor, args.inference_diffusion_steps, args.two_opt_iterations))
    device = 'cuda:3'

    # REI dataset
    all_results = dict()
    for scale in ['10K', '20K', '50K']:
        baseline = load_data('Dataset/rei/large/{}_base'.format(scale))
        dataset = load_data('Dataset/rei/large/{}_data'.format(scale))
        results = {
                'greedy': {
                        'len' : [],
                        'gap' : [],
                        'time': [], },
                '2-opt': {
                        'len' : [],
                        'gap' : [],
                        'time': [], },
        }
        model = TSPModel.load_from_checkpoint(args.ckpt_path, param_args=args, map_location=device)
        model.eval()
        for idx in range(20):
            print('sample {}, hk: {:.4f}, lkh: {:.4f}'.format(idx, baseline['hk'][idx], baseline['lkh']['len'][idx]))
            parsed_data = parse_data(idx, dataset[idx], args.sparse_factor)
            gd_time = time.time()
            gd_tours = inference(model, parsed_data, device)
            gd_lengths = []
            gd_gaps = []
            for seq_idx, gd_tour in enumerate(gd_tours):
                assert len(gd_tour.unique()) == len(dataset[idx])
                assert gd_tour.min() == 0 and gd_tour.max() == len(dataset[idx]) - 1
                gd_length = compute_tour_length(gd_tour, dataset[idx])
                gd_gap = (gd_length / baseline['hk'][idx] - 1) * 100
                gd_lengths += [gd_length]
                gd_gaps += [gd_gap]
            min_idx = np.argmin(gd_lengths)
            gd_tour = gd_tours[min_idx]
            gd_length = gd_lengths[min_idx]
            gd_gap = gd_gaps[min_idx]
            results['greedy']['len'] += [gd_length]
            results['greedy']['gap'] += [gd_gap]
            results['greedy']['time'] += [time.time() - gd_time]
            print('greedy_length: {:.4f}, gap: {:.4f}%, time: {:.4f}'.format(gd_length, gd_gap, time.time() - gd_time))

            iter_num = args.two_opt_iterations
            rf_tour, rf_time, it_num = cuda_2_opt(dataset[idx].to(device), gd_tour, iter_num=iter_num)
            assert len(rf_tour.unique()) == len(dataset[idx])
            assert rf_tour.min() == 0 and rf_tour.max() == len(dataset[idx]) - 1
            rf_length = compute_tour_length(rf_tour, dataset[idx])
            rf_gap = (rf_length / baseline['hk'][idx] - 1) * 100
            results['2-opt']['len'] += [rf_length]
            results['2-opt']['gap'] += [rf_gap]
            results['2-opt']['time'] += [rf_time]
            print('2-opt_length: {:.4f}, gap: {:.4f}%, time: {:.4f}, iter_num: {}'.format(
                    rf_length, rf_gap, rf_time, it_num))
            torch.cuda.empty_cache()
        print('avg_greedy: len: {:.2f}, gap: {:.2f}%, time: {:.2f}'.format(np.mean(results['greedy']['len']),
                                                                           np.mean(results['greedy']['gap']),
                                                                           np.mean(results['greedy']['time'])))
        print('avg_2-opt: len: {:.2f}, gap: {:.2f}%, time: {:.2f}'.format(np.mean(results['2-opt']['len']),
                                                                          np.mean(results['2-opt']['gap']),
                                                                          np.mean(results['2-opt']['time'])))
        print('')
        all_results[scale] = results
        save_data(all_results, 'results/comparative/rei/difusco.res', overwrite=True)

    # TSPLib/VLSI dataset
    for Dataset in ['tsplib', 'vlsi']:
        instances = []
        dataset = []
        all_results = dict()
        baseline = load_data('Dataset/{}/prev_baseline'.format(Dataset))
        model = TSPModel.load_from_checkpoint(args.ckpt_path, param_args=args, map_location=device)
        model.eval()
        for filename in os.listdir('Dataset/{}'.format(Dataset)):
            if '.tsp' in filename:
                ins = filename[:filename.index('.')]
                instances += [ins]
                dataset += [load_instance('Dataset/{}/{}'.format(Dataset, filename), scale=True)]
        for idx, ins in enumerate(instances):
            results = {
                    'greedy': {
                            'len' : [],
                            'gap' : [],
                            'time': [], },
                    '2-opt' : {
                            'len' : [],
                            'gap' : [],
                            'time': [], },
            }
            opt_val = baseline[ins]['opt']
            print('instance {}: {}, opt_len: {:.4f}'.format(idx, ins, opt_val))

            for run_idx in range(10):
                print('repeat_run: {}'.format(run_idx))
                parsed_data = parse_data(idx, dataset[idx], args.sparse_factor)
                gd_time = time.time()
                gd_tours = inference(model, parsed_data, device)
                gd_lengths = []
                gd_gaps = []
                for seq_idx, gd_tour in enumerate(gd_tours):
                    assert len(gd_tour.unique()) == len(dataset[idx])
                    assert gd_tour.min() == 0 and gd_tour.max() == len(dataset[idx]) - 1
                    gd_length = compute_tour_length(gd_tour, dataset[idx])
                    gd_gap = (gd_length / opt_val - 1) * 100
                    gd_lengths += [gd_length]
                    gd_gaps += [gd_gap]
                min_idx = np.argmin(gd_lengths)
                gd_tour = gd_tours[min_idx]
                gd_length = gd_lengths[min_idx]
                gd_gap = gd_gaps[min_idx]
                gd_time = time.time() - gd_time
                results['greedy']['len'] += [gd_length]
                results['greedy']['gap'] += [gd_gap]
                results['greedy']['time'] += [gd_time]
                print('greedy: len: {:.4f}, gap: {:.4f}%, time: {:.4f}'.format(gd_length, gd_gap, gd_time))

                iter_num = args.two_opt_iterations
                rf_tour, rf_time, it_num = cuda_2_opt(dataset[idx].to(device), gd_tour, iter_num=iter_num)
                assert len(rf_tour.unique()) == len(dataset[idx])
                assert rf_tour.min() == 0 and rf_tour.max() == len(dataset[idx]) - 1
                rf_length = compute_tour_length(rf_tour, dataset[idx])
                rf_gap = (rf_length / opt_val - 1) * 100
                results['2-opt']['len'] += [rf_length]
                results['2-opt']['gap'] += [rf_gap]
                results['2-opt']['time'] += [rf_time]
                print('2-opt: len: {:.4f}, gap: {:.4f}%, time: {:.4f}, iter_num: {}'.format(
                        rf_length, rf_gap, rf_time, it_num))
                torch.cuda.empty_cache()
            print('avg_greedy: len: {:.2f}, gap: {:.2f}%, time: {:.2f}'.format(np.mean(results['greedy']['len']),
                                                                               np.mean(results['greedy']['gap']),
                                                                               np.mean(results['greedy']['time'])))
            print('avg_2-opt: len: {:.2f}, gap: {:.2f}%, time: {:.2f}'.format(np.mean(results['2-opt']['len']),
                                                                              np.mean(results['2-opt']['gap']),
                                                                              np.mean(results['2-opt']['time'])))
            print('')
            all_results[ins] = results
            save_data(all_results, 'results/comparative/{}/difusco.res'.format(Dataset), overwrite=True)



if __name__ == '__main__':
    # Evaluation on TSP-Categorical with sampling decoding (4x sequential)
    args = arg_parser()
    args.diffusion_type = 'categorical'
    args.learning_rate = 0.0002
    args.weight_decay = 0.0001
    args.lr_scheduler = 'cosine-decay'
    args.inference_schedule = 'cosine'
    args.sparse_factor = 50  # used for large scale instances
    args.inference_diffusion_steps = 50
    args.sequential_sampling = 1
    args.two_opt_iterations = 5000
    # args.ckpt_path = './pretrained/difusco/tsp100.ckpt'
    args.ckpt_path = './pretrained/difusco/tsp10000.ckpt'
    args.resume_weight_only = True
    # args.use_activation_checkpoint = True

    with torch.no_grad():
        main(args)