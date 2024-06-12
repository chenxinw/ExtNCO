import time
import warnings

import matplotlib.pyplot as plt
import torch
import numpy as np
from tsp import Region
from divide import Divide
from solve import batch_pomo_solve, lkh_solve
from myutils import load_data, save_data, load_instance, compute_tour_length

from baselines.pomo.pomo import load_pomo, pomo_solve
from baselines.htsp.h_tsp import load_htsp, htsp_solve
from baselines.deepaco.net import Net
from baselines.deepaco.utils import load_my_dataset
from baselines.deepaco.script import test
from baselines.difusco.pl_tsp_model import TSPModel
from baselines.difusco.script import arg_parser, parse_data, inference
from baselines.difusco.tsp_utils import cuda_2_opt


def main(method, stage, dataset, device='cuda:0', hide_bar=True):
    print(f'method: {method}, stage: {stage}, dataset: {dataset}, device: {device}\n')

    # load dataset and optimal value
    names, samples, opts = [], [], []
    sample_num = 20 if stage == 'first' else 10
    if stage == 'first':
        if dataset == 'rei':
            for ins in ['10K', '20K', '50K']:
                data = load_data(f'Dataset/{dataset}/small/{ins}_data')
                base = load_data(f'Dataset/{dataset}/small/{ins}_base')
                for idx in range(len(data)):
                    names += [f'{dataset}-{ins}-sample_{idx}']
                    samples += [data[idx]]
                    opts += [base['hk'][idx]]
        else:
            if dataset == 'tsplib':
                instances = ['dsj1000', 'pr1002', 'u1060', 'vm1084', 'pcb1173', 'd1291', 'rl1304',
                             'rl1323', 'nrw1379', 'u1432', 'd1655', 'vm1748', 'u1817', 'rl1889',
                             'd2103', 'u2152', 'u2319', 'pr2392', 'pcb3038', 'fnl4461', 'rl5915',
                             'rl5934', 'pla7397', 'rl11849', 'usa13509', 'brd14051', 'd15112',
                             'd18512', 'pla33810', 'pla85900']
            elif dataset == 'vlsi':
                instances = ['xmc10150', 'xvb13584', 'xia16928', 'pjh17845', 'frh19289', 'fnc19402', 'ido21215',
                             'fma21553', 'lsb22777', 'xrh24104', 'bbz25234', 'irx28268', 'fyg28534', 'icx28698',
                             'boa28924', 'ird29514', 'pbh30440', 'xib32892', 'fry33203', 'bby34656', 'pba38478',
                             'ics39603', 'rbz43748', 'fht47608', 'fna52057', 'bna56769', 'dan59296']
            else:
                assert False, print(f'unsupported dataset: {dataset} at stage {stage}')
            base = load_data(f'Dataset/{dataset}/baseline')
            for ins in instances:
                data = load_instance(f'Dataset/{dataset}/{ins}.tsp', scale=True)
                for idx in range(sample_num):
                    names += [f'{dataset}-{ins}-repeat_{idx}']
                    samples += [data]
                    opts += [base[ins]['opt']]
    elif stage == 'second':
        if dataset == 'rei':
            for ins in [f'{scale}K' for scale in range(100, 1001, 100)]:
                data = load_data(f'Dataset/{dataset}/large/{ins}_data')
                base = load_data(f'Dataset/{dataset}/large/web_{ins}_base')
                for idx in range(len(data)):
                    names += [f'{dataset}-{ins}-sample_{idx}']
                    samples += [data[idx]]
                    opts += [base['hk'][idx]]
        elif dataset == 'vlsi':
            instances = ['sra104815', 'ara238025', 'lra498378', 'lrb744710']
            base = load_data(f'Dataset/{dataset}/large/baseline')
            for ins in instances:
                data = load_instance(f'Dataset/{dataset}/large/{ins}.tsp', scale=True)
                for idx in range(sample_num):
                    names += [f'{dataset}-{ins}-repeat_{idx}']
                    samples += [data]
                    opts += [base[ins]['opt']]
        else:
            assert False, print(f'unsupported dataset: {dataset} at stage {stage}')
    else:
        assert False, print(f'unsupported stage: {stage}')

    # load model
    if method == 'extnco-eff' or method == 'extnco-bal' or method == 'extnco-qlt':
        pomo = load_pomo('TSP-100', device)
        div_params = {
                'grid'     : 1000,
                'sub'      : 100,
                'rep_ratio': 0.005,
                'min_thr'  : 50,
                'nb_num'   : 10,
        }
        if method == 'extnco-eff':
            merge_params = {
                    'comb': 'mst',
                    'var' : 'eff',
                    'ord' : ['bts'], }
        elif method == 'extnco-bal':
            merge_params = {
                    'comb': 'mst',
                    'var' : 'bal',
                    'ord' : ['btp'],
            }
        else:
            assert method == 'extnco-qlt'
            merge_params = {
                    'comb': 'mst',
                    'var' : 'qlt',
                    'ord' : ['btp'],
            }
    elif method == 'htsp':
        htsp_model, htsp_env = load_htsp(upper='tsp10000', device=device)
    elif method == 'pomo':
        pomo = load_pomo('TSP-100', device)
    elif method == 'deepaco':
        deepaco = Net().to(device)
        deepaco.load_state_dict(torch.load('baselines/deepaco/tsp1000.pt', map_location=device))
    elif method == 'difusco' or method == 'difusco-2opt':
        args = arg_parser()
        args.diffusion_type = 'categorical'
        args.learning_rate = 0.0002
        args.weight_decay = 0.0001
        args.lr_scheduler = 'cosine-decay'
        args.inference_schedule = 'cosine'
        args.sparse_factor = 100  # 50 for REI-20K instances
        args.inference_diffusion_steps = 50
        args.sequential_sampling = 1
        args.two_opt_iterations = 5000
        args.ckpt_path = 'baselines/difusco/tsp10000.ckpt'
        args.resume_weight_only = True
        # load model
        difusco = TSPModel.load_from_checkpoint(args.ckpt_path, param_args=args, map_location=device)
        difusco.eval()

    # traverse instances
    lens, gaps, times = [], [], []
    for idx in range(len(samples)):
        name = names[idx]
        sample = samples[idx].to(device)
        opt = opts[idx]

        # solve instance
        if method == 'extnco-eff' or method == 'extnco-bal' or method == 'extnco-qlt':
            ExtNCO = Region(node_index=torch.arange(len(sample)))
            Divide(region=ExtNCO, coords=sample, type='hybrid', rep=True, params=div_params,
                   device=device, hide_bar=hide_bar)
            batch_pomo_solve(ExtNCO, sample, model=pomo, hide_bar=hide_bar)
            ExtNCO.merge(sample, merge_params, device, hide_bar)
            # ExtNCO.detail_time()
            assert ExtNCO.check_valid()
            lens += [compute_tour_length(ExtNCO.tour, sample)]
            times += [ExtNCO.total_time]
        elif method == 'htsp':
            tour, ut, sub_info = htsp_solve(sample.unsqueeze(0), htsp_model, htsp_env)
            lens += [compute_tour_length(tour, sample)]
            times += [ut]
        elif method == 'pomo':
            st = time.time()
            tours, _, _ = pomo_solve(sample.unsqueeze(0), pomo, pomo_size=5, enable_aug=True, device=device)
            lens += [compute_tour_length(tours[0], sample)]
            times += [time.time() - st]
        elif method == 'deepaco':
            sample = load_my_dataset(sample.unsqueeze(0), k_sparse=50, device=device, start_node=0)
            costs, tours, uts = test(sample, model=deepaco, n_ants=15, iterations=6)
            lens += [costs[0]]
            times += [uts[0]]
        elif method == 'difusco' or method == 'difusco-2opt':
            if len(sample) == 20000:
                args.sparse_factor = 50
            parsed_data = parse_data(0, sample.cpu(), args.sparse_factor)
            st = time.time()
            gd_tours = inference(difusco, parsed_data, device)
            assert len(gd_tours[0].unique()) == len(sample)
            assert gd_tours[0].min() == 0 and gd_tours[0].max() == len(sample) - 1
            print('len w/o 2-opt: {:.2f}, time: {:.2f}'.format(
                    compute_tour_length(gd_tours[0], sample.cpu()), time.time() - st))
            # refine tour using the 2-opt heuristic
            if method == 'difusco-2opt':
                rf_tour, rf_time, it_num = cuda_2_opt(sample, gd_tours[0], iter_num=args.two_opt_iterations)
                assert len(rf_tour.unique()) == len(sample)
                assert rf_tour.min() == 0 and rf_tour.max() == len(sample) - 1
                lens += [compute_tour_length(rf_tour, sample)]
            else:
                lens += [compute_tour_length(gd_tours[0], sample.cpu())]
            times += [time.time() - st]
        else:
            assert method == 'lkh', print(f'unsupported method: {method}')
            results = lkh_solve(sample.cpu().numpy())
            lens += [results['length']]
            times += [results['time']]
        gaps += [(lens[-1] / opt - 1) * 100]

        # output results
        print('sample: {}, len: {:.2f}, gap: {:.2f}%, time: {:.2f}'.format(name, lens[-1], gaps[-1], times[-1]))
        if idx % sample_num == sample_num - 1:
            print('len: {:.2f}, gap: {:.2f}%, time: {:.2f}\n'.format(
                np.mean(lens[-sample_num:]), np.mean(gaps[-sample_num:]), np.mean(times[-sample_num:])))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    method = 'extnco-eff'  # 'extnco-bal'/ 'extnco-qlt'/ 'htsp'/ 'pomo'/ 'deepaco'/ 'difusco'/ 'difusco-2opt'/ 'lkh'
    stage = 'first'  # 'second'
    dataset = 'rei'  # 'tsplib' / 'vlsi'
    with torch.no_grad():
        main(method, stage, dataset, device='cuda:0', hide_bar=True)
