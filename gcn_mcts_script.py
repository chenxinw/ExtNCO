import os
import torch
from myutils import load_data, load_instance, compute_tour_length
from baselines.gcn_mcts.gcn.net import load_model, build_map


# transform our dataset into format of GCN-MCTS
def transform_data(sample, outFile):
    scale = len(sample)
    fp = open(outFile, 'w')
    outStr = ''
    for xi, yi in sample:
        outStr += '{} {} '.format(xi, yi)
    outStr += 'output '
    for i in range(1, 1 + scale):
        outStr += '{} '.format(i)
    outStr += '1'
    print(outStr, file=fp)
    fp.close()


# check length of the solution
def check_tour_length(data, tFile):
    fp = open(tFile, 'r')
    for idx, tLine in enumerate(fp.readlines()):
        tour = []
        for node in tLine.split(' '):
            tour += [int(node) - 1]
        tour = torch.tensor(tour).long()
        length = compute_tour_length(tour, data[idx])
        print('sample: {}   length: {:.6f}'.format(idx, length))
    fp.close()


if __name__ == '__main__':
    dataset = 'tsplib'  # 'tsplib'/ 'vlsi'
    device = 'cuda:0'
    if not os.path.exists('baselines/gcn_mcts/data/'):
        os.mkdir('baselines/gcn_mcts/data/')
        os.mkdir('baselines/gcn_mcts/data/rei/')
        os.mkdir('baselines/gcn_mcts/data/tsplib/')
        os.mkdir('baselines/gcn_mcts/data/vlsi/')
    if not os.path.exists('baselines/gcn_mcts/results/'):
        os.mkdir('baselines/gcn_mcts/results/')
        os.mkdir('baselines/gcn_mcts/results/rei/')
        os.mkdir('baselines/gcn_mcts/results/tsplib/')
        os.mkdir('baselines/gcn_mcts/results/vlsi/')
    if not os.path.exists('baselines/gcn_mcts/heatmap/'):
        os.mkdir('baselines/gcn_mcts/heatmap/')
        os.mkdir('baselines/gcn_mcts/heatmap/rei/')
        os.mkdir('baselines/gcn_mcts/heatmap/tsplib/')
        os.mkdir('baselines/gcn_mcts/heatmap/vlsi/')

    # transform data format
    instances = []
    if dataset == 'rei':
        for ins in ['10K', '20K', '50K']:
            samples = load_data(f'Dataset/{dataset}/small/{ins}_data')
            scale = samples.size(1)
            for idx , sample in enumerate(samples):
                instances += [f'{scale}_{idx}']
                transform_data(sample, f'baselines/gcn_mcts/data/rei/{scale}_{idx}.txt')
                print(f'transform {instances[-1]} done')
    else:
        assert dataset == 'tsplib' or dataset == 'vlsi', print(f'unsupported dataset: {dataset}')
        if dataset == 'tsplib':
            instances = ['dsj1000', 'pr1002', 'u1060', 'vm1084', 'pcb1173', 'd1291', 'rl1304', 'rl1323',
                         'nrw1379', 'u1432', 'd1655', 'vm1748', 'u1817', 'rl1889', 'd2103', 'u2152',
                         'u2319', 'pr2392', 'pcb3038', 'fnl4461', 'rl5915', 'rl5934', 'pla7397',
                         'rl11849', 'usa13509', 'brd14051', 'd15112', 'd18512', 'pla33810', 'pla85900']
        else:
            instances = ['xmc10150', 'xvb13584', 'xia16928', 'pjh17845', 'frh19289', 'fnc19402', 'ido21215',
                         'fma21553', 'lsb22777', 'xrh24104', 'bbz25234', 'irx28268', 'fyg28534', 'icx28698',
                         'boa28924', 'ird29514', 'pbh30440', 'xib32892', 'fry33203', 'bby34656', 'pba38478',
                         'ics39603', 'rbz43748', 'fht47608', 'fna52057', 'bna56769', 'dan59296']
        samples = [load_instance(f'Dataset/{dataset}/{ins}.tsp', scale=True) for ins in instances]
        for idx, ins in enumerate(instances):
            transform_data(samples[idx], f'baselines/gcn_mcts/data/{dataset}/{ins}.txt')
            print(f'transform {ins} done')

    # apply GCN to generate heatmap
    model = load_model('tsp50', [0])
    with torch.no_grad():
        for sample, ins in zip(samples, instances):
            build_map(model, dataset, len(sample), ins, batch_size=64, K=50, K_expand=149, device=device)
