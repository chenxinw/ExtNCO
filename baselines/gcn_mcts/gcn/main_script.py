import time
import torch
import numpy as np
from myutils import load_data, compute_tour_length


# transform our dataset into format of GCN-MCTS
def transform_data(dataset, outFile, sample_num):
    assert len(dataset) >= sample_num
    scale = len(dataset[0])
    fp = open(outFile, 'a+')
    for idx in range(sample_num):
        outStr = ''
        for xi, yi in dataset[idx]:
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


# REI data
# dataset = load_data('Dataset/rei/large/50K_data')
# transform_data(dataset, 'baselines/gcn_mcts/data/rei/TSP-50000.txt', 20)

# TSPLIB data
instances = ['dsj1000', 'pr1002', 'u1060', 'vm1084', 'pcb1173', 'd1291', 'rl1304', 'rl1323',
             'nrw1379', 'u1432', 'd1655', 'vm1748', 'u1817', 'rl1889', 'd2103', 'u2152',
             'u2319', 'pr2392', 'pcb3038', 'fnl4461', 'rl5915', 'rl5934', 'pla7397']
dataset = [load_instance('Dataset/tsplib/{}.tsp'.format(ins), scale=True) for ins in instances]
for idx, ins in enumerate(instances):
    data = dataset[idx].unsqueeze(0)
    transform_data(data, 'baselines/gcn_mcts/data/tsplib/{}.txt'.format(ins), 1)

# VLSI dataset
# instances = ['xmc10150', 'xvb13584', 'xia16928', 'pjh17845', 'frh19289', 'fnc19402', 'ido21215',
#              'fma21553', 'lsb22777', 'xrh24104', 'bbz25234', 'irx28268', 'fyg28534', 'icx28698',
#              'boa28924', 'ird29514', 'pbh30440', 'xib32892', 'fry33203', 'bby34656', 'pba38478',
#              'ics39603', 'rbz43748', 'fht47608', 'fna52057', 'bna56769', 'dan59296']
# dataset = [load_instance('Dataset/vlsi/{}.tsp'.format(ins), scale=True) for ins in instances]
# for idx, ins in enumerate(instances):
#     data = dataset[idx].unsqueeze(0)
#     transform_data(data, 'baselines/gcn_mcts/data/vlsi/{}.txt'.format(ins), 1)


# apply GCN to generate heatmap
from baselines.gcn_mcts.gcn.net import load_model, build_map

model = load_model('tsp50', [0])
with torch.no_grad():

    # build_map(model, 10000, batch_size=64, K=50, K_expand=149)  # parameter for TSP-10K
    # build_map(model, 20000, batch_size=64, K=50, K_expand=149)  # parameter for TSP-20K
    # build_map(model, 50000, batch_size=64, K=50, K_expand=149, device=device)

    for idx, ins in enumerate(instances):
        build_map(model, len(dataset[idx]), batch_size=64, K=50, K_expand=149, type='tsplib', ins=ins)
        # build_map(model, int(ins[-5:]), batch_size=64, K=50, K_expand=149, type='vlsi', ins=ins)
