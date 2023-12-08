import torch
from tsp import Region
from divide import Divide
from solve import batch_pomo_solve
from myutils import load_data, save_data, load_instance, compute_tour_length
from baselines.pomo.pomo import load_pomo


def main(device='cuda:0', hide_bar=True):
    pomo = load_pomo('TSP-100', device)

    # load dataset and baseline
    # instances = ['10K', '20K', '50K']  # REI-Small
    # instances = ['{}K'.format(scale) for scale in range(100, 1001, 100)]  # REI-Large
    # for ins in instances:
    #     dataset = load_data('Dataset/rei/small/{}_data'.format(ins))
    #     baseline = load_data('Dataset/rei/small/{}_base'.format(ins))
    #     dataset = load_data('Dataset/rei/large/{}_data'.format(ins))
    #     baseline = load_data('Dataset/rei/large/new_{}_base'.format(ins))
    #     for idx, data in enumerate(dataset):
    #         opt = baseline['hk'][idx]  # REI dataset
    #         data = data.to(device)

    # TSPLib / VLSI
    # instances = ['dsj1000', 'pr1002', 'u1060', 'vm1084', 'pcb1173', 'd1291', 'rl1304',
    #              'rl1323', 'nrw1379', 'fl1400', 'u1432', 'fl1577', 'd1655', 'vm1748',
    #              'u1817', 'rl1889', 'd2103', 'u2152', 'u2319', 'pr2392', 'pcb3038',
    #              'fl3795', 'fnl4461', 'rl5915', 'rl5934', 'pla7397', 'rl11849',
    #              'usa13509', 'brd14051', 'd15112', 'd18512', 'pla33810', 'pla85900']
    # baseline = load_data('Dataset/tsplib/baseline')
    instances = ['xmc10150', 'xvb13584', 'xia16928', 'pjh17845', 'frh19289', 'fnc19402', 'ido21215',
                 'fma21553', 'lsb22777', 'xrh24104', 'bbz25234', 'irx28268', 'fyg28534', 'icx28698',
                 'boa28924', 'ird29514', 'pbh30440', 'xib32892', 'fry33203', 'bby34656', 'pba38478',
                 'ics39603', 'rbz43748', 'fht47608', 'fna52057', 'bna56769', 'dan59296']
    baseline = load_data('Dataset/vlsi/baseline')
    for idx, ins in enumerate(instances):
        opt = baseline[ins]['opt']
        # data = load_instance('Dataset/tsplib/{}.tsp'.format(ins), scale=True).to(device)
        data = load_instance('Dataset/vlsi/{}.tsp'.format(ins), scale=True).to(device)

        # ExtNCO
        DAC4ML = Region(node_index=torch.arange(len(data)))
        Divide(region=DAC4ML, coords=data, type='hybrid', rep=True, device=device, hide_bar=hide_bar,
               params={'grid': 1000, 'sub': 100, 'rep_ratio': 0.005, 'min_thr': 50, 'nb_num': 8})
        batch_pomo_solve(DAC4ML, data, model=pomo, hide_bar=hide_bar)
        DAC4ML.merge(data, {'comb': 'mst', 'var': 'eff', 'ord': ['bts']}, device, hide_bar)  # ExtNCO-Eff
        # DAC4ML.merge(data, {'comb': 'mst', 'var': 'bal', 'ord': ['btp']}, device, hide_bar)  # ExtNCO-Bal
        # DAC4ML.merge(data, {'comb': 'mst', 'var': 'qlt', 'ord': ['btp']}, device, hide_bar)  # ExtNCO-Qlt
        assert DAC4ML.check_valid()
        length = compute_tour_length(DAC4ML.tour, data)
        gap = (length / opt - 1) * 100
        print('len: {:.2f}, gap: {:.2f}%, time:{:.2f} sec'.format(length, gap, DAC4ML.total_time))
        print('divide: {:.2f} sec, solve: {:.2f} sec, evaluate: {:.2f} sec, execute: {:.2f} sec'.format(
                DAC4ML.divide_time, DAC4ML.solve_time, DAC4ML.eval_time, DAC4ML.trav_time))



if __name__ == '__main__':
    main(device='cuda:0', hide_bar=True)
