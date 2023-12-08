import time
import torch
from tqdm import tqdm
from tsp import Region
from myutils import find_sub_regions, find_neighbor_regions, remove_tiny_regions, check_nbs_connected


def Divide(region, coords, type, rep, params, device, hide_bar):
    region.divide_time = time.time()
    # cluster initialization
    if type == 'hybrid':
        HybridDivide(region, coords, params['grid'], params['sub'])
    elif type == 'kmeans':
        KmeansDivide(region, coords, params['sub'])
    elif type == 'grids':
        GridsDivide(region, coords, params['sub'])
    elif type == 'drori':
        from baselines.droridiv import DroriDivide
        xmin = coords[:, 0].min().item()
        xmax = coords[:, 0].max().item()
        ymin = coords[:, 1].min().item()
        ymax = coords[:, 1].max().item()
        region.area = {'xs': xmin, 'ys': ymin, 'xl': xmax - xmin, 'yl': ymax - ymin}
        DroriDivide(region, coords, params['sub'])
        region.sub_regs = find_sub_regions(region, name=True)
    else:
        print('unknown divide type!')

    # node repartition
    coords = coords.to(device)
    regions = region.sub_regs
    if rep:
        find_neighbor_regions(regions, coords, 'dist', params['nb_num'], device=device)
        regions = batch_repartition(regions, coords, params['rep_ratio'], device=device, hide_bar=hide_bar)
    find_neighbor_regions(regions, coords, 'dist', params['nb_num'], device=device)
    # remove tiny regions
    regions, updated = remove_tiny_regions(regions, coords, min_thr=params['min_thr'])
    # build neighbor relationship
    for nb_factor in range(1, 10):
        if nb_factor == 1 and updated:
            find_neighbor_regions(regions, coords, 'dist', params['nb_num'] * nb_factor, device=device)
        if nb_factor > 1:
            find_neighbor_regions(regions, coords, 'dist', params['nb_num'] * nb_factor, device=device)
        if check_nbs_connected(regions):
            break
    region.sub_regs = regions
    region.divide_time = time.time() - region.divide_time
    region.total_time += region.divide_time


def HybridDivide(region, coords, scale_grid, scale_sub):
    GridsDivide(region, coords, scale_grid)
    regions = region.sub_regs

    sub_regs = []
    while len(regions) != 0:
        regs_to_div = []
        for reg in regions:
            if reg.node_num >= 1.5 * scale_sub:
                regs_to_div += [reg]
            else:
                sub_regs += [reg]
        if len(regs_to_div) == 0:
            break

        regions = []
        for reg in regs_to_div:
            nodes = reg.node_index
            k_num = round(len(nodes) / scale_sub)
            sub_indices = k_means(coords[nodes], k_num, 0.01)
            for indices in sub_indices:
                regions += [Region(nodes[indices])]
    for reg_idx, reg in enumerate(sub_regs):
        reg.region_index = reg_idx
    region.sub_regs = sub_regs


def KmeansDivide(region, coords, scale_sub):
    k_num = round(len(coords) / scale_sub)
    sub_indices = k_means(coords, k_num, 0.01)
    for reg_idx, indices in enumerate(sub_indices):
        region.sub_regs += [Region(node_index=indices, region_index=reg_idx)]


# prev version
# def GridsDivide(region, coords, thr):
#     node_num = region.node_num
#     grid_num = round((node_num / thr) ** 0.5)
#     grid_len = 1 / grid_num
#     gis = coords.div(grid_len, rounding_mode='floor').long()
#     indices = torch.argwhere(gis == grid_num).squeeze(1)
#     for xi, yi in indices:
#         gis[xi, yi] -= 1
#     for xi in range(grid_num):
#         x_indices = torch.argwhere(gis[:, 0] == xi).squeeze(1)
#         yis = gis[x_indices, 1]
#         for yi in range(grid_num):
#             y_indices = torch.argwhere(yis == yi).squeeze(1)
#             if len(y_indices) == 0:
#                 continue
#             region.sub_regs += [Region(node_index=x_indices[y_indices].cpu(), region_index=len(region.sub_regs))]


def GridsDivide(region, coords, thr):
    node_num = region.node_num
    grid_num = round((node_num / thr) ** 0.5)
    x_coords = coords[:, 0]
    ver_width = max(x_coords).item() - min(x_coords).item()
    ver_len = ver_width / grid_num
    xis = x_coords.div(ver_len, rounding_mode='floor').long()
    indices = torch.argwhere(xis == grid_num).squeeze(1)
    xis[indices] -= 1
    y_coords = coords[:, 1]
    hor_width = max(y_coords).item() - min(y_coords).item()
    hor_len = hor_width / grid_num
    yis = y_coords.div(hor_len, rounding_mode='floor').long()
    indices = torch.argwhere(yis == grid_num).squeeze(1)
    yis[indices] -= 1

    for xi in range(grid_num):
        x_indices = torch.argwhere(xis == xi).squeeze(1)
        y_slice = yis[x_indices]
        for yi in range(grid_num):
            y_indices = torch.argwhere(y_slice == yi).squeeze(1)
            if len(y_indices) == 0:
                continue
            region.sub_regs += [Region(node_index=x_indices[y_indices].cpu(), region_index=len(region.sub_regs))]


def k_means(coords, k, ratio=0.01):
    node_num = len(coords)
    cis = torch.randperm(node_num)[:k]
    centroids = coords[cis]
    cluster = k * torch.ones(node_num).long().to(coords.device)

    cur_ratio = 1
    while cur_ratio > ratio:
        dists = ((coords.unsqueeze(1) - centroids) ** 2).sum(dim=2)
        new_cluster = dists.argmin(dim=1)
        node_changed_num = (new_cluster - cluster).bool().long().sum().item()
        cur_ratio = node_changed_num / node_num
        for idx in range(k):  # update cluster info
            indices = torch.argwhere(new_cluster == idx).squeeze(1)
            if len(indices) == 0:
                centroids[idx] = float('inf')
                continue
            centroids[idx] = coords[indices].mean(dim=0)
        cluster = new_cluster

    sub_indices = []  # collect clusters
    for idx in range(k):
        indices = torch.argwhere(cluster == idx).squeeze(1)
        if len(indices) != 0:
            sub_indices += [indices.cpu()]
    return sub_indices


def batch_repartition(regions, coords, rep_ratio, device, hide_bar):
    reg_num = len(regions)
    nb_num = len(regions[0].nb_regs) + 1
    nb_mat = []
    cluster = torch.zeros(len(coords)).long().to(device)
    centroids = torch.tensor([]).to(device)
    for reg_idx, reg in enumerate(regions):
        nb_mat += [reg.nb_regs + [reg_idx]]
        cluster[reg.node_index] = reg_idx
        centroids = torch.cat((centroids, reg.centroid.unsqueeze(0)), dim=0)
    nb_mat = torch.tensor(nb_mat).long().to(device)

    pbar = tqdm(desc='[node repartition]', unit='round', disable=hide_bar)
    ratio = 1
    round_num = 0
    while ratio > rep_ratio:
        round_num += 1
        reg_nums = torch.tensor([reg.node_num for reg in regions]).long().to(device)
        assert reg_nums.sum() == len(coords)
        reg_vals, reg_indices = torch.sort(reg_nums)  # sort regions by their scales
        scales, counts = torch.unique_consecutive(reg_vals, return_counts=True)
        counts = counts.tolist()

        cur_idx = 0
        node_changed_num = 0
        for scale_idx, scale in enumerate(scales):  # traverse regions
            if scale == 0:  # skip regions with no node
                cur_idx += counts[scale_idx]
                continue
            batch_coords = torch.tensor([]).to(device)
            batch_centroids = torch.tensor([]).to(device)
            for step_idx in range(counts[scale_idx]):
                reg_idx = reg_indices[cur_idx + step_idx]
                batch_coords = torch.cat((batch_coords, coords[regions[reg_idx].node_index].view(1, -1, 1, 2)), dim=0)
                batch_centroids = torch.cat((batch_centroids, centroids[nb_mat[reg_idx]].view(1, 1, -1, 2)), dim=0)
            batch_node_cent_dist = ((batch_coords - batch_centroids) ** 2).sum(dim=-1)  # [bsz,rsz,nb+1]
            min_indices = batch_node_cent_dist.argmin(dim=-1)  # [bsz,rsz]
            for step_idx in range(counts[scale_idx]):
                reg_idx = reg_indices[cur_idx + step_idx]
                node_changed_num += (min_indices[step_idx] - nb_num + 1).bool().long().sum().item()
                cluster[regions[reg_idx].node_index] = nb_mat[reg_idx][min_indices[step_idx]]
            cur_idx += counts[scale_idx]
        ratio = node_changed_num / len(coords)
        # print('node_changed_num: {}, node_changed_ratio: {:.2f}'.format(node_changed_num, ratio))

        # update regions according to cluster info
        vals, indices = cluster.sort()
        indices = indices.cpu()
        reg_indices, reg_scales = torch.unique_consecutive(vals, return_counts=True)
        reg_indices = reg_indices.tolist()
        reg_scales = reg_scales.tolist()
        cur_reg_idx = 0
        cur_node_idx = 0
        for reg_idx in range(reg_num):
            cur_reg = regions[reg_idx]
            if reg_indices[cur_reg_idx] != reg_idx:  # some region may be empty after repartition
                cur_reg.node_index = torch.tensor([])
                cur_reg.node_num = 0
                centroids[reg_idx] = float('inf')
                continue
            cur_reg.node_index = indices[cur_node_idx:cur_node_idx + reg_scales[cur_reg_idx]]
            cur_reg.node_num = reg_scales[cur_reg_idx]
            centroids[reg_idx] = coords[cur_reg.node_index].mean(dim=0)
            cur_node_idx += reg_scales[cur_reg_idx]
            cur_reg_idx += 1
        pbar.set_postfix(num=node_changed_num, ratio=ratio)
        pbar.update(1)
    # print('node repartition round: {}'.format(round_num))
    pbar.close()

    # remove empty subregions
    new_regions = []
    for reg_idx, reg in enumerate(regions):
        if reg.node_num == 0:
            continue
        reg.region_index = len(new_regions)
        new_regions += [reg]
    return new_regions
