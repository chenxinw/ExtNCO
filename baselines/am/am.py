import time
import math
import torch
import numpy as np
from torch.utils.data import DataLoader
# from baselines.am.am_utils import load_model

import json
import torch.nn.functional as F
from baselines.am.problems import TSP


def load_am(scale, decode_type='greedy', device='cuda:0'):
    model, _ = load_model(scale, device)
    model.set_decode_type(decode_type, temp=1)
    return model


def am_solve(model, dataset, batch_size, decode_type='greedy', device='cuda:0'):
    if dataset.dtype != torch.float32:
        dataset = dataset.float()
    if len(dataset.shape) != 3:
        assert len(dataset.shape) == 2
        dataset = dataset.unsqueeze(0)

    costs, tours, durations = _eval_dataset(model, dataset, batch_size, decode_type, 0, device)

    return costs, tours, durations


def load_model(scale, device='cuda:0'):
    from baselines.am.attention_model import AttentionModel

    assert scale == 'tsp-50' or scale == 'tsp-100', print('model not exists!')
    args = load_args('baselines/am/{}.json'.format(scale))
    model = AttentionModel(
            args['embedding_dim'],
            args['hidden_dim'],
            TSP,
            n_encode_layers=args['n_encode_layers'],
            mask_inner=True,
            mask_logits=True,
            normalization=args['normalization'],
            tanh_clipping=args['tanh_clipping'],
            checkpoint_encoder=args.get('checkpoint_encoder', False),
            shrink_size=args.get('shrink_size', None)
    )
    # Overwrite model parameters by parameters to load
    load_data = torch.load('baselines/am/{}.pt'.format(scale), map_location=device)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})
    model.eval()  # Put in eval mode
    model = model.to(device)

    return model, args


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]

    return args


def do_batch_rep(v, n):
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


def sample_many(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1):
    input = do_batch_rep(input, batch_rep)

    costs = []
    pis = []
    for i in range(iter_rep):
        _log_p, pi = inner_func(input)
        # pi.view(-1, batch_rep, pi.size(-1))
        cost, mask = get_cost_func(input, pi)

        costs.append(cost.view(batch_rep, -1).t())
        pis.append(pi.view(batch_rep, -1, pi.size(-1)).transpose(0, 1))

    max_length = max(pi.size(-1) for pi in pis)
    # (batch_size * batch_rep, iter_rep, max_length) => (batch_size, batch_rep * iter_rep, max_length)
    pis = torch.cat([F.pad(pi, (0, max_length - pi.size(-1))) for pi in pis], 1)  # .view(embeddings.size(0), batch_rep * iter_rep, max_length)
    costs = torch.cat(costs, 1)

    # (batch_size)
    mincosts, argmincosts = costs.min(-1)
    # (batch_size, minlength)
    minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]

    return minpis, mincosts


def compute_in_batches(f, calc_batch_size, *args, n=None):
    """
    Computes memory heavy function f(*args) in batches
    :param n: the total number of elements, optional if it cannot be determined as args[0].size(0)
    :param f: The function that is computed, should take only tensors as arguments and return tensor or tuple of tensors
    :param calc_batch_size: The batch size to use when computing this function
    :param args: Tensor arguments with equally sized first batch dimension
    :return: f(*args), this should be one or multiple tensors with equally sized first batch dimension
    """
    if n is None:
        n = args[0].size(0)
    n_batches = (n + calc_batch_size - 1) // calc_batch_size  # ceil
    if n_batches == 1:
        return f(*args)

    # Run all batches
    # all_res = [f(*batch_args) for batch_args in zip(*[torch.chunk(arg, n_batches) for arg in args])]
    # We do not use torch.chunk such that it also works for other classes that support slicing
    all_res = [f(*(arg[i * calc_batch_size:(i + 1) * calc_batch_size] for arg in args)) for i in range(n_batches)]

    # Allow for functions that return None
    def safe_cat(chunks, dim=0):
        if chunks[0] is None:
            assert all(chunk is None for chunk in chunks)
            return None
        return torch.cat(chunks, dim)

    # Depending on whether the function returned a tuple we need to concatenate each element or only the result
    if isinstance(all_res[0], tuple):
        return tuple(safe_cat(res_chunks, 0) for res_chunks in zip(*all_res))

    return safe_cat(all_res, 0)


def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]


def _eval_dataset(model, dataset, batch_size, decode_type, width, device):
    dataloader = DataLoader(dataset, batch_size=batch_size)

    lengths, tours, times = [], [], []
    for batch in dataloader:
        batch = batch.to(device)

        start = time.time()
        with torch.no_grad():
            assert decode_type in ('sample', 'greedy')
            if decode_type == 'greedy':
                assert width == 0, "Do not set width when using greedy"
                assert batch_size <= 10000, "eval_batch_size should be smaller than calc batch size"
                batch_rep = 1
                iter_rep = 1
            elif width * batch_size > 10000:
                assert batch_size == 1
                assert width % 10000 == 0
                batch_rep = 10000
                iter_rep = width // 10000
            else:
                batch_rep = width
                iter_rep = 1
            assert batch_rep > 0
            # This returns (batch_size, iter_rep shape)
            sequences, costs = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
            ids = torch.arange(len(costs), dtype=torch.int64, device=costs.device)

        assert sequences is not None
        assert ids is not None
        sequences, costs = get_best(sequences.cpu().numpy(), costs.cpu().numpy(), ids.cpu().numpy(), len(costs))
        duration = time.time() - start
        for seq, cost in zip(sequences, costs):
            lengths += [cost]
            tours += [seq.tolist()]
            times += [duration]
            # seq = seq.tolist()  # No need to trim as all are same length
            # results.append((cost, seq, duration))

    assert len(lengths) == len(dataset)
    lengths = np.array(lengths)
    tours = torch.tensor(tours).long()
    times = np.array(times)
    return lengths, tours, times