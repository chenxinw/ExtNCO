import time
import torch
from baselines.pomo.Model import POMO_Model


def load_pomo(scale, device):
    if scale == 'TSP-100':
        model_file = 'baselines/pomo/pomo_n100.pt'
    elif scale == 'TSP-50':
        model_file = 'baselines/pomo/pomo_n50.pt'
    else:
        print('unsupported model!')
        model_file = None
    checkpoint = torch.load(model_file, map_location=device)
    model = POMO_Model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def pomo_solve(coords, model, pomo_size, enable_aug, device='cuda:0'):  # coords: [gsz,2]
    pomo_time = time.time()
    coords = coords.float()
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
        enable_batch = False
    else:
        assert coords.dim() == 3
        enable_batch = True
    if enable_aug:
        coords = augment_xy_data_by_8_fold(coords)

    tours, lengths = model(coords, pomo_size, enable_aug, device)  # POMO Rollout
    tours = tours.squeeze(0).cpu() if not enable_batch else tours.cpu()
    lengths = lengths.item() if not enable_batch else lengths.tolist()
    pomo_time = time.time() - pomo_time

    model.encoded_nodes = None
    model.decoder.k = None
    model.decoder.v = None
    model.decoder.single_head_key = None
    model.decoder.q_first = None

    return tours, lengths, pomo_time


def augment_xy_data_by_8_fold(problems):  # [bsz,gsz,2]
    x = problems[:, :, [0]]  # [bsz,gsz,1]
    y = problems[:, :, [1]]  # [bsz,gsz,1]

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    return torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)  # [8*bsz,gsz,2]
