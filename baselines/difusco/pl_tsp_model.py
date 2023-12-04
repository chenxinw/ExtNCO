"""Lightning module for training the DIFUSCO TSP model."""

import torch
import torch.utils.data

from baselines.difusco.pl_meta_model import COMetaModel


class TSPModel(COMetaModel):
    def __init__(self,param_args=None):
        super(TSPModel, self).__init__(param_args=param_args, node_feature_only=False)

    def forward(self, x, adj, t, edge_index):
        return self.model(x, t, adj, edge_index)

    def categorical_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            x0_pred = self.forward(points.float().to(device), xt.float().to(device), t.float().to(device),
                                   edge_index.long().to(device) if edge_index is not None else None)

            if not self.sparse:
                x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
            else:
                x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)

            xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
            return xt

    def gaussian_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            pred = self.forward(points.float().to(device), xt.float().to(device), t.float().to(device),
                                edge_index.long().to(device) if edge_index is not None else None)
            pred = pred.squeeze(1)
            xt = self.gaussian_posterior(target_t, t, pred, xt)
        return xt
