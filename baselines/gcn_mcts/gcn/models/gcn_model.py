import torch
import torch.nn as nn
from baselines.gcn_mcts.gcn.models.gcn_layers import ResidualGatedGCNLayer, MLP
from baselines.gcn_mcts.gcn.utils.model_utils import *


class ResidualGatedGCNModel(nn.Module):

    def __init__(self, config, dtypeFloat, dtypeLong):
        super(ResidualGatedGCNModel, self).__init__()
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong

        # Define net parameters
        self.num_nodes = config.num_nodes
        self.node_dim = config.node_dim
        self.voc_nodes_in = config['voc_nodes_in']
        self.voc_nodes_out = config['num_nodes']  # config['voc_nodes_out']
        self.voc_edges_in = config['voc_edges_in']
        self.voc_edges_out = config['voc_edges_out']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.mlp_layers = config['mlp_layers']
        self.aggregation = config['aggregation']

        # Node and edge embedding layers/lookups
        self.nodes_coord_embedding = nn.Linear(self.node_dim, self.hidden_dim, bias=False)
        self.edges_values_embedding = nn.Linear(1, self.hidden_dim // 2, bias=False)
        self.edges_embedding = nn.Embedding(self.voc_edges_in, self.hidden_dim // 2)

        # Define GCN Layers
        gcn_layers = []
        for layer in range(self.num_layers):
            gcn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)

        # Define MLP classifiers
        self.mlp_edges = MLP(self.hidden_dim, self.voc_edges_out, self.mlp_layers)
        # self.mlp_nodes = MLP(self.hidden_dim, self.voc_nodes_out, self.mlp_layers)

    def forward(self, x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges,
                edge_cw, num_neg = 4, loss_type = "CE", gamma = 1):
        # Node and edge embedding
        x = self.nodes_coord_embedding(x_nodes_coord)  # B x V x H
        # e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(3))  # B x V x V x H
        # e_tags = self.edges_embedding(x_edges)  # B x V x V x H
        # e = torch.cat((e_vals, e_tags), dim=3)
        e = torch.cat((self.edges_values_embedding(x_edges_values.unsqueeze(3)),
                       self.edges_embedding(x_edges)), dim=3)
        
        # permute kaibin QIU
        x = x.permute(0, 2, 1)  # B x H x V
        e = e.permute(0, 3, 1, 2)  # B x H x V x V
        
        # GCN layers
        for layer in range(self.num_layers):
            x, e = self.gcn_layers[layer](x, e)  # B x V x H, B x V x V x H
        # MLP classifier
        y_pred_edges = self.mlp_edges(e)  # B x V x V x voc_edges_out
        # y_pred_nodes = self.mlp_nodes(x)  # B x V x voc_nodes_out
        
        # Compute loss
        edge_cw = torch.Tensor(edge_cw).type(self.dtypeFloat)  # Convert to tensors
        loss = loss_edges(y_pred_edges, y_edges, edge_cw, loss_type=loss_type, gamma=gamma)

        return y_pred_edges
