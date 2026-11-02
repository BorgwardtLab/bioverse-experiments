import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum


def _node_scalars(data, n_features):
    if hasattr(data, "residue_features"):
        return data.residue_features.float()
    n_nodes = data.vertex2molecule.shape[0]
    return torch.zeros(
        n_nodes, n_features, device=data.vertex2molecule.device, dtype=torch.float
    )


class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4):
        super(NeighborAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden

        self.W_V = nn.Sequential(
            nn.Linear(num_in, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, num_hidden),
        )
        self.Bias = nn.Sequential(
            nn.Linear(num_hidden * 3, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_heads),
        )
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def forward(self, h_V, h_E, center_id):
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)

        w = self.Bias(torch.cat([h_V[center_id], h_E], dim=-1)).view(-1, n_heads, 1)
        attend_logits = w / np.sqrt(d)

        V = self.W_V(h_E).view(-1, n_heads, d)
        attend = scatter_softmax(attend_logits, index=center_id, dim=0)
        h_V = scatter_sum(attend * V, center_id, dim=0).view([-1, self.num_hidden])
        return self.W_O(h_V)


class EdgeMLP(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1):
        super(EdgeMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, h_E, edge_idx):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]
        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E


class Context(nn.Module):
    def __init__(self, num_hidden):
        super(Context, self).__init__()
        self.V_MLP_g = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.Sigmoid(),
        )

    def forward(self, h_V, batch_id):
        c_V = scatter_mean(h_V, batch_id, dim=0)
        h_V = h_V * self.V_MLP_g(c_V[batch_id])
        return h_V


class GeneralGNN(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, update_edges=True):
        super(GeneralGNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(2)])
        self.attention = NeighborAttention(num_hidden, num_in, num_heads=4)
        self.edge_update = EdgeMLP(num_hidden, num_in) if update_edges else None
        self.context = Context(num_hidden)
        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden * 4),
            nn.ReLU(),
            nn.Linear(num_hidden * 4, num_hidden),
        )

    def forward(self, h_V, h_E, edge_idx, batch_id):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]
        dh = self.attention(h_V, torch.cat([h_E, h_V[dst_idx]], dim=-1), src_idx)
        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))
        if self.edge_update is not None:
            h_E = self.edge_update(h_V, h_E, edge_idx)
        h_V = self.context(h_V, batch_id)
        return h_V, h_E


class StructureEncoder(nn.Module):
    def __init__(self, hidden_dim, num_encoder_layers=3, dropout=0.1):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [
                GeneralGNN(
                    hidden_dim,
                    hidden_dim * 2,
                    dropout=dropout,
                    update_edges=i < num_encoder_layers - 1,
                )
                for i in range(num_encoder_layers)
            ]
        )

    def forward(self, h_V, h_E, edge_idx, batch_id):
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, edge_idx, batch_id)
        return h_V, h_E


class PifoldModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        node_dim, edge_dim = in_channels
        hidden_dim = 128
        self.node_dim = node_dim

        self.node_embedding = nn.Linear(node_dim, hidden_dim, bias=True)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim, bias=True)
        self.norm_nodes = nn.BatchNorm1d(hidden_dim)
        self.norm_edges = nn.BatchNorm1d(hidden_dim)

        self.W_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.encoder = StructureEncoder(hidden_dim, num_encoder_layers=3, dropout=0.1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        h_V = _node_scalars(data, self.node_dim)
        h_E = data.molecule_edge_features.float()
        edge_index = data.molecule_edges.T.long()
        batch_id = data.vertex2molecule

        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_E = self.W_e(self.norm_edges(self.edge_embedding(h_E)))
        h_V, h_E = self.encoder(h_V, h_E, edge_index, batch_id)
        return h_V
