import awkward as ak
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_scatter import scatter

from .gvp import GvpModel
from .mpnn import MpnnModel
from .pifold import PifoldModel


class TorchPointModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.tnet1_mlp = gnn.MLP([3, 128, 128])
        self.tnet1_head = gnn.MLP([128, 128, 3 * 3])
        self.tnet2_mlp = gnn.MLP([128, 128, 128])
        self.tnet2_head = gnn.MLP([128, 128, 128 * 128])
        self.mlp1 = gnn.MLP([in_channels + 3, 128, 128])
        self.mlp2 = gnn.MLP([128, 128, 128])
        self.pool = gnn.global_max_pool

    def forward(self, data):
        batch_id = data.vertex2molecule.long()
        pos = getattr(data, f"{self.resolution}_pos").float()
        if hasattr(data, f"{self.resolution}_features"):
            features = getattr(data, f"{self.resolution}_features").float()
        else:
            features = torch.ones(pos.shape[0], 1).to(pos.device)
        t1 = self.tnet1_head(self.pool(self.tnet1_mlp(pos), batch_id))
        pos = torch.einsum("bi,bij->bj", pos, t1.reshape(-1, 3, 3)[batch_id])
        x = torch.cat([pos, features], dim=1)
        x = self.mlp1(x)
        t2 = self.tnet2_head(self.pool(self.tnet2_mlp(x), batch_id))
        x = torch.einsum("bi,bij->bj", x, t2.reshape(-1, 128, 128)[batch_id])
        x = self.mlp2(x)
        return x


class TorchGraphModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.k = 8

        class EGNNConv(gnn.MessagePassing):
            def __init__(self, i_ch):
                super().__init__(aggr="add", node_dim=0)
                self.phi_e = gnn.MLP([2 * i_ch + 1, 128, 128], act="silu", norm=None)
                self.phi_x = nn.Linear(128, 1)
                self.phi_h = gnn.MLP([i_ch + 128, 128, 128], act="silu", norm=None)

            def forward(self, pos, features, edge_index):
                out = self.propagate(edge_index, x=features, pos=pos)
                m_i, dpos_i = out["m"], out["coord"]
                features = self.phi_h(torch.cat([features, m_i], dim=-1))
                pos = pos + dpos_i
                return pos, features

            def message(self, x_i, x_j, pos_i, pos_j):
                dij2 = (pos_i - pos_j).pow(2).sum(dim=-1, keepdim=True)
                msg_in = torch.cat([x_i, x_j, dij2], dim=-1)
                m_ij = self.phi_e(msg_in)  # [E, H]
                g_ij = self.phi_x(m_ij)  # [E, 1]
                coord = (pos_i - pos_j) * g_ij  # [E, 3]
                return {"m": m_ij, "coord": coord}

            def aggregate(self, inputs, index, ptr=None, dim_size=None):
                coord, m = inputs["coord"], inputs["m"]
                m = scatter(m, index, dim=0, dim_size=dim_size, reduce="add")
                coord = scatter(coord, index, dim=0, dim_size=dim_size, reduce="add")
                return {"m": m, "coord": coord}

        self.layers = nn.ModuleList(
            [EGNNConv(in_channels), EGNNConv(128), EGNNConv(128)]
        )

    def forward(self, data):
        batch_id = data.vertex2molecule.long()
        pos = getattr(data, f"{self.resolution}_pos").float()
        # edge_index = gnn.knn_graph(pos, self.k, batch_id)
        edge_index = data.molecule_edges.T.long()
        if hasattr(data, f"{self.resolution}_features"):
            features = getattr(data, f"{self.resolution}_features").float()
        else:
            features = torch.ones(pos.shape[0], 1).to(pos.device)
        for layer in self.layers:
            pos, features = layer(pos, features, edge_index)
        return features


class TorchVoxelModel(nn.Module):
    def __init__(self, in_channels, num_voxels=5, radius=10.0):
        super().__init__()
        self.k = 8

        class ConvLayer(nn.Module):
            def __init__(self, in_channels):
                super().__init__()
                self.buckets = torch.linspace(0, radius, num_voxels)
                self.kernel = nn.Parameter(
                    torch.randn(num_voxels, num_voxels, num_voxels, in_channels, 128)
                )
                self.aggr = gnn.global_add_pool

            def forward(self, pos, features, edges, R, vertex2molecule):
                a, b = edges
                rel_pos = torch.einsum("bi,bij->bj", pos[b] - pos[a], R[a])
                i, j, k = torch.bucketize(rel_pos, self.buckets.to(pos.device)).T
                features = torch.einsum("bi,bij->bj", features[b], self.kernel[i, j, k])
                features = self.aggr(features, a)
                return pos, features

        self.layers = nn.ModuleList(
            [ConvLayer(in_channels), ConvLayer(128), ConvLayer(128)]
        )

    def forward(self, data):
        batch_id = data.vertex2molecule.long()
        pos = getattr(data, f"{self.resolution}_pos").float()
        # edge_index = gnn.knn_graph(pos, self.k, batch_id)
        edge_index = data.molecule_edges.T.long()
        frames = getattr(data, f"{self.resolution}_frame_R").float()
        if hasattr(data, f"{self.resolution}_features"):
            features = getattr(data, f"{self.resolution}_features").float()
        else:
            features = torch.ones(pos.shape[0], 1).to(pos.device)
        for layer in self.layers:
            pos, features = layer(pos, features, edge_index, frames, batch_id)
        return features


class Model(nn.Module):
    def __init__(
        self,
        repr="point",
        node_level=True,
        in_channels=1,
        out_channels=10,
        loss="cross_entropy",
        learning_rate=0.001,
        resolution="atom",
    ):
        super().__init__()
        self.node_level = node_level
        self.resolution = resolution
        if repr == "point":
            self.trunk = TorchPointModel(in_channels)
        elif repr == "graph":
            self.trunk = TorchGraphModel(in_channels)
        elif repr == "voxel":
            self.trunk = TorchVoxelModel(in_channels)
        elif repr == "pifold":
            self.trunk = PifoldModel(in_channels)
        elif repr == "mpnn":
            self.trunk = MpnnModel(in_channels)
        elif repr == "gvp":
            self.trunk = GvpModel(in_channels)
        self.trunk.resolution = resolution
        if not node_level:
            self.pool = gnn.global_mean_pool
        self.head = nn.Linear(128, out_channels)
        self.loss = getattr(F, loss)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def train_step(self, Xy, data):
        x = self.trunk(data)
        if not self.node_level:
            x = self.pool(x, data.vertex2molecule)
        output = self.head(x)
        if output.shape[1] == 1:
            output = output.squeeze(1)
        if len(data.y.shape) == 3:
            output = output.reshape(*data.y.shape)
        if "molecule_mutation_pos" in vars(data):
            mutation_pos = ak.to_torch(Xy[1]["position"])
            output = output[mutation_pos[:, 0]]
        y = data.y.long() if self.loss == F.cross_entropy else data.y.float()
        loss = self.loss(output, y)
        return loss, output

    def eval_step(self, Xy, data):
        return self.train_step(Xy, data)
