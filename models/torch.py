import awkward as ak
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_scatter import scatter

from .gvp import GvpModel
from .mpnn import MpnnModel
from .pifold import PifoldModel


class EGNNConv(gnn.MessagePassing):
    """EGNN layer with residual features, LayerNorm, and mean aggregation.

    ``flow="target_to_source"`` matches Bioverse kNN edges ``(center, neighbor)``
    and the TensorFlow implementation, which both aggregate onto edge_index[0].
    """

    def __init__(self, hidden_dim=128, update_coords=True):
        super().__init__(aggr="mean", node_dim=0, flow="target_to_source")
        self.hidden_dim = hidden_dim
        self.update_coords = update_coords
        self.phi_e = gnn.MLP(
            [2 * hidden_dim + 1, hidden_dim, hidden_dim], act="silu", norm=None
        )
        self.phi_x = nn.Linear(hidden_dim, 1)
        self.phi_h = gnn.MLP(
            [2 * hidden_dim, hidden_dim, hidden_dim], act="silu", norm=None
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, pos, features, edge_index):
        out = self.propagate(edge_index, x=features, pos=pos)
        features = self.norm(features + self.phi_h(torch.cat([features, out["m"]], dim=-1)))
        if self.update_coords:
            pos = pos + out["coord"]
        return features, pos

    def message(self, x_i, x_j, pos_i, pos_j):
        dij2 = (pos_i - pos_j).pow(2).sum(dim=-1, keepdim=True)
        m_ij = self.phi_e(torch.cat([x_i, x_j, dij2], dim=-1))
        coord = (pos_i - pos_j) * torch.tanh(self.phi_x(m_ij))
        return {"m": m_ij, "coord": coord}

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        m = scatter(inputs["m"], index, dim=0, dim_size=dim_size, reduce="mean")
        coord = scatter(inputs["coord"], index, dim=0, dim_size=dim_size, reduce="mean")
        return {"m": m, "coord": coord}


def _make_egnn_layers(num_layers, hidden_dim, update_coords=True):
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    return [EGNNConv(hidden_dim, update_coords=update_coords) for _ in range(num_layers)]


def _run_egnn(pos, features, edge_index, layers, coord_output):
    pos0 = pos
    for layer in layers:
        features, pos = layer(pos, features, edge_index)
    return pos - pos0 if coord_output else features


def _node_features(data, pos, resolution):
    if hasattr(data, f"{resolution}_features"):
        features = getattr(data, f"{resolution}_features").float()
    else:
        features = torch.ones(pos.shape[0], 1, device=pos.device, dtype=pos.dtype)
    if getattr(data, "mutation_wt", None) is not None:
        return features
    idx = _mutation_positions(data, Xy=None)
    if idx is None:
        return features
    idx_t = torch.as_tensor(idx, device=features.device, dtype=torch.long)
    site = features.index_select(0, idx_t)
    n = features.shape[0]
    v2m = _vertex2molecule(data, n, device=features.device)
    if v2m.numel() == n and int(v2m.max()) + 1 == site.shape[0]:
        broadcast = site[v2m]
    else:
        broadcast = features.new_zeros(n, features.shape[1])
        broadcast[idx_t] = site
    marker = features.new_zeros(n, 1)
    marker[idx_t] = 1
    return torch.cat([features, broadcast, marker], dim=-1)


def _grid_centers(num_voxels, radius, device, dtype):
    edges = torch.linspace(0, radius, num_voxels + 1, device=device, dtype=dtype)
    c = 0.5 * (edges[:-1] + edges[1:])
    gx, gy, gz = torch.meshgrid(c, c, c, indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)


def _voxelize(pos, features, edge_index, frames, num_voxels, radius, buckets):
    n, c = features.shape
    nv = num_voxels
    center = nv // 2
    grid = features.new_zeros(n * nv**3, c)

    a, b = edge_index
    rel_pos = torch.einsum("bi,bij->bj", pos[b] - pos[a], frames[a])
    i, j, k = torch.bucketize(rel_pos, buckets.to(pos.device)).T
    i = i.clamp(0, nv - 1)
    j = j.clamp(0, nv - 1)
    k = k.clamp(0, nv - 1)
    flat_idx = a * nv**3 + i * nv**2 + j * nv + k
    grid = scatter(features[b], flat_idx, dim=0, dim_size=n * nv**3, reduce="add")

    self_idx = torch.arange(n, device=pos.device) * nv**3 + (
        center * nv**2 + center * nv + center
    )
    grid[self_idx] = grid[self_idx] + features
    return grid.view(n, nv**3 * c)


def _kpconv(pos, features, edge_index, frames, num_voxels, radius):
    n, c = features.shape
    nv = num_voxels
    k_size = nv**3
    kernel_pts = _grid_centers(nv, radius, pos.device, pos.dtype)
    sigma = radius / nv

    a, b = edge_index
    rel_pos = torch.einsum("bi,bij->bj", pos[b] - pos[a], frames[a])
    dist = ((rel_pos.unsqueeze(1) - kernel_pts.unsqueeze(0)).pow(2).sum(dim=-1) + 1e-8).sqrt()
    weight = (1 - dist / sigma).clamp(min=0)

    idx = (
        a.unsqueeze(1) * k_size + torch.arange(k_size, device=pos.device).unsqueeze(0)
    ).reshape(-1)
    contrib = (weight.unsqueeze(-1) * features[b].unsqueeze(1)).reshape(-1, c)
    grid = scatter(contrib, idx, dim=0, dim_size=n * k_size, reduce="add")

    center = nv // 2
    center_idx = torch.arange(n, device=pos.device) * k_size + (
        center * nv**2 + center * nv + center
    )
    grid[center_idx] = grid[center_idx] + features
    return grid.view(n, k_size * c)


class TorchGraphModel(nn.Module):
    def __init__(self, in_channels, coord_output=False, hidden_dim=128, num_layers=3, update_coords=True):
        super().__init__()
        self.coord_output = coord_output
        self.hidden_dim = hidden_dim
        self.embed = nn.Linear(in_channels, hidden_dim)
        self.layers = nn.ModuleList(
            _make_egnn_layers(num_layers, hidden_dim, update_coords=update_coords)
        )

    def forward(self, data):
        pos0 = getattr(data, f"{self.resolution}_pos").float()
        edge_index = data.molecule_edges.T.long()
        features = self.embed(_node_features(data, pos0, self.resolution))
        return _run_egnn(pos0, features, edge_index, self.layers, self.coord_output)


class TorchVoxelModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_voxels=5,
        radius=10.0,
        coord_output=False,
        hidden_dim=128,
        num_layers=3,
        update_coords=True,
    ):
        super().__init__()
        self.coord_output = coord_output
        self.hidden_dim = hidden_dim
        self.num_voxels = num_voxels
        self.radius = radius
        self.register_buffer("buckets", torch.linspace(0, radius, num_voxels))
        voxel_flat = num_voxels**3 * in_channels
        self.embed = gnn.MLP(
            [voxel_flat, hidden_dim, hidden_dim], act="silu", norm=None
        )
        self.layers = nn.ModuleList(
            _make_egnn_layers(num_layers, hidden_dim, update_coords=update_coords)
        )

    def forward(self, data):
        pos0 = getattr(data, f"{self.resolution}_pos").float()
        edge_index = data.molecule_edges.T.long()
        frames = getattr(data, f"{self.resolution}_frame_R").float()
        features = _node_features(data, pos0, self.resolution)
        x = self.embed(
            _voxelize(
                pos0,
                features,
                edge_index,
                frames,
                self.num_voxels,
                self.radius,
                self.buckets,
            )
        )
        return _run_egnn(pos0, x, edge_index, self.layers, self.coord_output)


class TorchPointModel(nn.Module):
    """KPConv-style neighbor aggregation in the local reference frame, then EGNN."""

    def __init__(
        self,
        in_channels,
        num_voxels=5,
        radius=10.0,
        coord_output=False,
        hidden_dim=128,
        num_layers=3,
        update_coords=True,
    ):
        super().__init__()
        self.coord_output = coord_output
        self.hidden_dim = hidden_dim
        self.num_voxels = num_voxels
        self.radius = radius
        kernel_flat = num_voxels**3 * in_channels
        self.embed = gnn.MLP(
            [kernel_flat, hidden_dim, hidden_dim], act="silu", norm=None
        )
        self.layers = nn.ModuleList(
            _make_egnn_layers(num_layers, hidden_dim, update_coords=update_coords)
        )

    def forward(self, data):
        pos0 = getattr(data, f"{self.resolution}_pos").float()
        edge_index = data.molecule_edges.T.long()
        frames = getattr(data, f"{self.resolution}_frame_R").float()
        features = _node_features(data, pos0, self.resolution)
        x = self.embed(
            _kpconv(pos0, features, edge_index, frames, self.num_voxels, self.radius)
        )
        return _run_egnn(pos0, x, edge_index, self.layers, self.coord_output)


def _as_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy") and not hasattr(value, "fields"):
        try:
            return np.asarray(value.numpy())
        except Exception:
            pass
    try:
        return np.asarray(ak.to_numpy(value))
    except Exception:
        return np.asarray(value)


def _n_nodes(data):
    for key in ("residue_pos", "atom_pos"):
        value = getattr(data, key, None)
        if value is not None:
            return int(value.shape[0])
    return None


def _graph_sizes(data, n_nodes=None):
    n_nodes = n_nodes if n_nodes is not None else _n_nodes(data)
    num_vertices = getattr(data, "num_vertices", None)
    if num_vertices is not None and n_nodes is not None:
        sizes = np.asarray(_as_numpy(num_vertices), dtype=np.int64).reshape(-1)
        if sizes.size and int(sizes.sum()) == int(n_nodes):
            return sizes
    v2m = getattr(data, "vertex2molecule", None)
    if v2m is not None and n_nodes is not None:
        v2m = np.asarray(_as_numpy(v2m)).reshape(-1)
        if v2m.size == n_nodes:
            return np.bincount(v2m.astype(np.int64))
    n_graphs = int(getattr(data, "num_molecules", 0) or 0)
    if n_nodes and n_graphs and n_nodes % n_graphs == 0:
        return np.full(n_graphs, n_nodes // n_graphs, dtype=np.int64)
    if n_nodes:
        return np.array([n_nodes], dtype=np.int64)
    return None


def _vertex2molecule(data, n_nodes, device=None):
    v2m = getattr(data, "vertex2molecule", None)
    if v2m is not None:
        tensor = v2m if torch.is_tensor(v2m) else torch.as_tensor(v2m)
        if tensor.numel() == n_nodes:
            return tensor.long().to(device or tensor.device)
    sizes = _graph_sizes(data, n_nodes)
    if sizes is None:
        raise ValueError("Cannot infer vertex-to-graph mapping")
    return torch.arange(len(sizes), device=device, dtype=torch.long).repeat_interleave(
        torch.as_tensor(sizes, device=device, dtype=torch.long)
    )


def _mutation_positions(data, Xy):
    if getattr(data, "mutation_pos", None) is not None:
        mutation_pos = _as_numpy(data.mutation_pos).reshape(-1)
    elif (
        Xy is not None
        and Xy[1] is not None
        and hasattr(Xy[1], "fields")
        and "position" in ak.fields(Xy[1])
    ):
        mutation_pos = _as_numpy(Xy[1]["position"]).reshape(-1)
    else:
        return None
    sizes = _graph_sizes(data)
    if sizes is not None and len(sizes) == len(mutation_pos):
        mutation_pos = mutation_pos + np.insert(np.cumsum(sizes), 0, 0)[:-1]
    return mutation_pos


def _mutation_tokens(data, Xy, field):
    value = getattr(data, f"mutation_{field}", None)
    if value is None and Xy is not None and Xy[1] is not None and hasattr(Xy[1], "fields"):
        if field in ak.fields(Xy[1]):
            value = Xy[1][field]
    if value is None:
        return None
    return np.asarray(_as_numpy(value), dtype=np.int64).reshape(-1)


def _gather_mutation_predictions(output, data, Xy):
    idx = _mutation_positions(data, Xy)
    if idx is None:
        return output
    wt = _mutation_tokens(data, Xy, "wt")
    mut = _mutation_tokens(data, Xy, "mut")
    gathered = output[torch.as_tensor(idx, device=output.device, dtype=torch.long)]
    if wt is None or mut is None or gathered.ndim != 2 or gathered.shape[-1] <= 1:
        return gathered.squeeze(-1) if gathered.ndim > 1 and gathered.shape[-1] == 1 else gathered
    wt_t = torch.as_tensor(wt, device=output.device, dtype=torch.long)
    mut_t = torch.as_tensor(mut, device=output.device, dtype=torch.long)
    return gathered.gather(1, mut_t.view(-1, 1)).squeeze(1) - gathered.gather(
        1, wt_t.view(-1, 1)
    ).squeeze(1)


def _make_trunk(
    repr,
    in_channels,
    num_voxels=5,
    radius=10.0,
    coord_output=False,
    hidden_dim=128,
    num_layers=3,
    update_coords=True,
):
    trunk_kwargs = dict(
        coord_output=coord_output,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        update_coords=update_coords,
    )
    if repr == "point":
        return TorchPointModel(
            in_channels,
            num_voxels=num_voxels,
            radius=radius,
            **trunk_kwargs,
        )
    if repr == "graph":
        return TorchGraphModel(in_channels, **trunk_kwargs)
    if repr == "voxel":
        return TorchVoxelModel(
            in_channels,
            num_voxels=num_voxels,
            radius=radius,
            **trunk_kwargs,
        )
    if repr == "pifold":
        return PifoldModel(in_channels)
    if repr == "mpnn":
        return MpnnModel(in_channels)
    if repr == "gvp":
        return GvpModel(in_channels)
    raise ValueError(f"Unknown repr: {repr}")


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
        pairwise=False,
        num_voxels=5,
        radius=10.0,
        coord_output=False,
        hidden_dim=128,
        num_layers=4,
        pool="mean",
        update_coords=True,
    ):
        super().__init__()
        self.node_level = node_level
        self.resolution = resolution
        self.pairwise = pairwise
        self.repr = repr
        self.coord_output = coord_output and repr in ("graph", "voxel", "point")
        trunk_kwargs = dict(
            num_voxels=num_voxels,
            radius=radius,
            coord_output=self.coord_output,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            update_coords=update_coords,
        )
        if pairwise:
            self.trunk1 = _make_trunk(repr, in_channels, **trunk_kwargs)
            self.trunk2 = _make_trunk(repr, in_channels, **trunk_kwargs)
            self.trunk1.resolution = resolution
            self.trunk2.resolution = resolution
            self.trunk = None
            head_in = hidden_dim * 2
        else:
            self.trunk = _make_trunk(repr, in_channels, **trunk_kwargs)
            self.trunk.resolution = resolution
            self.trunk1 = None
            self.trunk2 = None
            head_in = hidden_dim
        if not node_level:
            if pool == "sum":
                self.pool = gnn.global_add_pool
            elif pool == "mean":
                self.pool = gnn.global_mean_pool
            else:
                raise ValueError(f"Unknown pool: {pool}")
        self.head = (
            None
            if self.coord_output
            else gnn.MLP([head_in, hidden_dim, out_channels], act="silu", norm=None)
        )
        self.loss = getattr(F, loss)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=1e-6
        )

    def _encode(self, trunk, data):
        x = trunk(data)
        if isinstance(x, tuple):
            x = x[0]
        if not self.node_level:
            x = self.pool(x, _vertex2molecule(data, x.shape[0], device=x.device))
        return x

    def _predict(self, data):
        if self.pairwise:
            x = torch.cat(
                [
                    self._encode(self.trunk1, data.data1),
                    self._encode(self.trunk2, data.data2),
                ],
                dim=-1,
            )
        else:
            x = self._encode(self.trunk, data)
        if self.coord_output:
            output = x
        else:
            output = self.head(x)
        if output.shape[1] == 1:
            output = output.squeeze(1)
        if not self.pairwise and self.node_level and len(data.y.shape) == 3:
            output = output.reshape(*data.y.shape)
        return output, data

    def train_step(self, Xy, data):
        output, data = self._predict(data)
        output = _gather_mutation_predictions(output, data, Xy)
        y = data.y.long() if self.loss == F.cross_entropy else data.y.float()
        if self.loss == F.cross_entropy and y.ndim > 1:
            y = y.reshape(-1)
        elif y.ndim > 1 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        elif (
            output.ndim == 2
            and output.shape[-1] > 1
            and y.numel() == output.numel()
            and y.shape != output.shape
        ):
            y = y.reshape(output.shape)
        loss = self.loss(output, y)
        return loss, output

    def eval_step(self, Xy, data):
        return self.train_step(Xy, data)
