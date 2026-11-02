import awkward as ak
import numpy as np
import tensorflow as tf


def _unsorted_segment_max(x, seg):
    num = tf.cast(tf.reduce_max(seg) + 1, tf.int32)
    return tf.math.unsorted_segment_max(x, seg, num)


def _unsorted_segment_sum(x, seg, num=None):
    if num is None:
        num = tf.cast(tf.reduce_max(seg) + 1, tf.int32)
    return tf.math.unsorted_segment_sum(x, seg, num)


def _unsorted_segment_mean(x, seg, num=None):
    if num is None:
        num = tf.cast(tf.reduce_max(seg) + 1, tf.int32)
    s = tf.math.unsorted_segment_sum(x, seg, num)
    c = tf.math.unsorted_segment_sum(tf.ones_like(x[..., :1]), seg, num)
    return s / tf.maximum(c, 1.0)


class MLP(tf.keras.layers.Layer):
    def __init__(self, dims, act="silu", norm=None):
        super().__init__()
        self.blocks = []
        for i in range(len(dims) - 1):
            out_d = dims[i + 1]
            is_last = i == len(dims) - 2
            layers = [tf.keras.layers.Dense(out_d)]
            if not is_last:
                if norm == "batch_norm":
                    layers.append(tf.keras.layers.BatchNormalization())
                if act == "silu":
                    layers.append(tf.keras.layers.Activation(tf.nn.silu))
                elif act is not None:
                    layers.append(tf.keras.layers.ReLU())
            self.blocks.append(
                layers[0] if len(layers) == 1 else tf.keras.Sequential(layers)
            )

    def call(self, x, training=None):
        for block in self.blocks:
            if isinstance(block, tf.keras.Sequential):
                x = block(x, training=training)
            else:
                x = block(x)
        return x


class EGNNConv(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=128, update_coords=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.update_coords = update_coords
        self.phi_e = MLP([2 * hidden_dim + 1, hidden_dim, hidden_dim], act="silu", norm=None)
        self.phi_x = tf.keras.layers.Dense(1)
        self.phi_h = MLP([2 * hidden_dim, hidden_dim, hidden_dim], act="silu", norm=None)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, pos, features, edge_index, training=None):
        a = tf.cast(edge_index[0], tf.int32)
        b = tf.cast(edge_index[1], tf.int32)
        pos_i, pos_j = tf.gather(pos, a), tf.gather(pos, b)
        x_i, x_j = tf.gather(features, a), tf.gather(features, b)
        dij2 = tf.reduce_sum(tf.square(pos_i - pos_j), axis=-1, keepdims=True)
        m_ij = self.phi_e(tf.concat([x_i, x_j, dij2], axis=-1), training=training)
        num_nodes = tf.shape(pos)[0]
        m_i = _unsorted_segment_mean(m_ij, a, num_nodes)
        features = self.norm(
            features + self.phi_h(tf.concat([features, m_i], axis=-1), training=training),
            training=training,
        )
        if not self.update_coords:
            return features, pos
        coord = (pos_i - pos_j) * tf.tanh(self.phi_x(m_ij))
        dpos_i = _unsorted_segment_mean(coord, a, num_nodes)
        return features, pos + dpos_i


def _run_egnn(pos, features, edge_index, layers, coord_output, training=None):
    pos0 = pos
    for layer in layers:
        features, pos = layer(pos, features, edge_index, training=training)
    return pos - pos0 if coord_output else features


def _node_features(data, pos, resolution):
    if hasattr(data, f"{resolution}_features"):
        features = tf.cast(getattr(data, f"{resolution}_features"), tf.float32)
    else:
        features = tf.ones((tf.shape(pos)[0], 1), dtype=tf.float32)
    if getattr(data, "mutation_wt", None) is not None:
        return features
    idx = _mutation_positions(data, Xy=None)
    if idx is None:
        return features
    idx_t = tf.cast(idx, tf.int32)
    site = tf.gather(features, idx_t)
    n = tf.shape(features)[0]
    v2m = _vertex2molecule(data)
    n_graphs = tf.shape(site)[0]
    valid = tf.logical_and(
        tf.equal(tf.size(v2m), n),
        tf.equal(tf.reduce_max(v2m) + 1, n_graphs),
    )
    broadcast = tf.cond(
        valid,
        lambda: tf.gather(site, v2m),
        lambda: tf.tensor_scatter_nd_update(
            tf.zeros_like(features),
            tf.expand_dims(idx_t, 1),
            site,
        ),
    )
    marker = tf.scatter_nd(
        tf.expand_dims(idx_t, 1),
        tf.ones((tf.size(idx_t), 1), dtype=tf.float32),
        (n, 1),
    )
    return tf.concat([features, broadcast, marker], axis=-1)


def _grid_centers(num_voxels, radius):
    edges = tf.linspace(0.0, float(radius), num_voxels + 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    gx, gy, gz = tf.meshgrid(c, c, c, indexing="ij")
    return tf.reshape(tf.stack([gx, gy, gz], axis=-1), (-1, 3))


def _voxelize(pos, features, edge_index, frames, num_voxels, boundaries):
    n = tf.shape(features)[0]
    c = tf.shape(features)[1]
    nv = num_voxels
    center = nv // 2

    a = tf.cast(edge_index[0], tf.int32)
    b = tf.cast(edge_index[1], tf.int32)
    rel_pos = tf.einsum(
        "bi,bij->bj",
        tf.gather(pos, b) - tf.gather(pos, a),
        tf.gather(frames, a),
    )
    i = tf.raw_ops.Bucketize(input=rel_pos[:, 0], boundaries=boundaries)
    j = tf.raw_ops.Bucketize(input=rel_pos[:, 1], boundaries=boundaries)
    k = tf.raw_ops.Bucketize(input=rel_pos[:, 2], boundaries=boundaries)
    i = tf.clip_by_value(i, 0, nv - 1)
    j = tf.clip_by_value(j, 0, nv - 1)
    k = tf.clip_by_value(k, 0, nv - 1)
    flat_idx = (
        tf.cast(a, tf.int32) * nv**3
        + tf.cast(i, tf.int32) * nv**2
        + tf.cast(j, tf.int32) * nv
        + tf.cast(k, tf.int32)
    )
    grid = _unsorted_segment_sum(
        tf.gather(features, b),
        flat_idx,
        tf.cast(n * nv**3, tf.int32),
    )
    self_idx = tf.range(n, dtype=tf.int32) * nv**3 + (
        center * nv**2 + center * nv + center
    )
    grid = tf.tensor_scatter_nd_add(
        grid,
        tf.expand_dims(self_idx, 1),
        features,
    )
    return tf.reshape(grid, (n, nv**3 * c))


def _kpconv(pos, features, edge_index, frames, num_voxels, radius):
    n = tf.shape(features)[0]
    c = tf.shape(features)[1]
    nv = num_voxels
    k_size = nv**3
    kernel_pts = _grid_centers(nv, radius)
    sigma = radius / nv

    a = tf.cast(edge_index[0], tf.int32)
    b = tf.cast(edge_index[1], tf.int32)
    rel_pos = tf.einsum(
        "bi,bij->bj",
        tf.gather(pos, b) - tf.gather(pos, a),
        tf.gather(frames, a),
    )
    diff = tf.expand_dims(rel_pos, 1) - tf.expand_dims(kernel_pts, 0)
    dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + 1e-8)
    weight = tf.maximum(1.0 - dist / sigma, 0.0)

    k_idx = tf.range(k_size, dtype=tf.int32)
    idx = tf.reshape(tf.expand_dims(a, 1) * k_size + tf.expand_dims(k_idx, 0), (-1,))
    contrib = tf.reshape(
        tf.expand_dims(weight, -1) * tf.expand_dims(tf.gather(features, b), 1),
        (-1, c),
    )
    grid = _unsorted_segment_sum(contrib, idx, tf.cast(n * k_size, tf.int32))

    center = nv // 2
    center_idx = tf.range(n, dtype=tf.int32) * k_size + (
        center * nv**2 + center * nv + center
    )
    grid = tf.tensor_scatter_nd_add(
        grid,
        tf.expand_dims(center_idx, 1),
        features,
    )
    return tf.reshape(grid, (n, k_size * c))


def _egnn_layers(num_layers, hidden_dim, update_coords=True):
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    return [
        EGNNConv(hidden_dim, update_coords=update_coords) for _ in range(num_layers)
    ]


class TorchGraphModel:
    def __init__(
        self,
        in_channels,
        coord_output=False,
        hidden_dim=128,
        num_layers=3,
        update_coords=True,
    ):
        self.resolution = "atom"
        self.coord_output = coord_output
        self.hidden_dim = hidden_dim
        self.embed = tf.keras.layers.Dense(hidden_dim)
        self.layers_ = _egnn_layers(
            num_layers, hidden_dim, update_coords=update_coords
        )

    def forward_tensors(self, pos, features, edge_index, frames=None, training=None):
        x = self.embed(features, training=training)
        return _run_egnn(
            pos, x, edge_index, self.layers_, self.coord_output, training
        )

    def __call__(self, data, training=None):
        pos0 = tf.cast(getattr(data, f"{self.resolution}_pos"), tf.float32)
        edge_index = tf.cast(tf.transpose(data.molecule_edges), tf.int32)
        features = _node_features(data, pos0, self.resolution)
        return self.forward_tensors(pos0, features, edge_index, training=training)


class TorchVoxelModel:
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
        self.resolution = "atom"
        self.coord_output = coord_output
        self.hidden_dim = hidden_dim
        self.num_voxels = num_voxels
        self.radius = radius
        self.bucket_boundaries = np.linspace(
            0.0, float(radius), num_voxels
        ).astype(float).tolist()
        voxel_flat = num_voxels**3 * in_channels
        self.embed = MLP([voxel_flat, hidden_dim, hidden_dim], act="silu", norm=None)
        self.layers_ = _egnn_layers(
            num_layers, hidden_dim, update_coords=update_coords
        )

    def forward_tensors(self, pos, features, edge_index, frames, training=None):
        x = self.embed(
            _voxelize(
                pos, features, edge_index, frames, self.num_voxels, self.bucket_boundaries
            ),
            training=training,
        )
        return _run_egnn(pos, x, edge_index, self.layers_, self.coord_output, training)

    def __call__(self, data, training=None):
        pos0 = tf.cast(getattr(data, f"{self.resolution}_pos"), tf.float32)
        edge_index = tf.cast(tf.transpose(data.molecule_edges), tf.int32)
        frames = tf.cast(getattr(data, f"{self.resolution}_frame_R"), tf.float32)
        features = _node_features(data, pos0, self.resolution)
        return self.forward_tensors(pos0, features, edge_index, frames, training)


class TorchPointModel:
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
        self.resolution = "atom"
        self.coord_output = coord_output
        self.hidden_dim = hidden_dim
        self.num_voxels = num_voxels
        self.radius = radius
        kernel_flat = num_voxels**3 * in_channels
        self.embed = MLP([kernel_flat, hidden_dim, hidden_dim], act="silu", norm=None)
        self.layers_ = _egnn_layers(
            num_layers, hidden_dim, update_coords=update_coords
        )

    def forward_tensors(self, pos, features, edge_index, frames, training=None):
        x = self.embed(
            _kpconv(pos, features, edge_index, frames, self.num_voxels, self.radius),
            training=training,
        )
        return _run_egnn(pos, x, edge_index, self.layers_, self.coord_output, training)

    def __call__(self, data, training=None):
        pos0 = tf.cast(getattr(data, f"{self.resolution}_pos"), tf.float32)
        edge_index = tf.cast(tf.transpose(data.molecule_edges), tf.int32)
        frames = tf.cast(getattr(data, f"{self.resolution}_frame_R"), tf.float32)
        features = _node_features(data, pos0, self.resolution)
        return self.forward_tensors(pos0, features, edge_index, frames, training)


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
            return int(tf.shape(value)[0])
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


def _vertex2molecule(data, n_nodes=None):
    v2m = getattr(data, "vertex2molecule", None)
    if v2m is not None:
        return tf.reshape(tf.cast(v2m, tf.int32), (-1,))
    sizes = _graph_sizes(data, n_nodes)
    if sizes is None:
        raise ValueError("Cannot infer vertex-to-graph mapping")
    return tf.repeat(tf.range(len(sizes), dtype=tf.int32), repeats=sizes)


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
    gathered = tf.gather(output, idx)
    wt = _mutation_tokens(data, Xy, "wt")
    mut = _mutation_tokens(data, Xy, "mut")
    if wt is None or mut is None or gathered.shape.rank != 2 or gathered.shape[-1] == 1:
        return gathered
    wt_t = tf.constant(wt, dtype=tf.int32)
    mut_t = tf.constant(mut, dtype=tf.int32)
    return tf.gather(gathered, mut_t, batch_dims=1) - tf.gather(
        gathered, wt_t, batch_dims=1
    )


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
            in_channels, num_voxels=num_voxels, radius=radius, **trunk_kwargs
        )
    if repr == "graph":
        return TorchGraphModel(in_channels, **trunk_kwargs)
    if repr == "voxel":
        return TorchVoxelModel(
            in_channels, num_voxels=num_voxels, radius=radius, **trunk_kwargs
        )
    raise ValueError(f"Unknown repr: {repr}")


def _trunk_layers(trunk):
    if isinstance(trunk, (TorchPointModel, TorchVoxelModel, TorchGraphModel)):
        return [trunk.embed, *trunk.layers_]
    return []


class Model(tf.keras.Model):
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
        self.loss_name = loss
        self.coord_output = coord_output and repr in ("graph", "voxel", "point")
        # ProteinGym: 20-way site scores via LLR. Inverse folding also uses
        # out_channels=20 with cross-entropy, so only enable LLR for non-CE
        # losses. Do not use tf.cond — both branches are traced and the LLR
        # gather crashes on empty mutation indices.
        self.mutation_llr = (
            node_level and out_channels == 20 and self.loss_name != "cross_entropy"
        )
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
            self._track_trunk("trunk1", self.trunk1)
            self._track_trunk("trunk2", self.trunk2)
            head_in = hidden_dim * 2
        else:
            self.trunk = _make_trunk(repr, in_channels, **trunk_kwargs)
            self.trunk.resolution = resolution
            self.trunk1 = None
            self.trunk2 = None
            self._track_trunk("trunk", self.trunk)
            head_in = hidden_dim
        if not node_level:
            if pool == "sum":
                self.pool = _unsorted_segment_sum
            elif pool == "mean":
                self.pool = _unsorted_segment_mean
            else:
                raise ValueError(f"Unknown pool: {pool}")
        self.head = (
            None
            if self.coord_output
            else MLP([head_in, hidden_dim, out_channels], act="silu", norm=None)
        )
        self.out_channels = out_channels
        if loss == "cross_entropy":
            self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            )
        elif loss in ("mse", "mse_loss"):
            self.loss_fn = tf.keras.losses.MeanSquaredError()
        elif loss in ("l1_loss", "mae", "mean_absolute_error"):
            self.loss_fn = tf.keras.losses.MeanAbsoluteError()
        elif loss in ("bce_with_logits", "binary_cross_entropy_with_logits"):
            self.loss_fn = "bce_with_logits"
        else:
            self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, weight_decay=1e-6)
        self._clip_grad_norm = 0.0
        self._compiled_ready = False

    def _track_trunk(self, prefix, trunk):
        for idx, layer in enumerate(_trunk_layers(trunk)):
            setattr(self, f"{prefix}_layer_{idx}", layer)

    @property
    def checkpoint_layers(self):
        layers = [] if self.head is None else [self.head]
        for trunk in (self.trunk, self.trunk1, self.trunk2):
            if trunk is not None:
                layers.extend(_trunk_layers(trunk))
        return layers

    @property
    def checkpoint_variables(self):
        variables = []
        for layer in self.checkpoint_layers:
            variables.extend(layer.variables)
        return variables

    @property
    def trainable_variables(self):
        variables = []
        for layer in self.checkpoint_layers:
            variables.extend(layer.trainable_variables)
        return variables

    def _extract_single(self, data):
        pos = tf.cast(getattr(data, f"{self.resolution}_pos"), tf.float32)
        features = _node_features(data, pos, self.resolution)
        edge_index = tf.cast(tf.transpose(data.molecule_edges), tf.int32)
        if self.repr in ("voxel", "point"):
            frames = tf.cast(getattr(data, f"{self.resolution}_frame_R"), tf.float32)
        else:
            frames = tf.zeros((1, 3, 3), dtype=tf.float32)
        if self.node_level:
            v2m = tf.zeros((1,), dtype=tf.int32)
        else:
            v2m = _vertex2molecule(data)
        return pos, features, edge_index, frames, v2m

    def _mutation_index(self, data, Xy):
        idx = _mutation_positions(data, Xy)
        if idx is None:
            return tf.zeros((0,), dtype=tf.int32)
        return tf.constant(idx, dtype=tf.int32)

    def _mutation_aa(self, data, Xy, field):
        tok = _mutation_tokens(data, Xy, field)
        if tok is None:
            return tf.zeros((0,), dtype=tf.int32)
        return tf.constant(tok, dtype=tf.int32)

    def _encode_tensors(self, trunk, pos, features, edge_index, frames, v2m, training):
        x = trunk.forward_tensors(pos, features, edge_index, frames, training)
        if isinstance(x, (tuple, list)):
            x = x[0]
        if not self.node_level:
            x = self.pool(x, v2m)
        return x

    def _predict_from_tensors(
        self,
        pos,
        features,
        edge_index,
        frames,
        v2m,
        y,
        training,
        pos2=None,
        features2=None,
        edge_index2=None,
        frames2=None,
        v2m2=None,
    ):
        if self.pairwise:
            x = tf.concat(
                [
                    self._encode_tensors(
                        self.trunk1, pos, features, edge_index, frames, v2m, training
                    ),
                    self._encode_tensors(
                        self.trunk2,
                        pos2,
                        features2,
                        edge_index2,
                        frames2,
                        v2m2,
                        training,
                    ),
                ],
                axis=-1,
            )
        else:
            x = self._encode_tensors(
                self.trunk, pos, features, edge_index, frames, v2m, training
            )
        if self.coord_output:
            output = x
        else:
            output = self.head(x, training=training)
            if self.out_channels == 1:
                output = tf.reshape(output, [-1])
        if self.node_level and not self.pairwise and y.shape.rank == 3:
            output = tf.reshape(output, tf.shape(y))
        return output

    def _score_mutations(self, output, mut_idx, wt_aa, mut_aa):
        gathered = tf.gather(output, mut_idx)
        score = tf.gather(gathered, mut_aa, batch_dims=1) - tf.gather(
            gathered, wt_aa, batch_dims=1
        )
        return tf.reshape(score, [-1])

    def _loss_from_tensors(self, output, y, mut_idx, wt_aa, mut_aa):
        pred = output
        if self.mutation_llr:
            pred = self._score_mutations(output, mut_idx, wt_aa, mut_aa)
            y = tf.reshape(tf.cast(y, tf.float32), [-1])
            loss = tf.reduce_mean(tf.abs(pred - y))
            return loss, pred
        if self.loss_name == "cross_entropy":
            y = tf.reshape(tf.cast(y, tf.int32), [-1])
            loss = self.loss_fn(y, pred)
            return loss, pred
        if self.loss_fn == "bce_with_logits":
            y = tf.reshape(tf.cast(y, tf.float32), [-1])
            pred = tf.reshape(pred, [-1])
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=pred)
            )
            return loss, output
        if self.loss_name in ("l1_loss", "mae", "mean_absolute_error"):
            y = tf.reshape(tf.cast(y, tf.float32), [-1])
            pred = tf.reshape(output, [-1])
            loss = tf.reduce_mean(tf.abs(y - pred))
            return loss, output
        if self.loss_name == "cosine":
            pred = tf.math.l2_normalize(output, axis=-1)
            target = tf.math.l2_normalize(tf.cast(y, tf.float32), axis=-1)
            loss = 1.0 - tf.reduce_mean(tf.reduce_sum(pred * target, axis=-1))
            return loss, output
        y = tf.reshape(tf.cast(y, tf.float32), [-1])
        pred = tf.reshape(output, [-1])
        loss = tf.reduce_mean(tf.square(pred - y))
        return loss, output

    def _encode(self, trunk, data, training=None):
        pos, features, edge_index, frames, v2m = self._extract_single(data)
        return self._encode_tensors(
            trunk, pos, features, edge_index, frames, v2m, training
        )

    def _predict(self, data, training=None):
        if self.pairwise:
            output = self._predict_from_tensors(
                *self._extract_single(data.data1),
                data.y,
                training,
                *self._extract_single(data.data2),
            )
        else:
            output = self._predict_from_tensors(
                *self._extract_single(data), data.y, training
            )
        return output, data

    def _prepare_targets(self, output, y):
        if self.loss_name == "cross_entropy":
            return tf.reshape(tf.cast(y, tf.int32), [-1])
        y = tf.cast(y, tf.float32)
        if y.shape.rank is not None and y.shape.rank > 1 and y.shape[-1] == 1:
            y = tf.squeeze(y, axis=-1)
        elif (
            output.shape.rank == 2
            and output.shape[-1] is not None
            and output.shape[-1] > 1
            and y.shape.is_fully_defined()
            and output.shape.is_fully_defined()
            and int(np.prod(y.shape.as_list())) == int(np.prod(output.shape.as_list()))
            and y.shape != output.shape
        ):
            y = tf.reshape(y, tf.shape(output))
        return y

    @tf.function(reduce_retracing=True)
    def _compiled_eval_single(
        self, pos, features, edge_index, frames, v2m, y, mut_idx, wt_aa, mut_aa, training
    ):
        output = self._predict_from_tensors(
            pos, features, edge_index, frames, v2m, y, training
        )
        return self._loss_from_tensors(output, y, mut_idx, wt_aa, mut_aa)

    @tf.function(reduce_retracing=True)
    def _compiled_eval_pair(
        self,
        pos,
        features,
        edge_index,
        frames,
        v2m,
        pos2,
        features2,
        edge_index2,
        frames2,
        v2m2,
        y,
        mut_idx,
        wt_aa,
        mut_aa,
        training,
    ):
        output = self._predict_from_tensors(
            pos,
            features,
            edge_index,
            frames,
            v2m,
            y,
            training,
            pos2,
            features2,
            edge_index2,
            frames2,
            v2m2,
        )
        return self._loss_from_tensors(output, y, mut_idx, wt_aa, mut_aa)

    @tf.function(reduce_retracing=True)
    def _compiled_train_single(
        self, pos, features, edge_index, frames, v2m, y, mut_idx, wt_aa, mut_aa
    ):
        with tf.GradientTape() as tape:
            output = self._predict_from_tensors(
                pos, features, edge_index, frames, v2m, y, True
            )
            loss, output = self._loss_from_tensors(output, y, mut_idx, wt_aa, mut_aa)
        variables = self.trainable_variables
        grads = tape.gradient(loss, variables)
        if self._clip_grad_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, self._clip_grad_norm)
        self.optimizer.apply_gradients(zip(grads, variables))
        return loss, output

    @tf.function(reduce_retracing=True)
    def _compiled_train_pair(
        self,
        pos,
        features,
        edge_index,
        frames,
        v2m,
        pos2,
        features2,
        edge_index2,
        frames2,
        v2m2,
        y,
        mut_idx,
        wt_aa,
        mut_aa,
    ):
        with tf.GradientTape() as tape:
            output = self._predict_from_tensors(
                pos,
                features,
                edge_index,
                frames,
                v2m,
                y,
                True,
                pos2,
                features2,
                edge_index2,
                frames2,
                v2m2,
            )
            loss, output = self._loss_from_tensors(output, y, mut_idx, wt_aa, mut_aa)
        variables = self.trainable_variables
        grads = tape.gradient(loss, variables)
        if self._clip_grad_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, self._clip_grad_norm)
        self.optimizer.apply_gradients(zip(grads, variables))
        return loss, output

    def _ensure_compiled(self, data, training):
        if self._compiled_ready:
            return
        self._predict(data, training=training)
        self._compiled_ready = True

    def _run_compiled(self, Xy, data, training, apply_gradients):
        self._ensure_compiled(data, training)
        mut_idx = self._mutation_index(data, Xy)
        wt_aa = self._mutation_aa(data, Xy, "wt")
        mut_aa = self._mutation_aa(data, Xy, "mut")
        if self.pairwise:
            args = (
                *self._extract_single(data.data1),
                *self._extract_single(data.data2),
                data.y,
                mut_idx,
                wt_aa,
                mut_aa,
            )
            if apply_gradients:
                return self._compiled_train_pair(*args)
            return self._compiled_eval_pair(*args, training)
        args = (*self._extract_single(data), data.y, mut_idx, wt_aa, mut_aa)
        if apply_gradients:
            return self._compiled_train_single(*args)
        return self._compiled_eval_single(*args, training)

    def train_step(self, Xy, data):
        return self._run_compiled(Xy, data, training=True, apply_gradients=False)

    def eval_step(self, Xy, data):
        return self._run_compiled(Xy, data, training=False, apply_gradients=False)

    def device_train_step(self, Xy, data, clip_grad_norm=None):
        if clip_grad_norm is not None:
            self._clip_grad_norm = float(clip_grad_norm)
        return self._run_compiled(Xy, data, training=True, apply_gradients=True)
