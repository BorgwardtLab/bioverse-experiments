import tensorflow as tf


def _unsorted_segment_max(x, seg):
    num = tf.cast(tf.reduce_max(seg) + 1, tf.int32)
    return tf.math.unsorted_segment_max(x, seg, num)


def _unsorted_segment_sum(x, seg, num=None):
    if num is None:
        num = tf.cast(tf.reduce_max(seg) + 1, tf.int32)
    return tf.math.unsorted_segment_sum(x, seg, num)


def _unsorted_segment_mean(x, seg):
    num = tf.cast(tf.reduce_max(seg) + 1, tf.int32)
    s = tf.math.unsorted_segment_sum(x, seg, num)
    c = tf.math.unsorted_segment_sum(tf.ones_like(x[..., :1]), seg, num)
    return s / tf.maximum(c, 1.0)


class _MLP(tf.keras.layers.Layer):
    def __init__(self, dims, act="relu"):
        super().__init__()
        self.layers_ = []
        for i, d in enumerate(dims[1:]):
            self.layers_.append(tf.keras.layers.Dense(d))
        self.act = (
            None
            if act is None
            else (tf.nn.silu if act == "silu" else tf.keras.activations.get(act))
        )

    def call(self, x):
        for i, dense in enumerate(self.layers_):
            x = dense(x)
            if i < len(self.layers_) - 1 and self.act is not None:
                x = self.act(x)
        return x


class TorchPointModel(tf.keras.layers.Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.tnet1_mlp = _MLP([3, 128, 128])
        self.tnet1_head = _MLP([128, 128, 3 * 3])
        self.tnet2_mlp = _MLP([128, 128, 128])
        self.tnet2_head = _MLP([128, 128, 128 * 128])
        self.mlp1 = _MLP([in_channels + 3, 128, 128])
        self.mlp2 = _MLP([128, 128, 128])

    def call(self, pos, features, edges, R, vertex2molecule, training=None, **kwargs):
        vertex2molecule = tf.cast(vertex2molecule, tf.int32)
        t1 = self.tnet1_head(
            _unsorted_segment_max(self.tnet1_mlp(pos), vertex2molecule)
        )
        t1 = tf.reshape(t1, (-1, 3, 3))
        t1n = tf.gather(t1, vertex2molecule)
        pos = tf.einsum("bi,bij->bj", pos, t1n)
        x = tf.concat([pos, features], axis=1)
        x = self.mlp1(x)
        t2 = self.tnet2_head(_unsorted_segment_max(self.tnet2_mlp(x), vertex2molecule))
        t2 = tf.reshape(t2, (-1, 128, 128))
        t2n = tf.gather(t2, vertex2molecule)
        x = tf.einsum("bi,bij->bj", x, t2n)
        x = self.mlp2(x)
        return pos, x


class TorchGraphModel(tf.keras.layers.Layer):
    def __init__(self, in_channels):
        super().__init__()

        class EGNNConv(tf.keras.layers.Layer):
            def __init__(self, i_ch):
                super().__init__()
                self.phi_e = _MLP([2 * i_ch + 1, 128, 128], act="silu")
                self.phi_x = tf.keras.layers.Dense(1)
                self.phi_h = _MLP([i_ch + 128, 128, 128], act="silu")

            def call(self, pos, features, edges, training=None, **kwargs):
                a, b = tf.unstack(tf.cast(edges, tf.int32), axis=0)
                pos_i, pos_j = tf.gather(pos, a), tf.gather(pos, b)
                x_i, x_j = tf.gather(features, a), tf.gather(features, b)
                dij2 = tf.reduce_sum(tf.square(pos_i - pos_j), axis=-1, keepdims=True)
                m_ij = self.phi_e(tf.concat([x_i, x_j, dij2], axis=-1))
                g_ij = self.phi_x(m_ij)
                coord = (pos_i - pos_j) * g_ij
                num_nodes = tf.shape(pos)[0]
                m_i = _unsorted_segment_sum(m_ij, a, num_nodes)
                dpos_i = _unsorted_segment_sum(coord, a, num_nodes)
                features = self.phi_h(tf.concat([features, m_i], axis=-1))
                pos = pos + dpos_i
                return pos, features

        self.layers_ = [EGNNConv(in_channels), EGNNConv(128), EGNNConv(128)]

    def call(self, pos, features, edges, R, vertex2molecule, training=None, **kwargs):
        for layer in self.layers_:
            pos, features = layer(pos, features, edges)
        return pos, features


class TorchVoxelModel(tf.keras.layers.Layer):
    def __init__(self, in_channels, num_voxels=5, radius=10.0):
        super().__init__()
        self.layers_ = []
        for ch_in in [in_channels, 128, 128]:
            self.layers_.append(self._make_conv_layer(ch_in, num_voxels, radius))

    def _make_conv_layer(self, in_ch, num_voxels, radius):
        buckets = tf.linspace(0.0, float(radius), num_voxels)
        kernel = self.add_weight(
            shape=(num_voxels, num_voxels, num_voxels, in_ch, 128),
            initializer="glorot_uniform",
            trainable=True,
        )

        def layer_fn(pos, features, edges, R, vertex2molecule, training=None, **kwargs):
            a, b = tf.unstack(tf.cast(edges, tf.int32), axis=0)
            pos_a = tf.gather(pos, a)
            pos_b = tf.gather(pos, b)
            rel_pos = tf.einsum("bi,bij->bj", pos_b - pos_a, tf.gather(R, a))
            rx, ry, rz = tf.unstack(rel_pos, axis=1)
            i = tf.searchsorted(buckets, rx, side="left")
            j = tf.searchsorted(buckets, ry, side="left")
            k = tf.searchsorted(buckets, rz, side="left")
            i = tf.cast(i, tf.int32)
            j = tf.cast(j, tf.int32)
            k = tf.cast(k, tf.int32)
            kij = tf.gather_nd(kernel, tf.stack([i, j, k], axis=1))
            feats_b = tf.gather(features, b)
            out_e = tf.einsum("bi,bij->bj", feats_b, kij)
            num_nodes = tf.shape(pos)[0]
            features = _unsorted_segment_sum(out_e, a, num_nodes)
            return pos, features

        return layer_fn

    def call(self, pos, features, edges, R, vertex2molecule, training=None, **kwargs):
        for layer in self.layers_:
            pos, features = layer(pos, features, edges, R, vertex2molecule)
        return pos, features


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
        self.head = tf.keras.layers.Dense(out_channels)
        self.out_channels = out_channels
        if not node_level:
            self.pool = _unsorted_segment_mean

        if loss == "cross_entropy":
            self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        elif loss in ("mse", "mse_loss"):
            self.loss = tf.keras.losses.MeanSquaredError()
        elif loss in ("l1_loss", "mae", "mean_absolute_error"):
            self.loss = tf.keras.losses.MeanAbsoluteError()
        elif loss in ("bce_with_logits", "binary_cross_entropy_with_logits"):
            # expects y in {0,1}
            self.loss = lambda y_pred, y_true: tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.cast(y_true, tf.float32), logits=y_pred
                )
            )
        else:
            self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def forward(self, pos, features, edges, R, vertex2molecule):
        pos, x = self.trunk(pos, features, edges, R, vertex2molecule)
        if not self.node_level:
            x = self.pool(x, tf.cast(vertex2molecule, tf.int32))
        return self.head(x)

    def train_step(self, Xy, data):
        pos = tf.cast(getattr(data, f"{self.resolution}_pos"), tf.float32)
        feats = (
            tf.cast(getattr(data, f"{self.resolution}_features"), tf.float32)
            if hasattr(data, f"{self.resolution}_features")
            else tf.ones((tf.shape(pos)[0], 1), dtype=tf.float32)
        )
        edges = tf.cast(tf.transpose(data.molecule_edges), tf.int32)
        R = tf.cast(getattr(data, f"{self.resolution}_frame_R"), tf.float32)
        v2m = tf.cast(data.vertex2molecule, tf.int32)
        output = self.forward(pos, feats, edges, R, v2m)
        # Replace non-finite outputs to avoid NaNs in downstream metrics/loss.
        output = tf.where(tf.math.is_finite(output), output, tf.zeros_like(output))
        if self.out_channels == 1:
            output = tf.squeeze(output, axis=1)
        y = tf.cast(
            data.y,
            (
                tf.int32
                if isinstance(self.loss, tf.keras.losses.SparseCategoricalCrossentropy)
                else tf.float32
            ),
        )
        loss = (
            self.loss(y, output)
            if hasattr(self.loss, "__call__")
            else self.loss(output, y)
        )
        # Guard against non-finite loss values.
        loss = tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))
        return loss, output

    def eval_step(self, Xy, data):
        pos = tf.cast(getattr(data, f"{self.resolution}_pos"), tf.float32)
        feats = (
            tf.cast(getattr(data, f"{self.resolution}_features"), tf.float32)
            if hasattr(data, f"{self.resolution}_features")
            else tf.ones((tf.shape(pos)[0], 1), dtype=tf.float32)
        )
        edges = tf.cast(tf.transpose(data.molecule_edges), tf.int32)
        R = tf.cast(getattr(data, f"{self.resolution}_frame_R"), tf.float32)
        v2m = tf.cast(data.vertex2molecule, tf.int32)
        output = self.forward(pos, feats, edges, R, v2m)
        # Replace non-finite outputs to avoid NaNs in evaluation metrics.
        output = tf.where(tf.math.is_finite(output), output, tf.zeros_like(output))
        if self.out_channels == 1:
            output = tf.squeeze(output, axis=1)
        if len(data.y.shape) == 3:
            output = tf.reshape(output, data.y.shape)
        y = tf.cast(
            data.y,
            (
                tf.int32
                if isinstance(self.loss, tf.keras.losses.SparseCategoricalCrossentropy)
                else tf.float32
            ),
        )
        loss = (
            self.loss(y, output)
            if hasattr(self.loss, "__call__")
            else self.loss(output, y)
        )
        # Guard against non-finite loss values.
        loss = tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))
        return loss, output
