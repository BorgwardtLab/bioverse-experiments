from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree, to_dense_adj, to_dense_batch


def _incoming_degree(edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype) -> torch.Tensor:
    """Per-node incoming degree, with isolated nodes clamped to 1."""
    deg = degree(edge_index[1], num_nodes=num_nodes, dtype=dtype)
    return deg.clamp(min=1).unsqueeze(-1)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x)


class Encoder(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(aggr="add", flow="source_to_target")
        self.out_v = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.out_e = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.norm3 = nn.LayerNorm(hidden_channels)
        self.dense = PositionWiseFeedForward(hidden_channels, hidden_channels * 4)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_message = self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr)
        degree_scale = _incoming_degree(edge_index, x.size(0), h_message.dtype)
        dh = h_message / degree_scale
        x = self.norm1(x + self.dropout1(dh))
        dh = self.dense(x)
        x = self.norm2(x + self.dropout2(dh))
        row, col = edge_index
        x_i, x_j = x[row], x[col]
        h_e = torch.cat([x_i, x_j, edge_attr], dim=-1)
        h_e = self.out_e(h_e)
        edge_attr = self.norm3(edge_attr + self.dropout3(h_e))
        return x, edge_attr

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        h = torch.cat([x_i, x_j, edge_attr], dim=-1)  # [E, 2*d_v + d_e]
        h = self.out_v(h)  # [E, d_v]
        return h


class Decoder(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(aggr="add", flow="source_to_target")
        self.out_v = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.dense = PositionWiseFeedForward(hidden_channels, hidden_channels * 4)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        x_label: torch.Tensor,
        mask: torch.Tensor,
        x_enc: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:      [N, d_v]   decoder node states
        x_enc:  [N, d_v]   encoder node states (fixed during decoding)
        mask:   [E, 1]     1 for backward edges (source decoded), 0 for forward edges
        """
        h_message = self.propagate(
            edge_index=edge_index,
            x=x,
            x_enc=x_enc,
            x_label=x_label,
            edge_attr=edge_attr,
            mask=mask,
        )
        degree_scale = _incoming_degree(edge_index, x.size(0), h_message.dtype)
        dh = h_message / degree_scale
        x = self.norm1(x + self.dropout1(dh))
        dh = self.dense(x)
        x = self.norm2(x + self.dropout2(dh))
        return x

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        x_enc_j: torch.Tensor,
        x_label_j: torch.Tensor,
        edge_attr: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Use decoder state for backward edges, encoder state for forward edges
        x_src = mask * x_j + (1.0 - mask) * x_enc_j
        h_1 = torch.cat([x_src, edge_attr, x_label_j], dim=-1)
        h_0 = torch.cat([x_src, edge_attr, torch.zeros_like(x_label_j)], dim=-1)
        h = h_1 * mask + h_0 * (1.0 - mask)
        h = torch.cat([x_i, h], dim=-1)
        h = self.out_v(h)
        return h


class ProteinMPNN_Model(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dropout: float = 0.1,
        vocab_size: int = 33,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.node_dim = node_dim
        # Temporarily use internal featurization (_featurize) to build RBF edge features
        # from backbone coordinates, following the PyG reference implementation.
        # This yields 25 atom-pair combinations * 16 RBFs = 400-dim edge features.
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.node_mlp = nn.Sequential(nn.Linear(node_dim, self.hidden_dim))
        self.label_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_layers = nn.ModuleList(
            [
                Encoder(hidden_dim * 3, hidden_dim, dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                Decoder(hidden_dim * 4, hidden_dim, dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _rbf(self, D: torch.Tensor) -> torch.Tensor:
        """Radial basis function encoding, following the PyG ProteinMPNN example."""
        D_min, D_max, D_count = 2.0, 22.0, 16
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device).view(1, -1)
        D_sigma = (D_max - D_min) / D_count
        D_expand = D.unsqueeze(-1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _featurize(self, X: torch.Tensor, batch: torch.Tensor):
        """
        Temporary reimplementation of the PyG ProteinMPNN _featurize function,
        using only backbone coordinates and batch, ignoring masks.

        Args:
            X: [N, 4, 3] backbone coordinates (N, Ca, C, O)
            batch: [N] batch indices

        Returns:
            edge_index: [2, E]
            edge_attr: [E, 400] RBF features for all atom pair combinations.
        """
        N_atoms = X[:, 0, :]
        Ca = X[:, 1, :]
        C = X[:, 2, :]
        O = X[:, 3, :]

        b = Ca - N_atoms
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

        # Treat all positions as valid (no padding mask in concatenated representation).
        valid_mask = torch.ones(X.size(0), dtype=torch.bool, device=X.device)
        valid_Ca = Ca[valid_mask]
        valid_batch = batch[valid_mask]

        edge_index = knn_graph(valid_Ca, k=30, batch=valid_batch, loop=True)

        row, col = edge_index
        original_indices = torch.arange(Ca.size(0), device=X.device)[valid_mask]
        edge_index_original = torch.stack(
            [original_indices[row], original_indices[col]], dim=0
        )
        row, col = edge_index_original

        rbf_all = []
        for A, B in product([N_atoms, Ca, C, O, Cb], repeat=2):
            distances = torch.sqrt(torch.sum((A[row] - B[col]) ** 2, dim=1) + 1e-6)
            rbf = self._rbf(distances)
            rbf_all.append(rbf)

        edge_attr = torch.cat(rbf_all, dim=-1)  # [E, 400]
        return edge_index_original, edge_attr

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_attr: torch.Tensor | None,
        chain_seq_label: torch.Tensor,
        chain_mask_all: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        device = edge_attr.device
        N = chain_seq_label.size(0)
        # Encode structure once
        h_e = self.edge_mlp(edge_attr)  # [E, hidden_dim]
        if node_attr is None:
            node_attr = torch.zeros(N, self.node_dim, device=device)
        h_v = self.node_mlp(node_attr)
        for encoder in self.encoder_layers:
            h_v, h_e = encoder(h_v, edge_index, h_e)

        if self.training:
            # ---------------------------
            # Training: teacher forcing
            # ---------------------------
            label_ids = chain_seq_label
            h_label = self.label_embedding(label_ids)

            batch_chain_mask_all, _ = to_dense_batch(chain_mask_all, batch)
            decoding_order = torch.argsort(
                (batch_chain_mask_all + 1e-4)
                * (torch.abs(torch.randn(batch_chain_mask_all.shape, device=device)))
            )
            mask_size = batch_chain_mask_all.size(1)
            permutation_matrix_reverse = F.one_hot(
                decoding_order, num_classes=mask_size
            ).float()
            order_mask_backward = torch.einsum(
                "ij, biq, bjp->bqp",
                1 - torch.triu(torch.ones(mask_size, mask_size, device=device)),
                permutation_matrix_reverse,
                permutation_matrix_reverse,
            )
            adj = to_dense_adj(edge_index, batch)  # [B, N, N]
            mask_attend = order_mask_backward[adj.bool()].unsqueeze(-1)  # [E, 1]

            for decoder in self.decoder_layers:
                h_v = decoder(h_v, edge_index, h_e, h_label, mask_attend, h_v)
            return h_v

        # ---------------------------------------------------
        # Eval: simple iterative decoding without label bleed
        # ---------------------------------------------------
        src, dst = edge_index  # [E], [E]

        # Design only positions where chain_mask_all > 0
        design_mask = chain_mask_all > 0
        decoded = ~design_mask.bool()  # non-design positions treated as already decoded
        labels = torch.zeros(N, dtype=torch.long, device=device)

        # Maintain per-layer hidden states, like h_V_stack in the original implementation
        num_layers = len(self.decoder_layers)
        h_V_stack = [h_v.clone()] + [
            h_v.clone() for _ in range(num_layers)
        ]  # length = num_layers + 1

        # Decode in random order over designable nodes, updating only the current position
        while not decoded.all():
            # Current label embeddings from previous steps
            h_label = self.label_embedding(labels)  # [N, hidden_dim]

            # Edge mask: only use labels from already-decoded source nodes
            mask_edge = decoded[src].float().unsqueeze(-1)  # [E, 1]

            # Choose next undecoded design position (random among remaining)
            candidates = (~decoded & design_mask.bool()).nonzero(as_tuple=True)[0]
            if candidates.numel() == 0:
                break
            rand_idx = torch.randint(0, candidates.numel(), (1,), device=device)
            i = candidates[rand_idx].item()

            # For each decoder layer, update only position i using a subgraph centered at i
            for l, decoder in enumerate(self.decoder_layers):
                h_v_l = h_V_stack[l]  # [N, hidden_dim]

                # Select edges whose destination is i
                edge_mask_i = dst == i  # [E]
                if not edge_mask_i.any():
                    # No incoming edges (isolated node); skip update
                    h_V_stack[l + 1] = h_v_l
                    continue

                edge_index_i = edge_index[:, edge_mask_i]  # [2, E_i]
                edge_attr_i = h_e[edge_mask_i]  # [E_i, hidden_dim]
                mask_edge_i = mask_edge[edge_mask_i]  # [E_i, 1]

                # Run decoder on this subgraph; only node i will get non-zero update
                h_v_updated = decoder(
                    h_v_l, edge_index_i, edge_attr_i, h_label, mask_edge_i, h_v
                )  # [N, hidden_dim]

                h_v_next = h_v_l.clone()
                h_v_next[i] = h_v_updated[i]
                h_V_stack[l + 1] = h_v_next

            # Final hidden state for position i after all decoder layers
            h_v_i = h_V_stack[-1][i]  # [hidden_dim]

            # Predict token at position i using tied classifier on label embeddings
            logits_i = torch.matmul(h_v_i, self.label_embedding.weight.t())  # [vocab]
            labels[i] = logits_i.argmax(dim=-1)
            decoded[i] = True

        # After iterative decoding, return full decoded node embeddings
        return h_V_stack[-1]


class MpnnModel(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.model = ProteinMPNN_Model(
            node_dim=in_channels[0],
            edge_dim=in_channels[1],
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dropout=0.1,
            vocab_size=33,
        )

    def forward(self, data):
        # X = data.residue_backbone.float()  # [N, 4, 3]
        # edge_index, edge_attr = self.model._featurize(X, batch)  # [2,E], [E,400]
        batch = data.vertex2molecule.long()  # [N]
        edge_index = data.molecule_edges.T.long()  # [2, E]
        if hasattr(data, "residue_features"):
            node_attr = data.residue_features.float()
        else:
            node_attr = None
        edge_attr = data.molecule_edge_features.float()  # [E, d_e]
        chain_seq_label = data.residue_token.long()
        chain_mask_all = torch.ones_like(chain_seq_label, dtype=torch.float)
        return self.model(
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_attr=node_attr,
            chain_seq_label=chain_seq_label,
            chain_mask_all=chain_mask_all,
            batch=batch,
        )
