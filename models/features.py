import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E


def _compute_cb(N_atoms, Ca, C):
    b = Ca - N_atoms
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca
    return Cb


class PifoldFeaturizer(nn.Module):

    def __init__(
        self,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=30,
        radius=None,
        # PiFold flags
        node_dist=True,
        node_angle=True,
        node_direct=True,
        edge_dist=True,
        edge_angle=True,
        edge_direct=True,
    ):
        super().__init__()
        self.num_positional_embeddings = num_positional_embeddings
        self.num_rbf = num_rbf
        self.top_k = top_k
        self.radius = radius
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        # for MPNN edge projection compatibility
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, 128, bias=False)
        self.norm_edges = nn.LayerNorm(128)

        # PiFold flags
        self.node_dist = node_dist
        self.node_angle = node_angle
        self.node_direct = node_direct
        self.edge_dist = edge_dist
        self.edge_angle = edge_angle
        self.edge_direct = edge_direct

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _dist_concat(self, X, batch):
        # Vectorized exact top-k within graphs without loops/padding
        N = X.size(0)
        K = self.top_k
        diff = X.unsqueeze(1) - X.unsqueeze(0)  # [N,N,3]
        D = torch.sqrt((diff.pow(2).sum(-1)).add(1e-6))  # [N,N]
        same = batch.unsqueeze(1) == batch.unsqueeze(0)  # [N,N]
        D_adjust = D + (~same) * 1e6  # mask cross-graph
        K_eff = min(K, N)
        Dk, Ik = torch.topk(D_adjust, K_eff, dim=-1, largest=False)  # [N,K_eff]
        if K_eff < K:
            pad = K - K_eff
            Ik = torch.cat([Ik, Ik[:, :pad]], dim=-1)
            Dk = torch.cat([Dk, Dk[:, :pad]], dim=-1)
        return Dk, Ik

    def _get_rbf_concat(self, A, B, E_idx):
        Bn = B[E_idx]
        D = ((A.unsqueeze(1) - Bn) ** 2).sum(-1).add(1e-6).sqrt()
        return self._rbf(D)

    def build_mpnn(self, X, batch):
        """
        Build MPNN-style edge features (E [N,K,Fe], E_idx [N,K]) from concatenated input.
        """
        N_atoms = X[:, 0, :]
        Ca = X[:, 1, :]
        C = X[:, 2, :]
        O = X[:, 3, :]
        Cb = _compute_cb(N_atoms, Ca, C)

        D_neighbors, E_idx = self._dist_concat(Ca, batch)

        RBF_all = torch.cat(
            [
                self._rbf(D_neighbors),
                self._get_rbf_concat(N_atoms, N_atoms, E_idx),
                self._get_rbf_concat(C, C, E_idx),
                self._get_rbf_concat(O, O, E_idx),
                self._get_rbf_concat(Cb, Cb, E_idx),
                self._get_rbf_concat(Ca, N_atoms, E_idx),
                self._get_rbf_concat(Ca, C, E_idx),
                self._get_rbf_concat(Ca, O, E_idx),
                self._get_rbf_concat(Ca, Cb, E_idx),
                self._get_rbf_concat(N_atoms, C, E_idx),
                self._get_rbf_concat(N_atoms, O, E_idx),
                self._get_rbf_concat(N_atoms, Cb, E_idx),
                self._get_rbf_concat(Cb, C, E_idx),
                self._get_rbf_concat(Cb, O, E_idx),
                self._get_rbf_concat(O, C, E_idx),
                self._get_rbf_concat(N_atoms, Ca, E_idx),
                self._get_rbf_concat(C, Ca, E_idx),
                self._get_rbf_concat(O, Ca, E_idx),
                self._get_rbf_concat(Cb, Ca, E_idx),
                self._get_rbf_concat(C, N_atoms, E_idx),
                self._get_rbf_concat(O, N_atoms, E_idx),
                self._get_rbf_concat(Cb, N_atoms, E_idx),
                self._get_rbf_concat(C, Cb, E_idx),
                self._get_rbf_concat(O, Cb, E_idx),
                self._get_rbf_concat(C, O, E_idx),
            ],
            dim=-1,
        )

        # positional encodings: residue indices per graph computed from batch
        res_idx = torch.zeros(X.size(0), dtype=torch.long, device=X.device)
        for b in torch.unique(batch):
            idx = (batch == b).nonzero(as_tuple=True)[0]
            res_idx[idx] = torch.arange(idx.numel(), device=X.device)
        offset = res_idx.unsqueeze(1) - res_idx[E_idx]
        E_positional = self.embeddings(
            offset.long(), torch.ones_like(offset, dtype=torch.long)
        )

        E = torch.cat([E_positional, RBF_all], dim=-1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx

    def build_pifold(self, X, batch):
        """
        Build PiFold-style node and edge features, and edge_index, from concatenated input.
        Returns: h_V [N, Fv], h_E [E, Fe], edge_index [2,E], batch_id [N]
        """
        device = X.device
        # Vectorized scatter to padded without Python loops
        B = int(batch.max().item() + 1) if batch.numel() > 0 else 0
        sizes = torch.bincount(batch, minlength=B)
        N_max = int(sizes.max().item()) if sizes.numel() > 0 else 0
        if N_max == 0:
            return (
                torch.zeros(0, 0, device=device),
                torch.zeros(0, 0, device=device),
                torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
            )
        # local indices per-graph: pos = i - offset[batch[i]]
        offsets = torch.cumsum(sizes, dim=0) - sizes
        pos = torch.arange(X.shape[0], device=device) - offsets[batch]
        # Build padded X [B,N_max,4,3] and mask [B,N_max]
        X_pad = torch.zeros(B, N_max, 4, 3, device=device)
        mask = torch.zeros(B, N_max, dtype=torch.long, device=device)
        X_pad[batch, pos] = X
        mask[batch, pos] = 1

        # Reuse original PiFold feature builder once over padded batch

        # Dummy S tensor (content unused for geometric features)
        S_pad = torch.zeros(B, N_max, dtype=torch.long, device=device)
        Xp, S_sel, score, h_V, h_E, edge_index, batch_id, _, _, _ = _get_features(
            S=S_pad, score=None, X=X_pad, mask=mask
        )
        # _get_features returns concatenated outputs already; just return them
        return h_V, h_E, edge_index.long(), batch_id.long()


def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view(
        (neighbor_idx.shape[0], -1)
    )  # [4, 317, 30]-->[4, 9510]
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(
        -1, -1, nodes.size(2)
    )  # [4, 9510, dim]
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)  # [4, 9510, dim]
    return neighbor_features.view(
        list(neighbor_idx.shape)[:3] + [-1]
    )  # [4, 317, 30, 128]


def _rbf(D, num_rbf):
    D_min, D_max, D_count = 0.0, 20.0, num_rbf
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu.view([1, 1, 1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


def gather_edges(edges, neighbor_idx):
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)


def _get_rbf(A, B, E_idx=None, num_rbf=16):
    if E_idx is not None:
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = _rbf(D_A_B_neighbors, num_rbf)
    else:
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, :, None, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        RBF_A_B = _rbf(D_A_B, num_rbf)
    return RBF_A_B


def _orientations_coarse_gl_tuple(X, E_idx, eps=1e-6):
    V = X.clone()
    X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)
    dX = X[:, 1:, :] - X[:, :-1, :]  # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0, u_1 = U[:, :-2, :], U[:, 1:-1, :]
    n_0 = _normalize(torch.linalg.cross(u_0, u_1), dim=-1)
    b_1 = _normalize(u_0 - u_1, dim=-1)

    n_0 = n_0[:, ::3, :]
    b_1 = b_1[:, ::3, :]
    X = X[:, ::3, :]
    Q = torch.stack((b_1, n_0, torch.linalg.cross(b_1, n_0)), 2)
    Q = Q.view(list(Q.shape[:2]) + [9])
    Q = F.pad(Q, (0, 0, 0, 1), "constant", 0)  # [16, 464, 9]

    Q_neighbors = gather_nodes(Q, E_idx)  # [16, 464, 30, 9]
    X_neighbors = gather_nodes(V[:, :, 1, :], E_idx)  # [16, 464, 30, 3]
    N_neighbors = gather_nodes(V[:, :, 0, :], E_idx)
    C_neighbors = gather_nodes(V[:, :, 2, :], E_idx)
    O_neighbors = gather_nodes(V[:, :, 3, :], E_idx)

    Q = Q.view(list(Q.shape[:2]) + [3, 3]).unsqueeze(2)  # [16, 464, 1, 3, 3]
    Q_neighbors = Q_neighbors.view(
        list(Q_neighbors.shape[:3]) + [3, 3]
    )  # [16, 464, 30, 3, 3]

    dX = (
        torch.stack([X_neighbors, N_neighbors, C_neighbors, O_neighbors], dim=3)
        - X[:, :, None, None, :]
    )  # [16, 464, 30, 3]
    dU = torch.matmul(Q[:, :, :, None, :, :], dX[..., None]).squeeze(
        -1
    )  # [16, 464, 30, 3] 邻居的相对坐标
    B, N, K = dU.shape[:3]
    E_direct = _normalize(dU, dim=-1)
    E_direct = E_direct.reshape(B, N, K, -1)
    R = torch.matmul(Q.transpose(-1, -2), Q_neighbors)
    q = _quaternions(R)
    # edge_feat = torch.cat((dU, q), dim=-1) # 相对方向向量+旋转四元数

    dX_inner = V[:, :, [0, 2, 3], :] - X.unsqueeze(-2)
    dU_inner = torch.matmul(Q, dX_inner.unsqueeze(-1)).squeeze(-1)
    dU_inner = _normalize(dU_inner, dim=-1)
    V_direct = dU_inner.reshape(B, N, -1)
    return V_direct, E_direct, q


def _quaternions(R):
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(
        torch.abs(
            1 + torch.stack([Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz], -1)
        )
    )
    _R = lambda i, j: R[:, :, :, i, j]
    signs = torch.sign(
        torch.stack([_R(2, 1) - _R(1, 2), _R(0, 2) - _R(2, 0), _R(1, 0) - _R(0, 1)], -1)
    )
    xyz = signs * magnitudes
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.0
    Q = torch.cat((xyz, w), -1)
    return _normalize(Q, dim=-1)


def cal_dihedral(X, eps=1e-7):
    dX = X[:, 1:, :] - X[:, :-1, :]  # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0 = U[:, :-2, :]  # CA-N, C-CA, N-C,...
    u_1 = U[
        :, 1:-1, :
    ]  # C-CA, N-C, CA-N, ... 0, psi_{i}, omega_{i}, phi_{i+1} or 0, tau_{i},...
    u_2 = U[:, 2:, :]  # N-C, CA-N, C-CA, ...

    n_0 = _normalize(torch.linalg.cross(u_0, u_1), dim=-1)
    n_1 = _normalize(torch.linalg.cross(u_1, u_2), dim=-1)

    cosD = (n_0 * n_1).sum(-1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)

    v = _normalize(torch.linalg.cross(n_0, n_1), dim=-1)
    D = torch.sign((-v * u_1).sum(-1)) * torch.acos(cosD)  # TODO: sign

    return D


def _dihedrals(X, dihedral_type=0, eps=1e-7):
    B, N, _, _ = X.shape
    # psi, omega, phi
    X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)  # ['N', 'CA', 'C', 'O']
    D = cal_dihedral(X)
    D = F.pad(D, (1, 2), "constant", 0)
    D = D.view((D.size(0), int(D.size(1) / 3), 3))
    Dihedral_Angle_features = torch.cat((torch.cos(D), torch.sin(D)), 2)

    # alpha, beta, gamma
    dX = X[:, 1:, :] - X[:, :-1, :]  # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0 = U[:, :-2, :]  # CA-N, C-CA, N-C,...
    u_1 = U[:, 1:-1, :]  # C-CA, N-C, CA-N, ...
    cosD = (u_0 * u_1).sum(-1)  # alpha_{i}, gamma_{i}, beta_{i+1}
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.acos(cosD)
    D = F.pad(D, (1, 2), "constant", 0)
    D = D.view((D.size(0), int(D.size(1) / 3), 3))
    Angle_features = torch.cat((torch.cos(D), torch.sin(D)), 2)

    D_features = torch.cat((Dihedral_Angle_features, Angle_features), 2)
    return D_features


def nan_to_num(tensor, nan=0.0):
    idx = torch.isnan(tensor)
    tensor[idx] = nan
    return tensor


def _normalize(tensor, dim=-1):
    return nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _full_dist(X, mask, top_k=30, eps=1e-6):
    mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
    dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
    D = (1.0 - mask_2D) * 10000 + mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

    D_max, _ = torch.max(D, -1, keepdim=True)
    D_adjust = D + (1.0 - mask_2D) * (D_max + 1)
    D_neighbors, E_idx = torch.topk(
        D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False
    )
    return D_neighbors, E_idx


def _get_features(
    S,
    score,
    X,
    mask,
    top_k=30,
    radius=None,
    virtual_num=0,
    num_rbf=16,
    node_dist=True,
    node_angle=True,
    node_direct=True,
    edge_dist=True,
    edge_angle=True,
    edge_direct=True,
):
    device = X.device
    mask_bool = mask == 1
    B, N, _, _ = X.shape
    X_ca = X[:, :, 1, :]
    D_neighbors, E_idx = _full_dist(X_ca, mask, top_k)
    # optional radius mask over neighbors
    if radius is not None:
        within_radius = D_neighbors <= radius
    else:
        within_radius = torch.ones_like(D_neighbors, dtype=torch.bool)

    mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
    mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1
    mask_attend = mask_attend & within_radius
    edge_mask_select = lambda x: torch.masked_select(
        x, mask_attend.unsqueeze(-1)
    ).reshape(-1, x.shape[-1])
    node_mask_select = lambda x: torch.masked_select(
        x, mask_bool.unsqueeze(-1)
    ).reshape(-1, x.shape[-1])

    # print(mask_attend.shape)
    # x = mask_attend.float().sum(dim=2)
    # x[x == 0] = float("nan")
    # print(x.nanmean())
    # exit()

    randn = torch.rand(mask.shape, device=X.device) + 5
    decoding_order = torch.argsort(
        -mask * (torch.abs(randn))
    )  # 我们的mask=1代表数据可用, 而protein MPP的mask=1代表数据不可用，正好相反
    mask_size = mask.shape[1]
    permutation_matrix_reverse = torch.nn.functional.one_hot(
        decoding_order, num_classes=mask_size
    ).float()
    # 计算q已知的情况下, q->p的mask,
    order_mask_backward = torch.einsum(
        "ij, biq, bjp->bqp",
        (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
        permutation_matrix_reverse,
        permutation_matrix_reverse,
    )
    mask_attend2 = torch.gather(order_mask_backward, 2, E_idx)
    mask_1D = mask.view([mask.size(0), mask.size(1), 1])
    mask_bw = (mask_1D * mask_attend2).unsqueeze(-1)
    mask_fw = (mask_1D * (1 - mask_attend2)).unsqueeze(-1)
    # apply radius mask to edge-direction masks
    if radius is not None:
        mask_bw = mask_bw * within_radius.unsqueeze(-1)
        mask_fw = mask_fw * within_radius.unsqueeze(-1)
    mask_bw = edge_mask_select(mask_bw).squeeze()
    mask_fw = edge_mask_select(mask_fw).squeeze()

    # sequence
    S = torch.masked_select(S, mask_bool)
    if score is not None:
        score = torch.masked_select(score, mask_bool)

    # angle & direction
    V_angles = _dihedrals(X, 0)
    V_angles = node_mask_select(V_angles)

    V_direct, E_direct, E_angles = _orientations_coarse_gl_tuple(X, E_idx)
    V_direct = node_mask_select(V_direct)
    E_direct = edge_mask_select(E_direct)
    E_angles = edge_mask_select(E_angles)

    # distance
    atom_N = X[:, :, 0, :]
    atom_Ca = X[:, :, 1, :]
    atom_C = X[:, :, 2, :]
    atom_O = X[:, :, 3, :]
    b = atom_Ca - atom_N
    c = atom_C - atom_Ca
    a = torch.linalg.cross(b, c, dim=-1)

    if virtual_num > 0:
        virtual_atoms = virtual_atoms / torch.norm(virtual_atoms, dim=1, keepdim=True)
        for i in range(virtual_atoms.shape[0]):
            vars()["atom_v" + str(i)] = (
                virtual_atoms[i][0] * a
                + virtual_atoms[i][1] * b
                + virtual_atoms[i][2] * c
                + 1 * atom_Ca
            )

    node_list = ["Ca-N", "Ca-C", "Ca-O", "N-C", "N-O", "O-C"]
    node_dist = []
    for pair in node_list:
        atom1, atom2 = pair.split("-")
        node_dist.append(
            node_mask_select(
                _get_rbf(
                    vars()["atom_" + atom1],
                    vars()["atom_" + atom2],
                    None,
                    num_rbf,
                ).squeeze()
            )
        )

    if virtual_num > 0:
        for i in range(virtual_atoms.shape[0]):
            # # true atoms
            for j in range(0, i):
                node_dist.append(
                    node_mask_select(
                        _get_rbf(
                            vars()["atom_v" + str(i)],
                            vars()["atom_v" + str(j)],
                            None,
                            num_rbf,
                        ).squeeze()
                    )
                )
                node_dist.append(
                    node_mask_select(
                        _get_rbf(
                            vars()["atom_v" + str(j)],
                            vars()["atom_v" + str(i)],
                            None,
                            num_rbf,
                        ).squeeze()
                    )
                )
    V_dist = torch.cat(tuple(node_dist), dim=-1).squeeze()

    pair_lst = [
        "Ca-Ca",
        "Ca-C",
        "C-Ca",
        "Ca-N",
        "N-Ca",
        "Ca-O",
        "O-Ca",
        "C-C",
        "C-N",
        "N-C",
        "C-O",
        "O-C",
        "N-N",
        "N-O",
        "O-N",
        "O-O",
    ]

    edge_dist = []  # Ca-Ca
    for pair in pair_lst:
        atom1, atom2 = pair.split("-")
        rbf = _get_rbf(vars()["atom_" + atom1], vars()["atom_" + atom2], E_idx, num_rbf)
        edge_dist.append(edge_mask_select(rbf))

    if virtual_num > 0:
        for i in range(virtual_atoms.shape[0]):
            edge_dist.append(
                edge_mask_select(
                    _get_rbf(
                        vars()["atom_v" + str(i)],
                        vars()["atom_v" + str(i)],
                        E_idx,
                        num_rbf,
                    )
                )
            )
            for j in range(0, i):
                edge_dist.append(
                    edge_mask_select(
                        _get_rbf(
                            vars()["atom_v" + str(i)],
                            vars()["atom_v" + str(j)],
                            E_idx,
                            num_rbf,
                        )
                    )
                )
                edge_dist.append(
                    edge_mask_select(
                        _get_rbf(
                            vars()["atom_v" + str(j)],
                            vars()["atom_v" + str(i)],
                            E_idx,
                            num_rbf,
                        )
                    )
                )

    E_dist = torch.cat(tuple(edge_dist), dim=-1)

    h_V = []
    if node_dist:
        h_V.append(V_dist)
    if node_angle:
        h_V.append(V_angles)
    if node_direct:
        h_V.append(V_direct)

    h_E = []
    if edge_dist:
        h_E.append(E_dist)
    if edge_angle:
        h_E.append(E_angles)
    if edge_direct:
        h_E.append(E_direct)

    _V = torch.cat(h_V, dim=-1)
    _E = torch.cat(h_E, dim=-1)

    # edge index
    shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
    src = shift.view(B, 1, 1) + E_idx
    src = torch.masked_select(src, mask_attend).view(1, -1)
    dst = shift.view(B, 1, 1) + torch.arange(0, N, device=src.device).view(
        1, -1, 1
    ).expand_as(mask_attend)
    dst = torch.masked_select(dst, mask_attend).view(1, -1)
    E_idx = torch.cat((dst, src), dim=0).long()

    decoding_order = (
        node_mask_select((decoding_order + shift.view(-1, 1)).unsqueeze(-1))
        .squeeze()
        .long()
    )

    # 3D point
    sparse_idx = mask.nonzero()  # index of non-zero values
    X = X[sparse_idx[:, 0], sparse_idx[:, 1], :, :]
    batch_id = sparse_idx[:, 0]

    return X, S, score, _V, _E, E_idx, batch_id, mask_bw, mask_fw, decoding_order
