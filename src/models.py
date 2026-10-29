"""Pure-PyTorch models (no torch_geometric needed).

* BiGCN  – Bi-directional GCN with root enhancement (Bian et al., AAAI 2020),
           re-implemented with block-diagonal sparse adjacency.
* TextMLP – classifier on a precomputed source-text embedding (TF-IDF/SBERT/LLM).
* Fusion – BiGCN graph embedding + text embedding, joined by a learned gate.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import Tree, VOCAB


# --------------------------------------------------------------------------- batching
def _sp_to_torch(m: sp.spmatrix) -> torch.Tensor:
    m = m.tocoo()
    idx = torch.from_numpy(np.vstack([m.row, m.col]).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(m.data.astype(np.float32)), m.shape, check_invariants=False).coalesce()


def _norm_adj(edges_src, edges_dst, n) -> sp.csr_matrix:
    a = sp.coo_matrix((np.ones(len(edges_src)), (edges_dst, edges_src)), shape=(n, n))  # msg src->dst
    a = a + sp.eye(n)
    d = np.asarray(a.sum(1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(d))
    return (dinv @ a @ dinv).tocsr()


class Batch:
    """Block-diagonal batch of trees."""

    def __init__(self, trees: list[Tree], drop_edge: float = 0.0, rng: np.random.Generator | None = None,
                 text_emb: np.ndarray | None = None, x_transform=None):
        xs, td_s, td_d, roots, graph_id = [], [], [], [], []
        off = 0
        for g, t in enumerate(trees):
            child = np.where(t.parent >= 0)[0]
            par = t.parent[child]
            if drop_edge > 0 and len(child) > 0:
                keep = rng.random(len(child)) >= drop_edge
                child, par = child[keep], par[keep]
            td_s.append(par + off); td_d.append(child + off)
            xs.append(t.x)
            roots.append(off)
            graph_id.append(np.full(t.n, g))
            off += t.n
        n = off
        s, d = np.concatenate(td_s), np.concatenate(td_d)
        x = sp.vstack(xs).tocsr()
        if x_transform is not None:
            x = x_transform(x)
        self.x = _sp_to_torch(x)
        self.a_td = _sp_to_torch(_norm_adj(s, d, n))   # parent -> child (propagation)
        self.a_bu = _sp_to_torch(_norm_adj(d, s, n))   # child -> parent (dispersion)
        self.roots = torch.tensor(roots)
        self.graph_id = torch.from_numpy(np.concatenate(graph_id))
        self.num_graphs = len(trees)
        self.y = torch.tensor([t.label for t in trees])
        self.text = None if text_emb is None else torch.from_numpy(text_emb.astype(np.float32))


def mean_pool(h, graph_id, num_graphs):
    out = torch.zeros(num_graphs, h.shape[1], device=h.device).index_add_(0, graph_id, h)
    cnt = torch.bincount(graph_id, minlength=num_graphs).clamp(min=1).unsqueeze(1)
    return out / cnt


# --------------------------------------------------------------------------- BiGCN
class RootEnhancedGCN(nn.Module):
    """Two GCN layers; each layer's input is concatenated with the root's representation."""

    def __init__(self, in_dim, hid, out):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hid)
        self.w2_h = nn.Linear(hid, out)
        self.w2_root = nn.Linear(in_dim, out, bias=False)   # == concat(h1, x_root) @ W2
        self.out_dim = out + hid

    def forward(self, x_sp, a, roots, graph_id, num_graphs, dropout, training, node_mask=None):
        xw = torch.sparse.mm(x_sp, self.w1.weight.t()) + self.w1.bias
        if node_mask is not None:
            xw = xw * node_mask
        h1 = torch.sparse.mm(a, xw)
        root_x = torch.sparse.mm(x_sp, self.w2_root.weight.t())[roots]      # (G, out)
        h1r = F.relu(h1)
        h1r = F.dropout(h1r, dropout, training)
        h2 = torch.sparse.mm(a, self.w2_h(h1r) + root_x[graph_id])
        h2 = F.relu(h2)
        root_h1 = h1[roots][graph_id]                                       # root enhancement
        h = torch.cat([h2, root_h1], 1)
        return mean_pool(h, graph_id, num_graphs)


class BiGCN(nn.Module):
    def __init__(self, in_dim=VOCAB, hid=64, out=64, n_cls=4, dropout=0.5):
        super().__init__()
        self.td = RootEnhancedGCN(in_dim, hid, out)
        self.bu = RootEnhancedGCN(in_dim, hid, out)
        self.dropout = dropout
        self.emb_dim = self.td.out_dim + self.bu.out_dim
        self.fc = nn.Linear(self.emb_dim, n_cls)

    def embed(self, b: Batch, node_mask=None):
        ztd = self.td(b.x, b.a_td, b.roots, b.graph_id, b.num_graphs, self.dropout, self.training, node_mask)
        zbu = self.bu(b.x, b.a_bu, b.roots, b.graph_id, b.num_graphs, self.dropout, self.training, node_mask)
        return torch.cat([ztd, zbu], 1)

    def forward(self, b: Batch, node_mask=None):
        return self.fc(self.embed(b, node_mask))


class TextMLP(nn.Module):
    def __init__(self, text_dim, hid=128, n_cls=4, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(text_dim, hid), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hid, n_cls))

    def forward(self, b: Batch, node_mask=None):
        return self.net(b.text)


class Fusion(nn.Module):
    """Gated fusion of propagation structure (BiGCN) and source-text semantics."""

    def __init__(self, text_dim, in_dim=VOCAB, hid=64, n_cls=4, dropout=0.5):
        super().__init__()
        self.gnn = BiGCN(in_dim, hid, hid, n_cls, dropout)
        g = self.gnn.emb_dim
        self.proj_g = nn.Linear(g, hid * 2)
        self.proj_t = nn.Sequential(nn.Dropout(0.3), nn.Linear(text_dim, hid * 2))
        self.gate = nn.Linear(hid * 4, hid * 2)
        self.fc = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(hid * 2, n_cls))
        self.last_gate = None

    def forward(self, b: Batch, node_mask=None):
        zg = self.proj_g(self.gnn.embed(b, node_mask))
        zt = self.proj_t(b.text)
        a = torch.sigmoid(self.gate(torch.cat([zg, zt], 1)))
        self.last_gate = a.detach().mean(1)          # >0.5 = leaning on structure
        return self.fc(a * zg + (1 - a) * zt)
