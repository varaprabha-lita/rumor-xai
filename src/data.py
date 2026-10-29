"""Load Twitter15/16 propagation trees, source-tweet text and labels.

Sources (see scripts/download_data.sh):
  * Trees + node bag-of-words features: BiGCN release (Bian et al., AAAI 2020),
    file data.TD_RvNN.vol_5000.txt, originally from Ma et al. (ACL 2017).
    Columns: root_id, parent_idx|None, node_idx, max_degree, max_len, "word:count ..."
  * Labels: Twitter15_label_All.txt / Twitter16_label_All.txt (col 0 label, col 2 tweet id)
  * Source-tweet text: GLAN release (Yuan et al., ICDM 2019), twitter15.{train,dev,test}
"""
from __future__ import annotations

import os
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

LABELS = ["non-rumor", "false", "true", "unverified"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
VOCAB = 5000

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


@dataclass
class Tree:
    eid: str
    label: int
    text: str
    x: sp.csr_matrix              # (n_nodes, 5000) bag-of-words counts; row 0 = source tweet
    parent: np.ndarray            # parent index per node, -1 for root
    order: np.ndarray = field(default=None)  # BFS arrival order (proxy for time)
    group: str = ""                          # claim / source-event id (label file column 2)

    @property
    def n(self) -> int:
        return self.x.shape[0]

    def depth(self) -> np.ndarray:
        d = np.zeros(self.n, dtype=int)
        for i in self.order[1:]:
            d[i] = d[self.parent[i]] + 1
        return d

    def truncate(self, k: int) -> "Tree":
        """Keep the root plus the first k responses in arrival order (early detection)."""
        if k is None or k >= self.n - 1:
            return self
        keep = self.order[: k + 1]
        remap = -np.ones(self.n, dtype=int)
        remap[keep] = np.arange(len(keep))
        par = np.array([-1 if self.parent[i] < 0 else remap[self.parent[i]] for i in keep])
        t = Tree(self.eid, self.label, self.text, self.x[keep], par, group=self.group)
        t.order = np.arange(len(keep))
        return t

    def drop_nodes(self, nodes) -> "Tree":
        """Remove nodes (and their subtrees) — used for explanation faithfulness tests."""
        nodes = set(int(v) for v in nodes if v != 0)
        if not nodes:
            return self
        children = defaultdict(list)
        for i, p in enumerate(self.parent):
            if p >= 0:
                children[p].append(i)
        removed, q = set(), deque(nodes)
        while q:
            v = q.popleft()
            if v in removed:
                continue
            removed.add(v)
            q.extend(children[v])
        keep = [i for i in self.order if i not in removed]
        remap = -np.ones(self.n, dtype=int)
        remap[keep] = np.arange(len(keep))
        par = np.array([-1 if self.parent[i] < 0 else remap[self.parent[i]] for i in keep])
        t = Tree(self.eid, self.label, self.text, self.x[keep], par, group=self.group)
        t.order = np.arange(len(keep))
        return t


def _bfs_order(parent: np.ndarray) -> np.ndarray:
    children = defaultdict(list)
    for i, p in enumerate(parent):
        if p >= 0:
            children[p].append(i)
    order, q = [], deque([0])
    while q:
        v = q.popleft()
        order.append(v)
        q.extend(sorted(children[v]))
    # nodes unreachable from root (rare data glitches) are appended at the end, attached to root
    seen = set(order)
    for i in range(len(parent)):
        if i not in seen:
            parent[i] = 0
            order.append(i)
    return np.array(order)


def load_dataset(name: str = "Twitter15", root: str = ROOT) -> list[Tree]:
    assert name in ("Twitter15", "Twitter16")
    bdir = os.path.join(root, "bigcn", "data", name)
    labels, groups = {}, {}
    for line in open(os.path.join(bdir, f"{name}_label_All.txt"), encoding="utf8"):
        c = line.rstrip("\n").split("\t")
        labels[c[2]] = LABEL2ID[c[0]]
        groups[c[2]] = c[1]

    texts = {}
    tpath = os.path.join(root, "text", f"{name.lower()}_source.tsv")
    for line in open(tpath, encoding="utf8"):
        c = line.rstrip("\n").split("\t")
        if len(c) >= 2:
            texts[c[0]] = c[1]

    rows = defaultdict(list)
    for line in open(os.path.join(bdir, "data.TD_RvNN.vol_5000.txt"), encoding="utf8"):
        c = line.rstrip("\n").split("\t")
        if c[0] in labels:
            rows[c[0]].append(c)

    trees = []
    for eid, rs in rows.items():
        rs = sorted(rs, key=lambda r: int(r[2]))
        # make the root (parent None) node 0
        root_pos = next(i for i, r in enumerate(rs) if r[1] == "None")
        perm = [root_pos] + [i for i in range(len(rs)) if i != root_pos]
        rs = [rs[i] for i in perm]
        idx = {r[2]: i for i, r in enumerate(rs)}
        parent = np.array([-1 if r[1] == "None" else idx.get(r[1], 0) for r in rs])
        data, ri, ci = [], [], []
        for i, r in enumerate(rs):
            feats = r[5].strip() if len(r) > 5 else ""
            for tok in feats.split():
                w, cnt = tok.split(":")
                ri.append(i); ci.append(int(w)); data.append(float(cnt))
        x = sp.csr_matrix((data, (ri, ci)), shape=(len(rs), VOCAB), dtype=np.float32)
        t = Tree(eid, labels[eid], texts.get(eid, ""), x, parent, group=groups[eid])
        t.order = _bfs_order(t.parent)
        trees.append(t)
    trees.sort(key=lambda t: t.eid)
    missing = sum(1 for t in trees if not t.text)
    if missing:
        print(f"[data] warning: {missing} trees have no source text")
    return trees


def leakage_groups(trees: list[Tree], sim: float = 0.5) -> np.ndarray:
    """Group ids for leakage-free splitting.

    Two events share a group if (a) they carry the same claim/source-event id in the label file
    (e.g. CIKM_1000737, 'ferguson', 'whitehouse'), or (b) their source tweets are near-duplicates
    (TF-IDF cosine > sim). Every such group in Twitter15/16 is label-pure, so random splits let a
    model score by recognising the story rather than by judging veracity.
    """
    from scipy.sparse.csgraph import connected_components
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    n = len(trees)
    gid = {g: i for i, g in enumerate(sorted({t.group for t in trees}))}
    rows, cols = [], []
    first = {}
    for i, t in enumerate(trees):
        j = first.setdefault(gid[t.group], i)
        rows.append(i); cols.append(j)
    txt = [t.text.lower() for t in trees]
    X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(txt)
    S = cosine_similarity(X, dense_output=False)
    S = S.multiply(S > sim).tocoo()
    rows += S.row.tolist(); cols += S.col.tolist()
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, comp = connected_components(A, directed=False)
    return comp


def make_folds(trees: list[Tree], k: int = 5, seed: int = 42, split: str = "random"):
    """split='random' : stratified k-fold over events (protocol of BiGCN and most follow-ups).
    split='group'  : stratified *group* k-fold - no claim/story appears in both train and test."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    y = np.array([t.label for t in trees])
    if split == "random":
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(len(y)), y))
    if split == "group":
        g = leakage_groups(trees)
        sgk = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        return list(sgk.split(np.zeros(len(y)), y, g))
    raise ValueError(split)


def stats(trees: list[Tree]) -> dict:
    n = np.array([t.n for t in trees])
    d = np.array([t.depth().max() for t in trees])
    lab = np.bincount([t.label for t in trees], minlength=4)
    return {
        "events": len(trees),
        "labels": dict(zip(LABELS, lab.tolist())),
        "nodes_total": int(n.sum()),
        "nodes_mean": float(n.mean()),
        "nodes_median": float(np.median(n)),
        "depth_mean": float(d.mean()),
        "depth_max": int(d.max()),
    }


def load_replies(name: str = "Twitter15", root: str = ROOT, max_n: int | None = None) -> dict[str, list[str]]:
    """Reply texts per event (GLAN release; ~86% of Twitter15 and ~83% of Twitter16 events).
    Order is as released (not guaranteed chronological)."""
    d = os.path.join(root, "replies", name.lower())
    out = {}
    if not os.path.isdir(d):
        return out
    for f in os.listdir(d):
        with open(os.path.join(d, f), encoding="utf8", errors="ignore") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        out[f.split(".")[0]] = lines[:max_n] if max_n else lines
    return out
