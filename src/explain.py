"""Node-level explanations for BiGCN / Fusion, plus faithfulness tests.

For each test event:
  * saliency  : gradient of the predicted-class logit w.r.t. a per-node mask (gradient x input)
  * evidence  : the top-k nodes with structural descriptors (depth, replies received, arrival rank)
                -> this is what the LLM explainer is given (src/llm_explain.py)
Faithfulness (DeYoung et al., ERASER 2020), averaged over test events:
  * comprehensiveness@k : p(y_hat | G) - p(y_hat | G without the top-k% nodes)      (higher = better)
  * sufficiency@k       : p(y_hat | G) - p(y_hat | only top-k% nodes + their ancestors) (lower = better)
  each compared against removing / keeping the same number of *random* nodes.

  python src/train.py --dataset Twitter15 --split group --model bigcn --save-models
  python src/explain.py --run Twitter15_group_bigcn_none
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from data import LABELS, Tree, load_dataset, make_folds  # noqa: E402
from models import Batch  # noqa: E402
from train import build, log1p_tf  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def keep_only(t: Tree, nodes) -> Tree:
    """Keep the given nodes plus their ancestors (so the tree stays connected)."""
    keep = {0}
    for v in nodes:
        v = int(v)
        while v >= 0 and v not in keep:
            keep.add(v)
            v = int(t.parent[v])
    return _keep(t, keep)


def _keep(t: Tree, keep: set) -> Tree:
    order = [i for i in t.order if i in keep]
    remap = -np.ones(t.n, dtype=int)
    remap[order] = np.arange(len(order))
    par = np.array([-1 if t.parent[i] < 0 else remap[t.parent[i]] for i in order])
    out = Tree(t.eid, t.label, t.text, t.x[order], par, group=t.group)
    out.order = np.arange(len(order))
    return out


def prob(model, t: Tree, text_vec):
    b = Batch([t], text_emb=None if text_vec is None else text_vec[None], x_transform=log1p_tf)
    with torch.no_grad():
        return F.softmax(model(b), 1)[0].numpy()


def saliency(model, t: Tree, text_vec):
    b = Batch([t], text_emb=None if text_vec is None else text_vec[None], x_transform=log1p_tf)
    mask = torch.ones(t.n, 1, requires_grad=True)
    logits = model(b, node_mask=mask)
    c = int(logits.argmax(1))
    logits[0, c].backward()
    s = mask.grad.squeeze(1).numpy().copy()       # first-order effect of keeping each node
    s[0] = -np.inf                                 # the source tweet is always kept
    return s, c


def describe(t: Tree, nodes, scores):
    d = t.depth()
    kids = np.bincount(t.parent[t.parent >= 0], minlength=t.n)
    rank = np.empty(t.n, dtype=int)
    rank[t.order] = np.arange(t.n)
    return [{"node": int(v), "score": float(scores[v]), "depth": int(d[v]), "replies_received": int(kids[v]),
             "arrival_rank": int(rank[v]), "is_direct_reply": bool(t.parent[v] == 0),
             "n_tokens": int(t.x[v].sum())} for v in nodes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="checkpoint folder name under checkpoints/")
    ap.add_argument("--topk", type=float, nargs="*", default=[0.1, 0.2, 0.3])
    ap.add_argument("--evidence", type=int, default=5, help="nodes passed to the LLM explainer")
    ap.add_argument("--min-nodes", type=int, default=6, help="skip tiny trees in faithfulness stats")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cdir = os.path.join(ROOT, "checkpoints", a.run)
    ck0 = torch.load(os.path.join(cdir, "fold0.pt"), weights_only=False)
    args = argparse.Namespace(**ck0["args"])
    trees = load_dataset(args.dataset)
    folds = make_folds(trees, args.folds, args.seed, args.split)
    rng = np.random.default_rng(a.seed)

    faith = {k: {"comp": [], "comp_rand": [], "suff": [], "suff_rand": []} for k in a.topk}
    records = []
    for f, (_, te) in enumerate(folds):
        p = os.path.join(cdir, f"fold{f}.pt")
        if not os.path.exists(p):
            continue
        ck = torch.load(p, weights_only=False)
        model = build(args, ck["text_dim"])
        model.load_state_dict(ck["state"])
        model.eval()
        text = np.load(os.path.join(cdir, f"fold{f}_text.npy")) if ck["text_dim"] else None
        for i in te:
            t = trees[i]
            tv = None if text is None else text[i]
            s, c = saliency(model, t, tv)
            p_full = prob(model, t, tv)
            ranked = [int(v) for v in np.argsort(-s) if v != 0]
            rec = {"eid": t.eid, "group": t.group, "label": LABELS[t.label], "pred": LABELS[c],
                   "probs": dict(zip(LABELS, map(float, p_full))), "n_nodes": t.n,
                   "max_depth": int(t.depth().max()),
                   "direct_replies": int((t.parent == 0).sum()),
                   "evidence": describe(t, ranked[: a.evidence], s), "text": t.text}
            if hasattr(model, "last_gate") and model.last_gate is not None:
                rec["structure_gate"] = float(model.last_gate[0])
            records.append(rec)
            if t.n - 1 < a.min_nodes:
                continue
            for k in a.topk:
                m = max(1, int(round(k * (t.n - 1))))
                top = ranked[:m]
                rnd = rng.choice(np.arange(1, t.n), m, replace=False)
                faith[k]["comp"].append(p_full[c] - prob(model, t.drop_nodes(top), tv)[c])
                faith[k]["comp_rand"].append(p_full[c] - prob(model, t.drop_nodes(rnd), tv)[c])
                faith[k]["suff"].append(p_full[c] - prob(model, keep_only(t, top), tv)[c])
                faith[k]["suff_rand"].append(p_full[c] - prob(model, keep_only(t, rnd), tv)[c])
        print(f"fold {f} done ({len(records)} events)", flush=True)

    summary = {f"top{int(k * 100)}%": {m: float(np.mean(v)) for m, v in d.items()} | {"n": len(d["comp"])}
               for k, d in faith.items()}
    out = os.path.join(ROOT, "results", f"explain_{a.run}.json")
    with open(out, "w") as fh:
        json.dump({"faithfulness": summary, "events": records}, fh)
    print(json.dumps(summary, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
