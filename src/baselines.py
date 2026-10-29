"""Classical baselines on the same 5 folds.

  text_lr    : TF-IDF (word+char) of the source tweet -> logistic regression
  struct_rf  : hand-crafted propagation-structure features -> random forest
  cascade_lr : bag-of-words summed over the whole cascade (source + replies) -> logistic regression

  python src/baselines.py --dataset Twitter15
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from data import LABELS, Tree, load_dataset, make_folds  # noqa: E402
from text_features import clean  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def struct_features(t: Tree, k: int | None = None) -> np.ndarray:
    t = t.truncate(k) if k is not None else t
    d = t.depth()
    n = t.n
    outdeg = np.bincount(t.parent[t.parent >= 0], minlength=n)
    width = np.bincount(d)
    leaves = (outdeg == 0).sum()
    return np.array([
        np.log1p(n), d.max(), d.mean() if n > 1 else 0, np.log1p(outdeg[0]), outdeg[0] / max(n - 1, 1),
        width.max(), np.argmax(width), leaves / n, outdeg[1:].mean() if n > 1 else 0,
        np.log1p(outdeg[1:].max()) if n > 1 else 0, (d >= 2).mean(), (d >= 3).mean(),
        # Wiener-like spread: mean depth weighted by subtree
        np.std(d), np.log1p((outdeg > 1).sum()),
    ], dtype=np.float32)


def run(dataset: str, seed: int = 42, split: str = "group"):
    trees = load_dataset(dataset)
    y = np.array([t.label for t in trees])
    texts = [clean(t.text) for t in trees]
    S = np.stack([struct_features(t) for t in trees])
    C = sp.vstack([sp.csr_matrix(t.x.sum(0)) for t in trees]).tocsr()

    res = {m: [] for m in ("text_lr", "struct_rf", "cascade_lr", "text+struct_lr")}
    for f, (tr, te) in enumerate(make_folds(trees, 5, seed, split)):
        wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
        Xtr = sp.hstack([wv.fit_transform([texts[i] for i in tr]), cv.fit_transform([texts[i] for i in tr])]).tocsr()
        Xte = sp.hstack([wv.transform([texts[i] for i in te]), cv.transform([texts[i] for i in te])]).tocsr()
        lr = LogisticRegression(C=10, max_iter=3000).fit(Xtr, y[tr])
        res["text_lr"].append(lr.predict(Xte))

        rf = RandomForestClassifier(500, min_samples_leaf=2, random_state=seed, n_jobs=-1).fit(S[tr], y[tr])
        res["struct_rf"].append(rf.predict(S[te]))

        tf = TfidfTransformer(sublinear_tf=True)
        lr2 = LogisticRegression(C=5, max_iter=3000).fit(tf.fit_transform(C[tr]), y[tr])
        res["cascade_lr"].append(lr2.predict(tf.transform(C[te])))

        mu, sd = S[tr].mean(0), S[tr].std(0) + 1e-6
        Xtr2 = sp.hstack([Xtr, sp.csr_matrix((S[tr] - mu) / sd)]).tocsr()
        Xte2 = sp.hstack([Xte, sp.csr_matrix((S[te] - mu) / sd)]).tocsr()
        lr3 = LogisticRegression(C=10, max_iter=3000).fit(Xtr2, y[tr])
        res["text+struct_lr"].append(lr3.predict(Xte2))
        res.setdefault("_te", []).append(te)

    out = {}
    for m, preds in res.items():
        if m == "_te":
            continue
        accs = [accuracy_score(y[te], p) for te, p in zip(res["_te"], preds)]
        f1s = [f1_score(y[te], p, average="macro") for te, p in zip(res["_te"], preds)]
        pc = np.mean([f1_score(y[te], p, average=None, labels=range(4)) for te, p in zip(res["_te"], preds)], 0)
        out[m] = {"full": {"acc": [float(np.mean(accs)), float(np.std(accs))],
                           "macro_f1": [float(np.mean(f1s)), float(np.std(f1s))],
                           "f1_per_class": dict(zip(LABELS, pc.tolist()))}}
        print(f"== {dataset}_{split}_{m}: acc {np.mean(accs):.3f}±{np.std(accs):.3f}  macro-F1 {np.mean(f1s):.3f}±{np.std(f1s):.3f}")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    for m, s in out.items():
        with open(os.path.join(ROOT, "results", f"{dataset}_{split}_{m}.json"), "w") as fh:
            json.dump({"summary": s}, fh)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Twitter15")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="group", choices=["random", "group"])
    a = ap.parse_args()
    run(a.dataset, a.seed, a.split)
