"""Statistics for the LLM stage (Table 4 of the paper).

For each dataset, on the fixed class-balanced sample used by every LLM condition:
  * macro-F1 with 95% bootstrap CI (2,000 resamples) for every method
  * paired bootstrap CI of the macro-F1 difference + exact McNemar test for key comparisons
  * per-class F1 and confusion matrices
  * label-ambiguity analysis: macro-F1 after merging 'true' and 'non-rumor' into one class
    (both describe real events; they differ by source type, not truth)
Also scores the cascade-words + LR baseline (grouped split, out-of-fold) on the same events.

  python src/llm_stats.py --model gemma3_12b
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from data import LABELS, load_dataset, make_folds  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results")


def mf1(y, p, labels=range(4)):
    return f1_score(y, p, average="macro", labels=list(labels), zero_division=0)


def cascade_lr_oof(dataset):
    """Out-of-fold predictions of the cascade-words + LR baseline on the grouped split."""
    trees = load_dataset(dataset)
    y = np.array([t.label for t in trees])
    C = sp.vstack([sp.csr_matrix(t.x.sum(0)) for t in trees]).tocsr()
    pred = np.zeros(len(trees), dtype=int)
    for tr, te in make_folds(trees, 5, 42, "group"):
        tf = TfidfTransformer(sublinear_tf=True)
        lr = LogisticRegression(C=5, max_iter=3000).fit(tf.fit_transform(C[tr]), y[tr])
        pred[te] = lr.predict(tf.transform(C[te]))
    return {t.eid: int(p) for t, p in zip(trees, pred)}


def analyse(dataset, model, B=2000, seed=0):
    ex = json.load(open(os.path.join(R, f"explain_{dataset}_group_bigcn_none_{model}.json")))
    E = ex["explanations"]
    eids = [e["eid"] for e in E]
    y = np.array([LABELS.index(e["label"]) for e in E])
    preds = {"BiGCN (grouped, out-of-fold)": [LABELS.index(e["gnn"]) for e in E],
             "LLM + propagation evidence": [LABELS.index(e["llm"]) for e in E]}
    for ctx, name in (("source", "LLM zero-shot, source"), ("source_replies", "LLM zero-shot, source + replies")):
        p = os.path.join(R, f"{dataset}_classify_{ctx}_{model}.json")
        if os.path.exists(p):
            d = {q["eid"]: LABELS.index(q["pred"]) for q in json.load(open(p))["predictions"]}
            assert set(d) == set(eids), f"{p}: different event sample"
            preds[name] = [d[e] for e in eids]
    cl = cascade_lr_oof(dataset)
    preds["Cascade words + LR (grouped, out-of-fold)"] = [cl[e] for e in eids]
    preds = {k: np.array(v) for k, v in preds.items()}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (B, len(y)))
    merge = np.array([0, 1, 0, 3])          # non-rumor & true -> "real event"
    out = {"n": int(len(y)), "methods": {}, "comparisons": {}}
    for k, p in preds.items():
        bs = np.array([mf1(y[i], p[i]) for i in idx])
        out["methods"][k] = {
            "acc": float((p == y).mean()), "macro_f1": float(mf1(y, p)),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "f1_per_class": dict(zip(LABELS, f1_score(y, p, average=None, labels=range(4), zero_division=0).round(3).tolist())),
            "macro_f1_merged_true_nonrumor": float(mf1(merge[y], merge[p], labels=[0, 1, 3])),
            "confusion": confusion_matrix(y, p, labels=range(4)).tolist(),
        }
    pairs = [("LLM + propagation evidence", "BiGCN (grouped, out-of-fold)"),
             ("LLM + propagation evidence", "LLM zero-shot, source"),
             ("LLM + propagation evidence", "Cascade words + LR (grouped, out-of-fold)"),
             ("LLM zero-shot, source + replies", "LLM zero-shot, source")]
    for a, b in pairs:
        if a not in preds or b not in preds:
            continue
        pa, pb = preds[a], preds[b]
        d = np.array([mf1(y[i], pa[i]) - mf1(y[i], pb[i]) for i in idx])
        ca, cb = pa == y, pb == y
        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        out["comparisons"][f"{a} vs {b}"] = {
            "delta_macro_f1": float(mf1(y, pa) - mf1(y, pb)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "mcnemar_only_a_correct": n01, "mcnemar_only_b_correct": n10,
            "mcnemar_p": float(binomtest(n01, n01 + n10).pvalue) if n01 + n10 else 1.0}
    g, l = preds["BiGCN (grouped, out-of-fold)"], preds["LLM + propagation evidence"]
    dis = g != l
    out["disagreement"] = {"n": int(dis.sum()), "llm_right": int((l[dis] == y[dis]).sum()),
                           "gnn_right": int((g[dis] == y[dis]).sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3_12b")
    a = ap.parse_args()
    res = {}
    for ds in ("Twitter15", "Twitter16"):
        if not os.path.exists(os.path.join(R, f"explain_{ds}_group_bigcn_none_{a.model}.json")):
            continue
        r = res[ds] = analyse(ds, a.model)
        print(f"\n===== {ds} (n = {r['n']})")
        for k, m in r["methods"].items():
            print(f"{k:45s} acc {m['acc']:.3f}  macro-F1 {m['macro_f1']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]"
                  f"  merged T+NR {m['macro_f1_merged_true_nonrumor']:.3f}  per-class {list(m['f1_per_class'].values())}")
        for k, c in r["comparisons"].items():
            print(f"  {k}: dF1 {c['delta_macro_f1']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}]"
                  f"  McNemar {c['mcnemar_only_a_correct']} vs {c['mcnemar_only_b_correct']}, p = {c['mcnemar_p']:.3f}")
        print("  disagreements:", r["disagreement"])
    json.dump(res, open(os.path.join(R, f"llm_stats_{a.model}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
