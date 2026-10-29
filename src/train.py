"""Train/evaluate BiGCN, TextMLP and Fusion with stratified 5-fold CV.

Examples
  python src/train.py --dataset Twitter15 --model bigcn
  python src/train.py --dataset Twitter15 --model fusion --text tfidf
  python src/train.py --dataset Twitter16 --model fusion --text llm --text-model Qwen/Qwen2.5-7B-Instruct
  python src/train.py --dataset Twitter15 --model bigcn --early 0 5 10 20 40   # early detection
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from data import LABELS, load_dataset, make_folds  # noqa: E402
from models import BiGCN, Batch, Fusion, TextMLP  # noqa: E402
from text_features import fold_text_features  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def log1p_tf(x):
    x = x.copy()
    x.data = np.log1p(x.data)
    return x


def metrics(y, p):
    return {
        "acc": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro")),
        "f1_per_class": dict(zip(LABELS, f1_score(y, p, average=None, labels=range(4)).tolist())),
    }


def batches(trees, idx, bs, text, shuffle, rng, drop_edge=0.0, truncate=None):
    idx = np.array(idx)
    if shuffle:
        idx = rng.permutation(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        ts = [trees[k] if truncate is None else trees[k].truncate(truncate) for k in j]
        yield j, Batch(ts, drop_edge, rng, None if text is None else text[j], log1p_tf)


def predict(model, trees, idx, text, bs=128, truncate=None):
    model.eval()
    probs = []
    with torch.no_grad():
        for _, b in batches(trees, idx, bs, text, False, None, 0.0, truncate):
            probs.append(F.softmax(model(b), 1).numpy())
    return np.concatenate(probs)


def build(args, text_dim):
    if args.model == "bigcn":
        return BiGCN(hid=args.hid, out=args.hid, dropout=args.dropout)
    if args.model == "text":
        return TextMLP(text_dim)
    if args.model == "fusion":
        return Fusion(text_dim, hid=args.hid, dropout=args.dropout)
    raise ValueError(args.model)


def run_fold(args, trees, tr, te, fold, rng):
    # inner validation split for early stopping
    tr = rng.permutation(tr)
    n_val = max(1, int(0.1 * len(tr)))
    va, tr = tr[:n_val], tr[n_val:]

    text = None
    text_dim = 0
    if args.model in ("text", "fusion"):
        texts = [t.text for t in trees]
        fit_idx = np.concatenate([tr, va])
        z_fit, z_te = fold_text_features(args.text, texts, fit_idx, te, args.text_model)
        text = np.zeros((len(trees), z_fit.shape[1]), dtype=np.float32)
        text[fit_idx], text[te] = z_fit, z_te
        text_dim = text.shape[1]

    torch.manual_seed(args.seed + fold)
    model = build(args, text_dim)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best, best_state, bad = -1, None, 0
    yv = np.array([trees[i].label for i in va])
    for ep in range(args.epochs):
        model.train()
        tot = 0
        for _, b in batches(trees, tr, args.bs, text, True, rng, args.drop_edge):
            opt.zero_grad()
            loss = F.cross_entropy(model(b), b.y)
            loss.backward()
            opt.step()
            tot += loss.item() * b.num_graphs
        pv = predict(model, trees, va, text).argmax(1)
        f1 = f1_score(yv, pv, average="macro")
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if args.verbose:
            print(f"  fold {fold} ep {ep:3d} loss {tot / len(tr):.3f} val-F1 {f1:.3f}")
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)

    out = {"fold": fold, "epochs": ep + 1, "val_f1": float(best), "test_idx": te.tolist()}
    yt = np.array([trees[i].label for i in te])
    for k in (args.early or [None]):
        p = predict(model, trees, te, text, truncate=k)
        key = "full" if k is None else f"k{k}"
        out[key] = metrics(yt, p.argmax(1))
        out[key]["probs"] = p.tolist()
    if args.save_models:
        d = os.path.join(ROOT, "checkpoints", f"{args.dataset}_{args.split}_{args.model}_{args.text if text is not None else 'none'}")
        os.makedirs(d, exist_ok=True)
        torch.save({"state": best_state, "args": vars(args), "text_dim": text_dim}, os.path.join(d, f"fold{fold}.pt"))
        if text is not None:
            np.save(os.path.join(d, f"fold{fold}_text.npy"), text)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Twitter15", choices=["Twitter15", "Twitter16"])
    ap.add_argument("--model", default="bigcn", choices=["bigcn", "text", "fusion"])
    ap.add_argument("--text", default="tfidf", choices=["tfidf", "sbert", "llm"])
    ap.add_argument("--text-model", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--split", default="group", choices=["random", "group"],
                    help="group = leakage-free (no shared claim/story between train and test)")
    ap.add_argument("--only-fold", type=int, default=None, help="run a single fold (quick test)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--drop-edge", type=float, default=0.2)
    ap.add_argument("--early", type=int, nargs="*", default=None,
                    help="also test with only the first k responses (0 = source tweet only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.early is not None and None not in args.early:
        args.early = [None] + args.early

    trees = load_dataset(args.dataset)
    folds = make_folds(trees, args.folds, args.seed, args.split)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    results = []
    for f, (tr, te) in enumerate(folds):
        if args.only_fold is not None and f != args.only_fold:
            continue
        r = run_fold(args, trees, tr, te, f, rng)
        results.append(r)
        print(f"[{args.dataset}/{args.model}] fold {f}: acc {r['full']['acc']:.3f}  macro-F1 {r['full']['macro_f1']:.3f}"
              f"  ({r['epochs']} ep, {time.time() - t0:.0f}s)", flush=True)

    keys = [k for k in results[0] if k == "full" or (k.startswith("k") and k[1:].isdigit())]
    summary = {}
    for k in keys:
        summary[k] = {m: [float(np.mean([r[k][m] for r in results])), float(np.std([r[k][m] for r in results]))]
                      for m in ("acc", "macro_f1")}
        summary[k]["f1_per_class"] = {l: float(np.mean([r[k]["f1_per_class"][l] for r in results])) for l in LABELS}
    name = f"{args.dataset}_{args.split}_{args.model}" + (f"_{args.text}" if args.model != "bigcn" else "") + (f"_{args.tag}" if args.tag else "")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", name + ".json"), "w") as fh:
        json.dump({"args": vars(args), "summary": summary, "folds": results}, fh)
    for k, s in summary.items():
        print(f"== {name} [{k}] acc {s['acc'][0]:.3f}±{s['acc'][1]:.3f}  macro-F1 {s['macro_f1'][0]:.3f}±{s['macro_f1'][1]:.3f}")


if __name__ == "__main__":
    main()
