"""Collect results/*.json into paper-ready tables (results/tables.md)."""
import glob
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results")

ORDER = [("struct_rf", "Structure features + RF"), ("text_lr", "Source text TF-IDF + LR"),
         ("text_tfidf", "Source text TF-IDF + MLP"), ("cascade_lr", "Cascade words (source+replies) + LR"),
         ("text+struct_lr", "Source text + structure + LR"), ("bigcn", "BiGCN (re-impl.)"),
         ("fusion_tfidf", "BiGCN + text (TF-IDF), gated fusion"), ("text_sbert", "Source text SBERT + MLP"),
         ("fusion_sbert", "BiGCN + text (SBERT), gated fusion"), ("text_llm", "Source text LLM-emb + MLP"),
         ("fusion_llm", "BiGCN + text (LLM-emb), gated fusion")]


def load(ds, split, key):
    p = os.path.join(R, f"{ds}_{split}_{key}.json")
    return json.load(open(p))["summary"] if os.path.exists(p) else None


def fmt(s, m):
    return f"{s[m][0]:.3f} ± {s[m][1]:.3f}"


lines = ["# Results (5-fold CV, mean ± std)\n",
         "*random* = stratified event-level folds (BiGCN protocol). *group* = leakage-free: no claim/story "
         "group in both train and test. Chance macro-F1 = 0.25.\n"]
for metric, title in (("macro_f1", "Macro-F1"), ("acc", "Accuracy")):
    lines += [f"\n## {title}\n", "| Model | T15 random | T15 group | T16 random | T16 group |", "|---|---|---|---|---|"]
    for key, name in ORDER:
        cells = []
        for ds in ("Twitter15", "Twitter16"):
            for sp in ("random", "group"):
                s = load(ds, sp, key)
                cells.append(fmt(s["full"], metric) if s else "–")
        if any(c != "–" for c in cells):
            lines.append(f"| {name} | " + " | ".join(cells) + " |")

# LLM runs (fixed balanced sample; no training, so there is no train/test leakage to split around)
rows = []
for p in sorted(glob.glob(os.path.join(R, "*_classify_*.json"))):
    o = json.load(open(p))
    n = sum(o["overall"].get("pred_counts", {}).values()) or "all"
    rows.append((os.path.basename(p)[:-5], n, o["overall"]))
for p in sorted(glob.glob(os.path.join(R, "explain_*_group_bigcn_none_*.json"))):
    o = json.load(open(p))
    base = os.path.basename(p)[:-5]
    n = sum(o["gnn"].get("pred_counts", {}).values()) or "all"
    for k in ("gnn", "llm_with_evidence") + tuple(k for k in o if k.startswith("hybrid")):
        rows.append((f"{base} [{k}]", n, o[k]))
if rows:
    lines += ["\n## LLM stage (same class-balanced events in every row of a dataset)\n",
              "| Run | n | Accuracy | Macro-F1 |", "|---|---|---|---|"]
    for name, n, s_ in rows:
        lines.append(f"| {name} | {n} | {s_['acc']:.3f} | {s_['macro_f1']:.3f} |")

# early detection
lines += ["\n## Early detection (BiGCN, macro-F1 with only the first k responses at test time)\n",
          "| Dataset / split | k=0 | k=5 | k=10 | k=20 | full |", "|---|---|---|---|---|---|"]
for ds in ("Twitter15", "Twitter16"):
    for sp in ("random", "group"):
        s = load(ds, sp, "bigcn")
        if s and "k0" in s:
            lines.append(f"| {ds} {sp} | " + " | ".join(f"{s[k]['macro_f1'][0]:.3f}" for k in ("k0", "k5", "k10", "k20", "full")) + " |")

# explanation faithfulness
for p in sorted(glob.glob(os.path.join(R, "explain_*.json"))):
    if "_classify_" in p or p.endswith(".jsonl"):
        continue
    o = json.load(open(p))
    if "faithfulness" not in o:
        continue
    lines += [f"\n## Explanation faithfulness – {os.path.basename(p)[8:-5]}\n",
              "| Nodes | Comprehensiveness ↑ (saliency) | (random) | Sufficiency ↓ (saliency) | (random) | n |",
              "|---|---|---|---|---|---|"]
    for k, v in o["faithfulness"].items():
        lines.append(f"| {k} | {v['comp']:.3f} | {v['comp_rand']:.3f} | {v['suff']:.3f} | {v['suff_rand']:.3f} | {v['n']} |")

open(os.path.join(R, "tables.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
