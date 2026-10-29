"""LLM step: zero-shot veracity classification and grounded explanations.

Backends (no extra Python deps except for `hf`):
  --backend openai  any OpenAI-compatible server: Ollama (http://localhost:11434/v1), vLLM
                    (http://localhost:8000/v1), LM Studio, or a hosted API (set OPENAI_API_KEY).
  --backend hf      local transformers pipeline (GPU recommended).
  --backend dry     only write the prompts to results/llm_prompts_*.jsonl (to inspect them).

Modes
  classify : LLM alone. --context source | source+replies
  explain  : LLM is given the GNN's verdict + the salient propagation evidence from
             src/explain.py and the replies, and must return a verdict and a grounded explanation.
             Also reports a late-fusion "hybrid" (average of GNN and LLM probabilities).

Examples
  ollama pull qwen2.5:7b-instruct
  python src/llm_explain.py classify --dataset Twitter16 --context source --model qwen2.5:7b-instruct
  python src/llm_explain.py classify --dataset Twitter16 --context source+replies --model qwen2.5:7b-instruct
  python src/llm_explain.py explain --run Twitter16_group_bigcn_none --model qwen2.5:7b-instruct

Caveat to report in the paper: Twitter15/16 events are from 2013-2016 and fact-checked on
Snopes, so an LLM may have memorised some verdicts. The `--limit` option lets you
compare against a recent dataset (e.g. XFacta) to quantify this.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from data import LABELS, load_dataset, load_replies, make_folds  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")

SYSTEM = (
    "You are a careful fact-checking analyst. You judge social-media posts about news events. "
    "Use these four labels exactly:\n"
    "- non-rumor: an ordinary news report from a reliable source, not a rumour\n"
    "- false: a rumour whose claim is false\n"
    "- true: a rumour whose claim turned out to be true\n"
    "- unverified: a rumour that cannot be confirmed or refuted\n"
    "Respond with JSON only."
)

CLASSIFY = """Post: "{text}"
{replies_block}
Classify the post. Return JSON:
{{"label": "<non-rumor|false|true|unverified>",
  "probs": {{"non-rumor": p, "false": p, "true": p, "unverified": p}},
  "rationale": "<2-3 sentences>"}}"""

EXPLAIN = """Post: "{text}"

A graph neural network analysed how this post spread on Twitter.
- Cascade: {n_nodes} posts, {direct} direct replies, maximum reply depth {depth}.
- GNN verdict: {pred} (probabilities: {probs}).
- Most influential parts of the cascade for that verdict (by gradient saliency; checked for
  faithfulness by deletion tests):
{evidence}
{replies_block}
Task: give your own final verdict and explain it. Ground the explanation in (1) the claim itself,
(2) the propagation evidence above and (3) what repliers say (e.g. questioning, denial, support).
Say explicitly whether you agree with the GNN and why. Return JSON:
{{"label": "<non-rumor|false|true|unverified>",
  "probs": {{"non-rumor": p, "false": p, "true": p, "unverified": p}},
  "agrees_with_gnn": true|false,
  "explanation": "<3-5 sentences>",
  "evidence_used": ["claim"|"propagation"|"replies", ...]}}"""


def replies_block(replies, n):
    if not replies or n == 0:
        return ""
    rs = [re.sub(r"@\w+", "@user", re.sub(r"\s+", " ", r))[:200] for r in replies[:n]]  # mask handles
    return "\nReplies (first {}):\n".format(len(rs)) + "\n".join(f"- {r}" for r in rs) + "\n"


def evidence_lines(ev):
    out = []
    for e in ev:
        kind = "direct reply" if e["is_direct_reply"] else f"reply at depth {e['depth']}"
        out.append(f"  * a {kind}, arrived #{e['arrival_rank']} in the cascade, itself drew "
                   f"{e['replies_received']} further replies")
    return "\n".join(out) if out else "  * (cascade too small for node-level evidence)"


# ------------------------------------------------------------------ backends
class Backend:
    def __init__(self, a):
        self.a = a
        if a.backend == "hf":
            from transformers import pipeline
            self.pipe = pipeline("text-generation", model=a.model, device_map="auto", torch_dtype="auto")

    def __call__(self, user: str) -> str:
        a = self.a
        if a.backend == "dry":
            return ""
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        if a.backend == "hf":
            out = self.pipe(msgs, max_new_tokens=a.max_tokens, do_sample=False)
            return out[0]["generated_text"][-1]["content"]
        body = {"model": a.model, "messages": msgs, "temperature": 0, "max_tokens": a.max_tokens}
        if a.json_mode:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(a.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "none")})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    return json.loads(r.read())["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
                print("  retry:", e)


def parse(resp: str):
    """Robustly pull label + probability vector out of an LLM response."""
    m = re.search(r"\{.*\}", resp or "", re.S)
    obj = {}
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = {}
    lab = str(obj.get("label", "")).lower().replace("_", "-").replace(" ", "-")
    if lab in ("non-rumour", "nonrumor", "not-rumor"):
        lab = "non-rumor"
    if lab not in LABELS:  # fall back to the first label mentioned in the text
        hits = [(resp.lower().find(l), l) for l in LABELS if l in (resp or "").lower()]
        lab = min(hits)[1] if hits else "unverified"
    p = np.full(4, 1e-3)
    probs = obj.get("probs") or {}
    for i, l in enumerate(LABELS):
        try:
            p[i] = max(float(probs.get(l, probs.get(l.replace("-", ""), 0))), 1e-3)
        except (TypeError, ValueError):
            pass
    if p.sum() < 0.05 or np.argmax(p) != LABELS.index(lab):
        p = np.full(4, 0.1)
        p[LABELS.index(lab)] = 0.7
    return lab, p / p.sum(), obj


def balanced_sample(labels, n, seed=0):
    """Indices of a class-balanced random sample of size ~n (for pilot runs)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    per = max(1, n // 4)
    idx = np.concatenate([rng.permutation(np.where(labels == c)[0])[:per] for c in range(4)])
    return np.sort(idx)


def score(y, pred):
    return {"acc": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=range(4), zero_division=0)),
            "f1_per_class": dict(zip(LABELS, f1_score(y, pred, average=None, labels=range(4),
                                                       zero_division=0).tolist())),
            "pred_counts": dict(zip(LABELS, np.bincount(pred, minlength=4).tolist()))}


def fold_avg(y, pred, folds):
    a = [accuracy_score(y[te], pred[te]) for _, te in folds]
    f = [f1_score(y[te], pred[te], average="macro", labels=range(4), zero_division=0) for _, te in folds]
    return {"acc": [float(np.mean(a)), float(np.std(a))], "macro_f1": [float(np.mean(f)), float(np.std(f))]}


def run_items(a, items, tag):
    """items: list of (key, prompt). Resumable: answers are cached in a jsonl file."""
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    if a.backend == "dry":
        p = os.path.join(ROOT, "results", f"llm_prompts_{tag}.jsonl")
        with open(p, "w") as fh:
            for k, pr in items:
                fh.write(json.dumps({"key": k, "system": SYSTEM, "user": pr}) + "\n")
        print(f"[dry] wrote {len(items)} prompts to {p}")
        return None
    cache_p = os.path.join(ROOT, "results", f"llm_cache_{tag}_{a.model.replace('/', '_').replace(':', '_')}.jsonl")
    done = {}
    if os.path.exists(cache_p):
        for line in open(cache_p):
            r = json.loads(line)
            done[r["key"]] = r["response"]
    llm = Backend(a)
    with open(cache_p, "a") as fh:
        for i, (k, pr) in enumerate(items):
            if k in done:
                continue
            done[k] = llm(pr)
            fh.write(json.dumps({"key": k, "response": done[k]}) + "\n")
            fh.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(items)}", flush=True)
    return done


def cmd_classify(a):
    trees = load_dataset(a.dataset)
    if a.limit:  # class-balanced random pilot sample
        trees = [trees[i] for i in balanced_sample([t.label for t in trees], a.limit)]
    reps = load_replies(a.dataset) if a.context == "source+replies" else {}
    items = [(t.eid, CLASSIFY.format(text=t.text, replies_block=replies_block(reps.get(t.eid), a.n_replies)))
             for t in trees]
    tag = f"{a.dataset}_classify_{a.context.replace('+', '_')}"
    resp = run_items(a, items, tag)
    if resp is None:
        return
    y = np.array([t.label for t in trees])
    parsed = [parse(resp[t.eid]) for t in trees]
    pred = np.array([LABELS.index(p[0]) for p in parsed])
    out = {"args": vars(a), "overall": score(y, pred)}
    if not a.limit:
        for split in ("random", "group"):
            out[split] = fold_avg(y, pred, make_folds(trees, 5, 42, split))
    out["predictions"] = [{"eid": t.eid, "label": LABELS[t.label], "pred": p[0], "probs": p[1].tolist(),
                           "rationale": p[2].get("rationale", "")} for t, p in zip(trees, parsed)]
    name = f"{tag}_{a.model.replace('/', '_').replace(':', '_')}"
    json.dump(out, open(os.path.join(ROOT, "results", name + ".json"), "w"), indent=1)
    print(f"== {name}: acc {out['overall']['acc']:.3f} macro-F1 {out['overall']['macro_f1']:.3f}")
    print("   predicted:", out["overall"]["pred_counts"], " true:", dict(zip(LABELS, np.bincount(y, minlength=4).tolist())))


def cmd_explain(a):
    ex = json.load(open(os.path.join(ROOT, "results", f"explain_{a.run}.json")))
    events = ex["events"]
    if a.limit:  # same balanced sample (same event ids) as `classify --limit`
        trees = load_dataset(a.run.split("_")[0])
        keep = {trees[i].eid for i in balanced_sample([t.label for t in trees], a.limit)}
        events = sorted([e for e in events if e["eid"] in keep], key=lambda e: e["eid"])
    ds = a.run.split("_")[0]
    reps = load_replies(ds)
    items = []
    for e in events:
        probs = ", ".join(f"{k} {v:.2f}" for k, v in e["probs"].items())
        items.append((e["eid"], EXPLAIN.format(text=e["text"], n_nodes=e["n_nodes"], direct=e["direct_replies"],
                                               depth=e["max_depth"], pred=e["pred"], probs=probs,
                                               evidence=evidence_lines(e["evidence"]),
                                               replies_block=replies_block(reps.get(e["eid"]), a.n_replies))))
    tag = f"explain_{a.run}"
    resp = run_items(a, items, tag)
    if resp is None:
        return
    y = np.array([LABELS.index(e["label"]) for e in events])
    gnn_p = np.array([[e["probs"][l] for l in LABELS] for e in events])
    parsed = [parse(resp[e["eid"]]) for e in events]
    llm_p = np.stack([p[1] for p in parsed])
    hybrid = a.alpha * gnn_p + (1 - a.alpha) * llm_p
    agree = np.mean([p[2].get("agrees_with_gnn") is True for p in parsed])
    out = {"args": vars(a), "gnn": score(y, gnn_p.argmax(1)), "llm_with_evidence": score(y, llm_p.argmax(1)),
           f"hybrid_alpha{a.alpha}": score(y, hybrid.argmax(1)), "llm_self_reported_agreement": float(agree),
           "llm_actual_agreement": float(np.mean(llm_p.argmax(1) == gnn_p.argmax(1))),
           "explanations": [{"eid": e["eid"], "label": e["label"], "gnn": e["pred"], "llm": p[0],
                             "explanation": p[2].get("explanation", ""), "evidence_used": p[2].get("evidence_used", [])}
                            for e, p in zip(events, parsed)]}
    name = f"{tag}_{a.model.replace('/', '_').replace(':', '_')}"
    json.dump(out, open(os.path.join(ROOT, "results", name + ".json"), "w"), indent=1)
    for k in ("gnn", "llm_with_evidence", f"hybrid_alpha{a.alpha}"):
        print(f"== {name} [{k}] acc {out[k]['acc']:.3f} macro-F1 {out[k]['macro_f1']:.3f}")
    print(f"   agreement: self-reported {agree:.2f}, actual {out['llm_actual_agreement']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("classify", "explain"):
        s = sub.add_parser(name)
        s.add_argument("--backend", default="openai", choices=["openai", "hf", "dry"])
        s.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"))
        s.add_argument("--model", default="qwen2.5:7b-instruct")
        s.add_argument("--max-tokens", type=int, default=400)
        s.add_argument("--json-mode", action="store_true", help="send response_format=json_object")
        s.add_argument("--n-replies", type=int, default=10)
        s.add_argument("--limit", type=int, default=None, help="class-balanced random sample of N events (pilot runs)")
        if name == "classify":
            s.add_argument("--dataset", default="Twitter15", choices=["Twitter15", "Twitter16"])
            s.add_argument("--context", default="source", choices=["source", "source+replies"])
        else:
            s.add_argument("--run", required=True, help="e.g. Twitter16_group_bigcn_none")
            s.add_argument("--alpha", type=float, default=0.5, help="GNN weight in the hybrid")
    a = ap.parse_args()
    {"classify": cmd_classify, "explain": cmd_explain}[a.cmd](a)


if __name__ == "__main__":
    main()
