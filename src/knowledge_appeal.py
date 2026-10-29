"""How often do the LLM's explanations appeal to prior knowledge (fact-checks, news coverage)?

Counts explanations matching a fixed keyword pattern and compares LLM accuracy on events with and
without such an appeal (evidence-grounded condition, 200-event samples). Writes
results/knowledge_appeal_<model>.json.

  python src/knowledge_appeal.py --model gemma3_12b
"""
import argparse
import json
import os
import re

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = os.path.join(ROOT, "results")
PATTERN = re.compile(r"widely (reported|documented)|fact-check|debunk|confirmed by|news (outlets|sources|reports) "
                     r"(confirm|report)|demonstrably|well-documented|numerous (news|reliable)", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3_12b")
    a = ap.parse_args()
    out = {"pattern": PATTERN.pattern}
    for ds in ("Twitter15", "Twitter16"):
        E = json.load(open(os.path.join(R, f"explain_{ds}_group_bigcn_none_{a.model}.json")))["explanations"]
        k = [bool(PATTERN.search(e["explanation"] or "")) for e in E]
        ok = [e["llm"] == e["label"] for e in E]
        with_k = [o for o, kk in zip(ok, k) if kk]
        without = [o for o, kk in zip(ok, k) if not kk]
        prop = sum(("propagat" in (e["explanation"] or "").lower() or "cascade" in (e["explanation"] or "").lower()) for e in E)
        out[ds] = {"n": len(E), "appeal_n": sum(k), "appeal_frac": sum(k) / len(E),
                   "acc_with_appeal": sum(with_k) / len(with_k), "acc_without_appeal": sum(without) / len(without),
                   "mentions_propagation_frac": prop / len(E)}
        print(ds, json.dumps(out[ds]))
    json.dump(out, open(os.path.join(R, f"knowledge_appeal_{a.model}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
