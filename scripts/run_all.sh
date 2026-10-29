#!/usr/bin/env bash
# Stage 1 (CPU, ~15 min): baselines, BiGCN, fusion, early detection, explanation faithfulness.
# Stage 2 (GPU / Ollama): uncomment the blocks at the bottom.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=$(command -v python3 || command -v python)

for DS in Twitter15 Twitter16; do
  for SPLIT in random group; do
    $PY src/baselines.py --dataset $DS --split $SPLIT
    $PY src/train.py --dataset $DS --split $SPLIT --model bigcn --early 0 5 10 20 --save-models
    $PY src/train.py --dataset $DS --split $SPLIT --model text   --text tfidf
    $PY src/train.py --dataset $DS --split $SPLIT --model fusion --text tfidf --save-models
  done
  $PY src/explain.py --run ${DS}_group_bigcn_none
done

# ---------------- Stage 2: stronger text encoders (GPU) ----------------
# for DS in Twitter15 Twitter16; do for SPLIT in random group; do
#   $PY src/train.py --dataset $DS --split $SPLIT --model text   --text sbert
#   $PY src/train.py --dataset $DS --split $SPLIT --model fusion --text sbert
#   $PY src/train.py --dataset $DS --split $SPLIT --model text   --text llm --text-model Qwen/Qwen2.5-7B-Instruct
#   $PY src/train.py --dataset $DS --split $SPLIT --model fusion --text llm --text-model Qwen/Qwen2.5-7B-Instruct --save-models
# done; done

# ---------------- Stage 2: LLM zero-shot + grounded explanations (Ollama) ----------------
# ollama pull qwen2.5:7b-instruct      # or llama3.1:8b-instruct
# for DS in Twitter15 Twitter16; do
#   $PY src/llm_explain.py classify --dataset $DS --context source         --model qwen2.5:7b-instruct
#   $PY src/llm_explain.py classify --dataset $DS --context source+replies --model qwen2.5:7b-instruct
#   $PY src/llm_explain.py explain  --run ${DS}_group_bigcn_none          --model qwen2.5:7b-instruct
# done

$PY src/make_tables.py
