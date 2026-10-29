# Evidence-Grounded LLM Reasoning over Propagation Graphs for Rumour Detection on Unseen Stories

Code, leakage-free splits, prompts and results for the paper of the same title
(G. Poornima, Mohamadi Begum Y, Sukruth Gowda M A, Mrutyunjaya M S; submitted to *Big Data*,
special issue "Intelligent Analytics for Social and Web Information Networks").

## What is here

| Path | Purpose |
|---|---|
| `src/data.py` | Loads Twitter15/16 trees, labels, source and reply text; `make_folds(split="random"\|"group")` builds random or leakage-free grouped folds |
| `src/baselines.py` | Structure features + RF, source-text TF-IDF + LR, cascade words + LR, text + structure + LR |
| `src/models.py` | BiGCN (plain PyTorch, no torch_geometric), text MLP, gated fusion |
| `src/train.py` | 5-fold training/evaluation, early-detection evaluation, checkpoints |
| `src/explain.py` | Node saliency and faithfulness (comprehensiveness / sufficiency vs. random) |
| `src/llm_explain.py` | LLM zero-shot classification and evidence-grounded explanations (Ollama / vLLM / Hugging Face) |
| `src/llm_stats.py` | Bootstrap confidence intervals, paired tests and McNemar tests for the LLM stage |
| `src/knowledge_appeal.py` | How often LLM explanations appeal to prior knowledge |
| `src/make_tables.py`, `src/make_figures.py` | Tables (`results/tables.md`) and figures (`figures/`) |
| `scripts/download_data.sh` | Downloads the public data releases (no Twitter/X API needed) |
| `scripts/run_all.sh` | Trained models, explanations and tables (CPU, about 20 minutes) |
| `scripts/run_llm.bat` | LLM stage on Windows with Ollama (200 class-balanced events per dataset) |

## Reproduce

```bash
pip install -r requirements.txt
bash scripts/download_data.sh
bash scripts/run_all.sh
# LLM stage (needs Ollama: https://ollama.com)
ollama pull gemma3:12b
python src/llm_explain.py classify --dataset Twitter16 --context source         --limit 200 --model gemma3:12b
python src/llm_explain.py classify --dataset Twitter16 --context source+replies --limit 200 --model gemma3:12b
python src/llm_explain.py explain  --run Twitter16_group_bigcn_none            --limit 200 --model gemma3:12b
# repeat for Twitter15, then
python src/llm_stats.py && python src/knowledge_appeal.py && python src/make_tables.py && python src/make_figures.py
```

On Windows, run `scripts\run_llm.bat` from an Anaconda Prompt instead of the LLM commands.

## Notes

* The grouped split merges events that share a claim identifier or have near-duplicate source tweets
  (TF-IDF cosine > 0.5), building on the event-separated evaluation of Wu and Hooi (ECML PKDD 2022)
  and the chronological splits of Mu et al. (EACL Findings 2023; LREC-COLING 2024).
* The released trees have no timestamps; early detection uses breadth-first order as a proxy.
* User handles are masked before any text is sent to the LLM.

## Data

Twitter15/16 (Ma et al., ACL 2017) as released by Bian et al. (AAAI 2020) and Yuan et al. (ICDM 2019).
The raw data are not redistributed here; `scripts/download_data.sh` fetches them from the original releases.
