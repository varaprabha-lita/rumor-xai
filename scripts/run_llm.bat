@echo off
REM LLM stage on a fixed, class-balanced sample of 200 events per dataset (same events in every condition).
REM Run inside Anaconda Prompt after:  conda activate rumour
REM Safe to stop (Ctrl+C) and re-run: finished answers are cached and skipped.
cd /d %~dp0..
set MODEL=gemma3:12b
set N=200
for %%D in (Twitter16 Twitter15) do (
  python src/llm_explain.py classify --dataset %%D --context source         --limit %N% --model %MODEL%
  python src/llm_explain.py classify --dataset %%D --context source+replies --limit %N% --model %MODEL%
  python src/llm_explain.py explain  --run %%D_group_bigcn_none            --limit %N% --model %MODEL%
)
python src/make_tables.py
echo All LLM runs finished.
