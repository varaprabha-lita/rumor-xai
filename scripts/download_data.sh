#!/usr/bin/env bash
# Fetch Twitter15/16 propagation trees, labels, source-tweet text and reply text.
# Everything comes from public GitHub releases; no Twitter/X API access needed.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=$(command -v python3 || command -v python)
mkdir -p data/raw/text data/raw/replies data/tmp

# 1) Trees + node bag-of-words + labels (Ma et al. 2017, as released with BiGCN, AAAI 2020)
[ -d data/raw/bigcn ] || git clone --depth 1 https://github.com/TianBian95/BiGCN.git data/raw/bigcn

# 2) Source tweets + replies (as released with GLAN, Yuan et al. ICDM 2019)
if [ ! -d data/tmp/glan ]; then
  git clone --depth 1 --filter=blob:none --no-checkout https://github.com/chunyuanY/RumorDetection.git data/tmp/glan
  (cd data/tmp/glan && git checkout HEAD -- dataset/twitter15 dataset/twitter16)
fi
for d in twitter15 twitter16; do
  cat data/tmp/glan/dataset/$d/$d.train data/tmp/glan/dataset/$d/$d.dev data/tmp/glan/dataset/$d/$d.test \
    > data/raw/text/${d}_source.tsv
done

# 3) Reply text (RAR archives). Needs libarchive (`pip install libarchive-c`) or unrar/7z.
$PY - <<'PY'
import os
src = "data/tmp/glan/dataset/{}/replies.rar"
for d in ("twitter15", "twitter16"):
    out = f"data/raw/replies/{d}"
    os.makedirs(out, exist_ok=True)
    if os.listdir(out):
        continue
    try:
        import libarchive
        with libarchive.file_reader(src.format(d)) as a:
            for e in a:
                if e.isdir:
                    continue
                with open(os.path.join(out, os.path.basename(e.pathname)), "wb") as f:
                    for b in e.get_blocks():
                        f.write(b)
    except Exception as ex:  # fall back to command-line tools
        print("libarchive failed:", ex, "- trying 7z/unrar")
        if os.system(f"7z e -y -o{out} {src.format(d)} >/dev/null") != 0:
            os.system(f"unrar e -y {src.format(d)} {out}/")
    print(d, len(os.listdir(out)), "reply files")
PY
wc -l data/raw/text/*.tsv
echo "done."
