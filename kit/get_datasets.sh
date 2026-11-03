#!/bin/bash
# Download the four public benchmarks used by qeval.py (parquet, from the
# Hugging Face Hub or a mirror) into $HET_RESULTS/qeval/data.
#   HF_ENDPOINT=https://hf-mirror.com bash get_datasets.sh   # mirror
set -eu
EP=${HF_ENDPOINT:-https://huggingface.co}
D=${QEVAL_DATA:-${HET_RESULTS:-./results}/qeval/data}
mkdir -p "$D"
while read -r src dst; do
  [ -z "$src" ] && continue
  repo=$(echo "$src" | cut -d/ -f1-2); path=$(echo "$src" | cut -d/ -f3-)
  curl -sfL --max-time 300 -o "$D/$dst.parquet" \
    "$EP/datasets/$repo/resolve/main/$path"
  echo "ok $dst $(stat -c %s "$D/$dst.parquet") B"
done <<'EOF'
openai/gsm8k/main/test-00000-of-00001.parquet gsm8k_test
openai/gsm8k/main/train-00000-of-00001.parquet gsm8k_train
cais/mmlu/all/test-00000-of-00001.parquet mmlu_test
cais/mmlu/all/dev-00000-of-00001.parquet mmlu_dev
allenai/ai2_arc/ARC-Challenge/test-00000-of-00001.parquet arcc_test
allenai/ai2_arc/ARC-Challenge/train-00000-of-00001.parquet arcc_train
Salesforce/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet wikitext2_test
EOF
