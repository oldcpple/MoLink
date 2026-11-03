#!/bin/bash
# Bubble-rate driver (README §7): one fresh engine per window, torch
# profiler on GPU kernels only, saturating fixed_mix workload sent at t=0.
#
#   set -a; . ./cluster.env; set +a
#   bash bubble.sh TAG BAND [het.py start flags...]
#   python bubble.py --clip 2:32 $HET_RESULTS/bubble/TAG/prof [...]
#   python nccl_split.py --clip 2:32 $HET_RESULTS/bubble/TAG/prof
#
# TP2xPP2, Qwen3-30B-A3B (§7.2):
#   bash bubble.sh native_250 250
#   bash bubble.sh nativev2_250 250 --runner v2
#   bash bubble.sh ours_250 250 --best-tpc
# PP2, one 4090 + one 3090, Qwen3-8B (§7.4; GPUS_A=<one GPU>, B_NUM_GPUS=1):
#   MIX_X=3 bash bubble.sh q8_native_250 250 --model q8 --tp 1
#   MIX_X=3 bash bubble.sh q8_nativev1_250 250 --model q8 --tp 1 --runner v1
#   MIX_X=3 bash bubble.sh q8_ours_250 250 --model q8 --tp 1 --best --partition 24,12
#
# The profiler window is seconds 15..55 after each worker's first busy step;
# analyse only the steady seconds 2..32 of it (--clip 2:32): stopping the
# profiler exports the trace synchronously and stalls the pipeline, so the
# tail of the window is a measurement artefact. MIX_X multiplies the
# 72-request mix (distinct prompts) so a fast system is still saturated at
# the end of the window; check that the replay "wall" exceeds ~60 s.
set -u
cd "$(dirname "$0")"
TAG=${1:?usage: bubble.sh TAG BAND [start flags]}; BAND=${2:?}; shift 2
PY=$A_VENV/bin/python
RES=${HET_RESULTS:-./results}; OUT=$RES/bubble/$TAG; mkdir -p "$OUT/prof" "$RES/runs"
PORT=${PORT:-26000}

$PY het.py stop >/dev/null 2>&1; sleep 8
for i in $(seq 1 30); do
  n=$(pgrep -fc '[v]llm.entrypoints|[E]ngineCore' || true)
  used=$(nvidia-smi -i "$GPUS_A" --query-gpu=memory.used \
    --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
  [ "$n" = 0 ] && [ "$used" -lt 2000 ] && break
  [ $i = 30 ] && { echo "CLEAN-FAIL procs=$n gpu_mib=$used"; exit 1; }
  sleep 5
done
rm -rf "$RES/sched_ctrl.json" "$RES/prof"
$PY het.py up >/dev/null 2>&1
t0=$(date +%s)
$PY het.py start --prof t15:55 --prof-cuda-only "$@" >/dev/null 2>&1
for i in $(seq 1 720); do
  c=$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")
  [ "$c" = 200 ] && break; sleep 5
done
dt=$(( $(date +%s) - t0 )); echo "$TAG healthy=$c in ${dt}s ($*)"
[ "$c" = 200 ] && [ $dt -gt 20 ] || { echo "BOOT-FAIL"; exit 1; }
$PY het.py tc set "$BAND" | tail -1
$PY replay.py replay --port "$PORT" --trace fixed_mix --mix-x "${MIX_X:-1}" \
  --rate 1 --dur 45 --timeout 1800 --out "$RES/runs/bub_${TAG}_mix_${BAND}.json" | \
  grep -oE '"(wall|out_tps_wall)": [0-9.]+' | tr '\n' ' '; echo
sleep 20   # let the profiler finish writing its traces
$PY het.py pull-prof "$OUT/prof"
$PY het.py stop >/dev/null 2>&1; $PY het.py tc clear >/dev/null 2>&1
$PY bubble.py --clip 2:32 "$OUT/prof"
