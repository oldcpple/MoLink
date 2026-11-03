#!/bin/bash
# One-shot paired acceptance benchmark: ours (het.py start --best) vs native
# vLLM, back-to-back in one window, identical workloads.
#
#   set -a; . ./cluster.env; set +a
#   bash bench.sh [a3b|d32]            # ~2 h (a3b), longer for d32
#
# A3B also runs native vLLM a second time with Model Runner V2 forced on
# (vLLM picks V1 for MoE models; V2 is a launch flag, no code change) so
# ours is compared against both (NATIVE_V2=0 skips it). Dense Qwen3 (32B)
# already defaults to V2.
#
# Per system: clean stop -> fresh ray -> start -> health (must take >20 s:
# a faster "healthy" is a zombie engine still bound to the port) -> tc 250
# verify -> [ours: greedy capture vs native] -> 4 bandwidths x 10 cells ->
# sat_long@0.5@250 SLO run -> stop. Results in $HET_RESULTS/runs, summary
# table printed at the end (accept_tab.py) and figures via make_figs.py.
set -u
cd "$(dirname "$0")"
MODEL=${1:-a3b}
PY=$A_VENV/bin/python
RES=${HET_RESULTS:-./results}
R=$RES/runs
mkdir -p "$R"
# layer partition A/B, same for both systems: A3B 28,20, 32B 32,32
OPART=28,20; NPART=28,20
[ "$MODEL" = d32 ] && { OPART=32,32; NPART=32,32; }
OT=${MODEL}_ours; NT=${MODEL}_native
PORT=${PORT:-26000}

clean() {
  $PY het.py stop >/dev/null 2>&1; sleep 8
  for i in $(seq 1 30); do
    n=$(pgrep -fc '[v]llm.entrypoints|[E]ngineCore' || true)
    used=$(nvidia-smi -i "$GPUS_A" --query-gpu=memory.used \
      --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    [ "$n" = 0 ] && [ "$used" -lt 2000 ] && return 0
    sleep 5
  done
  echo "CLEAN-FAIL procs=$n gpu_mib=$used"; exit 1
}

boot() {  # partition, start flags...
  part=$1; shift
  clean
  rm -f "$RES/sched_ctrl.json"
  $PY het.py up >/dev/null 2>&1
  t0=$(date +%s)
  $PY het.py start --model "$MODEL" --partition "$part" "$@" >/dev/null 2>&1
  for i in $(seq 1 720); do
    c=$(curl -s -m 2 -o /dev/null -w '%{http_code}' \
      "http://127.0.0.1:$PORT/health")
    if [ "$c" = 200 ]; then
      dt=$(( $(date +%s) - t0 ))
      [ $dt -gt 20 ] || { echo "ZOMBIE: healthy in ${dt}s"; exit 1; }
      echo "healthy in ${dt}s ($*)"; return 0
    fi
    sleep 5
  done
  echo "BOOT-TIMEOUT"; tail -30 "$RES/engine.log"; exit 1
}

cells() {  # tag band
  RUN="$PY replay.py replay --port $PORT --timeout 1800"
  for M in 1 4 16 32; do
    $RUN --trace dec_sat --rate $M --dur 45 --out $R/$1_dec${M}_$2.json
  done
  $RUN --trace dec_sat --rate 16 --dur 45 --in-len 256 \
    --out $R/$1_dec16in256_$2.json
  $RUN --trace dec_sat --rate 16 --dur 45 --in-len 4096 \
    --out $R/$1_dec16in4K_$2.json
  for L in 2048 4096 8192; do
    $RUN --trace pre_sat --rate 1 --dur 45 --in-len $L \
      --out $R/$1_pre${L}_$2.json
  done
  $RUN --trace fixed_mix --rate 1 --dur 45 --out $R/$1_mix_$2.json
}

tc_bytes() { ${A_SUDO:-} tc -s class show dev "$A_IF" 2>/dev/null | \
  grep -oE "Sent [0-9]+ bytes" | head -1 | grep -oE "[0-9]+"; }

run_sys() {  # tag partition start-flags...
  tag=$1; shift
  boot "$@"
  $PY het.py tc set 250 || exit 1
  $PY greedy.py --tag "$tag" --port $PORT     # native runs first
  [ "$tag" = "$OT" ] && $PY greedy_cmp.py "$NT" "$OT"
  rm -f $R/${tag}_wan.jsonl
  for B in 100 250 500 1000; do
    $PY het.py tc set $B >/dev/null || exit 1
    b0=$(tc_bytes)
    cells "$tag" $B > /dev/null
    b1=$(tc_bytes)
    echo "{\"band\": $B, \"tc_bytes\": $((b1 - b0))}" >> $R/${tag}_wan.jsonl
    echo "  $tag @$B done $(date +%T)"
  done
  $PY het.py tc set 250 >/dev/null || exit 1
  $PY replay.py replay --port $PORT --trace sat_long --rate 0.5 --dur 300 \
    --timeout 1800 --out $R/${tag}_satlong05_250.json | tail -3
  echo "== $tag done $(date +%T) =="
}

echo "start $(date +%T) load $(cut -d' ' -f1-3 /proc/loadavg)"
run_sys "$NT" "$NPART"
if [ "$MODEL" = a3b ] && [ "${NATIVE_V2:-1}" = 1 ]; then
  run_sys "${NT}_v2" "$NPART" --runner v2
fi
run_sys "$OT" "$OPART" --best
clean
$PY het.py tc clear >/dev/null
$PY accept_tab.py "$OT" "$NT"
[ -e "$R/${NT}_v2_dec1_100.json" ] && $PY accept_tab.py "$OT" "${NT}_v2"
$PY ../make_figs.py --runs "$R" --ours "$OT" --native "$NT" --out "$RES" \
  --title "$MODEL"
echo "ALL-DONE $(date +%T)"
