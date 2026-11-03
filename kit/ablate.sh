#!/bin/bash
# Ablation driver (EVALUATION §5, REPRODUCE §4c): one fresh engine per config,
# identical cells in identical order, prefix caching off:
#   bands 100/250/1000 x {dec4, dec32, pre8192, mix}  +  sat_long@0.5@250 SLO
#
#   set -a; . ./cluster.env; set +a
#   bash ablate.sh cfg.txt [PFX]       # cfg.txt: one "tag het.py-start-flags" per line
#   python ablate_tab.py --pfx PFX --base natv1 --tags nat2,best,...
#
# Example cfg.txt (the configs of EVALUATION §5):
#   natv1
#   nat2       --runner v2
#   nat2t      --runner v2 --tok-tail
#   L2t        --link --codec bf16 --net-bw 0 --runner v2 --tok-tail
#   LQ2t       --link --codec int4 --net-bw 0 --runner v2 --tok-tail
#   LM2t       --link --codec bf16 --merge --net-bw 0 --runner v2 --tok-tail
#   best       --best
#   bestsync   --best --link-sync
#   besttpc    --best-tpc
# "Native without MoE tuning" = move fused_moe/configs/E=128,N=384,*.json
# away on both machines before that config and back afterwards.
# Results: $HET_RESULTS/runs/{PFX}_{tag}_{cell}_{band}.json (+ .reqs.json),
# greedy capture $HET_RESULTS/{PFX}_{tag}_gen_fixed.json.
set -u
cd "$(dirname "$0")"
CFG=${1:?usage: ablate.sh cfg.txt [PFX]}; PFX=${2:-ab}
PY=$A_VENV/bin/python
RES=${HET_RESULTS:-./results}; R=$RES/runs; mkdir -p "$R"
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

boot() {  # start flags...
  clean; rm -f "$RES/sched_ctrl.json"
  $PY het.py up >/dev/null 2>&1
  t0=$(date +%s)
  $PY het.py start --no-prefix-cache "$@" >/dev/null 2>&1
  for i in $(seq 1 720); do
    c=$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")
    if [ "$c" = 200 ]; then
      dt=$(( $(date +%s) - t0 ))
      [ $dt -gt 20 ] || { echo "ZOMBIE: healthy in ${dt}s"; exit 1; }
      echo "healthy in ${dt}s ($*)"; return 0
    fi
    [ $i -gt 6 ] && ! pgrep -f '[v]llm.entrypoints.openai' >/dev/null && {
      echo "ENGINE-DIED"; tail -30 "$RES/engine.log"; exit 1; }
    sleep 5
  done
  echo "BOOT-TIMEOUT"; tail -30 "$RES/engine.log"; exit 1
}

run() {  # tag band name replay-args...
  tag=$1; b=$2; name=$3; shift 3
  $PY replay.py replay --port "$PORT" --timeout 1800 "$@" \
    --out "$R/${PFX}_${tag}_${name}_${b}.json" | \
    grep -oE '"(wall|out_tps_wall|goodput)": [0-9.]+' | tr '\n' ' '
  echo " <- $tag $name@$b"
}

cells() {  # tag
  for B in 100 250 1000; do
    $PY het.py tc set $B | tail -1 | grep -q verified || { echo "TC-FAIL $B"; exit 1; }
    run "$1" $B dec4 --trace dec_sat --rate 4 --dur 45
    run "$1" $B dec32 --trace dec_sat --rate 32 --dur 45
    run "$1" $B pre8192 --trace pre_sat --rate 1 --dur 45 --in-len 8192
    run "$1" $B mix --trace fixed_mix --rate 1 --dur 45
  done
  $PY het.py tc set 250 | tail -1 | grep -q verified || { echo "TC-FAIL 250"; exit 1; }
  run "$1" 250 satlong05 --trace sat_long --rate 0.5 --dur 300
}

echo "start $(date +%T)"
while read -u 3 tag flags; do
  case "$tag" in ''|'#'*) continue ;; esac
  # shellcheck disable=SC2086
  boot $flags
  $PY greedy.py --tag "${PFX}_$tag" >/dev/null 2>&1
  cells "$tag"
  echo "== $tag done $(date +%T) =="
done 3< "$CFG"
clean; $PY het.py tc clear >/dev/null 2>&1
echo "ABLATE-DONE $(date +%T)"
