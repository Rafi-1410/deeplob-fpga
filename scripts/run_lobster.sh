#!/usr/bin/env bash
set -e
PY=${PY:-python}
FILES=${FILES:-"lobster/[IM][NS]*_message_10.csv"}
SEEDS=${SEEDS:-"0 1 2"}
DEEPLOB_US=${DEEPLOB_US:-1110}
FPGA_US=${FPGA_US:-563}
mkdir -p runs logs

run() {
  local tag=$1; shift
  if [ -f "runs/$tag.json" ]; then echo "skip $tag"; return; fi
  echo ">> $tag ($(date +%H:%M))"
  $PY train.py --tag "$tag" "$@" 2>&1 | tee "logs/$tag.log"
}

for S in $SEEDS; do
  SUF=$([ "$S" = 0 ] && echo "" || echo "_s$S")
  L=(--dataset lobster --train-files "$FILES" --seed "$S")
  run lobL_deeplob_fp32$SUF "${L[@]}" --model deeplob
  run lobL_light_fp32$SUF   "${L[@]}" --model light
  run lobL_light_w4a4$SUF   "${L[@]}" --model light --bits 4 --init runs/lobL_light_fp32$SUF.pt --lr 5e-4
  run lobL_deeplob_fp32_d$DEEPLOB_US$SUF "${L[@]}" --model deeplob --delay-us "$DEEPLOB_US"
done
run lobL_light_w4a4_d$FPGA_US --dataset lobster --train-files "$FILES" --seed 0 \
    --model light --bits 4 --init runs/lobL_light_fp32.pt --lr 5e-4 --delay-us "$FPGA_US"

[ -f latency.csv ] || cp results/lobster/latency.csv latency.csv
$PY - <<'PYEOF'
import csv, glob, os, re
rows = list(csv.DictReader(open("latency.csv")))
have = {(r["tag"], r["platform"]) for r in rows}
tags = {os.path.basename(p)[:-5] for p in glob.glob("runs/lobL_*.json")}
new = []
for t in sorted(tags):
    base = re.sub(r"_s\d+$", "", t)
    if base == t:
        continue
    for r in rows:
        if r["tag"] == base and (t, r["platform"]) not in have:
            new.append(dict(r, tag=t))
with open("latency.csv", "a", newline="") as f:
    csv.DictWriter(f, fieldnames=["tag", "platform", "latency_us"]).writerows(new)
print("latency.csv: added", len(new), "seed rows")
PYEOF
$PY latency_eval.py --runs "runs/lobL_*.json" --latency latency.csv
$PY scripts/lobster_seed_summary.py results/latency/f1_vs_delay.csv results/latency/effective.csv
$PY scripts/fig_latency.py results/latency/f1_vs_delay.csv results/latency/effective.csv figures/fig_latency_consistent_f1_3seed
