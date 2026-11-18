#!/usr/bin/env bash
HORIZONS=${HORIZONS:-"10 100"}
SEEDS=${SEEDS:-"0"}
EPOCHS=${EPOCHS:-50}
mkdir -p runs logs

run() {
  local tag=$1; shift
  if [ -f "runs/$tag.json" ]; then echo "skip $tag (done)"; return; fi
  echo ">> $tag  ($(date +%H:%M))"
  python train.py "$@" 2>&1 | tee "logs/$tag.log"
  if [ ! -f "runs/$tag.json" ]; then echo "!! $tag FAILED, see logs/$tag.log"; fi
}

for K in $HORIZONS; do
  for S in $SEEDS; do
    B="--horizon $K --seed $S --epochs $EPOCHS"
    run light_fp32_k${K}_s$S        --model light $B
    for Q in "8 8" "4 4" "2 2" "2 4"; do
      set -- $Q
      run light_w$1a$2_k${K}_s$S    --model light --bits $1 --abits $2 $B --init runs/light_fp32_k${K}_s$S.pt --lr 5e-4
    done
    run light2_fp32_k${K}_s$S       --model light2 $B
    for Q in "8 8" "4 4" "2 4"; do
      set -- $Q
      run light2_w$1a$2_k${K}_s$S   --model light2 --bits $1 --abits $2 $B --init runs/light2_fp32_k${K}_s$S.pt --lr 5e-4
    done
    run deeplob_fp32_k${K}_s$S        --model deeplob $B
    run deeplob_nolstm_fp32_k${K}_s$S --model deeplob_nolstm $B
  done
done

python measure.py --runs runs
python shift_eval.py --runs "runs/*.json"
echo "ALL DONE $(date)"
