#!/usr/bin/env bash
set -e
K=${K:-10}
SEEDS=${SEEDS:-"0 1 2"}

for S in $SEEDS; do
  python train.py --model deeplob        --horizon $K --seed $S
  python train.py --model deeplob_nolstm --horizon $K --seed $S
  python train.py --model light          --horizon $K --seed $S
  for B in 8 4 2; do
    python train.py --model light --bits $B --horizon $K --seed $S --init runs/light_fp32_k${K}_s${S}.pt --lr 5e-4
  done
done

TRAIN="lobster/train/*_message_10.csv"
TEST="lobster/test/*_message_10.csv"
if ls lobster/train/*_message_10.csv >/dev/null 2>&1; then
  for S in $SEEDS; do
    L=(--dataset lobster --train-files "$TRAIN" --test-files "$TEST" --horizon $K --seed $S)
    python train.py "${L[@]}" --model deeplob
    python train.py "${L[@]}" --model light
    for B in 8 4 2; do
      python train.py "${L[@]}" --model light --bits $B --init runs/lob_light_fp32_k${K}_s${S}.pt --lr 5e-4
    done
  done
fi

python measure.py --runs runs
python shift_eval.py --runs "runs/*.json"
if [ -f latency.csv ]; then python latency_eval.py --runs "runs/lob_*.json" --latency latency.csv; fi
python export_qonnx.py --ckpt runs/light_w4a4_k${K}_s0.pt --bits 4 --out finn/light_w4a4.onnx
