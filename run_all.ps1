$K = 10
$Seeds = 0, 1, 2
$Quick = ""
$q = $Quick.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)

foreach ($S in $Seeds) {
  python train.py --model deeplob        --horizon $K --seed $S @q
  python train.py --model deeplob_nolstm --horizon $K --seed $S @q
  python train.py --model light          --horizon $K --seed $S @q
  foreach ($B in 8, 4, 2) {
    python train.py --model light --bits $B --horizon $K --seed $S --init "runs/light_fp32_k${K}_s${S}.pt" --lr 5e-4 @q
  }
}

if (Test-Path "lobster/train/*_message_10.csv") {
  foreach ($S in $Seeds) {
    $L = @("--dataset", "lobster", "--train-files", "lobster/train/*_message_10.csv",
           "--test-files", "lobster/test/*_message_10.csv", "--horizon", $K, "--seed", $S)
    python train.py @L --model deeplob @q
    python train.py @L --model light @q
    foreach ($B in 8, 4, 2) {
      python train.py @L --model light --bits $B --init "runs/lob_light_fp32_k${K}_s${S}.pt" --lr 5e-4 @q
    }
  }
}

python measure.py --runs runs
python shift_eval.py --runs "runs/*.json"
if (Test-Path "latency.csv") { python latency_eval.py --runs "runs/lob_*.json" --latency latency.csv }
python export_qonnx.py --ckpt "runs/light_w4a4_k${K}_s0.pt" --bits 4 --out finn/light_w4a4.onnx
