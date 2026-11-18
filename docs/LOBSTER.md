# LOBSTER data and labels

## Data

The experiments use LOBSTER's free sample day (21 June 2012; AAPL, AMZN, GOOG, INTC, MSFT;
level 10) from <https://lobsterdata.com> (Data Samples page). The LOBSTER site requires a login.
A public re-host of the same sample files exists on Hugging Face (`totalorganfailure/lobster-data`):

```bash
pip install -U huggingface_hub
hf download totalorganfailure/lobster-data --repo-type dataset --local-dir lobster_raw
```

Put the `*_message_10.csv` / `*_orderbook_10.csv` pairs of the five stocks in `lobster/`.
Check that the names look like `INTC_2012-06-21_34200000_57600000_message_10.csv`. Do not commit
LOBSTER files to a public repository (`.gitignore` excludes `lobster/`).

Test the pipeline first on synthetic random-walk files (F1 must stay near chance, ~33%):

```bash
python make_synthetic_lobster.py --out lobster_synth
python train.py --dataset lobster --train-files "lobster_synth/train/*_message_10.csv" \
    --test-files "lobster_synth/test/*_message_10.csv" --model light --epochs 2 --tag synth_test
```

## Windows, labels and splits (`lobster.py`)

* One snapshot every `--sample-every` (10) events, windows of 100 snapshots, 40 features (10 levels).
* `--lob-norm relative` (default): prices relative to the last mid price of the window, in units of
  10 bp; sizes `log1p` and z-scored with pooled training statistics. `relative_ps` z-scores sizes
  per stock; `zscore` is plain z-scoring.
* Label of a decision at event t with execution delay d: the start price is the mid price of the
  **last event with timestamp ≤ t + d** (d = 0 gives t itself), the end price is k sampled steps
  later, returns with |r| < 1e-9 are set to 0, and the up/stationary/down threshold alpha is set per
  ticker on the training part.
* No `--test-files`: time split of every stock-day, 60% train, 20% validation, 20% test, with
  labels purged at the boundaries. With `--test-files`: separate (e.g. unseen) stock-days.
* `--delay-us D` trains on labels shifted by D µs (latency-aware training).

## Regimes

INTC and MSFT are large-tick stocks (spread mostly one tick, many stationary labels); AAPL, GOOG
and AMZN are small-tick. Training on one regime and testing on the other failed for every model
(F1 0.10 to 0.27, predictions collapse to one class), so the main experiment uses INTC + MSFT with a
time split: `bash scripts/run_lobster.sh`.

## Burst effect

Many messages share one timestamp. At d = 0 the label starts inside such a burst, which an order
sent at t could not trade on; labels at d = 0 are optimistic by up to 1.5 F1 points compared with
d = 1 µs. Report d ≥ 1 µs (or the model's own latency) rather than d = 0.
