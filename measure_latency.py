import argparse
import csv
import glob
import os
import time

import numpy as np
import torch

from runutil import load_run


def time_calls(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts = np.array(ts)
    return float(np.median(ts)), float(np.percentile(ts, 99)), float(ts.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run .json files or globs")
    ap.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--cpu-threads", type=int, default=1)
    ap.add_argument("--out", default="results/latency_measured.csv")
    a = ap.parse_args()
    torch.set_num_threads(a.cpu_threads)

    paths = sorted({p for pat in a.runs for p in glob.glob(pat)} - {"runs/summary.json"})
    rows = []
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    for path in paths:
        res, model, _ = load_run(path)
        tag = res["tag"]
        x_np = np.random.randn(1, 1, 100, 40).astype(np.float32)
        for dev in a.devices:
            if dev == "cuda" and gpu_name is None:
                continue
            m = model.to(dev).eval()

            if dev == "cuda":
                def call():
                    with torch.no_grad():
                        x = torch.from_numpy(x_np).to("cuda", non_blocking=False)
                        y = m(x)
                        y.cpu()
                        torch.cuda.synchronize()
                platform = f"GPU {gpu_name}"
            else:
                def call():
                    with torch.no_grad():
                        m(torch.from_numpy(x_np)).numpy()
                platform = f"CPU ({a.cpu_threads} thread)"
            med, p99, mn = time_calls(call, a.reps, a.warmup)
            rows.append({"tag": tag, "platform": platform, "median_us": round(med, 1),
                         "p99_us": round(p99, 1), "min_us": round(mn, 1)})
            print(f"{tag:28s} {platform:32s} median {med:9.1f} us   p99 {p99:9.1f} us   min {mn:9.1f} us",
                  flush=True)
            model = m.cpu()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", a.out)
    print("\nrows for latency.csv (tag,platform,latency_us), using the median:")
    for r in rows:
        print(f"{r['tag']},{r['platform']},{r['median_us']}")


if __name__ == "__main__":
    main()
