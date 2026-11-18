import argparse, csv, os
import numpy as np
from sklearn.metrics import f1_score

from runutil import load_run, run_paths, predict_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/*.json")
    ap.add_argument("--out", default="results/shift")
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from train import load_data

    rows, cache = [], {}
    paths = run_paths(a.runs)
    print(f"{len(paths)} runs, device {a.device}", flush=True)
    for path in paths:
        res, model, args = load_run(path, device=a.device)
        key = (args.dataset, args.data, args.norm, args.train_files, args.test_files, args.horizon,
               args.sample_every, args.delay_us, args.test_stride, str(args.alpha), getattr(args, "lob_norm", "relative"))
        if key not in cache:
            _, val, _, tests, _ = load_data(args)
            cache[key] = (val, tests)
        val, tests = cache[key]
        p, y = predict_dataset(model, val, device=a.device)
        row = {"tag": res["tag"], "dataset": args.dataset, "model": res["model"],
               "precision": "FP32" if res["bits"] is None else f"W{res['bits']}A{res.get('abits') or res['bits']}",
               "horizon": args.horizon, "seed": args.seed, "delay_us": args.delay_us,
               "val_f1": f1_score(y, p, average="macro")}
        parts = []
        for name, ds in tests.items():
            p, y = predict_dataset(model, ds, device=a.device)
            row[f"f1_{name}"] = f1_score(y, p, average="macro")
            parts.append(row[f"f1_{name}"])
        row["mean_test_f1"] = float(np.mean(parts))
        row["worst_test_f1"] = float(np.min(parts))
        row["drop"] = row["val_f1"] - row["mean_test_f1"]
        rows.append(row)
        print(f"{res['tag']}: val {row['val_f1']:.4f}  mean test {row['mean_test_f1']:.4f}  drop {row['drop']:.4f}",
              flush=True)

    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(a.out, "shift.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(a.out, "shift.md"), "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(f"{r[c]:.4f}" if isinstance(r.get(c), float) else str(r.get(c, "-"))
                                      for c in cols) + " |\n")

    group = lambda r: (r["dataset"], r["model"], r["horizon"], r["seed"], r["delay_us"])
    fp32 = {group(r): r for r in rows if r["model"] in ("light", "light2") and r["precision"] == "FP32"}
    pen = []
    for r in rows:
        if r["model"] in ("light", "light2") and r["precision"] != "FP32" and group(r) in fp32:
            b = fp32[group(r)]
            pen.append({"tag": r["tag"], "model": r["model"], "precision": r["precision"], "dataset": r["dataset"],
                        "horizon": r["horizon"], "seed": r["seed"],
                        "val_f1_loss_vs_fp32": round(b["val_f1"] - r["val_f1"], 5),
                        "test_f1_loss_vs_fp32": round(b["mean_test_f1"] - r["mean_test_f1"], 5),
                        "penalty_under_shift": round(r["drop"] - b["drop"], 5)})
    if pen:
        with open(os.path.join(a.out, "penalty.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pen[0])); w.writeheader(); w.writerows(pen)
        print("\nQuantisation penalty under shift (positive = quantised model degrades more out of sample):")
        for p in pen:
            print(f"  {p['tag']:<40} {p['penalty_under_shift']:+.4f}")
    print(f"\nwrote results to {a.out}")


if __name__ == "__main__":
    main()
