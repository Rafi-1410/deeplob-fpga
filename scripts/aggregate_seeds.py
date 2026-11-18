import argparse
import collections
import csv
import glob
import json
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/*.json")
    ap.add_argument("--dataset", default="fi2010", choices=["fi2010", "lobster", "all"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    groups = collections.defaultdict(list)
    for p in sorted(glob.glob(a.runs)):
        r = json.load(open(p))
        if "test" not in r or (a.dataset != "all" and r.get("dataset") != a.dataset):
            continue
        prec = "fp32" if r.get("bits") is None else f"w{r['bits']}a{r.get('abits') or r['bits']}"
        key = (r.get("dataset"), r.get("horizon"), r.get("model"), prec, r.get("delay_us", 0.0))
        groups[key].append(r["test"]["f1_macro"])
    rows = []
    for (ds, k, model, prec, d), v in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        rows.append({"dataset": ds, "k": k, "model": model, "prec": prec, "delay_us": d,
                     "mean": round(statistics.mean(v), 4),
                     "std": round(statistics.stdev(v), 4) if len(v) > 1 else 0.0, "count": len(v)})
    for r in rows:
        print(f"{r['dataset']:8s} k={r['k']:<4} {r['model']:15s} {r['prec']:5s} d={r['delay_us']:<7g} "
              f"F1 {100 * r['mean']:.1f} ± {100 * r['std']:.1f}  (n={r['count']})")
    if a.out and rows:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
