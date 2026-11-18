import argparse, csv, os
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score

from runutil import load_run, run_paths, predict_lobster_all
from train import load_data

DEFAULT_GRID = "0,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000"


def pooled_f1(days, preds, delay_us):
    ys, ps = [], []
    for name, day in days.items():
        day.relabel(delay_us)
        ys.append(day.y_all[day.valid]); ps.append(preds[name][day.valid])
    y, p = np.concatenate(ys), np.concatenate(ps)
    return f1_score(y, p, average="macro"), len(y)


def crossing(grid, f1s, target):
    grid, f1s = np.asarray(grid, float), np.asarray(f1s, float)
    if f1s[0] < target:
        return None
    below = np.where(f1s < target)[0]
    if len(below) == 0:
        return float("inf")
    i = below[0]
    x0, x1, y0, y1 = grid[i - 1], grid[i], f1s[i - 1], f1s[i]
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/lob_*.json")
    ap.add_argument("--latency", default=None, help="CSV: tag,platform,latency_us")
    ap.add_argument("--grid-us", default=DEFAULT_GRID)
    ap.add_argument("--extra-us", type=float, default=0.0,
                    help="overhead added to every model (feature extraction, network)")
    ap.add_argument("--out", default="results/latency")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    lat = defaultdict(list)
    if a.latency:
        for r in csv.DictReader(open(a.latency)):
            lat[r["tag"]].append((r["platform"], float(r["latency_us"]) + a.extra_us))
    own = sorted({d for v in lat.values() for _, d in v})
    grid = sorted(set(float(g) for g in a.grid_us.split(",")) | set(own))

    curves, ev_rows, data_cache = {}, [], {}
    for path in run_paths(a.runs):
        res, model, args = load_run(path)
        if args.dataset != "lobster":
            print(f"skip {res['tag']}: latency evaluation needs LOBSTER timestamps (FI-2010 has none)")
            continue
        key = (args.train_files, args.test_files, args.horizon, args.sample_every, args.test_stride, str(args.alpha), getattr(args, "lob_norm", "relative"))
        if key not in data_cache:
            data_cache[key] = load_data(args)[3]
            for name, day in data_cache[key].items():
                for d in grid:
                    ev_rows.append({"test_day": name, "delay_us": d,
                                    "median_events": float(np.median(day.events_elapsed(d)))})
        days = data_cache[key]
        preds = {n: predict_lobster_all(model, d) for n, d in days.items()}
        curves[res["tag"]] = [pooled_f1(days, preds, d) for d in grid]
        print(f"{res['tag']}: F1 at d=0 {curves[res['tag']][0][0]:.4f}, "
              f"at {grid[-1]:g} us {curves[res['tag']][-1][0]:.4f}", flush=True)
    if not curves:
        raise SystemExit("no LOBSTER runs found")

    with open(os.path.join(a.out, "f1_vs_delay.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tag", "delay_us", "f1_macro", "n"])
        for tag, c in curves.items():
            for d, (f1, n) in zip(grid, c):
                w.writerow([tag, d, round(f1, 5), n])
    seen = set()
    with open(os.path.join(a.out, "events_elapsed.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test_day", "delay_us", "median_events"]); w.writeheader()
        for r in ev_rows:
            if (r["test_day"], r["delay_us"]) not in seen:
                seen.add((r["test_day"], r["delay_us"])); w.writerow(r)

    eff = []
    for tag, entries in lat.items():
        if tag not in curves:
            print(f"warning: {tag} in latency file but no matching run")
            continue
        for platform, d in entries:
            eff.append({"tag": tag, "platform": platform, "latency_us": d,
                        "effective_f1": round(curves[tag][grid.index(d)][0], 5),
                        "f1_at_zero_delay": round(curves[tag][0][0], 5)})
    if eff:
        eff.sort(key=lambda r: -r["effective_f1"])
        with open(os.path.join(a.out, "effective.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(eff[0])); w.writeheader(); w.writerows(eff)
        rows = []
        for A in eff:
            for B in eff:
                if A is B:
                    continue
                x = crossing(grid, [c[0] for c in curves[A["tag"]]], B["effective_f1"])
                rows.append({"model_A": f"{A['tag']} ({A['platform']})", "model_B": f"{B['tag']} ({B['platform']})",
                             "B_effective_f1": B["effective_f1"], "A_latency_us": A["latency_us"],
                             "A_break_even_us": "never" if x is None else (">grid" if x == float("inf") else round(x, 2)),
                             "A_beats_B_now": A["effective_f1"] > B["effective_f1"]})
        with open(os.path.join(a.out, "break_even.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print("\nEffective F1 (each model at its own latency):")
        for r in eff:
            print(f"  {r['tag']:<40} {r['platform']:<20} {r['latency_us']:>9g} us  F1 {r['effective_f1']:.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = [max(g, 0.5) for g in grid]
        for tag, c in curves.items():
            line, = ax.plot(x, [100 * f for f, _ in c], label=tag)
            for r in eff:
                if r["tag"] == tag:
                    ax.scatter(max(r["latency_us"], 0.5), 100 * r["effective_f1"], color=line.get_color(), zorder=3)
                    ax.annotate(r["platform"], (max(r["latency_us"], 0.5), 100 * r["effective_f1"]),
                                textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("Execution delay d (us, log scale; leftmost point = 0)")
        ax.set_ylabel("Macro F1 against delayed label (%)")
        ax.set_title("Latency-consistent F1 (dots = each model at its measured latency)")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "f1_vs_delay.png"), dpi=200)
    except ImportError:
        pass
    print(f"\nwrote results to {a.out}")


if __name__ == "__main__":
    main()
