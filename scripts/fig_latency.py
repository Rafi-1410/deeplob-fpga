import csv
import collections
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

curve_csv, eff_csv, out = sys.argv[1], sys.argv[2], sys.argv[3]
import statistics
raw = collections.defaultdict(dict)
for r in csv.DictReader(open(curve_csv)):
    raw[r["tag"]][float(r["delay_us"])] = float(r["f1_macro"])
curve, spread = {}, {}
for base in ("lobL_deeplob_fp32", "lobL_light_fp32", "lobL_light_w4a4"):
    runs = [raw[t] for t in (base, base + "_s1", base + "_s2")]
    ds = sorted(set.intersection(*[set(r) for r in runs]))
    curve[base] = {d: statistics.mean(r[d] for r in runs) for d in ds}
    spread[base] = {d: statistics.stdev(r[d] for r in runs) for d in ds}
lat = {}
for r in csv.DictReader(open(eff_csv)):
    lat[(r["tag"], r["platform"])] = float(r["latency_us"])
eff = [{"tag": t, "platform": p, "latency_us": l, "effective_f1": curve[t][l]} for (t, p), l in lat.items() if t in curve]
names = {"lobL_deeplob_fp32": "DeepLOB (FP32)", "lobL_light_fp32": "LightLOB (FP32)",
         "lobL_light_w4a4": "LightLOB W4A4 (FPGA model)"}
colors = {"lobL_deeplob_fp32": "#2a5ea8", "lobL_light_fp32": "#d9822b", "lobL_light_w4a4": "#2e9e5b"}

fig, ax = plt.subplots(figsize=(7.4, 4.5), dpi=200)
ax.axvspan(500, 2000, color="0.92", zorder=0)
ax.text(1000, 67.6, "steepest loss\n(2–6 market events)", ha="center", va="top", fontsize=7.5, color="0.35")
for tag, pts in curve.items():
    if tag not in names:
        continue
    xs = sorted(d for d in pts if d > 0)
    ax.plot(xs, [100 * pts[d] for d in xs], color=colors[tag], lw=1.8, label=names[tag])
    ax.fill_between(xs, [100 * (pts[d] - spread[tag][d]) for d in xs], [100 * (pts[d] + spread[tag][d]) for d in xs],
                    color=colors[tag], alpha=0.15, lw=0)
    ax.scatter([0.6], [100 * pts[0.0]], color=colors[tag], marker="_", s=120, lw=2)
show = {("lobL_deeplob_fp32", "GPU median"): ("DeepLOB · GPU median", (10, 4)),
        ("lobL_deeplob_fp32", "CPU median"): ("DeepLOB · CPU median", (8, 5)),
        ("lobL_deeplob_fp32", "GPU p99"): ("DeepLOB · GPU p99", (-70, -14)),
        ("lobL_light_fp32", "CPU median"): ("LightLOB · CPU median", (-100, 12)),
        ("lobL_light_w4a4", "FPGA accel median"): ("FPGA median", (-62, -13)),
        ("lobL_light_w4a4", "FPGA accel p99"): ("FPGA p99", (8, 1)),
        ("lobL_light_w4a4", "FPGA end-to-end"): ("FPGA incl. ARM I/O", (-96, -12))}
for r in eff:
    key = (r["tag"], r["platform"])
    if key not in show:
        continue
    m = "X" if "p99" in r["platform"] else ("s" if "end-to-end" in r["platform"] else "o")
    x, y = float(r["latency_us"]), 100 * float(r["effective_f1"])
    ax.scatter([x], [y], color=colors[r["tag"]], marker=m, s=46, edgecolor="black", lw=0.6, zorder=5)
    txt, off = show[key]
    ax.annotate(txt, (x, y), xytext=off, textcoords="offset points", fontsize=7.2, color=colors[r["tag"]])
ax.set_xscale("log")
ax.set_xlim(0.45, 40000)
ax.set_ylim(40, 68)
ax.set_xlabel("Execution delay d (µs, log scale; short bars at left = d = 0)")
ax.set_ylabel("Macro-F1 vs. delay-shifted label (%)")
ax.set_title("Latency-consistent evaluation on LOBSTER (INTC, MSFT; mean ± std over 3 seeds)", fontsize=10)
ax.grid(alpha=0.3, which="both")
h, _ = ax.get_legend_handles_labels()
h += [Line2D([], [], marker="o", ls="", color="0.5", mec="black", label="measured median latency"),
      Line2D([], [], marker="X", ls="", color="0.5", mec="black", label="measured p99 latency"),
      Line2D([], [], marker="s", ls="", color="0.5", mec="black", label="FPGA incl. ARM-side I/O")]
ax.legend(handles=h, fontsize=7.2, loc="lower left", frameon=True)
fig.tight_layout()
fig.savefig(out + ".png")
fig.savefig(out + ".pdf")
