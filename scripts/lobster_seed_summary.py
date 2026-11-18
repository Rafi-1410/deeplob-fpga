import collections
import csv
import re
import statistics
import sys

curve_csv, eff_csv = sys.argv[1], sys.argv[2]
curve = collections.defaultdict(dict)
for r in csv.DictReader(open(curve_csv)):
    curve[r["tag"]][float(r["delay_us"])] = float(r["f1_macro"])
lat = {}
for r in csv.DictReader(open(eff_csv)):
    base = re.sub(r"_s\d+$", "", r["tag"])
    lat[(base, r["platform"])] = float(r["latency_us"])

seeds = collections.defaultdict(list)
for tag in curve:
    seeds[re.sub(r"_s\d+$", "", tag)].append(tag)


def ms(vals):
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{100 * statistics.mean(vals):5.1f} ± {100 * sd:3.1f} (n={len(vals)})"


for base in sorted(seeds):
    tags = sorted(seeds[base])
    print(f"\n{base}")
    print(f"  {'zero delay':28s} {ms([curve[t][0.0] for t in tags])}")
    platforms = {p: d for (b, p), d in lat.items() if b == base or base.startswith(b + "_d")}
    for p, d in sorted(platforms.items(), key=lambda x: x[1]):
        vals = [curve[t][d] for t in tags if d in curve[t]]
        print(f"  {p + f' ({d:g} us)':28s} {ms(vals)}")

print("\nLatency-aware training, paired by seed:")
for base in sorted(seeds):
    m = re.match(r"(.+)_d(\d+(?:\.\d+)?)", base)
    if not m or m.group(1) not in seeds:
        continue
    ref, d = m.group(1), float(m.group(2))
    pairs = []
    for t in seeds[base]:
        s = re.search(r"_s(\d+)$", t)
        rt = f"{ref}_s{s.group(1)}" if s else ref
        dd = min(curve[t], key=lambda x: abs(x - d))
        if rt in curve:
            pairs.append(curve[t][dd] - curve[rt][dd])
    if pairs:
        sd = statistics.stdev(pairs) if len(pairs) > 1 else 0.0
        print(f"  {base} vs {ref} at {dd:g} us: {100 * statistics.mean(pairs):+.1f} ± {100 * sd:.1f} points "
              f"(per seed: {', '.join(f'{100 * x:+.1f}' for x in pairs)})")
