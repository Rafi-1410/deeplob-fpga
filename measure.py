import argparse, glob, json, os, statistics, time
import torch
import torch.nn as nn
from models import build


def macs(model):
    total = [0]

    def conv_hook(m, i, o):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        total[0] += o.numel() // o.shape[0] * k

    def lin_hook(m, i, o):
        total[0] += m.in_features * m.out_features

    def lstm_hook(m, i, o):
        steps = i[0].shape[1]
        total[0] += steps * 4 * m.hidden_size * (m.input_size + m.hidden_size)

    hs = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hs.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hs.append(m.register_forward_hook(lin_hook))
        elif isinstance(m, nn.LSTM):
            hs.append(m.register_forward_hook(lstm_hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, 100, 40))
    for h in hs:
        h.remove()
    return total[0]


def weight_kbit(model, bits):
    if bits is None:
        return sum(p.numel() for p in model.parameters()) * 32 / 1000
    tot = 0
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            wb = getattr(m, "weight_quant", None)
            b = int(m.quant_weight().bit_width.item()) if wb is not None else 32
            tot += m.weight.numel() * b
    return tot / 1000


def cpu_latency_us(model, runs=300):
    torch.set_num_threads(1)
    x = torch.randn(1, 1, 100, 40)
    model.eval()
    with torch.no_grad():
        for _ in range(30):
            model(x)
        ts = []
        for _ in range(runs):
            t = time.perf_counter(); model(x); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.runs, "*.json"))):
        if f.endswith("summary.json"):
            continue
        r = json.load(open(f))
        m = build(r["model"], r["bits"], r.get("abits"))
        m.load_state_dict(torch.load(f[:-5] + ".pt", map_location="cpu"))
        rows.append({
            "tag": r["tag"], "model": r["model"], "precision": "FP32" if r["bits"] is None else f"W{r['bits']}A{r.get('abits') or r['bits']}",
            "horizon": r["horizon"], "params": r["params"], "MACs": macs(m),
            "weight_kbit": round(weight_kbit(m, r["bits"]), 1),
            "cpu_latency_us": round(cpu_latency_us(m), 1) if r["bits"] is None else None,
            "test_f1": round(100 * r["test"]["f1_macro"], 2), "test_acc": round(100 * r["test"]["accuracy"], 2),
            "epochs": r["epochs_run"], "train_windows": r["train_windows"], "train_stride": r["train_stride"],
        })
    json.dump(rows, open(os.path.join(a.runs, "summary.json"), "w"), indent=2)
    cols = list(rows[0].keys())
    with open(os.path.join(a.runs, "summary.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")
    with open(os.path.join(a.runs, "summary.md"), "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join("-" if r[c] is None else str(r[c]) for c in cols) + " |\n")
    print(open(os.path.join(a.runs, "summary.md")).read())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for r in rows:
            ax.scatter(r["weight_kbit"], r["test_f1"])
            ax.annotate(f"{r['model']} {r['precision']}", (r["weight_kbit"], r["test_f1"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("Weight memory (kbit, log scale)")
        ax.set_ylabel("Test macro F1 (%)")
        ax.set_title("F1 vs model size on FI-2010 (FPGA latency pending)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(a.runs, "frontier.png"), dpi=200)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
