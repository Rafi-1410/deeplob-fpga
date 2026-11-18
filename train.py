import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from models import build, count_params


def metrics(y, p):
    return {
        "accuracy": accuracy_score(y, p),
        "precision_macro": precision_score(y, p, average="macro", zero_division=0),
        "recall_macro": recall_score(y, p, average="macro", zero_division=0),
        "f1_macro": f1_score(y, p, average="macro"),
        "n": int(len(y)),
    }


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x.to(device)).argmax(1).cpu())
            ys.append(y)
    return metrics(torch.cat(ys).numpy(), torch.cat(ps).numpy())


def to_device(model, device):
    model = model.to(device)
    for m in model.modules():
        for name, val in list(vars(m).items()):
            if torch.is_tensor(val) and not isinstance(val, nn.Parameter) and val.device != torch.device(device):
                setattr(m, name, val.to(device))
    return model


def label_counts(ds):
    if isinstance(ds, ConcatDataset):
        return sum(label_counts(d) for d in ds.datasets)
    if hasattr(ds, "idx") and hasattr(ds, "y"):
        y = np.asarray(ds.y)[ds.idx]
    elif hasattr(ds, "y"):
        y = np.asarray(ds.y)
    else:
        y = np.array([int(ds[i][1]) for i in range(len(ds))])
    return np.bincount(y, minlength=3)[:3].astype(np.int64)


def load_data(a):
    if a.dataset == "fi2010":
        from data import get_datasets, get_test_days
        if a.horizon not in (10, 20, 30, 50, 100):
            raise SystemExit("FI-2010 horizons are 10, 20, 30, 50, 100")
        z = getattr(a, "feat_norm", "zscore") == "zscore"
        train, val, test = get_datasets(a.data, a.horizon, a.train_stride, a.test_stride, norm=a.norm, zscore=z)
        tests = get_test_days(a.data, a.horizon, a.test_stride, norm=a.norm, zscore=z)
        return train, val, test, tests, {}
    from lobster import get_lobster_datasets
    if not a.train_files:
        raise SystemExit("--dataset lobster needs --train-files (and --test-files, or none for a time split)")
    train, val, tests, meta = get_lobster_datasets(
        a.train_files, a.test_files, horizon=a.horizon, sample_every=a.sample_every,
        delay_us=a.delay_us, train_stride=a.train_stride, test_stride=a.test_stride, alpha=a.alpha,
        lob_norm=getattr(a, "lob_norm", "relative"), test_frac=getattr(a, "test_frac", 0.2))
    return train, val, ConcatDataset(list(tests.values())), tests, meta


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fi2010", choices=["fi2010", "lobster"])
    ap.add_argument("--data", default="data", help="FI-2010 folder")
    ap.add_argument("--norm", default="DecPre", choices=["DecPre", "ZScore", "MinMax"])
    ap.add_argument("--feat-norm", default="zscore", choices=["zscore", "none"],
                    help="FI-2010: z-score the 40 features with training mean/std")
    ap.add_argument("--class-weight", default="balanced", choices=["balanced", "none"],
                    help="weight the loss by inverse class frequency of the training labels")
    ap.add_argument("--train-files", default=None, help="LOBSTER: glob of training *_message_*.csv")
    ap.add_argument("--test-files", default=None,
                    help="LOBSTER: glob of test *_message_*.csv (other stocks/days). Omit for a time split "
                         "of the training days: last --test-frac of each day is the test set")
    ap.add_argument("--test-frac", type=float, default=0.2, help="LOBSTER time split: test share of each day")
    ap.add_argument("--sample-every", type=int, default=10, help="LOBSTER: snapshot every n events")
    ap.add_argument("--delay-us", type=float, default=0.0, help="LOBSTER: train labels with this delay")
    ap.add_argument("--alpha", type=float, default=None, help="LOBSTER: label threshold (default: balanced per stock)")
    ap.add_argument("--lob-norm", default="relative", choices=["relative", "relative_ps", "zscore"],
                    help="LOBSTER input: prices relative to the window's last mid (default); relative_ps also "
                         "z-scores log sizes per stock; zscore = old per-stock price/size z-score")
    ap.add_argument("--model", default="deeplob", choices=["deeplob", "deeplob_nolstm", "light", "light2"])
    ap.add_argument("--bits", type=int, default=None, help="weight bits 2/4/8 for Brevitas QAT (light, light2)")
    ap.add_argument("--abits", type=int, default=None, help="activation bits (default = --bits), e.g. --bits 2 --abits 4")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--train-stride", type=int, default=1, help=">1 only for quick CPU tests")
    ap.add_argument("--test-stride", type=int, default=1)
    ap.add_argument("--init", default=None, help="FP32 checkpoint to start QAT from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="runs")
    return ap


def main():
    a = parser().parse_args()
    if a.bits is not None and a.model not in ("light", "light2"):
        raise SystemExit("--bits is only implemented for --model light / light2")
    if a.abits is not None and a.bits is None:
        raise SystemExit("--abits needs --bits")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prec = "fp32" if a.bits is None else f"w{a.bits}a{a.abits or a.bits}"
    ds = "" if a.dataset == "fi2010" else "lob_"
    dl = "" if a.delay_us == 0 else f"_d{a.delay_us:g}us"
    tag = a.tag or f"{ds}{a.model}_{prec}_k{a.horizon}{dl}_s{a.seed}"
    os.makedirs(a.out, exist_ok=True)

    train, val, test, tests, meta = load_data(a)
    nw = 2 if device == "cuda" else 0
    tl = DataLoader(train, batch_size=a.batch, shuffle=True, num_workers=nw)
    vl = DataLoader(val, batch_size=512, num_workers=nw)
    te = DataLoader(test, batch_size=512, num_workers=nw)
    print(f"[{tag}] device={device} train={len(train)} val={len(val)} test={len(test)}", flush=True)

    model = build(a.model, a.bits, a.abits)
    if a.init:
        missing, unexpected = model.load_state_dict(torch.load(a.init, map_location="cpu"), strict=False)
        print(f"init from {a.init}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    model = to_device(model, device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    counts = label_counts(train)
    print(f"[{tag}] train labels up/stat/down = {counts.tolist()}", flush=True)
    w = None
    if a.class_weight == "balanced":
        w = torch.tensor(counts.sum() / (3.0 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    best, best_ep, wait, hist = -1, -1, 0, []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        tot, n = 0.0, 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y); n += len(y)
        v = evaluate(model, vl, device)
        hist.append({"epoch": ep, "train_loss": tot / n, "val_f1": v["f1_macro"]})
        print(f"[{tag}] ep {ep} loss {tot / n:.4f} val_f1 {v['f1_macro']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if v["f1_macro"] > best:
            best, best_ep, wait = v["f1_macro"], ep, 0
            torch.save(model.state_dict(), f"{a.out}/{tag}.pt")
        else:
            wait += 1
            if wait >= a.patience:
                break

    model.load_state_dict(torch.load(f"{a.out}/{tag}.pt", map_location=device))
    res = {
        "tag": tag, "dataset": a.dataset, "model": a.model, "bits": a.bits, "abits": a.abits or a.bits, "horizon": a.horizon,
        "seed": a.seed, "delay_us": a.delay_us, "params": count_params(model), "best_epoch": best_ep,
        "epochs_run": len(hist), "train_windows": len(train), "train_stride": a.train_stride,
        "test_stride": a.test_stride, "train_seconds": round(time.time() - t0, 1), "device": device,
        "val": evaluate(model, vl, device), "test": evaluate(model, te, device),
        "test_parts": {k: evaluate(model, DataLoader(d, batch_size=512), device) for k, d in tests.items()},
        "lobster_meta": {k: v for k, v in meta.items() if k != "norms"} if meta else None,
        "args": vars(a), "history": hist,
    }
    with open(f"{a.out}/{tag}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{tag}] TEST f1_macro {res['test']['f1_macro']:.4f} acc {res['test']['accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()
