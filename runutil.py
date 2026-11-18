import argparse, glob, json
import numpy as np
import torch

from models import build
from train import parser

T = 100


def load_run(json_path, device="cpu"):
    res = json.load(open(json_path))
    if "args" not in res:
        raise SystemExit(f"{json_path} was made by an older train.py; re-run training to use this script")
    a = argparse.Namespace(**{**vars(parser().parse_args([])), **res["args"]})
    if a.dataset == "lobster" and res.get("lobster_meta"):
        a.alpha = res["lobster_meta"]["alpha"]
    model = build(res["model"], res["bits"], res.get("abits"))
    model.load_state_dict(torch.load(json_path[:-5] + ".pt", map_location=device))
    return res, model.to(device).eval(), a


def run_paths(pattern):
    paths = [p for p in sorted(glob.glob(pattern))
             if not p.endswith(("summary.json",)) and "/results/" not in p.replace("\\", "/")]
    if not paths:
        raise SystemExit(f"no run files match {pattern!r}")
    return paths


@torch.no_grad()
def predict(model, windows, device="cpu", batch=512):
    out = []
    for i in range(0, len(windows), batch):
        x = torch.stack(windows[i:i + batch]).to(device)
        out.append(model(x).argmax(1).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, dtype=np.int64)


@torch.no_grad()
def predict_dataset(model, ds, device="cpu", batch=512):
    ps, ys = [], []
    loader = torch.utils.data.DataLoader(ds, batch_size=batch)
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ps).numpy(), torch.cat(ys).numpy()


@torch.no_grad()
def predict_lobster_all(model, day, device="cpu", batch=512):
    return predict(model, [day.window(j) for j in day.j_all], device, batch)
