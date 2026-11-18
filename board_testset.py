import argparse
import numpy as np
import torch

from models import build
from data import get_test_days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="light", choices=["light", "light2"])
    ap.add_argument("--bits", type=int, default=None)
    ap.add_argument("--abits", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    model = build(a.model, a.bits, a.abits)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    model.eval()

    days = get_test_days(a.data, a.horizon, test_stride=1)
    per_day = a.n // len(days)
    xs, ys, ds = [], [], []
    for name, d in days.items():
        for i in np.linspace(0, len(d) - 1, per_day).astype(int):
            x, y = d[i]
            xs.append(x); ys.append(int(y)); ds.append(int(name[3:]))
    x = torch.stack(xs)
    if a.model == "light2":
        x = x.permute(0, 3, 2, 1).contiguous()

    with torch.no_grad():
        logits = torch.cat([model(x[i:i + 256]) for i in range(0, len(x), 256)])
    pred = logits.argmax(1).numpy()
    y = np.array(ys)

    x_nchw = x.numpy().astype(np.float32)
    np.savez_compressed(
        a.out, x_nchw=x_nchw, x_nhwc=np.transpose(x_nchw, (0, 2, 3, 1)).copy(),
        y=y, torch_logits=logits.numpy().astype(np.float32), torch_pred=pred, day=np.array(ds))
    print(f"wrote {a.out}: {len(y)} windows, x_nchw {x_nchw.shape}, "
          f"PyTorch accuracy on them {np.mean(pred == y):.4f}")


if __name__ == "__main__":
    main()
