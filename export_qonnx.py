import argparse, os
import torch
from brevitas.export import export_qonnx
from models import build, EXPORT_SHAPE

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="light", choices=["light", "light2"])
ap.add_argument("--ckpt", required=True)
ap.add_argument("--bits", type=int, required=True)
ap.add_argument("--abits", type=int, default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()

model = build(a.model, a.bits, a.abits)
model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
model.eval()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
export_qonnx(model, torch.randn(*EXPORT_SHAPE[a.model]), export_path=a.out)
print("wrote", a.out, "input shape", EXPORT_SHAPE[a.model])
