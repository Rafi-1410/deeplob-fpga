import argparse
import json
import re

import numpy as np
from pynq import Device
from pynq.ps import Clocks

from driver import io_shape_dict
from driver_base import FINNExampleOverlay


def default_platform():
    src = open("driver.py").read()
    m = re.search(r"--platform['\"].*?default\s*=\s*['\"]([^'\"]+)", src, re.S)
    return m.group(1) if m else "zynq-iodma"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitfile", default="../bitfile/finn-accel.bit")
    ap.add_argument("--fclk", type=float, required=True, help="PL clock in MHz, e.g. 66.67")
    ap.add_argument("--npz", required=True, help="test set from board_testset.py")
    ap.add_argument("--reps", type=int, default=200, help="batch-1 latency repetitions")
    ap.add_argument("--chunk", type=int, default=250, help="batch used for the correctness check")
    ap.add_argument("--big-batch", type=int, default=500)
    ap.add_argument("--out", default="board_result.json")
    a = ap.parse_args()

    plat = default_platform()
    accel = FINNExampleOverlay(bitfile_name=a.bitfile, platform=plat, io_shape_dict=io_shape_dict,
                               batch_size=1, fclk_mhz=a.fclk, weight_dir="weights/",
                               device=Device.devices[0])
    fclk = float(Clocks.fclk0_mhz)
    print(f"platform {plat}  requested {a.fclk} MHz  PL clock now {fclk:.3f} MHz", flush=True)

    d = np.load(a.npz)
    x = d["x_nhwc"].astype(np.float32)
    accel.batch_size = a.chunk
    outs = []
    for i in range(0, len(x), a.chunk):
        xb = x[i:i + a.chunk]
        nb = len(xb)
        if nb < a.chunk:
            xb = np.concatenate([xb, np.zeros((a.chunk - nb,) + xb.shape[1:], np.float32)])
        o = accel.execute([xb])
        o = np.asarray(o[0] if isinstance(o, list) else o).reshape(a.chunk, -1)
        outs.append(o[:nb])
    out = np.concatenate(outs)
    pred = out.argmax(1)
    agree = float(np.mean(pred == d["torch_pred"]))
    res = {"fclk_mhz": fclk, "n": int(len(x)),
           "agreement_with_pytorch": agree,
           "fpga_acc": float(np.mean(pred == d["y"])),
           "pytorch_acc": float(np.mean(d["torch_pred"] == d["y"])),
           "max_abs_logit_diff": float(np.abs(out - d["torch_logits"]).max())}
    print(f"agreement with PyTorch {agree:.4f}   FPGA acc {res['fpga_acc']:.4f}   "
          f"PyTorch acc {res['pytorch_acc']:.4f}   max |logit diff| {res['max_abs_logit_diff']:.2e}",
          flush=True)

    accel.batch_size = 1
    runs = [accel.throughput_test() for _ in range(a.reps)]
    keys = ["runtime[ms]", "fold_input[ms]", "pack_input[ms]", "copy_input_data_to_device[ms]",
            "copy_output_data_from_device[ms]", "unpack_output[ms]", "unfold_output[ms]"]
    lat = {}
    for k in keys:
        v = np.array([r[k] for r in runs if k in r])
        if len(v):
            lat[k] = {"median": float(np.median(v)), "p99": float(np.percentile(v, 99)),
                      "min": float(v.min())}
    host = sum(lat[k]["median"] for k in keys[1:] if k in lat)
    res["batch1"] = lat
    res["batch1_end_to_end_median_ms"] = lat["runtime[ms]"]["median"] + host
    print(f"batch 1 accelerator runtime: median {lat['runtime[ms]']['median']:.4f} ms  "
          f"p99 {lat['runtime[ms]']['p99']:.4f} ms  min {lat['runtime[ms]']['min']:.4f} ms", flush=True)
    print(f"batch 1 end-to-end incl. ARM-side packing/copies: ~{res['batch1_end_to_end_median_ms']:.3f} ms",
          flush=True)

    accel.batch_size = a.big_batch
    r = accel.throughput_test()
    res["throughput"] = {"batch_size": a.big_batch, "runtime_ms": r["runtime[ms]"],
                         "samples_per_s": r["throughput[images/s]"]}
    print(f"batch {a.big_batch}: {r['throughput[images/s]']:.0f} samples/s", flush=True)

    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
