import json
import re
import subprocess
import sys


def with_seed(name, seed):
    return re.sub(r"_s\d+$", "", name) + f"_s{seed}"


src, seed = sys.argv[1], int(sys.argv[2])
a = json.load(open(f"runs/{src}.json"))["args"]
a["seed"] = seed
a["tag"] = with_seed(src, seed)
if a.get("init"):
    init = a["init"]
    stem = init.rsplit("/", 1)[-1][:-3]
    a["init"] = init[: -len(stem) - 3] + with_seed(stem, seed) + ".pt"
cmd = [sys.executable, "train.py"]
for k, v in a.items():
    if v is None:
        continue
    cmd += [f"--{k.replace('_', '-')}", str(v)]
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
