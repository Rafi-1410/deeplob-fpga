import sys
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

from runutil import load_run, predict_dataset
from train import load_data


def show(name, ds, model):
    p, y = predict_dataset(model, ds)
    f1 = f1_score(y, p, average="macro")
    print(f"\n== {name}: n={len(y)}  F1={f1:.3f}  acc={np.mean(p == y):.3f}")
    print("   true dist  ", np.round(np.bincount(y, minlength=3) / len(y), 2).tolist())
    print("   pred dist  ", np.round(np.bincount(p, minlength=3) / len(p), 2).tolist())
    print("   confusion (rows=true up/stat/down, cols=pred):")
    for row in confusion_matrix(y, p, labels=[0, 1, 2]):
        print("    ", row.tolist())


def main():
    res, model, args = load_run(sys.argv[1])
    _, val, _, tests, _ = load_data(args)
    print(f"run {res['tag']}  model {res['model']}  dataset {args.dataset}")
    show("validation", val, model)
    for name, ds in tests.items():
        show(name, ds, model)


if __name__ == "__main__":
    main()
