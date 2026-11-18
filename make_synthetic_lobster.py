import argparse, os
import numpy as np


def make_day(path_prefix, n=20000, start_price=100.0, tick=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t = 34200 + np.cumsum(rng.exponential(0.001, n))
    steps = rng.choice([-1, 0, 0, 0, 0, 1], size=n)
    best_bid = np.round(start_price / tick) + np.cumsum(steps)
    spread = rng.choice([1, 1, 1, 2], size=n)
    best_ask = best_bid + spread
    types = rng.choice([1, 1, 1, 2, 3, 3, 4, 5], size=n)
    types[rng.integers(0, n)] = 7
    msg = np.column_stack([t, types, rng.integers(1, 10**7, n), rng.integers(1, 500, n),
                           (best_bid * tick * 10000).astype(np.int64), rng.choice([-1, 1], n)])
    ob = np.zeros((n, 40), dtype=np.int64)
    for lvl in range(10):
        ob[:, 4 * lvl + 0] = ((best_ask + lvl) * tick * 10000).astype(np.int64)
        ob[:, 4 * lvl + 1] = rng.integers(100, 2000, n)
        ob[:, 4 * lvl + 2] = ((best_bid - lvl) * tick * 10000).astype(np.int64)
        ob[:, 4 * lvl + 3] = rng.integers(100, 2000, n)
    np.savetxt(path_prefix + "_message_10.csv", msg, delimiter=",", fmt=["%.9f"] + ["%d"] * 5)
    np.savetxt(path_prefix + "_orderbook_10.csv", ob, delimiter=",", fmt="%d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="lobster_synth")
    ap.add_argument("--events", type=int, default=20000)
    a = ap.parse_args()
    for split, specs in {"train": [("AAA", "2012-06-18", 100.0), ("AAA", "2012-06-19", 101.0),
                                   ("BBB", "2012-06-18", 30.0)],
                         "test": [("AAA", "2012-06-21", 102.0), ("CCC", "2012-06-21", 55.0)]}.items():
        os.makedirs(os.path.join(a.out, split), exist_ok=True)
        for i, (tk, day, p0) in enumerate(specs):
            prefix = os.path.join(a.out, split, f"{tk}_{day}_34200000_57600000")
            make_day(prefix, a.events, p0, seed=(1000 if split == "test" else 0) + i)
    print(f"wrote synthetic LOBSTER files to {a.out}/train and {a.out}/test (CCC is an unseen ticker)")


if __name__ == "__main__":
    main()
