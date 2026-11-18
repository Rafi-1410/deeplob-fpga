import glob
import os
import re

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset, ConcatDataset
except ImportError:
    torch = None
    Dataset = object
    ConcatDataset = None

T = 100
LEVELS = 10
DUMMY_ASK = 9999999999
DUMMY_BID = -9999999999


def find_days(pattern):
    out = []
    for msg in sorted(glob.glob(pattern)):
        if "_message_" not in os.path.basename(msg):
            continue
        ob = msg.replace("_message_", "_orderbook_")
        if not os.path.exists(ob):
            raise FileNotFoundError(f"orderbook file missing for {msg}: expected {ob}")
        base = os.path.basename(msg)
        m = re.match(r"([A-Za-z.]+)_(\d{4}-\d{2}-\d{2})", base)
        name = f"{m.group(1)}_{m.group(2)}" if m else base.split("_message_")[0]
        out.append((name, msg, ob))
    if not out:
        raise FileNotFoundError(f"no LOBSTER message files match {pattern!r}")
    return out


def ticker_of(name):
    return name.split("_")[0]


def load_day(msg_path, ob_path, levels=LEVELS):
    msg = pd.read_csv(msg_path, header=None, usecols=range(6),
                      names=["time", "type", "id", "size", "price", "dir"])
    ob = pd.read_csv(ob_path, header=None).iloc[:, :4 * levels]
    if ob.shape[1] < 4 * levels:
        raise ValueError(f"{ob_path} has fewer than {levels} levels")
    if len(msg) != len(ob):
        raise ValueError(f"{msg_path} and {ob_path} have different row counts")
    keep = ~msg["type"].isin([5, 7]).to_numpy()
    lob = ob.to_numpy(dtype=np.float64)
    prices = lob[:, 0::2]
    keep &= ~((prices == DUMMY_ASK) | (prices == DUMMY_BID)).any(axis=1)
    lob = lob[keep]
    lob[:, 0::2] /= 10000.0
    time = msg["time"].to_numpy(dtype=np.float64)[keep]
    mid = (lob[:, 0] + lob[:, 2]) / 2.0
    return {"time": time, "lob": lob, "mid": mid, "dropped": int((~keep).sum())}


def decision_points(n_events, sample_every, stride=1):
    idx_s = np.arange(sample_every - 1, n_events, sample_every)
    j_all = np.arange(T - 1, len(idx_s), stride)
    return idx_s, j_all


def start_index(time, t_idx, delay_s):
    if delay_s <= 0:
        return np.asarray(t_idx).copy()
    s = np.searchsorted(time, time[t_idx] + delay_s, side="right") - 1
    return np.maximum(s, t_idx)


def returns(time, mid, t_idx, delay_s, horizon_events):
    s = start_index(time, t_idx, delay_s)
    cs = np.concatenate([[0.0], np.cumsum(mid)])
    end = s + 1 + horizon_events
    ok = end <= len(mid)
    r = np.full(len(t_idx), np.nan)
    sv, ev = s[ok], end[ok]
    p_end = (cs[ev] - cs[sv + 1]) / horizon_events
    r[ok] = (p_end - mid[sv]) / mid[sv]
    r[np.abs(r) < 1e-9] = 0.0
    return r


def classify(r, alpha):
    y = np.full(len(r), -1, dtype=np.int64)
    ok = ~np.isnan(r)
    y[ok & (r > alpha)] = 0
    y[ok & (r < -alpha)] = 2
    y[ok & (np.abs(r) <= alpha)] = 1
    return y


def balanced_alpha(r_list):
    r = np.concatenate([x[~np.isnan(x)] for x in r_list])
    return float(np.quantile(np.abs(r), 1.0 / 3.0))


def _day_returns(d, sample_every, horizon_events, lo_frac=0.0, hi_frac=1.0, stride=1):
    n = len(d["time"])
    idx_s, j_all = decision_points(n, sample_every, stride)
    t = idx_s[j_all]
    keep = (t >= int(lo_frac * n)) & (t + 1 + horizon_events < int(hi_frac * n))
    return returns(d["time"], d["mid"], t[keep], 0.0, horizon_events)


def events_elapsed(time, t_idx, delay_s):
    return start_index(time, t_idx, delay_s) - t_idx


def fit_norm(lobs):
    p = np.concatenate([l[:, 0::2].ravel() for l in lobs])
    v = np.concatenate([l[:, 1::2].ravel() for l in lobs])
    return {"p_mean": float(p.mean()), "p_std": float(p.std() + 1e-12),
            "v_mean": float(v.mean()), "v_std": float(v.std() + 1e-12)}


def fit_rel_norm(lobs):
    v = np.log1p(np.concatenate([l[:, 1::2].ravel() for l in lobs]))
    return {"mode": "relative", "v_mean": float(v.mean()), "v_std": float(v.std() + 1e-12)}


def apply_norm(lob, st):
    x = np.empty_like(lob, dtype=np.float32)
    x[:, 0::2] = (lob[:, 0::2] - st["p_mean"]) / st["p_std"]
    x[:, 1::2] = (lob[:, 1::2] - st["v_mean"]) / st["v_std"]
    return x


class LobsterDay(Dataset):

    def __init__(self, name, day, norm, alpha, horizon, sample_every, delay_us=0.0,
                 stride=1, first_frac=0.0, last_frac=1.0):
        self.name, self.alpha = name, alpha
        self.time, self.mid = day["time"], day["mid"]
        self.horizon_events = horizon * sample_every
        idx_s, j_all = decision_points(len(self.time), sample_every, stride)
        lo, hi = int(first_frac * len(idx_s)), int(last_frac * len(idx_s))
        j_all = j_all[(j_all >= max(lo, T - 1)) & (j_all < hi)]
        self.idx_s, self.j_all = idx_s, j_all
        self.label_end_limit = idx_s[hi] if hi < len(idx_s) else len(self.time)
        self.t_idx = idx_s[j_all]
        self.relative = norm.get("mode") == "relative"
        if self.relative:
            snap = day["lob"][idx_s]
            self.p = snap[:, 0::2]
            self.v = ((np.log1p(snap[:, 1::2]) - norm["v_mean"]) / norm["v_std"]).astype(np.float32)
            self.mid_s = day["mid"][idx_s]
        else:
            x = apply_norm(day["lob"][idx_s], norm)
            self.x = torch.from_numpy(x) if torch is not None else x
        self.relabel(delay_us)

    def relabel(self, delay_us):
        self.delay_us = float(delay_us)
        r = returns(self.time, self.mid, self.t_idx, delay_us * 1e-6, self.horizon_events)
        s = start_index(self.time, self.t_idx, delay_us * 1e-6)
        r[s + self.horizon_events >= self.label_end_limit] = np.nan
        self.y_all = classify(r, self.alpha)
        self.valid = self.y_all >= 0
        self.j = self.j_all[self.valid]
        self.y = self.y_all[self.valid]
        return self

    def events_elapsed(self, delay_us):
        return events_elapsed(self.time, self.t_idx, delay_us * 1e-6)

    def window(self, j):
        if not self.relative:
            return self.x[j - T + 1:j + 1].unsqueeze(0)
        m = self.mid_s[j]
        x = np.empty((T, self.p.shape[1] * 2), dtype=np.float32)
        x[:, 0::2] = (self.p[j - T + 1:j + 1] - m) / m * 1e3
        x[:, 1::2] = self.v[j - T + 1:j + 1]
        return torch.from_numpy(x).unsqueeze(0) if torch is not None else x[None]

    def __len__(self):
        return len(self.j)

    def __getitem__(self, i):
        return self.window(self.j[i]), int(self.y[i])


def get_lobster_datasets(train_glob, test_glob, horizon=10, sample_every=10, delay_us=0.0,
                         train_stride=1, test_stride=1, alpha=None, val_frac=0.2,
                         warmup_frac=0.1, levels=LEVELS, verbose=True, lob_norm="relative",
                         test_frac=0.2):
    train_days = [(n, load_day(m, o, levels)) for n, m, o in find_days(train_glob)]
    temporal = not test_glob
    test_days = [] if temporal else [(n, load_day(m, o, levels)) for n, m, o in find_days(test_glob)]
    if not temporal:
        test_frac = 0.0
    tr_end = 1.0 - val_frac - test_frac

    norms = {}
    if lob_norm == "relative":
        rel = fit_rel_norm([d["lob"][:int(len(d["lob"]) * tr_end)] for _, d in train_days])
        norms = {ticker_of(n): rel for n, _ in train_days + test_days}
    elif lob_norm == "relative_ps":
        for tk in sorted({ticker_of(n) for n, _ in train_days}):
            norms[tk] = fit_rel_norm([d["lob"][:int(len(d["lob"]) * tr_end)]
                                      for n, d in train_days if ticker_of(n) == tk])
    else:
        for tk in sorted({ticker_of(n) for n, _ in train_days}):
            cut_lobs = []
            for n, d in train_days:
                if ticker_of(n) == tk:
                    cut_lobs.append(d["lob"][:int(len(d["lob"]) * tr_end)])
            norms[tk] = fit_norm(cut_lobs)
    train_tickers = {ticker_of(n) for n, _ in train_days}

    H = horizon * sample_every
    if isinstance(alpha, (int, float)):
        alphas = {ticker_of(n): float(alpha) for n, _ in train_days + test_days}
    else:
        alphas = dict(alpha or {})
        by_tk = {}
        for n, d in train_days:
            if ticker_of(n) not in alphas:
                by_tk.setdefault(ticker_of(n), []).append(
                    _day_returns(d, sample_every, H, 0.0, tr_end, train_stride))
        for n, d in test_days:
            if ticker_of(n) not in alphas and ticker_of(n) not in by_tk:
                by_tk.setdefault(ticker_of(n), []).append(
                    _day_returns(d, sample_every, H, 0.0, warmup_frac))
        alphas.update({tk: balanced_alpha(rs) for tk, rs in by_tk.items()})

    kw = dict(horizon=horizon, sample_every=sample_every, delay_us=delay_us)
    train = [LobsterDay(n, d, norms[ticker_of(n)], alphas[ticker_of(n)], stride=train_stride,
                        last_frac=tr_end, **kw) for n, d in train_days]
    val = [LobsterDay(n, d, norms[ticker_of(n)], alphas[ticker_of(n)], stride=test_stride,
                      first_frac=tr_end, last_frac=1 - test_frac, **kw) for n, d in train_days]
    tests, unseen = {}, []
    if temporal:
        for n, d in train_days:
            tests[n] = LobsterDay(n, d, norms[ticker_of(n)], alphas[ticker_of(n)], stride=test_stride,
                                  first_frac=1 - test_frac, **kw)
    for n, d in test_days:
        tk = ticker_of(n)
        if tk in train_tickers:
            tests[n] = LobsterDay(n, d, norms[tk], alphas[tk], stride=test_stride, **kw)
        else:
            cut = int(len(d["lob"]) * warmup_frac)
            nm = (norms[tk] if lob_norm == "relative"
                  else fit_rel_norm([d["lob"][:cut]]) if lob_norm == "relative_ps"
                  else fit_norm([d["lob"][:cut]]))
            tests[n] = LobsterDay(n, d, nm, alphas[tk], stride=test_stride,
                                  first_frac=warmup_frac, **kw)
            unseen.append(n)

    meta = {"alpha": alphas, "norms": norms, "unseen_test_tickers": unseen,
            "train_days": [n for n, _ in train_days], "test_days": [n for n, _ in test_days],
            "dropped_rows": {n: d["dropped"] for n, d in train_days + test_days}}
    if verbose:
        print("LOBSTER label threshold per ticker: "
              + ", ".join(f"{k}={v:.3g}" for k, v in sorted(alphas.items())), flush=True)
        for name, part in [("train", train), ("val", val), ("test", list(tests.values()))]:
            for t in part:
                c = np.bincount(t.y, minlength=3)
                print(f"  {name:5s} {t.name:18s} up/stat/down = {c.tolist()}  "
                      f"({np.round(c / max(c.sum(), 1), 2).tolist()})", flush=True)
        print(f"  unseen test tickers: {unseen}", flush=True)
    return ConcatDataset(train), ConcatDataset(val), tests, meta
