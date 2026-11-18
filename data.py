import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

HORIZONS = {10: -5, 20: -4, 30: -3, 50: -2, 100: -1}
T = 100


def load_file(path):
    cache = path[:-4] + ".npy"
    if os.path.exists(cache):
        return np.load(cache)
    try:
        import pandas as pd
        m = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float32, engine="c").to_numpy()
    except ImportError:
        m = np.loadtxt(path, dtype=np.float32)
    np.save(cache, m)
    return m


class LOBWindows(Dataset):

    def __init__(self, matrix, horizon=10, stride=1, stats=None):
        x = matrix[:40, :].T.copy()
        if stats is not None:
            x = (x - stats[0]) / stats[1]
        y = matrix[HORIZONS[horizon], :].astype(np.int64) - 1
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.idx = np.arange(T - 1, x.shape[0], stride)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        t = self.idx[i]
        window = self.x[t - T + 1:t + 1].unsqueeze(0)
        return window, self.y[t]


def train_stats(data_dir, val_frac=0.2, norm="DecPre"):
    m = load_file(os.path.join(data_dir, f"Train_Dst_NoAuction_{norm}_CF_7.txt"))
    cut = int(m.shape[1] * (1 - val_frac))
    x = m[:40, :cut].astype(np.float64)
    mean, std = x.mean(axis=1), x.std(axis=1)
    std[std < 1e-12] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def get_datasets(data_dir, horizon=10, train_stride=1, test_stride=1, val_frac=0.2, norm="DecPre",
                 zscore=True):
    train_m = load_file(os.path.join(data_dir, f"Train_Dst_NoAuction_{norm}_CF_7.txt"))
    stats = train_stats(data_dir, val_frac, norm) if zscore else None
    n = train_m.shape[1]
    cut = int(n * (1 - val_frac))
    train = LOBWindows(train_m[:, :cut], horizon, train_stride, stats)
    val = LOBWindows(train_m[:, cut:], horizon, test_stride, stats)
    tests = list(get_test_days(data_dir, horizon, test_stride, norm, zscore, val_frac).values())
    return train, val, ConcatDataset(tests)


def get_test_days(data_dir, horizon=10, test_stride=1, norm="DecPre", zscore=True, val_frac=0.2):
    stats = train_stats(data_dir, val_frac, norm) if zscore else None
    return {
        f"day{d + 1}": LOBWindows(load_file(os.path.join(data_dir, f"Test_Dst_NoAuction_{norm}_CF_{d}.txt")),
                                  horizon, test_stride, stats)
        for d in (7, 8, 9)
    }
