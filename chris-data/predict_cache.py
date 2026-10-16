"""Cache whole-mesh predictions so several plots of the same case (and checkpoint) share one forward pass."""
from pathlib import Path

import numpy as np
import torch

from train_geotransolver import predict_field


def cached_predict_field(cache_dir, ckpt_path, case, model, coords_n, g, geo, chunk):
    """predict_field, but stored as <cache_dir>/<ckpt>_<mtime>_<case>.npy (standardised units, (N, C)).
    The key includes the checkpoint's modification time, so retraining invalidates old entries."""
    if cache_dir is None:
        return predict_field(model, coords_n, g, geo, chunk)
    ck = Path(ckpt_path)
    f = Path(cache_dir) / f"{ck.stem}_{int(ck.stat().st_mtime)}_{case}.npy"
    if f.exists():
        return torch.from_numpy(np.load(f))
    out = predict_field(model, coords_n, g, geo, chunk)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, out.numpy())
    return out
