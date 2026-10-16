"""Plot training curves (train/validation loss per epoch) from saved training output.

Inputs can be
  * history.csv written by the training scripts (next to best.pt; on Modal: /<run_name>/history.csv), or
  * the text the training scripts print, saved to a file (for Modal: copy the app log, or run with output
    redirected). Any line containing `epoch N train_mse X | val_mse Y ...` is picked up; other lines are ignored.

    python plot_training_curves.py history.csv
    python plot_training_curves.py volume.log surface.log --labels volume surface --out curves.png
    python plot_training_curves.py run.log --channels ro vx vr vt p

Panels: train and validation MSE (log scale), per-channel validation relative L2, and the extra validation errors the
scripts report (wall MSE for the whole-mesh model, blade / plane MSE for the surface model) when present.
Fetch the CSV:  modal volume get geotransolver-runs /geotransolver/history.csv .
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NUM = r"([-+0-9.eE]+|nan|inf)"
LINE = re.compile(rf"epoch\s+(\d+)\s+train_mse\s+{NUM}")
VAL = re.compile(rf"val_mse\s+{NUM}")
BLADE_PLANE = re.compile(rf"blade\s+{NUM}\s+plane\s+{NUM}")
WALL = re.compile(rf"val_wall_mse\s+{NUM}")
REL = re.compile(r"val_relL2((?:\s+[-+0-9.eE]+|\s+nan)+)")


def read_log(path):
    """Parse printed training output -> {name: array} (later lines for the same epoch win)."""
    rows = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        m = LINE.search(line)
        if not m:
            continue
        r = {"epoch": int(m.group(1)), "train_mse": float(m.group(2))}
        for key, rx in (("val_mse", VAL), ("val_wall_mse", WALL)):
            mm = rx.search(line)
            if mm:
                r[key] = float(mm.group(1))
        mm = BLADE_PLANE.search(line)
        if mm:
            r["val_blade_mse"], r["val_plane_mse"] = float(mm.group(1)), float(mm.group(2))
        mm = REL.search(line)
        if mm:
            for i, v in enumerate(mm.group(1).split()):
                r[f"val_relL2_{i}"] = float(v)
        rows[r["epoch"]] = r
    return rows


def read_csv(path):
    with open(path, newline="") as fh:
        return {int(float(r["epoch"])): {k: float(v) for k, v in r.items() if v != ""} for r in csv.DictReader(fh)}


def load(path):
    rows = read_csv(path) if Path(path).suffix.lower() == ".csv" else read_log(path)
    if not rows:
        sys.exit(f"no epoch lines found in {path}")
    keys = sorted({k for r in rows.values() for k in r})
    ep = np.array(sorted(rows))
    return ep, {k: np.array([rows[e].get(k, np.nan) for e in ep]) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="history.csv or saved training output, one or more runs")
    ap.add_argument("--labels", nargs="+", default=None, help="one name per file (default: file names)")
    ap.add_argument("--channels", nargs="+", default=None,
                    help="names of the output channels for the relL2 panel (default: numbered, or taken from the CSV)")
    ap.add_argument("--linear", action="store_true", help="linear y axis instead of log")
    ap.add_argument("--out", default=None, help="save to this file instead of showing")
    a = ap.parse_args()
    labels = a.labels or [Path(f).stem for f in a.files]
    if len(labels) != len(a.files):
        sys.exit("--labels needs one name per file")

    runs = [load(f) for f in a.files]
    extra = [("val_wall_mse", "validation MSE at the wall points"), ("val_blade_mse", "validation MSE, blade points"),
             ("val_plane_mse", "validation MSE, plane points")]
    extra = [(k, t) for k, t in extra if any(k in d and np.isfinite(d[k]).any() for _, d in runs)]
    rel_keys = sorted({k for _, d in runs for k in d if k.startswith("val_relL2_")},
                      key=lambda k: (not k.rsplit("_", 1)[1].isdigit(), int(k.rsplit("_", 1)[1]) if k.rsplit("_", 1)[1].isdigit() else 0, k))

    n_panels = 2 + (1 if extra else 0)
    fig, axs = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 4.8), constrained_layout=True)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    scale = "linear" if a.linear else "log"

    ax = axs[0]
    for i, ((ep, d), lab) in enumerate(zip(runs, labels)):
        c = colours[i % len(colours)]
        ax.plot(ep, d["train_mse"], color=c, ls="--", lw=1.3, label=f"{lab} train")
        if "val_mse" in d:
            ax.plot(ep, d["val_mse"], color=c, lw=1.8, label=f"{lab} val")
            b = np.nanargmin(d["val_mse"])
            ax.plot(ep[b], d["val_mse"][b], "o", color=c, ms=6)  # best checkpoint
            print(f"{lab}: {len(ep)} epochs; best val_mse {d['val_mse'][b]:.4e} at epoch {ep[b]}; "
                  f"final train {d['train_mse'][-1]:.4e}, val {d['val_mse'][-1]:.4e}")
    ax.set_yscale(scale)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (standardised units)")
    ax.set_title("train / validation MSE (dot = best epoch)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axs[1]
    for i, ((ep, d), lab) in enumerate(zip(runs, labels)):
        c = colours[i % len(colours)]
        keys = [k for k in rel_keys if k in d and np.isfinite(d[k]).any()]
        for j, k in enumerate(keys):
            name = a.channels[j] if a.channels and j < len(a.channels) else k.replace("val_relL2_", "")
            ax.plot(ep, d[k], color=c, alpha=1 - 0.55 * j / max(len(keys), 1), lw=1.5,
                    label=f"{lab} {name}" if len(runs) > 1 or len(keys) > 1 else name)
    ax.set_yscale(scale)
    ax.set_xlabel("epoch")
    ax.set_ylabel("relative L2 error")
    ax.set_title("validation relative L2 per channel")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, ncol=2)
    else:
        ax.text(0.5, 0.5, "no per-channel errors found", ha="center", transform=ax.transAxes)
    ax.grid(alpha=0.3)

    if extra:
        ax = axs[2]
        for i, ((ep, d), lab) in enumerate(zip(runs, labels)):
            for j, (k, title) in enumerate(extra):
                if k in d and np.isfinite(d[k]).any():
                    ax.plot(ep, d[k], color=colours[(i * len(extra) + j) % len(colours)], lw=1.6,
                            label=f"{lab} {k.replace('val_', '').replace('_mse', '')}")
        ax.set_yscale(scale)
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE (standardised units)")
        ax.set_title("validation MSE by region")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    if a.out:
        fig.savefig(a.out, dpi=150)
        print("saved", a.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
