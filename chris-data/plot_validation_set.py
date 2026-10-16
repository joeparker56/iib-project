"""Make the plots for every validation case and save them to a folder.

    python plot_validation_set.py --ckpt best.pt                  # -> val_plots/
    python plot_validation_set.py --ckpt best.pt --out_dir my_plots --plots blade yp

Per case:  blade/<case>.png (blade pressure), exit_p/, exit_ske/, exit_yp/ (exit-plane maps, 2D)
Summary:   summary_mse_heatmap.png, summary_loss_parity.png, loss.csv
Predicting a whole mesh is slow on CPU (minutes per case), so predictions are cached in <out_dir>/.cache and shared
by all plots of a case. Existing images are skipped, so an interrupted run can simply be restarted
(--overwrite to redo them). Each plot runs as its own process, so one failure does not stop the rest.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
PER_CASE = {  # name -> (script, extra args, folder)
    "blade": ("plot_blade_pressure.py", [], "blade"),
    "p": ("plot_exit_plane.py", ["--var", "p"], "exit_p"),
    "ske": ("plot_exit_plane.py", ["--var", "ske"], "exit_ske"),
    "yp": ("plot_exit_plane.py", ["--var", "yp"], "exit_yp"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=str(ROOT / "grid125_R23"))
    ap.add_argument("--out_dir", default=str(ROOT / "val_plots"))
    ap.add_argument("--plots", nargs="+", default=list(PER_CASE) + ["summary"], choices=list(PER_CASE) + ["summary"])
    ap.add_argument("--cases", nargs="+", default=None, help="only these validation case ids")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    out = Path(a.out_dir)
    cache = out / ".cache"
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    # train_geotransolver trains on evenly spread `picks`; val_idx indexes that list
    import csv

    with open(Path(a.data) / "cases.csv", newline="") as fh:
        ids = [r["case"] for r in csv.DictReader(fh)]
    n_used = len(ckpt["train_idx"]) + len(ckpt["val_idx"])
    picks = torch.linspace(0, len(ids) - 1, n_used).round().long().tolist()
    val_cases = sorted(ids[picks[i]] for i in ckpt["val_idx"])
    if a.cases:
        unknown = set(a.cases) - set(val_cases)
        if unknown:
            sys.exit(f"not validation cases: {sorted(unknown)}")
        val_cases = [c for c in val_cases if c in a.cases]
    print(f"{len(val_cases)} validation cases: {' '.join(val_cases)}", flush=True)

    env = dict(os.environ, MPLBACKEND="Agg")
    common = ["--ckpt", a.ckpt, "--data", a.data]
    failures, t0 = [], time.time()

    def run(script, args, target, use_cache=True):
        if target.exists() and not a.overwrite:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        cache_args = ["--cache", str(cache)] if use_cache else []  # the heatmap predicts its own random subsets
        r = subprocess.run([sys.executable, str(ROOT / script), *common, *cache_args, *args, "--out", str(target)],
                           env=env, capture_output=True, text=True)
        if r.returncode != 0 or not target.exists():
            failures.append(target.name if target.parent == out else f"{target.parent.name}/{target.name}")
            print(f"  FAILED {failures[-1]}:\n{r.stderr[-600:]}", flush=True)

    for n, case in enumerate(val_cases, 1):
        for name in (p for p in a.plots if p in PER_CASE):
            script, extra, folder = PER_CASE[name]
            run(script, ["--case", case, *extra], out / folder / f"{case}.png")
        print(f"[{n}/{len(val_cases)}] {case} done ({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)

    if "summary" in a.plots and not a.cases:
        run("plot_mse_heatmap.py", ["--split", "val"], out / "summary_mse_heatmap.png", use_cache=False)
        run("plot_loss_coefficient.py", ["--split", "val", "--csv", str(out / "loss.csv")], out / "summary_loss_parity.png")
    print(f"finished in {(time.time() - t0) / 60:.1f} min; plots in {out}", flush=True)
    if failures:
        print("failed:", ", ".join(failures))


if __name__ == "__main__":
    main()
