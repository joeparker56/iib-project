This folder is to test the geotransolver architecture on Chris' dataset of turbine blades.

These are using parabolic lean and sweep profiles (symetrical half span models).

These test sets have been generated based on the Harrison cascade:

Inlet blade/flow angle (from axial) 40.0°
Exit blade angle (from axial) -66.8°
 
Inlet boundary layer (x/C,, = -0.56) :
Displacement thickness 2.6 mm
Momentum thickness 1.9 mm
Kinetic energy thickness 3.3 mm

## Surrogate model (GeoTransolver)

Predicts the flow field (`ro vx vr vt p` by default) from mesh coordinates plus the three design parameters
(sweep, lean, theta). The dataset is a 5x5x5 grid of sweep x lean x theta (125 cases). Run all commands below from
this folder. Needs `torch`, `numpy` and `matplotlib`.

Files: `train_geotransolver.py` (training), `modal_train.py` (train on a Modal GPU),
`plot_blade_pressure.py` (3D blade pressure plot), `plot_exit_plane.py` (plane downstream of the trailing edge), `plot_loss_coefficient.py` (total pressure loss, predicted vs actual), `plot_validation_set.py` (all of these for every validation case), `plot_compare_models.py` (actual vs volume model vs surface model), `plot_training_curves.py` (loss per epoch), `plot_loss_compare.py` (loss at the surface model's plane: actual vs both models vs the dataset's loss), `plot_mse_heatmap.py` (MSE over the design space).

### Training

Cases are split randomly (seeded) by whole case into train/validation, 80/20 by default, so validation designs are
unseen. Fields are standardised per channel; coordinates and parameters are scaled to [-1, 1]. The checkpoint with
the lowest validation MSE is saved as `best.pt`.

Locally (the default `--data` path is wrong when run from this folder, so pass it explicitly):

    python train_geotransolver.py --data grid125_R23 --epochs 100 --batch_size 2

`--params both|global|none` (where sweep/lean/theta enter: `both` = per point and as the global embedding, the default;
`global` = only the global embedding, so the local input is just [x, r, rt]; `none` = not at all, so a case is known only
through the blade-surface geometry, and theta, which does not change the geometry, cannot be learned; the plot scripts
read this from the checkpoint),
Useful options: `--max_cases N` (quick debug run), `--n_points` (mesh points sampled per case per step),
`--wall_frac` (fraction of each step's points drawn from the layers next to the blade walls, default 0; try 0.3-0.5 if the blade pressure loading is poorly predicted), `--wall_layers`,
`--val_frac`, `--lr`, `--n_layers`, `--n_hidden`, `--seed`, `--out` (default `runs/geotransolver`).

On Modal (A100, 64 GB RAM). One-off dataset upload:

    modal volume create geotransolver-data
    modal volume put geotransolver-data grid125_R23 /grid125_R23

Train (extra arguments go straight to `train_geotransolver.py`):

    modal run modal_train.py --args "--epochs 100 --batch_size 4"
    modal run --detach modal_train.py --args "--epochs 300"     # keeps running if you disconnect

Download the checkpoint (saved even if the run fails or is interrupted):

    modal volume get geotransolver-runs /geotransolver/best.pt .

### Plotting

Both scripts take the checkpoint with `--ckpt` and save to a file with `--out` (otherwise a window opens).

Blade pressure on the 3D blade: actual, predicted and error, each seen from the pressure side (top row) and the
suction side (bottom row). Uses the first validation case unless `--case` is given. The checkpoint must have been
trained with `p` in `--variables` (the default).

    python plot_blade_pressure.py --ckpt best.pt
    python plot_blade_pressure.py --ckpt best.pt --case g222 --out blade.png

Several cases in one run (model and dataset loaded once, one PNG per case plus an `errors.csv` of blade-surface relative
L2 errors in the folder). `--split val|train|all` selects every case in a split, `--case` takes several ids, and
`--skip_existing` makes an interrupted batch restartable. Add `--cache DIR` to reuse whole-mesh predictions.

    python plot_blade_pressure.py --ckpt best.pt --split val --out_dir blade_val --cache pred_cache
    python plot_blade_pressure.py --ckpt best.pt --case g001 g134 g222 --out_dir blade_some

MSE heat map with sweep on x and lean on y, averaged over theta (`--per_theta` gives one panel per theta).
MSE is in standardised units. `--split val` scores only unseen cases (the default `all` includes training
cases, so looks better than reality); `--channel p` scores a single variable.

    python plot_mse_heatmap.py --ckpt best.pt --split val
    python plot_mse_heatmap.py --ckpt best.pt --per_theta --channel p --out mse.png

Plane downstream of the trailing edge: actual, predicted and error on a constant-axial-station plane (a full pitch by
span slice), drawn in 3D with the blade in grey, from upstream (top row) and downstream (bottom row). `--i` picks the
station from 66 (trailing edge) to 84 (end of the mesh), default 75; `--var` picks the variable (default `p`).

    python plot_exit_plane.py --ckpt best.pt
    python plot_exit_plane.py --ckpt best.pt --var vx --i 84 --out plane.png

`--var ske` plots secondary kinetic energy, 0.5 rho V_sec^2, computed from the predicted (and actual) `ro vx vr vt`.
V_sec is the velocity minus its component along the reference exit direction in the axial-tangential plane, i.e. the
spanwise vr plus the pitchwise deviation from that direction. The reference is the mass-flux-averaged exit yaw at
midspan (`--ref midspan`, default) or at each span (`--ref local`), taken from the actual field and used for both.
Check this matches the definition you use before quoting numbers.

    python plot_exit_plane.py --ckpt best.pt --var ske --i 84

`--var yp` plots the local stagnation pressure loss coefficient, Yp = (p01 - p0) / (p02 - p2), with p0 derived from the
predicted fields, p01 the case's `pp_p0_inlet`, and p02, p2 the mass-averaged exit total and static pressure of the
actual field (used for both maps so they are comparable). It also prints the mass-averaged Yp, actual vs predicted.
For `yp` and `ske` the default station is the product plane (I=84); other variables default to I=75.

    python plot_exit_plane.py --ckpt best.pt --var yp

Note on inference: the model's slice attention pools over every input point, so a point's prediction depends on the
other points in the same forward pass. Never predict a lone subset (a wall, a plane) - it is out of distribution and
gives badly wrong values. Use `predict_field` in `train_geotransolver.py`, which predicts the whole mesh in random
chunks like training and lets you read off any points afterwards (the plot scripts do this).

Total pressure loss, predicted vs actual, one point per case (parity plot): Yp = (p01 - p02)/(p02 - p2) and the
dataset's own 1 - p02/p01 (`pp_p0_loss`), from mass-averaged (rho*vx) exit-plane fields at each case's `pp_I_0based`
station. p01 is the case's `pp_p0_inlet`. Needs a model trained on `ro vx vr vt p`. It predicts whole meshes, so it is
slow on CPU (many minutes for 25 cases); use `--split val` and ideally a GPU.

    python plot_loss_coefficient.py --ckpt best.pt --split val --out loss.png --csv loss.csv

All plots for every validation case, saved to a folder (`val_plots/` by default): blade pressure and exit-plane
`p`, `ske` and `yp` maps per case, plus a validation MSE heat map and the loss parity plot. Whole-mesh predictions are
cached in `<out_dir>/.cache` and shared between plots, and existing images are skipped, so an interrupted run can be
restarted. Slow on CPU (a few minutes per case); the `--cache DIR` option on the individual plot scripts does the same.

    python plot_validation_set.py --ckpt best.pt
    python plot_validation_set.py --ckpt best.pt --plots blade yp --out_dir my_plots

### Blade-surface + plane model (`train_geotransolver_surface.py`)

A second training script that predicts the flow only where it is needed instead of on the whole mesh. The tokens are
the blade surface points (pressure and suction faces, leading to trailing edge, all spans; ~7.9k) and the plane at
axial index 75 (~4.5k), each case ~12k points, all of them used every step.
- local input (6): coordinates (x, r, rt), normalised, plus the unit surface normal. Blade normals point into the flow;
  the plane's normals are the true mesh-surface normals at I=75 (pointing downstream, close to but not exactly axial). The stored (x, r, rt) frame is treated as locally Cartesian.
- geometry context (3): the blade-surface coordinates.
- global input: `cases.csv` columns given by `--global_cols`, default `A1 M2 theta_unit` (inlet angle, exit Mach number,
  theta). In this dataset only sweep, lean and theta vary, so `A1`/`M2` are constant and carry no information; sweep
  and lean reach the model through the blade geometry. There is no inlet-velocity column.
- output (5): the stored `ro rovx rovr rorvt roe` at those points. Trained on standardised values; the checkpoint
  holds `f_mean`/`f_std` to turn predictions back into table values.
Because every forward pass sees the same point set, the whole-mesh context problem described above does not arise.

    python train_geotransolver_surface.py --data grid125_R23 --epochs 300
    modal run --detach modal_train.py --script train_geotransolver_surface.py --run-name surface --args "--epochs 300"

The log reports validation MSE split into blade and plane points.

Plotting: `plot_blade_pressure.py`, `plot_exit_plane.py`, `plot_mse_heatmap.py`, `plot_loss_coefficient.py` and
`plot_validation_set.py` detect which kind of checkpoint they were given and take the same arguments for both.
For a surface-model checkpoint (helper: `surface_model.py`):
- The model only predicts the blade surface and its own plane, so `plot_exit_plane.py` is fixed to that plane (`--i` must
  match, default 75) and `plot_loss_coefficient.py` evaluates Yp and 1 - p02/p01 at that plane. The dataset's
  `pp_p0_loss` is defined at the product plane (I=84), so it does not apply there. SKE and `yp` maps are likewise at I=75.
- `plot_exit_plane.py --var` accepts anything derivable from the five predicted conserved fields: `ro rovx rovr rorvt
  roe vx vr vt p M p0`, plus `ske` and `yp`. `plot_blade_pressure.py` derives p the same way.
- `plot_mse_heatmap.py` scores exactly the points the model predicts; `--region blade|plane|all` picks which.
- No whole-mesh prediction or `--cache` is needed (one forward pass of ~12k points), so these are fast.

    python plot_blade_pressure.py --ckpt best_surface.pt --split val --out_dir blade_val
    python plot_exit_plane.py --ckpt best_surface.pt --var M
    python plot_mse_heatmap.py --ckpt best_surface.pt --split val --region plane
    python plot_validation_set.py --ckpt best_surface.pt

Comparing the two models: `plot_compare_models.py` shows actual, the whole-mesh (volume) model and the blade-surface
model side by side, plus both errors, on the pressure face, suction face and the surface model's plane (I=75). Both
models are scored on the same points, and it prints the relative L2 error of each. `--var` is any of `ro vx vr vt p M p0`.
With the default settings both training scripts draw the same validation split, and each case is labelled train/val per
model. The volume model predicts the whole mesh, so use `--cache` to reuse those predictions.

    python plot_compare_models.py --volume_ckpt best.pt --surface_ckpt best_surface.pt --var p --cache pred_cache
    python plot_compare_models.py --volume_ckpt best.pt --surface_ckpt best_surface.pt --split all --out_dir compare --cache pred_cache --skip_existing

Training curves: both training scripts now write `history.csv` (one row per epoch: train and validation MSE, per-channel
validation relative L2, and the wall / blade / plane errors) next to `best.pt`. Runs made before this have no such file
and the checkpoint does not store the history, but `plot_training_curves.py` can also read the text the scripts print,
saved to a file (copy it from the Modal app log), and picks out the `epoch ... train_mse ... val_mse ...` lines.

    modal volume get geotransolver-runs /geotransolver/history.csv .
    python plot_training_curves.py history.csv
    python plot_training_curves.py volume.log surface_history.csv --labels volume surface --out curves.png

Loss at the surface model's plane: `plot_loss_compare.py` computes Yp and 1 - p02/p01 at the surface model's plane (I=75)
from the actual fields, the volume model and the surface model, and compares them with the dataset's `pp_p0_loss`, which
is defined further downstream at I=84 (so it is systematically larger). It prints RMSE / MAE / bias / R2 for each and
writes the per-case values with `--csv`. `--split val|train|all` picks the cases; `--cache` reuses volume predictions.

    python plot_loss_compare.py --volume_ckpt best.pt --surface_ckpt best_surface.pt --split val --cache val_plots/.cache --csv loss75.csv
