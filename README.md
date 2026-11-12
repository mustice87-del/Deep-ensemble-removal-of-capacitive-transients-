# Data and code: removal of capacitive transients from cell-attached NaV1.6 recordings

Companion data and code for the manuscript on deep-ensemble (1D U-Net) baseline estimation of capacitive transients in a cell-attached NaV1.6 patch-clamp recording (CHO cell, one patch, 493 sweeps, 100 kHz).

## Contents
- `data/step3_windows.npz` — window of step 3 (+120 to +40 mV command step) for all 493 sweeps; 1500 samples each (15 ms), starting 6 samples (0.06 ms) after the command step. Arrays (493 x 1500, pA): `raw_current_pA`, `manual_baseline_pA`, `unet_ensemble_baseline_pA`, `qlp_template_baseline_pA`, `qlp_baseline_pA`, `asls_baseline_pA`; `time_s` (time in the sweep, s), `sweep_number` (1-493).
  `manual_baseline_pA` is the raw trace minus the operator's manual capacitive-current subtraction (not ground truth).
- `data/channel_counts_per_sweep.csv` — per sweep: peak of the smoothed channel current (pA) for each estimator, and channel counts at unitary currents of 1.5, 2.0 and 2.5 pA (the manuscript's main value is 1.8 pA: count = round(peak / 1.8)).
- `weights/unet_s0.pt, unet_s1.pt, unet_s2.pt` — the three ensemble members (PyTorch state dicts).
- `code/` — simulator, estimators, training, evaluation and figure scripts (Python; NumPy, SciPy, PyTorch, matplotlib). `code/common.py` holds the paths; edit them to your layout. Some scripts still contain an extra estimator that is not part of the manuscript.

## Not included
The raw Axon Binary Format file (about 430 MB, 493 sweeps, four voltage steps each) is too large for this repository. The real-sweep evaluation of steps s1, s2 and s4 (Tables 4 and 5) and the simulator calibration need it; it is available at: [DOI to be added].

## Reproducing
The step 3 tables (RMS difference to the manual baseline, channel counts, late openings) can be recomputed from `data/step3_windows.npz` with `code/cmp_manual.py`, `code/cnt.py`, `code/cnt2.py` and `code/late.py` (adapt the input paths).

## Licence
Code: MIT. Data: CC BY 4.0. [to be confirmed by the authors]

## Citation
[manuscript citation to be added]
