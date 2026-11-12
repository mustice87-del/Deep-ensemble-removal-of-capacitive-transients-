# Data and code: removal of capacitive transients from a cell-attached NaV1.6 recording

Minimal dataset and code for the manuscript on deep-ensemble (1D U-Net) estimation of the capacitive baseline in a cell-attached NaV1.6 recording (CHO cell, one patch, 493 sweeps, 100 kHz, four voltage steps per sweep).

## Data (`data/`)
All windows are 1500 samples (15 ms) long and start 6 samples (0.06 ms) after the command step; step positions in the raw recording (sample index): s1 3437, s2 103437, s3 123437, s4 173437. Currents are in pA. `time_ms` gives the time of each sample after the command step.

- `step3_windows.npz` / `step3_windows.mat` — step 3 (+120 to +40 mV), all 493 sweeps: `raw_current_pA`, `manual_baseline_pA` (raw trace minus the operator's manual capacitive-current subtraction; an estimate, not ground truth), and the baselines of `unet_ensemble`, `qlp_template`, `qlp` and `asls`; `time_s`, `sweep_number`.
- `evaluation_windows_s1_s4.npz` — the 600 real evaluation windows (150 per step) with all estimator baselines: `s1_...` to `s4_...` arrays (`_sweep_index_1based`, `_raw_current_pA`, `_unet_ensemble_baseline_pA`, `_qlp_template_baseline_pA`, `_qlp_baseline_pA`, `_asls_baseline_pA`). Steps s1 and s2 are sign-flipped so that all steps are negative-going, as in the manuscript.
- `quiet_sweep_windows.npz` — the 180 quiet source windows used to build the simulator and the 60 held-out windows (`src_*`, `held_*`; `*_meta` columns: step 0-3 = s1-s4, 0-based sweep index, time block 0-4; `*_p`: fitted exponential parameters), plus the noise standard deviation, the fitted Bessel cutoff and the ringing templates.
- `channel_counts_per_sweep.csv` — per sweep (step 3): peak of the smoothed channel current for each estimator and the channel counts at unitary currents of 1.5, 2.0 and 2.5 pA (the manuscript's main value is 1.8 pA: count = round(peak / 1.8)).

## Weights (`weights/`)
`unet_s0.pt`, `unet_s1.pt`, `unet_s2.pt`: the three ensemble members (PyTorch state dicts). Network definition: `code/unet.py`.

## Code (`code/`)
- `reproduce_results.py` runs on the data in this repository only (numpy, scipy). It recomputes the residual RMS of the four estimators on the 600 evaluation windows and the step 3 channel-count histograms, e.g. the manual reference `[32, 290, 145, 25, 1]` sweeps with 0-4 channels at 1.8 pA.
  Run from the repository root: `python code/reproduce_results.py`.
- The other scripts (simulator, estimators, training, evaluation, figures) are the analysis pipeline. They read the original recording (Axon Binary Format) through a NumPy array whose path is set at the top of `common.py`; that recording is not part of this repository and is available from the authors on request.

## Not included
The complete raw recording (about 430 MB) is not in the repository. All numbers that the manuscript reports for step 3 and for the 600 evaluation sweeps can be recomputed from the files above; recalibrating the simulator from the full recording needs the raw file.

## Licence
Code: MIT. Data: CC BY 4.0. [to be confirmed by the authors]
