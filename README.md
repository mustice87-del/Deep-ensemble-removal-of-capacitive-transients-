[README.md](https://github.com/user-attachments/files/34732525/README.md)
# Minimal dataset: Deep-ensemble removal of capacitive transients in cell-attached NaV1.6 patch-clamp recordings

Companion to the manuscript by M. Demirtaş (Bursa Uludağ University). Status of each item is stated below;

## Contents
| Path | Description | Status |
|---|---|---|
| `data/before_Icapsub_first100sweeps.mat` / `.csv` | Raw (before capacitive subtraction) current, step s3, sweeps 1–100, first 20 ms of the cut | included |
| `tables/Table1.csv … Table9.csv` | Numerical values of manuscript Tables 1–9 (summary results) | included |
| `code/inspect_example_data.py` | Loads the example cut and prints sampling rate, step position and noise | included |
| Full recording (ABF 2.6, 493 sweeps, 2.2 s, 100 kHz) | Basis of all analyses (Section 2.1) 
| `after_Icapsub_allsweeps.mat` | Experimenter's manual subtraction, step s3, 493 sweeps (Section 2.5) 
| Simulator, estimators (AsLS, QLP, QLP+template), U-Net training/evaluation, injection test, counting and late-opening code | Used for all results 
| Trained weights (3 ensemble members, v1 and v2) | Section 2.4 
| Stored source-fit parameters, templates E(t), noise spectrum, quiet/evaluation sweep indices | Section 2.2 
| Per-sweep result files behind Tables 4–9 and Figures 3–8 | Primary data of the figures 

## Example data: `before_Icapsub_first100sweeps.mat`
| Variable | Size | Unit | Definition |
|---|---|---|---|
| `samplerate` | 1×1 | Hz | Sampling rate (100000) |
| `sweepcut0` | 2000×100 | pA | Raw current, rows = samples, columns = sweeps 1–100, recorded sign |
| `timecut` | 2000×1 | ms | Time of each row from the first sample of the cut (0 to 19.99 ms) |

The CSV has the same data (column `time_ms`, then `sweep1…sweep100`).
Facts verified from the file: the median trace is flat (~1.3 pA) until row ~441 (4.4 ms), then falls to −17.4 pA within ~0.1–0.2 ms, which is the negative-going s3 transient (+120 to +40 mV); last-500-sample noise SD (per-sweep median) is 0.24 pA, consistent with the 0.25 pA noise floor in the manuscript.

## Recording protocol (from the manuscript)
Cell-attached CHO cell expressing NaV1.6; pCLAMP 10.3.2.1, Axon Binary Format 2.6, 21 July 2022; protocol "cell att Late current"; 493 sweeps, each 2.2 s, 220,000 samples at 100 kHz, 5 s apart (41.1 min). Command potentials (applied to the pipette): −30 mV 34.4 ms, +80 mV 1.0 s, +120 mV 0.2 s, +40 mV 0.5 s, −30 mV 0.47 s. Steps: s1 −30→+80, s2 +80→+120, s3 +120→+40, s4 +40→−30 mV. Analysis windows: 1500 samples (15 ms) starting 6 samples (60 µs) after the command step. Membrane potential unknown (resting potential not recorded).

## Tables
`tables/TableN.csv` reproduce the numbers printed in the manuscript (Table 1 simulated benchmark; 2 by channel number; 3 robustness; 4 real-sweep residuals; 5 injection test; 6 vs manual baseline; 7 channel counts; 8 simulated counts; 9 late openings). Units: pA unless stated; charge in pA·ms. Brackets are 95% bootstrap intervals.

## Software (as in the manuscript)
Python 3.13, NumPy 2.5.3, SciPy 1.18.1 (HiGHS), MATLAB R2026a (ABF reading).

## Privacy and licence
No personal or confidential data (cell-line recording only). Suggested licence: CC-BY 4.0 for data, an OSI licence (e.g. MIT) for code. Deposit in a trusted repository (e.g. Zenodo) to obtain a DOI and cite it in the Data Availability Statement.
