"""Loads the example raw cut and reports basic facts (no manuscript analysis).
Usage: python -I inspect_example_data.py ../data/before_Icapsub_first100sweeps.mat
"""
import sys
import numpy as np
from scipy.io import loadmat

m = loadmat(sys.argv[1])
fs = int(m["samplerate"][0, 0])          # Hz
x = m["sweepcut0"]                       # (2000 samples, 100 sweeps), pA
t = m["timecut"][:, 0]                   # ms
med = np.median(x, axis=1)
edge = int(np.argmax(np.abs(np.diff(med))))  # steepest sample of the median trace
print(f"sampling rate {fs} Hz, {x.shape[1]} sweeps x {x.shape[0]} samples (0 to {t[-1]:.2f} ms)")
print(f"steepest change of the median trace after sample {edge} (t = {t[edge]:.2f} ms)")
print(f"median first peak {med[edge:edge+30].min():.1f} pA")
print(f"pre-step SD {x[:edge-5].std():.2f} pA; tail SD (last 500 samples, per-sweep median) "
      f"{np.median(x[-500:].std(axis=0)):.3f} pA")
