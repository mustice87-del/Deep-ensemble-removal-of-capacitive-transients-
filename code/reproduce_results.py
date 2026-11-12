"""Recompute Table 4-type residual RMS and the step 3 channel counts from the packaged data only.
Usage: python code/reproduce_results.py   (run from the repository root; needs numpy and scipy)"""
import numpy as np
from scipy.ndimage import uniform_filter1d
E_ = np.load('data/evaluation_windows_s1_s4.npz')
print('Residual RMS (pA), RMS over all evaluation sweeps of a step; early = 0-0.5 ms, mid = 0.5-6 ms')
for s in ('s1', 's2', 's3', 's4'):
    W = E_[s + '_raw_current_pA'].astype(float)
    for m in ('unet_ensemble', 'qlp_template', 'qlp', 'asls'):
        r = W - E_[f'{s}_{m}_baseline_pA'].astype(float)
        print(f'{s} {m:14s} early {np.sqrt((r[:, :50]**2).mean()):.2f}  mid {np.sqrt((r[:, 50:600]**2).mean()):.2f}')
D = np.load('data/step3_windows.npz')
W = D['raw_current_pA'].astype(float)
print('\nChannel counts per sweep at i = 1.8 pA (step 3, 493 sweeps): histogram of 0,1,2,3,4,... channels')
for m in ('manual', 'unet_ensemble', 'qlp_template', 'qlp', 'asls'):
    ch = uniform_filter1d(W - D[f'{m}_baseline_pA'].astype(float), 20, axis=1, mode='nearest')
    k = np.clip(np.round(ch[:, 50:600].max(1) / 1.8), 0, None).astype(int)
    print(f'{m:14s}', np.bincount(k, minlength=5).tolist(), ' largest count', k.max())
