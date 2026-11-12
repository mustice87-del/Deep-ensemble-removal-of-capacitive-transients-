# Classical estimators on (i) simulated test set, (ii) the real evaluation sweeps (never used for calibration)
import numpy as np, json, time, warnings; warnings.filterwarnings('ignore')
from sim import *; from methods import *
tuned=json.load(open('out/tuned.json')); ev=json.load(open('out/eval_idx.json'))
AT=[design(E[s]) for s in range(4)]; A0=design(None)
x,B,s,N,u,st=make(600,777,HELD); out={}
t0=time.time(); out['asls']=np.array([asls(v,*tuned['asls']) for v in x]); out['t_asls']=(time.time()-t0)/600
t0=time.time(); out['qlp']=np.array([qlp(v,tuned['qlp'],None,A0) for v in x]); out['t_qlp']=(time.time()-t0)/600
t0=time.time(); out['qlp_t']=np.array([qlp(v,tuned['qlp_t'],E[st[i]],AT[st[i]]) for i,v in enumerate(x)]); out['t_qlp_t']=(time.time()-t0)/600
print('sim done',flush=True)
for si,k in enumerate(SNAMES):
    W=windows(k); idx=np.array([i for b in range(5) for i in ev[f'{k}_{b}']])
    xs=W[idx]; out[f'{k}_idx']=idx
    out[f'{k}_asls']=np.array([asls(v,*tuned['asls']) for v in xs]); out[f'{k}_qlp']=np.array([qlp(v,tuned['qlp'],None,A0) for v in xs])
    out[f'{k}_qlp_t']=np.array([qlp(v,tuned['qlp_t'],E[si],AT[si]) for v in xs]); print(k,flush=True)
np.savez('out/classical.npz',**out); print('done')
