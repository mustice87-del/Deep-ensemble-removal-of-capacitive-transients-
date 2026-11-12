# Tune classical methods on TRAIN-pool simulations only
import numpy as np, json, warnings; warnings.filterwarnings('ignore')
from sim import *; from methods import *
xt,Bt,_,Nt,_,st=make(80,31,SRC); n=40
rm=lambda b,B:np.sqrt(((b-B)**2).mean(1))
AT=[design(E[s]) for s in range(4)]; A0=design(None)
r={(l,p):rm(np.array([asls(x,l,p) for x in xt[:n]]),Bt[:n]).mean() for l in [0.1,0.3,1.0,3.0,10.0,100.0,1e3,1e4,1e5] for p in [0.001,0.01,0.1,0.3]}
lam,p=min(r,key=r.get); print('AsLS',lam,p,round(r[(lam,p)],3),flush=True)
best={'asls':[lam,p]}
for name in ['qlp','qlp_t']:
    res={}
    for tau in [0.02,0.05,0.1,0.2,0.3,0.4,0.5]:
        if name=='qlp': b=np.array([qlp(x,tau,None,A0) for x in xt[:n]])
        else: b=np.array([qlp(x,tau,E[st[i]],AT[st[i]]) for i,x in enumerate(xt[:n])])
        res[tau]=rm(b,Bt[:n]).mean()
    tau=min(res,key=res.get); best[name]=tau; print(name,tau,{k:round(v,3) for k,v in res.items()},flush=True)
json.dump(best,open('out/tuned.json','w'))
