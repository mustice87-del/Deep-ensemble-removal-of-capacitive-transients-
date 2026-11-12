# Simulator ingredients from the FULL recording: quiet sweeps of all 4 steps x 5 time blocks
import numpy as np, json
from scipy.signal import welch, bessel, sosfreqz
from scipy.optimize import minimize_scalar
from common import *
Wk={k:windows(k) for k in SNAMES}
src=[];held=[];ev={}   # entries (step,sweep)
P={}
for k in SNAMES:
    act=activity(Wk[k])
    for b,(lo,hi) in enumerate(BLOCKS):
        o=lo+np.argsort(act[lo:hi])
        q=o[:12]; hl=[q[2],q[6],q[10]]; sr=[i for j,i in enumerate(q) if j not in (2,6,10)]
        for i in sr: src.append((k,int(i),b))
        for i in hl: held.append((k,int(i),b))
        ev[f'{k}_{b}']=[int(i) for i in o[12:42]]
for (k,i,b) in src+held: P[(k,i)]=fit_exp(Wk[k][i],nstart=25)
print('fits',len(P),flush=True)
E={}
for k in SNAMES:
    ids=[(kk,i) for (kk,i,b) in src if kk==k]
    res=np.array([Wk[k][i]-bmodel(P[(k,i)]) for (_,i) in ids]); e=np.median(res,0)
    n=int(0.6/DT); w=np.ones(NSEG); k0=int(0.35/DT); w[k0:n]=0.5*(1+np.cos(np.pi*(np.arange(k0,n)-k0)/(n-k0))); w[n:]=0
    E[k]=e*w; print(k,'E min %.2f at %.2f ms'%(E[k].min(),T[E[k].argmin()]))
# noise PSD from tails of all sweeps of all steps
tail=np.array([(Wk[k][i]-bmodel(P[(k,i)]))[1000:] for (k,i,b) in src+held]); tail=tail-tail.mean(1,keepdims=True)
f,Pxx=welch(tail,fs=FS*1e3,nperseg=256,axis=1); Pm=Pxx.mean(0)
def loss(fc):
    sos=bessel(4,fc*1e3,'low',fs=FS*1e3,output='sos',norm='mag'); _,h=sosfreqz(sos,worN=f,fs=FS*1e3)
    m=(f>500)&(f<40000); model=np.abs(h[m])**2; plat=Pm[(f>200)&(f<1500)].mean()
    return np.mean((np.log(Pm[m])-np.log(plat*model))**2)
r=minimize_scalar(loss,bounds=(2,15),method='bounded'); print('Bessel-4 fc',r.x)
sd=float(tail.std()); print('noise sd',sd)
np.savez('out/ingredients.npz',E=np.array([E[k] for k in SNAMES]),fc=r.x,sd=sd,psd_f=f,psd=Pm,
  src_meta=np.array([[SNAMES.index(k),i,b] for k,i,b in src]),held_meta=np.array([[SNAMES.index(k),i,b] for k,i,b in held]),
  src_p=np.array([P[(k,i)] for k,i,b in src]),held_p=np.array([P[(k,i)] for k,i,b in held]))
json.dump(ev,open('out/eval_idx.json','w'))
print(len(src),len(held))
