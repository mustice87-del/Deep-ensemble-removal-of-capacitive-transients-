import numpy as np, json, glob, torch, warnings; warnings.filterwarnings('ignore')
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize
d=np.load('out/manual_cmp.npz'); W=d['W']
M={'Manual':d['Bref'],'U-Net ens.':d['U-Net_v2'],'QLP + template':d['QLP+template'],'QLP':d['QLP']}
a,b=400,1500   # late window 4-15 ms
sm=lambda s:uniform_filter1d(s,20,axis=-1,mode='nearest')
S={n:sm(W-B)[:,a:b] for n,B in M.items()}
print('smoothed noise SD (quiet tail, manual)',np.std(S['Manual'][:,-300:]))
# --- i from all-points histogram: mixture of Gaussians at 0,i,2i,3i with common width
def fit_i(x):
    x=x.ravel()[::5]
    def nll(p):
        i,sg,w1,w2,w3=p
        if i<0.8 or i>3.5 or sg<0.03 or min(w1,w2,w3)<1e-4: return 1e9
        w=np.array([1,w1,w2,w3]); w=w/w.sum(); lv=np.arange(4)*i
        pdf=(w[None,:]*np.exp(-0.5*((x[:,None]-lv[None,:])/sg)**2)/(sg*np.sqrt(2*np.pi))).sum(1)
        return -np.log(pdf+1e-12).sum()
    best=None
    for i0 in (1.2,1.6,2.0,2.4,2.8):
        r=minimize(nll,[i0,0.25,.3,.1,.03],method='Nelder-Mead',options={'maxiter':3000,'xatol':1e-3,'fatol':1e-2})
        if best is None or r.fun<best.fun: best=r
    return best.x
res={}
for n in S:
    p=fit_i(S[n]); res[n]={'i':float(p[0]),'sd':float(p[1]),'w':[float(v) for v in (np.array([1,*p[2:]])/np.array([1,*p[2:]]).sum())]}
    print(n,'i=%.2f sd=%.3f'%(p[0],p[1]),np.round(res[n]['w'],3))
iM=res['Manual']['i']
# histogram for figure
# --- late openings & occupancy with i=iM and i=2.0
def analyse(s,i):
    thr=(np.arange(1,6)-0.5)*i
    k=np.clip(np.round(s/i),0,None).astype(int)          # open level per sample
    ev=((k[:,1:]>=1)&(k[:,:-1]==0)).sum(1)              # openings from closed
    ev+= (k[:,0]>=1)
    mean_cur=s.mean(1); po=mean_cur/(i)                  # sum k /N later
    return k,ev
out={'fit':res,'late_window_ms':[4,15]}
for I in (iM,2.0):
    r={}
    km={}
    for n in S:
        k,ev=analyse(S[n],I); km[n]=(k,ev)
    Nm=int(km['Manual'][0].max())
    for n in S:
        k,ev=km[n]; N=int(k.max()); evm=km['Manual'][1]
        po=k.mean()/Nm
        r[n]={'N_max_late':N,'mean_open_ch':float(k.mean()),'Po_with_manual_N':float(po),'late_openings_per_sweep':float(ev.mean()),'sweeps_with_opening':float((ev>0).mean()),
              'corr_events_vs_manual':float(np.corrcoef(ev,evm)[0,1]),'exact_event_agree':float((ev==evm).mean()),'level_hist':[float((k==j).mean()) for j in range(5)],
              'level_agree_manual':float((k==km['Manual'][0]).mean()),'corr_meancur_vs_manual':float(np.corrcoef(k.mean(1),km['Manual'][0].mean(1))[0,1])}
    # events vs Po (per sweep), Chatelier Fig 4f analogue
    k,ev=km['Manual']; r['Manual']['corr_events_vs_Po']=float(np.corrcoef(ev,k.mean(1)/Nm)[0,1])
    # binomial test: per-sample open count vs binomial(N,p)
    pp=k.mean()/Nm; from scipy.stats import binom
    obs=np.array([(k==j).mean() for j in range(Nm+1)]); exp=binom.pmf(np.arange(Nm+1),Nm,pp)
    r['Manual']['binom_expected']=[float(v) for v in exp]; r['Manual']['binom_observed']=[float(v) for v in obs]
    out[f'i={I:.2f}']=r
json.dump(out,open('out/late.json','w'),indent=1)
for key in out:
    if key.startswith('i='):
        print(key)
        for n,v in out[key].items(): print(' ',n,{kk:(round(vv,3) if isinstance(vv,float) else vv) for kk,vv in v.items() if kk not in('binom_expected','binom_observed','level_hist')}, [round(x,3) for x in v['level_hist']])
        print('  binom exp',[round(x,3) for x in out[key]['Manual']['binom_expected']],'obs',[round(x,3) for x in out[key]['Manual']['binom_observed']])
np.savez('out/late_S.npz',**{n.replace(' ','_'):S[n] for n in S})
