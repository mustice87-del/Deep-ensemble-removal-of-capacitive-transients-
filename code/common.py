import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import median_filter
XPATH='/tmp/claude-0/-home-claude/f203aef0-e204-5e23-8173-a0b5de6668b3/scratchpad/newdata/X.npy'
FS=100.0; DT=0.01; NSEG=1500; T=np.arange(NSEG)*DT
STEPS={'s1':3437,'s2':103437,'s3':123437,'s4':173437}; SIGN={'s1':-1,'s2':-1,'s3':1,'s4':1}; OFF=6
SNAMES=['s1','s2','s3','s4']; STEPTXT={'s1':'-30→+80 mV','s2':'+80→+120 mV','s3':'+120→+40 mV','s4':'+40→-30 mV'}
BLOCKS=[(0,100),(100,200),(200,300),(300,400),(400,493)]
_X=None
def X():
    global _X
    if _X is None: _X=np.load(XPATH,mmap_mode='r')
    return _X
def windows(k):      # (493,1500) in model orientation (negative-going step response)
    i0=STEPS[k]+OFF; return (SIGN[k]*np.asarray(X()[:,i0:i0+NSEG])).astype(np.float64)
def activity(W):
    med=median_filter(W,size=(51,1),mode='nearest'); return (W-med)[:,50:400].std(1)
def bmodel(p,t=T):
    c=p[0]; a=p[1:4]; r=p[4:7]
    return c-(a[:,None]*np.exp(-r[:,None]*t[None,:])).sum(0)
def fit_exp(x, nstart=30, rng=None):
    rng=rng or np.random.default_rng(0)
    best=None
    lo=[-3,0,0,0,0.02,0.02,0.02]; hi=[3,60,60,60,400,400,400]
    for s in range(nstart):
        r=np.sort(10**rng.uniform(-1,2.3,3))[::-1]
        a=np.abs(rng.normal(5,3,3))
        p0=np.r_[x[-50:].mean(),a,r]
        p0=np.clip(p0,np.array(lo)+1e-6,np.array(hi)-1e-6)
        try:
            res=least_squares(lambda p:bmodel(p)-x,p0,bounds=(lo,hi),loss='soft_l1',f_scale=0.5,max_nfev=300)
        except Exception: continue
        if best is None or res.cost<best.cost: best=res
    return best.x
