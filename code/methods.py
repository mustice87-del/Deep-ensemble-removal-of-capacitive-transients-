import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.optimize import linprog
from common import *
# ---------- 1) ABC exactly as in the manuscript ----------
def abc_baseline(x,seed=0,NP=100,maxc=1000,limit=20,lam=28.,lo=-60.,hi=60.,wfirst=300):
    rng=np.random.default_rng(seed); t=T; D=6
    def cost(Pm):
      with np.errstate(all='ignore'):
        B=-(Pm[:,0:1]*np.exp(-Pm[:,1:2]*t)+Pm[:,2:3]*np.exp(-Pm[:,3:4]*t)+Pm[:,4:5]*np.exp(-Pm[:,5:6]*t))
        jf=-(2*B[:,:wfirst].sum(1)+B[:,wfirst:].sum(1))
        v=np.minimum(x[None,:]-B,0.); jp=lam*(v**2).sum(1); out=jf+jp
      return np.where(np.isfinite(out),np.minimum(out,1e30),1e30)
    def fit(f): return np.where(f>=0,1/(1+f),1+np.abs(f))
    Xp=rng.uniform(lo,hi,(NP,D)); f=cost(Xp); tr=np.zeros(NP,int)
    bi=f.argmin(); best=Xp[bi].copy(); bf=f[bi]
    def neighbor(Xp):
        k=(np.arange(NP)+rng.integers(1,NP,NP))%NP; j=rng.integers(0,D,NP); phi=rng.uniform(-1,1,NP)
        V=Xp.copy(); V[np.arange(NP),j]=np.clip(Xp[np.arange(NP),j]+phi*(Xp[np.arange(NP),j]-Xp[k,j]),lo,hi); return V
    for c in range(maxc):
        V=neighbor(Xp); fv=cost(V); b=fv<f; Xp[b]=V[b]; f[b]=fv[b]; tr[b]=0; tr[~b]+=1
        pr=fit(f); pr=pr/pr.sum(); sel=rng.choice(NP,NP,p=pr)       # onlookers
        V=neighbor(Xp[sel]); fv=cost(V)
        for i,s in enumerate(sel):
            if fv[i]<f[s]: Xp[s]=V[i]; f[s]=fv[i]; tr[s]=0
            else: tr[s]+=1
        bi=f.argmin()
        if f[bi]<bf: bf=f[bi]; best=Xp[bi].copy()
        m=tr.argmax()
        if tr[m]>limit: Xp[m]=rng.uniform(lo,hi,D); f[m]=cost(Xp[m:m+1])[0]; tr[m]=0
    d=best; return -(d[0]*np.exp(-d[1]*T)+d[2]*np.exp(-d[3]*T)+d[4]*np.exp(-d[5]*T)), d
# ---------- 2) AsLS (Eilers & Boelens 2005) ----------
_D=sp.diags([1.,-2.,1.],[0,1,2],shape=(NSEG-2,NSEG)); _DtD=(_D.T@_D).tocsc()
def asls(x,lam=1e5,p=0.01,it=10):
    w=np.ones(NSEG)
    for _ in range(it):
        W=sp.diags(w); z=spsolve((W+lam*_DtD).tocsc(),w*x); w=p*(x>z)+(1-p)*(x<=z)
    return z
# ---------- 3/4) Quantile regression LP on exponential dictionary (+ optional shared early-transient template) ----------
RATES=np.geomspace(0.05,60,12)
def design(template=None):
    cols=[np.ones(NSEG)]+[-np.exp(-r*T) for r in RATES]
    if template is not None: cols.append(template)
    return np.array(cols).T
def qlp(x,tau=0.1,template=None,A=None):
    A=design(template) if A is None else A; K=A.shape[1]; n=NSEG
    # min tau*sum(u)+(1-tau)*sum(v)  s.t.  A th + u - v = x ;  u,v>=0 ; amplitudes (cols 1..12)>=0
    c=np.r_[np.zeros(K),tau*np.ones(n),(1-tau)*np.ones(n)]
    Aeq=sp.hstack([sp.csr_matrix(A),sp.identity(n),-sp.identity(n)],format='csc')
    bnds=[(None,None)]+[(0,None)]*len(RATES)+([(None,None)] if template is not None else [])+[(0,None)]*(2*n)
    r=linprog(c,A_eq=Aeq,b_eq=x,bounds=bnds,method='highs')
    if r.x is None:
        for mth in ('highs-ipm','highs-ds'):
            r=linprog(c,A_eq=Aeq,b_eq=x,bounds=bnds,method=mth)
            if r.x is not None: break
    th=r.x[:K]; return A@th
