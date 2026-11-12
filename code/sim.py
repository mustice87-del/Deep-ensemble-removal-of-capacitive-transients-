import numpy as np
from scipy.signal import bessel, sosfilt
from common import *
ING=np.load('out/ingredients.npz')
E=ING['E']; SD=float(ING['sd']); psd_f=ING['psd_f']; psd=ING['psd']      # E: (4,1500), one ringing template per step type
SRC=(ING['src_p'],ING['src_meta'][:,0]); HELD=(ING['held_p'],ING['held_meta'][:,0])
_pl=psd[(psd_f>200)&(psd_f<1500)].mean(); FC3=float(psd_f[np.argmax((psd_f>1500)&(psd<_pl/2))])
FCK=float(ING['fc']); SOS=bessel(4,FCK*1e3,'low',fs=FS*1e3,output='sos',norm='mag')
_fr=np.fft.rfftfreq(NSEG,DT*1e-3); _amp=np.sqrt(np.interp(_fr,psd_f,psd)); _amp[0]=0
def noise(M,rng):
    W=np.fft.rfft(rng.standard_normal((M,NSEG)),axis=1)*_amp
    n=np.fft.irfft(W,NSEG,axis=1); n*= SD/ n.std(); return n
def sample_baselines(M,rng,pool):
    Pp,Ss=pool; B=np.empty((M,NSEG)); st=np.empty(M,int)
    for m in range(M):
        ia=rng.integers(len(Pp)); a=Pp[ia]; s=Ss[ia]
        if rng.random()<0.5:
            same=np.flatnonzero(Ss==s); b=Pp[same[rng.integers(len(same))]]; w=rng.random()
            p=np.empty(7); p[1:]=np.exp(w*np.log(a[1:]+1e-3)+(1-w)*np.log(b[1:]+1e-3)); p[0]=w*a[0]+(1-w)*b[0]
        else: p=a.copy()
        p=p.copy(); p[1:4]*=np.exp(rng.normal(0,.2,3)); p[4:7]*=np.exp(rng.normal(0,.2,3)); p[0]+=rng.normal(0,.1)
        g=max(0.3,rng.normal(1,.25))
        B[m]=bmodel(p)+g*E[s]; st[m]=s
    return B,st
EV={'unit':(1.0,2.2),'ml':(.15,.9),'md':(.15,.7)}
def events(M,rng,pmax=4):
    cur=np.zeros((M,NSEG+1)); Ns=rng.choice(np.arange(pmax+1),size=M,p=[.15,.2,.25,.25,.15])
    unit=rng.uniform(*EV['unit'],M); ml=rng.uniform(*EV['ml'],M); md=rng.uniform(*EV['md'],M)
    for ch in range(pmax):
        act=Ns>ch
        t=rng.gamma(2,ml/2)
        for o in range(3):
            d=rng.exponential(md); s=np.round(t/DT).astype(int); e=np.round((t+d)/DT).astype(int)
            ok=act&(s<NSEG)
            idx=np.where(ok)[0]; np.add.at(cur,(idx,s[idx]),unit[idx]); np.add.at(cur,(idx,np.minimum(e[idx],NSEG)),-unit[idx])
            reopen=rng.random(M)<0.3
            act=act&reopen; t=t+d+rng.exponential(.15,M)
    s=np.cumsum(cur[:,:NSEG],axis=1); s=sosfilt(SOS,s,axis=1); return s,Ns,unit
def make(M,seed,pool):
    rng=np.random.default_rng(seed); B,st=sample_baselines(M,rng,pool); s,N,u=events(M,rng); n=noise(M,rng)
    return B+s+n,B,s,N,u,st
