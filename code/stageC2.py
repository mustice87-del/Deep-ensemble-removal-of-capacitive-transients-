import numpy as np, json, glob, torch, warnings; warnings.filterwarnings('ignore')
import sim; from sim import *; from unet import *; from methods import *
torch.set_num_threads(2)
models=[]
for p in sorted(glob.glob('out/unet_s*.pt')):
    m=UNet(); m.load_state_dict(torch.load(p)); models.append(m)
tuned=json.load(open('out/tuned.json')); E0=sim.E.copy(); SD0=sim.SD; EV0=dict(sim.EV)
def shift(E,k):
    o=np.zeros_like(E); o[:,k:]=E[:,:-k]; return o
variants={'default':{},'transient shifted +1 sample (10 µs)':{'E':shift(E0,1)},'transient shifted +2 samples (20 µs)':{'E':shift(E0,2)},
 'noise x1.5':{'SD':1.5*SD0},'larger/longer events (unit 2-3.5 pA, latency up to 2 ms)':{'EV':{'unit':(2.0,3.5),'ml':(.5,2.0),'md':(.3,1.5)}},
 'shift +1 sample and noise x1.5':{'E':shift(E0,1),'SD':1.5*SD0},'ringing template x2 (gain)':{'E':2*E0}}
AT=[design(E0[j]) for j in range(4)]; A0=design(None)
out={}
for name,v in variants.items():
    sim.E=v.get('E',E0); sim.SD=v.get('SD',SD0); sim.EV=v.get('EV',EV0)
    x,B,s,N,u,st=make(300,888,HELD)
    P=predict(models,x).mean(0)
    bq=np.array([qlp(xx,tuned['qlp_t'],E0[st[i]],AT[st[i]]) for i,xx in enumerate(x)])     # the QLP+T template is always the UNshifted one
    b2=np.array([qlp(xx,tuned['qlp'],None,A0) for xx in x])
    r=lambda b,sl:float(np.sqrt(((b-B)[:,sl]**2).mean()))
    out[name]={k:{'RMSE 0-4 ms':r(b,slice(0,400)),'RMSE 0-0.5 ms':r(b,slice(0,50))} for k,b in [('U-Net ensemble',P),('Quantile-LP + template',bq),('Quantile-LP',b2)]}
    print(name,out[name],flush=True)
json.dump(out,open('out/robust.json','w'),indent=1)
