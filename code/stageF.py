import numpy as np, json, glob, time, torch, warnings; warnings.filterwarnings('ignore')
import sim; from sim import *; from unet import *; from methods import *
torch.set_num_threads(2)
models=[]
for p in sorted(glob.glob('out/unet_s*.pt')):
    m=UNet(); m.load_state_dict(torch.load(p)); models.append(m)
tuned=json.load(open('out/tuned.json')); cl=np.load('out/classical.npz'); AT=[design(E[j]) for j in range(4)]; A0=design(None)
out={'ood_hybrid':{}}
sim.EV={'unit':(2.0,3.5),'ml':(.5,2.0),'md':(.3,1.5)}
rng=np.random.default_rng(77)
for si,k in enumerate(SNAMES):
    W=windows(k); idx=cl[f'{k}_idx']; Wx=W[idx]
    sh_,Nh,_=events(len(idx)*2,rng); sh_=sh_[Nh>=1][:len(idx)]; Wh=Wx+sh_
    base={'U-Net ensemble':predict(models,Wx.astype(np.float32)).mean(0),'Quantile-LP + template':cl[f'{k}_qlp_t'],'Quantile-LP':cl[f'{k}_qlp'],'AsLS':cl[f'{k}_asls']}
    hyb={'U-Net ensemble':predict(models,Wh.astype(np.float32)).mean(0),'Quantile-LP + template':np.array([qlp(v,tuned['qlp_t'],E[si],AT[si]) for v in Wh]),'Quantile-LP':np.array([qlp(v,tuned['qlp'],None,A0) for v in Wh]),'AsLS':np.array([asls(v,*tuned['asls']) for v in Wh])}
    o={'charge_injected':float((sh_[:,50:600].sum(1)*DT).mean())}
    for n in base:
        db=hyb[n]-base[n]; q=db[:,50:600].sum(1)*DT; o[n]={'rms_db_0.5-6':float(np.sqrt((db[:,50:600]**2).mean())),'abs_charge_leak':float(np.abs(q).mean()),'signed_charge_leak':float(q.mean())}
    out['ood_hybrid'][k]=o; print(k,o,flush=True)
# timing (one CPU thread, per sweep)
torch.set_num_threads(1)
W=windows('s3')[:150].astype(np.float32); t=time.time(); predict(models,W); out['time_unet_ens_per_sweep_s']=(time.time()-t)/150
W64=W.astype(np.float64); t=time.time(); [asls(w,*tuned['asls']) for w in W64[:50]]; out['time_asls']=(time.time()-t)/50
t=time.time(); [qlp(w,tuned['qlp_t'],E[2],AT[2]) for w in W64[:10]]; out['time_qlp_t']=(time.time()-t)/10
t=time.time(); [abc_baseline(w,seed=1) for w in W64[:3]]; out['time_abc']=(time.time()-t)/3; print(out['time_unet_ens_per_sweep_s'],out['time_abc'],flush=True)
# ABC repeatability on real sweeps: 3 sweeps x 5 seeds
ev=json.load(open('out/eval_idx.json')); W4=windows('s4'); res={}
for i in ev['s4_2'][:3]:
    bs=[];ds=[]
    for sd in range(5): b,d=abc_baseline(W4[i],seed=sd); bs.append(b); ds.append(d)
    bs=np.array(bs); ds=np.array(ds); mb=bs.mean(0)
    res[int(i)]={'rms_between_runs_full':float(np.sqrt(((bs-mb)**2).mean(1)).mean()),'rms_between_runs_0-4ms':float(np.sqrt(((bs-mb)[:,:400]**2).mean(1)).mean()),'max_pair_rms_0-4ms':float(max(np.sqrt(((bs[a]-bs[b])[:400]**2).mean()) for a in range(5) for b in range(a)) ),'param_sd_max':float(ds.std(0).max())}
    print(i,res[int(i)],flush=True)
out['abc_repeat']=res
json.dump(out,open('out/stageF.json','w'),indent=1)
