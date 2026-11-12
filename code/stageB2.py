import numpy as np, json, glob, torch, warnings, time; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr, wilcoxon
from sim import *; from unet import *; from methods import *
torch.set_num_threads(2)
tuned=json.load(open('out/tuned.json')); ev=json.load(open('out/eval_idx.json')); cl=np.load('out/classical.npz'); abc=np.load('out/abc_results.npz')
models=[]
for p in sorted(glob.glob('out/unet_s*.pt')):
    m=UNet(); m.load_state_dict(torch.load(p)); models.append(m)
print('models',len(models))
x,B,s,N,u,st=make(600,777,HELD); Psim=predict(models,x)
est={'ABC envelope fit':None,'AsLS':cl['asls'],'Quantile-LP':cl['qlp'],'Quantile-LP + template':cl['qlp_t'],'U-Net (single)':Psim[0],'U-Net (ensemble)':Psim.mean(0)}
ma=lambda a:np.apply_along_axis(lambda r:np.convolve(r,np.ones(5)/5,'same'),1,a)
def metrics(b,idx,per=False):
    xs,Bs,ss,Ns=x[idx],B[idx],s[idx],N[idx]; e=b-Bs; sh=xs-b
    q=(sh[:,50:600].sum(1)-ss[:,50:600].sum(1))*DT; pk=ma(sh)[:,50:600].max(1)-ma(ss)[:,50:600].max(1)
    d={'RMSE 0-4 ms (pA)':np.sqrt((e[:,:400]**2).mean(1)),'RMSE 0-0.5 ms (pA)':np.sqrt((e[:,:50]**2).mean(1)),'RMSE all (pA)':np.sqrt((e**2).mean(1)),
       'tail bias 8-15 ms (pA)':e[:,800:].mean(1),'charge err (pA·ms)':q,'peak err (pA)':pk}
    return d
idx120=np.arange(120); tab={}; per={}
for k,b in est.items():
    if b is None: b=abc['sim']
    d=metrics(b,idx120) if k=='ABC envelope fit' else metrics(b[:120],idx120)
    per[k]=d
    tab[k]={'RMSE 0-4 ms (pA)':float(d['RMSE 0-4 ms (pA)'].mean()),'RMSE 0-0.5 ms (pA)':float(d['RMSE 0-0.5 ms (pA)'].mean()),'RMSE all (pA)':float(d['RMSE all (pA)'].mean()),
            'tail bias 8-15 ms (pA)':float(d['tail bias 8-15 ms (pA)'].mean()),'|charge err| (pA·ms)':float(np.abs(d['charge err (pA·ms)']).mean()),'charge err signed':float(d['charge err (pA·ms)'].mean()),
            '|peak err| (pA)':float(np.abs(d['peak err (pA)']).mean()),'peak err signed':float(d['peak err (pA)'].mean())}
    sh=(x[:120]-(abc['sim'] if k=='ABC envelope fit' else b[:120])); nul=N[:120]==0
    tab[k]['false activity null sweeps 0.5-6 ms (pA RMS)']=float(np.sqrt((sh[nul][:,50:600]**2).mean()))
# bootstrap CI + paired tests on RMSE 0-4 ms (n=120)
rng=np.random.default_rng(0); stats={}
for k in per:
    v=per[k]['RMSE 0-4 ms (pA)']; bs=[v[rng.integers(0,120,120)].mean() for _ in range(2000)]; stats[k]={'ci':[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]}
ens=per['U-Net (ensemble)']['RMSE 0-4 ms (pA)']
for k in per:
    if k!='U-Net (ensemble)': stats[k]['wilcoxon_p_vs_ens']=float(wilcoxon(ens,per[k]['RMSE 0-4 ms (pA)']).pvalue); stats[k]['ens_better_frac']=float((ens<per[k]['RMSE 0-4 ms (pA)']).mean())
# 600-sweep table (no ABC), by N and by step type, uncertainty
E600={k:v for k,v in est.items() if k!='ABC envelope fit'}
tab600={k:{m:float(np.mean(d)) for m,d in metrics(b,np.arange(600)).items()} for k,b in E600.items()}
unc={}; sdv=Psim.std(0); err=np.abs(Psim.mean(0)-B)
unc['spearman(ens SD, |err|) early']=float(spearmanr(sdv[:,:400].ravel(),err[:,:400].ravel())[0]); unc['mean SD early']=float(sdv[:,:400].mean()); unc['mean SD tail']=float(sdv[:,800:].mean())
unc['members RMSE']=[float(np.sqrt(((P-B)**2).mean())) for P in Psim]
pe=Psim.mean(0)
byN={int(n):{'n':int((N==n).sum()),'ens':float(np.sqrt(((pe[N==n]-B[N==n])[:,:400]**2).mean())),'qlp_t':float(np.sqrt(((cl['qlp_t'][N==n]-B[N==n])[:,:400]**2).mean())),'qlp':float(np.sqrt(((cl['qlp'][N==n]-B[N==n])[:,:400]**2).mean())),'asls':float(np.sqrt(((cl['asls'][N==n]-B[N==n])[:,:400]**2).mean()))} for n in range(5)}
byS={SNAMES[j]:{'n':int((st==j).sum()),'ens':float(np.sqrt(((pe[st==j]-B[st==j])[:,:400]**2).mean())),'qlp_t':float(np.sqrt(((cl['qlp_t'][st==j]-B[st==j])[:,:400]**2).mean())),'qlp':float(np.sqrt(((cl['qlp'][st==j]-B[st==j])[:,:400]**2).mean()))} for j in range(4)}
# ---------------- REAL: evaluation sweeps (never used for calibration) ----------------
V1=np.load('../proj/out/full_real.npz'); sig=SD
AT=[design(E[j]) for j in range(4)]; A0=design(None)
rng=np.random.default_rng(21)
real={}; hyb={}; raw={}; HEX={}
ABCsel={(int(a),int(b)):i for i,(a,b) in enumerate(abc['sel'])}
for si,k in enumerate(SNAMES):
    W=windows(k); idx=cl[f'{k}_idx']; Wx=W[idx]
    blk=np.array([b for b in range(5) for _ in ev[f'{k}_{b}']])
    # neighbour median (excluding the sweep itself, +-25 sweeps)
    nb=np.array([np.median(np.delete(W[max(0,i-25):i+26],i-max(0,i-25),axis=0),axis=0) for i in idx])
    ue=predict(models,Wx.astype(np.float32)).mean(0)
    E_={'U-Net v2 (full-recording calibration)':ue,'U-Net v1 (first 100 sweeps only)':V1[f'{k}_unet'][idx],'Quantile-LP + template':cl[f'{k}_qlp_t'],'Quantile-LP':cl[f'{k}_qlp'],'AsLS':cl[f'{k}_asls'],'Neighbour-sweep median (reference)':nb}
    real[k]={}
    for name,b in E_.items():
        r=Wx-b; real[k][name]={'early':float(np.sqrt((r[:,:50]**2).mean())),'mid':float(np.sqrt((r[:,50:600]**2).mean())),'late':float(np.sqrt((r[:,600:]**2).mean())),'tailoff':float(r[:,800:].mean()),
                   'by_block_early':[float(np.sqrt((r[blk==b_][:,:50]**2).mean())) for b_ in range(5)],'by_block_mid':[float(np.sqrt((r[blk==b_][:,50:600]**2).mean())) for b_ in range(5)]}
    raw[k]=(idx,blk,Wx,E_)
    # hybrid: add simulated events to the real sweeps
    sh_,Nh,_=events(len(idx)*2,rng); sh_=sh_[Nh>=1][:len(idx)]
    Wh=Wx+sh_; uh=predict(models,Wh.astype(np.float32)).mean(0)
    Hh={'U-Net v2 (full-recording calibration)':uh,'Quantile-LP + template':np.array([qlp(v,tuned['qlp_t'],E[si],AT[si]) for v in Wh]),'Quantile-LP':np.array([qlp(v,tuned['qlp'],None,A0) for v in Wh]),'AsLS':np.array([asls(v,*tuned['asls']) for v in Wh])}
    hyb[k]={'charge_injected':float((sh_[:,50:600].sum(1)*DT).mean())}
    HEX[k]={'Wx':Wx[:4],'Wh':Wh[:4],'s':sh_[:4],**{'h_'+n.replace(' ','_'):v[:4] for n,v in Hh.items()},**{'o_'+n.replace(' ','_'):E_[n][:4] for n in Hh}}
    for name,bh in Hh.items():
        if bh is None: continue
        db=bh-E_[name]; q=db[:,50:600].sum(1)*DT
        hyb[k][name]={'rms_db_0.5-6':float(np.sqrt((db[:,50:600]**2).mean())),'rms_db_0-0.5':float(np.sqrt((db[:,:50]**2).mean())),'abs_charge_leak':float(np.abs(q).mean()),'signed_charge_leak':float(q.mean())}
    # ABC subset
    sub=[(ABCsel[(si,int(i))],j) for j,i in enumerate(idx) if (si,int(i)) in ABCsel]
    if sub:
        ia=np.array([a for a,_ in sub]); ja=np.array([j for _,j in sub]); b_abc=abc['real'][ia]; r=Wx[ja]-b_abc
        real[k]['ABC envelope fit (subset)']={'n':len(ja),'early':float(np.sqrt((r[:,:50]**2).mean())),'mid':float(np.sqrt((r[:,50:600]**2).mean())),'late':float(np.sqrt((r[:,600:]**2).mean())),'tailoff':float(r[:,800:].mean())}
        real[k]['ref subset U-Net v2']={'early':float(np.sqrt(((Wx[ja]-ue[ja])[:,:50]**2).mean())),'mid':float(np.sqrt(((Wx[ja]-ue[ja])[:,50:600]**2).mean())),'late':float(np.sqrt(((Wx[ja]-ue[ja])[:,600:]**2).mean()))}
        # ABC hybrid
        rows=[ia_ for ia_ in ia]; 
    print(k,'real done',flush=True)
# wilcoxon on pooled evaluation sweeps: v2 vs v1 / QLP+T (mid window RMS per sweep, early too)
pool={}
for nm in ['U-Net v1 (first 100 sweeps only)','Quantile-LP + template','Quantile-LP','AsLS']:
    for w,(a,b) in {'early':(0,50),'mid':(50,600)}.items():
        v2=np.concatenate([np.sqrt(((raw[k][2]-raw[k][3]['U-Net v2 (full-recording calibration)'])[:,a:b]**2).mean(1)) for k in SNAMES]); o=np.concatenate([np.sqrt(((raw[k][2]-raw[k][3][nm])[:,a:b]**2).mean(1)) for k in SNAMES])
        pool[f'v2 vs {nm} [{w}]']={'p':float(wilcoxon(v2,o).pvalue),'frac_v2_lower':float((v2<o).mean()),'v2_mean':float(v2.mean()),'other_mean':float(o.mean())}
# ABC pathology on real
d=abc['real_d']; patho={'frac at bound':float(((np.abs(d)>59.4).any(1)).mean()),'frac negative rate':float(((d[:,[1,3,5]]<0).any(1)).mean()),'frac mixed-sign amplitudes':float(((np.sign(d[:,0])!=np.sign(d[:,2]))|(np.sign(d[:,0])!=np.sign(d[:,4]))).mean()),'abc sec per sweep':float(abc['sec_per'])}
json.dump({'sim120':tab,'sim120_stats':stats,'sim600':tab600,'unc':unc,'byN':byN,'byStep':byS,'real':real,'hybrid':hyb,'pooled_tests':pool,'abc_patho':patho,'noise_sd':float(SD),'timing':{'asls':float(cl['t_asls']),'qlp':float(cl['t_qlp']),'qlp_t':float(cl['t_qlp_t'])}},open('out/results.json','w'),indent=1,default=float)
np.savez('out/hybrid_ex.npz',**{f'{k}_{n}':v for k,d in HEX.items() for n,v in d.items()})
np.savez('out/preds.npz',Psim=Psim,**{f'{k}_unet':raw[k][3]['U-Net v2 (full-recording calibration)'] for k in SNAMES})
print('saved')
