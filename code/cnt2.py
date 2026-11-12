import numpy as np, glob, torch, json, warnings; warnings.filterwarnings('ignore')
from sim import *; from unet import *; from methods import *
from scipy.ndimage import uniform_filter1d, label
tuned=json.load(open('out/tuned.json'))
I=1.8; MA=20
pk=lambda s:uniform_filter1d(s,MA,axis=1,mode='nearest')[:,50:600].max(1)
d=np.load('out/manual_cmp.npz'); Wr=d['W']
names=['Manual','U-Net ens.','QLP + template','QLP']
Mb={'Manual':d['Bref'],'U-Net ens.':d['U-Net_v2'],'QLP + template':d['QLP+template'],'QLP':d['QLP']}
out={}
# real: N by block, ensemble-average Po, sweeps with k>=3
blk=[(0,100),(100,200),(200,300),(300,400),(400,493)]
real={}
for n in names:
    s=Wr-Mb[n]; k=np.clip(np.round(pk(s)/I),0,None).astype(int)
    ens=uniform_filter1d(s.mean(0),20,mode='nearest')
    ip=int(np.argmax(ens[50:600]))+50; pe=float(ens[ip])
    late=uniform_filter1d(s,20,axis=1,mode='nearest')[:,400:]
    kl=np.clip(np.round(late/I),0,None).astype(int)
    real[n]={'k_hist':np.bincount(k,minlength=6)[:6].tolist(),'N_early':int(k.max()),'n_sweeps_k>=3':int((k>=3).sum()),'n_sweeps_k>=4':int((k>=4).sum()),
      'N_by_block':[int(k[a:b].max()) for a,b in blk],'mean_k':float(k.mean()),'ens_peak_pA':pe,'ens_peak_ms':float(ip*0.01),
      'late_mean_pA':float(s[:,400:].mean()),'Nlate':int(kl.max())}
    for N in (3,4): real[n][f'Po_peak_N{N}']=pe/(I*N)
    real[n]['Po_late_N3']=real[n]['late_mean_pA']/(I*3); real[n]['Po_late_N4']=real[n]['late_mean_pA']/(I*4)
out['real']=real
# sim, two fresh pools
models=[]
for p in sorted(glob.glob('out/unet_s*.pt')):
    m=UNet(); m.load_state_dict(torch.load(p)); models.append(m)
AT=[design(E[j]) for j in range(4)]; A0=design(None)
rows=[]
for seed in (777,4242):
    x,B,s,N,u,st=make(600,seed,HELD)
    ue=predict(models,x).mean(0)
    est={'U-Net ens.':ue,'QLP + template':np.array([qlp(v,tuned['qlp_t'],E[st[i]],AT[st[i]]) for i,v in enumerate(x)]),
         'QLP':np.array([qlp(v,tuned['qlp'],None,A0) for v in x]),'AsLS':np.array([asls(v,*tuned['asls']) for v in x])}
    kt=np.round(pk(s)/u).astype(int)
    rows.append((x,s,u,kt,est,N)); print('seed',seed,'done',flush=True)
simres={}
for n in ['U-Net ens.','QLP + template','QLP','AsLS']:
    ex=[];p1=[];bi=[];ox=[];op1=[];pe=[]
    for x,s,u,kt,est,N in rows:
        ke=np.clip(np.round(pk(x-est[n])/I),0,None).astype(int)
        ko=np.clip(np.round(pk(x-est[n])/u),0,None).astype(int)
        ex+=list(ke==kt);p1+=list(abs(ke-kt)<=1);bi+=list(ke-kt);ox+=list(ko==kt);op1+=list(abs(ko-kt)<=1);pe+=list((pk(x-est[n])-pk(s))**2)
    simres[n]={'exact':float(np.mean(ex)),'pm1':float(np.mean(p1)),'bias':float(np.mean(bi)),'oracle_exact':float(np.mean(ox)),'oracle_pm1':float(np.mean(op1)),'peak_rmse':float(np.sqrt(np.mean(pe)))}
simres['n']=1200
kt_all=np.concatenate([r[3] for r in rows]); simres['true_hist']=np.bincount(kt_all,minlength=6)[:6].tolist(); simres['true_N']=int(kt_all.max())
# sim late-opening agreement (known events) at i=1.8 for min 30 samples
def runs(k):
    lab,n=label(k>=1); return n
def lateruns(sig):
    sm=uniform_filter1d(sig,20,axis=1,mode='nearest')[:,400:]
    k=np.clip(np.round(sm/I),0,None).astype(int); c=[]
    for kk in k:
        lab,n=label(kk>=1); c.append(sum((lab==j).sum()>=30 for j in range(1,n+1)))
    return np.array(c)
tru=np.concatenate([lateruns(r[1]) for r in rows])
simres['late_events']={'true_total':int(tru.sum())}
for n in ['U-Net ens.','QLP + template','QLP','AsLS']:
    c=np.concatenate([lateruns(r[0]-r[4][n]) for r in rows]); simres['late_events'][n]={'total':int(c.sum()),'same_count':float((c==tru).mean()),'r':float(np.corrcoef(c,tru)[0,1]) if c.std()>0 else None}
out['sim']=simres
json.dump(out,open('out/count_i18.json','w'),indent=1); print(json.dumps(out,indent=1))
