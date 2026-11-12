import numpy as np, glob, torch, json, warnings; warnings.filterwarnings('ignore')
from sim import *; from unet import *
from scipy.ndimage import uniform_filter1d
MA=20
def pk(s): return uniform_filter1d(s,MA,axis=1,mode='nearest')[:,50:600].max(1)
d=np.load('out/manual_cmp.npz'); Br=d['Bref'];W=d['W']
M={'Manual':Br,'U-Net ens.':d['U-Net_v2'],'QLP + template':d['QLP+template'],'QLP':d['QLP'],'AsLS':d['AsLS']}
out={}
# real
for I in (1.5,2.0,2.5):
    k={n:np.clip(np.round(pk(W-b)/I),0,None).astype(int) for n,b in M.items()}
    out[f'real_{I}']={n:{'agree_exact':float((k[n]==k['Manual']).mean()),'agree_pm1':float((abs(k[n]-k['Manual'])<=1).mean()),'mean_diff':float((k[n]-k['Manual']).mean()),
        'hist':np.bincount(np.minimum(k[n],6),minlength=7).tolist(),'pmax':int(k[n].max())} for n in M}
out['peak_pA']={n:[float(np.mean(pk(W-b))),float(np.percentile(pk(W-b),95))] for n,b in M.items()}
out['peak_rms_diff_to_manual']={n:float(np.sqrt(np.mean((pk(W-b)-pk(W-Br))**2))) for n,b in M.items()}
out['peak_bias_to_manual']={n:float(np.mean(pk(W-b)-pk(W-Br))) for n,b in M.items()}
# simulated truth
models=[]
for p in sorted(glob.glob('out/unet_s*.pt')):
    m=UNet(); m.load_state_dict(torch.load(p)); models.append(m)
cl=np.load('out/classical.npz')
x,B,s,N,u,st=make(600,777,HELD); ue=predict(models,x).mean(0)
SB={'U-Net ens.':ue,'QLP + template':cl['qlp_t'],'QLP':cl['qlp'],'AsLS':cl['asls']}
ktrue=np.round(pk(s)/u[:,None].ravel() if False else pk(s)/u).astype(int)
out['sim_true_hist']=np.bincount(np.minimum(ktrue,6),minlength=7).tolist()
sim={}
for I in (1.5,2.0,2.5):
    sim[str(I)]={}
    for n,b in SB.items():
        ke=np.clip(np.round(pk(x-b)/I),0,None).astype(int)
        sim[str(I)][n]={'exact':float((ke==ktrue).mean()),'pm1':float((abs(ke-ktrue)<=1).mean()),'bias':float((ke-ktrue).mean())}
# unit-known oracle (isolates baseline error)
sim['oracle_unit']={n:{'exact':float((np.clip(np.round(pk(x-b)/u),0,None)==ktrue).mean()),'pm1':float((abs(np.clip(np.round(pk(x-b)/u),0,None)-ktrue)<=1).mean())} for n,b in SB.items()}
sim['peak_err_rms']={n:float(np.sqrt(np.mean((pk(x-b)-pk(s))**2))) for n,b in SB.items()}
# baseline-free: no subtraction of the exponential, i.e. Aslike 'null' sweeps false counts
nul=N==0; sim['false_count_null_I2']={n:float((np.round(pk(x-b)/2.0)[nul]>=1).mean()) for n,b in SB.items()}
out['sim']=sim
json.dump(out,open('out/count.json','w'),indent=1); print(json.dumps(out,indent=0)[:6000])
