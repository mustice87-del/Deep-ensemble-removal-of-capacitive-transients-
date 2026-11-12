import numpy as np, json, glob, torch, warnings, scipy.io as sio; warnings.filterwarnings('ignore')
from sim import *; from unet import *; from methods import *
torch.set_num_threads(4)
tuned=json.load(open('out/tuned.json'))
m=sio.loadmat('/mnt/user-data/uploads/after_Icapsub_allsweeps.mat'); S=m['sweeps'].T  # (493,49700)
W=windows('s3'); Sg=S[:,157:157+1500]; Bref=W-Sg
print('Bref roughness (diff SD)',np.diff(Bref,axis=1).std(),'noise diff SD',np.diff(Sg[:,800:],axis=1).std())
print('Bref mean first/last',Bref[:,:5].mean(),Bref[:,-100:].mean())
models=[]
for p in sorted(glob.glob('out/unet_s*.pt')):
    mm=UNet(); mm.load_state_dict(torch.load(p)); models.append(mm)
Wf=W.astype(np.float32); ue=predict(models,Wf).mean(0)
V1=np.load('../proj/out/full_real.npz')['s3_unet']
AT=design(E[2]); A0=design(None)
est={'U-Net v2':ue,'U-Net v1':V1,
 'AsLS':np.array([asls(v,*tuned['asls']) for v in W]),
 'QLP':np.array([qlp(v,tuned['qlp'],None,A0) for v in W]),
 'QLP+template':np.array([qlp(v,tuned['qlp_t'],E[2],AT) for v in W])}
abc=np.load('out/abc_results.npz'); sel=abc['sel']
np.savez('out/manual_cmp.npz',Bref=Bref,W=W,**{k.replace(' ','_'):v for k,v in est.items()},sel=sel,abc_real=abc['real'])
