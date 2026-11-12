import sys, time, numpy as np, torch
from sim import *; from unet import *
seed=int(sys.argv[1]); steps=int(sys.argv[2]); nthr=int(sys.argv[3]); torch.set_num_threads(nthr); torch.manual_seed(seed)
M=24000
t0=time.time(); x,B,s,N,u,_=make(M,1000+seed,SRC); xv,Bv,_,_,_,_=make(1500,5000+seed,SRC)
x=torch.tensor(x,dtype=torch.float32); B=torch.tensor(B,dtype=torch.float32); xv=torch.tensor(xv,dtype=torch.float32); Bv=torch.tensor(Bv,dtype=torch.float32)
print('data',time.time()-t0,flush=True)
net=UNet(); print('params',sum(p.numel() for p in net.parameters()),flush=True)
opt=torch.optim.AdamW(net.parameters(),2e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.OneCycleLR(opt,3e-3,total_steps=steps,pct_start=.1)
w=torch.ones(1500); w[:400]=3.; w=w/w.mean(); bs=48; best=1e9
for st in range(steps):
    net.train(); idx=torch.randint(0,M,(bs,)); 
    loss=(((net(x[idx])-B[idx])**2)*w).mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step(); sch.step()
    if st%100==0 or st==steps-1:
        net.eval()
        with torch.no_grad(): vl=np.mean([(((net(xv[i:i+150])-Bv[i:i+150])**2)*w).mean().item() for i in range(0,1500,150)])
        print(st,'train %.4f val %.4f (val rmse %.3f) t=%.0fs'%(loss.item(),vl,np.sqrt(vl),time.time()-t0),flush=True)
        if vl<best: best=vl; torch.save(net.state_dict(),f'out/unet_s{seed}.pt')
