import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
L=1536
def blk(i,o): return nn.Sequential(nn.Conv1d(i,o,5,padding=2),nn.GroupNorm(4,o),nn.GELU(),nn.Conv1d(o,o,5,padding=2),nn.GroupNorm(4,o),nn.GELU())
class UNet(nn.Module):
    def __init__(s,ch=(16,32,64,96)):
        super().__init__(); s.e=nn.ModuleList(); i=1
        for c in ch: s.e.append(blk(i,c)); i=c
        s.mid=blk(ch[-1],ch[-1]*2); s.up=nn.ModuleList(); s.d=nn.ModuleList(); i=ch[-1]*2
        for c in reversed(ch): s.up.append(nn.ConvTranspose1d(i,c,2,stride=2)); s.d.append(blk(2*c,c)); i=c
        s.out=nn.Conv1d(ch[0],1,1)
    def forward(s,x):                        # x: (B,1500) in pA
        x=x/10.; x=F.pad(x,(0,L-x.shape[1]),mode='replicate').unsqueeze(1); sk=[]
        for e in s.e: x=e(x); sk.append(x); x=F.max_pool1d(x,2)
        x=s.mid(x)
        for u,d in zip(s.up,s.d): x=u(x); x=d(torch.cat([x,sk.pop()],1))
        return s.out(x)[:,0,:1500]*10.
def predict(models,x,bs=128):
    out=[]
    with torch.no_grad():
        for m in models:
            m.eval(); out.append(torch.cat([m(torch.as_tensor(x[i:i+bs],dtype=torch.float32)) for i in range(0,len(x),bs)]).numpy())
    return np.array(out)               # (n_models, n, 1500)
