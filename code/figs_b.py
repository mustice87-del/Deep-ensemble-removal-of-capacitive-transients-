import numpy as np, json, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sim import *; from methods import *
T=T+0.06   # plot axis: window starts 6 samples (0.06 ms) after the command step
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':220,'savefig.bbox':'tight'})
C={'ABC envelope fit':'#d95f02','AsLS':'#7f7f7f','Quantile-LP':'#1b9e77','Quantile-LP + template':'#7570b3','U-Net (ensemble)':'#e7298a','U-Net v2':'#e7298a','U-Net v1':'#f0a3c8','Neighbour median':'#222222'}
x,B,s,N,u,st=make(600,777,HELD); cl=np.load('out/classical.npz'); abc=np.load('out/abc_results.npz'); pr=np.load('out/preds.npz'); R=json.load(open('out/results.json'))
Psim=pr['Psim']; ens=Psim.mean(0)
# ---- Fig 3: simulated examples (one per step type if possible)
sel=[]
for j in [2,3,0]:
    c=[k for k in range(120) if st[k]==j and N[k] in (2,3)]
    if c: sel.append(c[0])
fig,ax=plt.subplots(2,3,figsize=(9.5,5),sharex='col',gridspec_kw={'height_ratios':[2,1.2]})
for j,k in enumerate(sel):
    a=ax[0,j]; a.plot(T,x[k],c='0.55',lw=.7,label='data'); a.plot(T,B[k],'k',lw=1.3,label='true baseline')
    for nm,b in [('ABC envelope fit',abc['sim'][k]),('Quantile-LP',cl['qlp'][k]),('Quantile-LP + template',cl['qlp_t'][k]),('U-Net (ensemble)',ens[k])]: a.plot(T,b,c=C[nm],lw=.9,label=nm)
    a.set_xlim(0,6); a.set_title(f'step type {SNAMES[st[k]][1]} ({STEPTXT[SNAMES[st[k]]]}), N = {N[k]}',fontsize=8)
    a=ax[1,j]; a.plot(T,s[k],'k',lw=1.2)
    for nm,b in [('ABC envelope fit',abc['sim'][k]),('Quantile-LP + template',cl['qlp_t'][k]),('U-Net (ensemble)',ens[k])]: a.plot(T,x[k]-b,c=C[nm],lw=.8)
    a.set_xlim(0,6); a.set_xlabel('Time after command step (ms)')
ax[0,0].set_ylabel('Current (pA)'); ax[1,0].set_ylabel('Recovered event signal (pA)'); ax[0,0].legend(frameon=False,fontsize=6.5,loc='lower right')
plt.tight_layout(); plt.savefig('out/fig3_sim_examples.png'); plt.close()
# ---- Fig 4: metrics with bootstrap CI (n=120)
est={'ABC envelope fit':abc['sim'][:120],'AsLS':cl['asls'][:120],'Quantile-LP':cl['qlp'][:120],'Quantile-LP + template':cl['qlp_t'][:120],'U-Net (ensemble)':ens[:120]}
def per(b):
    e=b-B[:120]; sh=x[:120]-b
    return {'RMSE 0–4 ms (pA)':np.sqrt((e[:,:400]**2).mean(1)),'RMSE 0–0.5 ms (pA)':np.sqrt((e[:,:50]**2).mean(1)),'|charge error| 0.5–6 ms (pA·ms)':np.abs((sh[:,50:600].sum(1)-s[:120,50:600].sum(1))*DT),'tail bias 8–15 ms (pA)':e[:,800:].mean(1)}
PM={k:per(b) for k,b in est.items()}; rng=np.random.default_rng(0)
fig,ax=plt.subplots(1,4,figsize=(10,2.8))
for j,met in enumerate(PM['AsLS']):
    for i,(k,v) in enumerate(PM.items()):
        m=v[met].mean(); bs=[v[met][rng.integers(0,120,120)].mean() for _ in range(1000)]; lo,hi=np.percentile(bs,[2.5,97.5])
        ax[j].bar(i,m,color=C[k],yerr=[[m-lo],[hi-m]],capsize=2)
    ax[j].set_xticks(range(len(PM))); ax[j].set_xticklabels(['ABC','AsLS','QLP','QLP+T','U-Net'],fontsize=7); ax[j].set_title(met,fontsize=8); ax[j].axhline(0,c='k',lw=.5)
plt.tight_layout(); plt.savefig('out/fig4_metrics.png'); plt.close()
# ---- Fig 5: real evaluation sweeps
real=R['real']; HE=np.load('out/hybrid_ex.npz')
from common import windows
fig,ax=plt.subplots(2,3,figsize=(10,6.2),gridspec_kw={'height_ratios':[1.5,1.1]})
NM=['U-Net v2 (full-recording calibration)','U-Net v1 (first 100 sweeps only)','Quantile-LP + template','Neighbour-sweep median (reference)']; LAB=['U-Net ensemble (v2)','U-Net ensemble (v1: first 100 sweeps only)','QLP + template','Neighbour-sweep median']; COL=[C['U-Net v2'],C['U-Net v1'],C['Quantile-LP + template'],C['Neighbour median']]
pr_=np.load('out/preds.npz'); cl_idx={k:cl[f'{k}_idx'] for k in SNAMES}
for j,(k,blk) in enumerate([('s3',3),('s4',3)]):
    a=ax[0,j]; W=windows(k); ii=cl_idx[k]; sub=[n for n in range(len(ii)) if (n//30)==blk][3]   # a typical evaluation sweep from block 4
    idx=ii[sub]; a.plot(T,W[idx],c='0.78',lw=1.4,label='data')
    a.plot(T,pr_[f'{k}_unet'][sub],c=C['U-Net v2'],lw=1.0,label='U-Net ensemble (v2)'); a.plot(T,cl[f'{k}_qlp_t'][sub],c=C['Quantile-LP + template'],lw=.9,label='QLP + template'); a.plot(T,cl[f'{k}_asls'][sub],c='#333333',lw=.6,ls='--',label='AsLS')
    a.set_xlim(0,6); a.set_title(f'{k[1]}: {STEPTXT[k]} (sweep {idx}, orientation of the model)',fontsize=8); a.set_xlabel('Time after command step (ms)'); a.set_ylabel('Current (pA)')
h,l=ax[0,0].get_legend_handles_labels(); ax[0,2].axis('off'); ax[0,2].legend(h,l,frameon=False,fontsize=7,loc='upper left')
a=ax[0,2]; a.text(0,.35,'Residuals of well-fitted baselines\nreach the noise floor (0.25 pA).\nAsLS follows the data (residual\n0.04 pA ≪ noise), so lower\nresidual does not mean better.',fontsize=8,va='center')
for j,k in enumerate(['s3','s4']):
    a=ax[1,j]
    for nm,lab,col in zip(NM,LAB,COL): a.plot(range(1,6),real[k][nm]['by_block_early'],'o-',c=col,label=lab,lw=1.2,ms=3)
    a.set_xlabel('Time block of the recording (≈ 8 min each)'); a.set_ylabel('Residual RMS 0–0.5 ms (pA)'); a.set_xticks(range(1,6)); a.set_title(f'{k[1]}: {STEPTXT[k]}',fontsize=8)
ax[1,0].legend(frameon=False,fontsize=6,loc='upper center',bbox_to_anchor=(.5,-.28),ncol=1)
a=ax[1,2]
for nm,lab,col in zip(NM,LAB,COL): a.plot(range(1,5),[real[k][nm]['early'] for k in SNAMES],'o-',c=col,lw=1.2,ms=3)
a.set_xticks(range(1,5)); a.set_xticklabels([f's{i}' for i in range(1,5)]); a.set_xlabel('Voltage step'); a.set_ylabel('Residual RMS 0–0.5 ms (pA)'); a.set_title('All blocks pooled',fontsize=8)
plt.tight_layout(); plt.savefig('out/fig5_real.png'); plt.close()
# ---- Fig 6: injection of simulated events into real sweeps
H=R['hybrid']; fig,ax=plt.subplots(1,3,figsize=(10,3.1),gridspec_kw={'width_ratios':[1.3,1,1]})
a=ax[0]; k='s3'; i=1
a.plot(T,HE[f'{k}_Wh'][i],c='0.6',lw=.7,label='real sweep + injected events'); a.plot(T,HE[f'{k}_s'][i]+0,c='k',lw=1.1,label='injected events')
for nm,col in [('AsLS',C['AsLS']),('U-Net_v2_(full-recording_calibration)',C['U-Net v2'])]:
    a.plot(T,HE[f'{k}_Wh'][i]-HE[f'{k}_h_{nm}'][i],c=col,lw=1.0,label=('AsLS' if nm=='AsLS' else 'U-Net')+': corrected signal')
a.set_xlim(0,6); a.set_xlabel('Time after command step (ms)'); a.set_ylabel('Current (pA)'); a.legend(frameon=False,fontsize=6.5); a.set_title('a  Event recovery (step 3)',loc='left',fontsize=9)
names=[('AsLS','AsLS'),('Quantile-LP','QLP'),('Quantile-LP + template','QLP+T'),('U-Net v2 (full-recording calibration)','U-Net')]
a=ax[1]; w=.2
for j,(nm,lab) in enumerate(names): a.bar(np.arange(4)+j*w,[H[k][nm]['rms_db_0.5-6'] for k in SNAMES],w,color=C.get(nm.split(' (')[0] if nm.startswith('U-Net') else nm,'k') if not nm.startswith('U-Net') else C['U-Net v2'],label=lab)
a.set_xticks(np.arange(4)+1.5*w); a.set_xticklabels(['s1','s2','s3','s4']); a.set_ylabel('Event leakage into baseline,\nRMS 0.5–6 ms (pA)'); a.set_ylim(0,1.4); a.legend(frameon=False,fontsize=6.5,ncol=2,loc='upper right'); a.set_title('b  Baseline change caused by events',loc='left',fontsize=9)
a=ax[2]
for j,(nm,lab) in enumerate(names): a.bar(np.arange(4)+j*w,[H[k][nm]['abs_charge_leak'] for k in SNAMES],w,color=C.get(nm,'k') if not nm.startswith('U-Net') else C['U-Net v2'])
a.set_xticks(np.arange(4)+1.5*w); a.set_xticklabels(['s1','s2','s3','s4']); a.set_ylabel('|charge leakage| 0.5–6 ms (pA·ms)'); a.set_title('c  Charge lost to the baseline',loc='left',fontsize=9)
plt.tight_layout(); plt.savefig('out/fig6_hybrid.png'); plt.close()
# ---- Fig 7: ABC parameters and ensemble uncertainty
fig,ax=plt.subplots(1,2,figsize=(8,3.2)); d=abc['real_d']
ax[0].scatter(d[:,2],d[:,4],s=10,c='#d95f02'); ax[0].axhline(0,c='k',lw=.4); ax[0].axvline(0,c='k',lw=.4); ax[0].set_xlabel('ABC d₃ (pA)'); ax[0].set_ylabel('ABC d₅ (pA)'); ax[0].set_title('a  ABC amplitude parameters, %d real sweeps'%len(d),loc='left',fontsize=9)
sd=Psim.std(0)[:,:400].mean(1); er=np.sqrt(((ens-B)[:,:400]**2).mean(1)); ax[1].scatter(sd,er,s=5,alpha=.4,c='#e7298a'); ax[1].set_xlabel('Ensemble SD, 0–4 ms (pA)'); ax[1].set_ylabel('Ensemble RMSE, 0–4 ms (pA)'); ax[1].set_title('b  Uncertainty vs error (600 test sweeps)',loc='left',fontsize=9)
plt.tight_layout(); plt.savefig('out/fig7_abc_unc.png'); plt.close(); print('figs ok')
