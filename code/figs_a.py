import numpy as np, json, warnings, pyabf; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.signal import welch
from scipy.ndimage import median_filter
from common import *
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':220,'savefig.bbox':'tight'})
SC={'s1':'#1f77b4','s2':'#2ca02c','s3':'#d62728','s4':'#9467bd'}
Xa=np.asarray(X()); a_=pyabf.ABF('../newdata/data.abf'); a_.setSweep(0)
ING=np.load('out/ingredients.npz')
fig=plt.figure(figsize=(9.5,6.2)); gs=fig.add_gridspec(2,6,height_ratios=[1,1.05])
# a: protocol and sweep
ax=fig.add_subplot(gs[0,:3]); t=np.arange(220000)/1e5
ax2=ax.twinx(); ax2.spines['right'].set_visible(True); ax2.plot(t,a_.sweepC,c='0.4',lw=1.0); ax2.set_ylabel('Command (mV)',color='0.4'); ax2.set_ylim(-60,260)
ax.plot(t,Xa[0],c='k',lw=.35); ax.set_ylim(-45,25); ax.set_xlabel('Time in sweep (s)'); ax.set_ylabel('Current (pA)'); ax.set_title('a  One 2.2 s sweep and its command potential',loc='left',fontsize=9)
for k,i0 in STEPS.items(): ax.annotate(k[1],(i0/1e5,-42),fontsize=8,color=SC[k],ha='center')
# b: the four transients (median over first 100 sweeps), recorded sign
ax=fig.add_subplot(gs[0,3:])
for k,i0 in STEPS.items():
    m=np.median(Xa[:100,i0-20:i0+1500],0); tt=(np.arange(-20,1500))*DT-OFF*DT
    ax.plot(tt,m-np.median(Xa[:100,i0-20:i0-5]),c=SC[k],lw=1,label=f'{k[1]}: {STEPTXT[k]}')
ax.set_xlim(-0.2,6); ax.set_xlabel('Time after command step (ms)'); ax.set_ylabel('Median current (pA)'); ax.legend(frameon=False,fontsize=7,loc='lower right'); ax.set_title('b  The four transients per sweep',loc='left',fontsize=9)
# c: drift across the recording
ax=fig.add_subplot(gs[1,:2]); 
for k in ['s3','s4']:
    W=windows(k); v=W[:,600:].mean(1); ax.scatter(np.arange(493),v,s=3,c=SC[k],alpha=.35); ax.plot(median_filter(v,size=25,mode='nearest'),c=SC[k],lw=1.6,label=f'{k[1]}: {STEPTXT[k]}')
for b0,b1 in BLOCKS[1:]: ax.axvline(b0-.5,c='0.8',lw=.6)
ax.set_xlabel('Sweep number (5 s apart; 41 min)'); ax.set_ylabel('Mean current 6–15 ms\nafter step (pA)'); ax.legend(frameon=False,fontsize=7); ax.set_title('c  Slow drift of the transient',loc='left',fontsize=9)
# d: template E_k
ax=fig.add_subplot(gs[1,2:4])
for j,k in enumerate(SNAMES): ax.plot(T[:70]+0.06,ING['E'][j][:70]*SIGN[k],c=SC[k],lw=1.3)
ax.set_xlabel('Time after step (ms)'); ax.set_ylabel('Ringing template E(t), recorded sign (pA)'); ax.set_title('d  Early ringing not captured\nby exponentials',loc='left',fontsize=9)
# e: noise spectrum
ax=fig.add_subplot(gs[1,4:]); f=ING['psd_f']; ax.loglog(f[1:],ING['psd'][1:],'k',label='measured')
import sys; sys.path.insert(0,'.'); import sim
n=sim.noise(400,np.random.default_rng(3)); ff,pp=welch(n,fs=FS*1e3,nperseg=256,axis=1); ax.loglog(ff[1:],pp.mean(0)[1:],'r--',lw=1,label='simulated')
ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('PSD (pA²/Hz)'); ax.legend(frameon=False,fontsize=7); ax.set_title('e  Noise spectrum',loc='left',fontsize=9)
plt.tight_layout(); plt.savefig('out/fig1_data.png'); plt.close()
# pipeline schematic
fig,ax=plt.subplots(figsize=(9,2.9)); ax.axis('off'); ax.set_xlim(0,100); ax.set_ylim(0,32)
boxes=[(0.5,'493 real sweeps ×\n4 voltage steps\n(one patch, 41 min)'),(20.5,'Quiet sweeps:\nexp. fits (180 source,\n60 held out);\nE(t) from the 180;\nnoise from all 240'),(42,'Simulator:\nbaseline + Markov\nchannel events\n+ noise'),(63,'1D U-Net ×3\n(deep ensemble)'),(83,'Baseline b̂(t),\nensemble spread')]
for x0,t_ in boxes:
    ax.add_patch(FancyBboxPatch((x0,9),16,16,boxstyle='round,pad=0.4',fc='#eef3fb',ec='#335')); ax.text(x0+8,17,t_,ha='center',va='center',fontsize=7.5)
for x0 in [17.2,37.2,58.7,79.7]: ax.annotate('',xy=(x0+3.6,17),xytext=(x0,17),arrowprops=dict(arrowstyle='->'))
ax.text(50,3.2,'evaluation: (i) simulated held-out baselines (known truth); (ii) real sweeps never used for calibration, all 4 steps, 5 time blocks;\n(iii) real sweeps with injected simulated channel events (event leakage into the baseline)',ha='center',fontsize=7.5,style='italic')
plt.savefig('out/fig2_pipeline.png'); plt.close(); print('ok')
