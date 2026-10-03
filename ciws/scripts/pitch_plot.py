import cv2, numpy as np, glob, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

files = sorted(glob.glob('frames/ljs1f/*.jpg'))
F0 = 720.0
y0,y1,x0,x1 = 55,165,230,730
prev=None; data=[]
for i,f in enumerate(files):
    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if im is None: continue
    w = im[y0:y1,x0:x1].astype(np.float32)
    if prev is not None:
        (dx,dy),resp = cv2.phaseCorrelate(prev,w)
        t=i+1
        if 219<=t<=532: data.append((t,float(dy),float(resp)))
    prev=w

# reliable dy only
reldy=[]; gaps=[]
for t,dy,resp in data:
    ok = resp>=0.08 and abs(dy)<=25
    reldy.append((t, dy if ok else None))

# cumulative relative theta (deg); integrate over consecutive pairs, linearly bridge short gaps
theta={}
cum=0.0
prevt=None
pend=[]
for t,dy in reldy:
    if dy is None:
        pend.append(t); continue
    if prevt is None:
        theta[t]=cum
    elif t-prevt==1:
        cum += -dy/F0*180/np.pi
        theta[t]=cum
    else:
        gapn=t-prevt-1
        bridge=(-dy/F0*180/np.pi) if gapn<=5 else 0.0   # bridge short gaps, longer gaps: carry value
        step=bridge/(gapn+1)
        for k in range(1,gapn+2):
            if prevt+k in dict(reldy) or True:
                pass
        # assign bridged values for gap frames
        v=cum
        for k in range(1,gapn+1):
            v+=step; theta[prevt+k]=v
        cum+=bridge
        theta[t]=cum
    prevt=t
# ensure all t present
for t,_ in reldy:
    if t not in theta: theta[t]=cum

ts=np.array(sorted(theta.keys()))
th=np.array([theta[t] for t in ts])
# anchors
def val_at(t):
    i=np.argmin(abs(ts-t)); return th[i]
a0,a1=300,507
o0,o1=val_at(a0),val_at(a1)
off=np.zeros_like(th)
i0=np.argmin(abs(ts-a0)); i1=np.argmin(abs(ts-a1))
off[:i0]=o0
off[i0:i1]=o0+(o1-o0)*(np.arange(i0,i1)-i0)/max(1,(i1-i0))
off[i1:]=o1
th_abs=th-off

# quality flags
gapsq=[t for t,dy in reldy if dy is None]

# save csv
with open('out/pitch_evolution.csv','w',newline='') as fh:
    wr=csv.writer(fh); wr.writerow(['t_s','theta_down_deg_f720','reliable'])
    for t,a in zip(ts,th_abs):
        wr.writerow([int(t), round(float(a),3), 0 if t in gapsq else 1])

# plot
fig,ax=plt.subplots(figsize=(13,5),dpi=150)
ax.plot(ts,th_abs,color='#1f77b4',lw=1.6)
# mark unreliable spans as shaded
if gapsq:
    g=np.array(sorted(gapsq)); sv=g[0]; pv=g[0]
    for v in g[1:]:
        if v==pv+1: pv=v
        else:
            ax.axvspan(sv,pv,color='red',alpha=0.12); sv=v; pv=v
    ax.axvspan(sv,pv,color='red',alpha=0.12)
for a in (300,507):
    i=np.argmin(abs(ts-a))
    ax.plot(ts[i],th_abs[i],'o',color='green',ms=7)
    ax.annotate(f'VP-тест: уровень (t={a})',(ts[i],th_abs[i]),textcoords='offset points',xytext=(6,-14),fontsize=9,color='green')
ax.axhline(0,color='k',lw=0.8)
ax.set_xlabel('время в сессии, с')
ax.set_ylabel('относительный тангаж (низ +), град')
ax.set_title('Эволюция угла наведения CIWS за сессию оператора (ljsZXyhDekQ), 960x540, f≈720px\n'
             'якоря: кадры с VP-строкой ровно на центре (0°). Красным — ненадёжные участки (низкая корреляция/монтаж).',
             fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('out/pitch_evolution.png')
print("saved out/pitch_evolution.png and .csv;", len(ts), "samples;", len(gapsq), "unreliable")
print("min %.2f max %.2f deg"%(th_abs.min(), th_abs.max()))
