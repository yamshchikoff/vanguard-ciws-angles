import numpy as np, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

F0 = 720.0  # px focal at 960x540 (sight = 1.5x of default 90-deg base FOV)
flow = np.load('out/flow4f.npy')  # rows: t, dy, resp

ts = flow[:,0]; dy = flow[:,1]; resp = flow[:,2]

reliable = resp >= 0.12
# integrate
theta = np.full(len(flow), np.nan)
cum = 0.0
theta[0] = 0.0
last_ok = 0
for i in range(1,len(flow)):
    if reliable[i] and (ts[i]-ts[i-1]) <= 0.26:
        cum += -dy[i]/F0*180/np.pi
        theta[i] = cum
        last_ok = i
    else:
        theta[i] = cum  # carry (gap); mark separately

# anchors
def idx_at(t): return int(np.argmin(abs(ts-t)))
i0, i1 = idx_at(300), idx_at(507)
# segment-wise linear detrend between anchors
off = np.zeros_like(theta)
o0, o1 = theta[i0], theta[i1]
off[:i0] = o0
if i1 > i0:
    off[i0:i1] = o0 + (o1-o0)*(np.arange(i0,i1)-i0)/max(1,(i1-i0))
off[i1:] = o1
theta_abs = theta - off

# gaps for shading
gap_idx = ~reliable
gap_t = ts[gap_idx]

# save csv
with open('out/pitch_evolution_4f.csv','w',newline='') as fh:
    wr = csv.writer(fh); wr.writerow(['t_s','theta_down_deg_f720','reliable'])
    for t,a,r in zip(ts, theta_abs, reliable):
        wr.writerow([round(float(t),2), round(float(a),3), int(r)])

# stats
rel_theta = theta_abs[reliable]
print("theta(reliable) min %.2f max %.2f deg ; n=%d/%d"%(rel_theta.min(), rel_theta.max(), reliable.sum(), len(flow)))
# deepest-down stretches
down = np.where((theta_abs > 1.5) & reliable)[0]
if len(down):
    segs=[]; s=down[0]; p=down[0]
    for b in down[1:]:
        if b==p+1: p=b
        else: segs.append((ts[s],ts[p])); s=b; p=b
    segs.append((ts[s],ts[p]))
    print("stretches > +1.5 deg down:", [(round(a,1),round(b,1)) for a,b in segs][:10])
up = np.where((theta_abs < -1.0) & reliable)[0]
if len(up):
    segs=[]; s=up[0]; p=up[0]
    for b in up[1:]:
        if b==p+1: p=b
        else: segs.append((ts[s],ts[p])); s=b; p=b
    segs.append((ts[s],ts[p]))
    print("stretches < -1.0 deg (up):", [(round(a,1),round(b,1)) for a,b in segs][:10])

fig,ax=plt.subplots(figsize=(14,5.2),dpi=150)
ax.plot(ts, theta_abs, color='#1f77b4', lw=1.2)
if len(gap_t):
    g=np.array(sorted(gap_t)); sv=g[0]; pv=g[0]
    for v in g[1:]:
        if abs(v-pv)<=0.26: pv=v
        else:
            ax.axvspan(sv,pv,color='red',alpha=0.10); sv=v; pv=v
    ax.axvspan(sv,pv,color='red',alpha=0.10)
for a in (300,507):
    i=idx_at(a)
    ax.plot(ts[i],theta_abs[i],'o',color='green',ms=7,zorder=5)
    ax.annotate(f'VP-тест: уровень (t={a})',(ts[i],theta_abs[i]),textcoords='offset points',xytext=(6,-16),fontsize=9,color='green')
ax.axhline(0,color='k',lw=0.9)
ax.set_xlabel('время в сессии, с')
ax.set_ylabel('тангаж относительно горизонта (вниз +), град')
ax.set_title('Эволюция угла наведения CIWS (видео ljsZXyhDekQ, сессия оператора). '
             '4 к/с, f≈720px (дефолт FOV 90° × 1.5x).\n'
             'Зелёные точки — независимая привязка по VP-тесту (0°). Красные полосы — неинтегрируемые участки.', fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('out/pitch_evolution.png')
print("saved out/pitch_evolution.png")
