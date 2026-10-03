import csv, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv)>1 else 'out/pitch_perframe_2fps.csv'
out_png  = sys.argv[2] if len(sys.argv)>2 else 'out/pitch_evolution_lines.png'

rows=[r for r in csv.reader(open(csv_path))][1:]
pts=[(float(r[0]), r[1], float(r[2])) for r in rows]
print("points:", len(pts), " A:", sum(1 for _,g,_ in pts if g=='A'))

ts=np.array([p[0] for p in pts])
ps=np.array([p[2] for p in pts])
gg=np.array([p[1] for p in pts])

fig,ax=plt.subplots(figsize=(14,5),dpi=150)
# evolution line: connect consecutive measurements (<=3 s apart)
for i in range(1,len(ts)):
    if ts[i]-ts[i-1] <= 3.0:
        ax.plot(ts[i-1:i+1], ps[i-1:i+1], color='#1f77b4', lw=1.3, zorder=2)
# shots at all points
mB = gg=='B'; mA = gg=='A'
ax.scatter(ts[mB], ps[mB], s=16, color='#b0b0b0', zorder=3, label='B: одно семейство (предварительно)')
ax.scatter(ts[mA], ps[mA], s=48, color='#d62728', edgecolor='k', zorder=4, label='A: два семейства согласны (надёжно)')
ax.axhline(0, color='k', lw=1)
ax.set_xlabel('время в сессии, с')
ax.set_ylabel('наклон по линиям отн. горизонта (вниз +), град (f≈720px)')
ax.set_title(f'Эволюция наклона CIWS по линейным замерам (без интегрирования). {csv_path}\n'
             'Линия — соединение соседних замеров (≤3 с); разрывы линии — отсутствие валидного замера.', fontsize=10)
ax.grid(alpha=0.3); ax.legend(loc='upper left', fontsize=9)
fig.tight_layout(); fig.savefig(out_png)
print("saved", out_png)
if len(ps):
    print("range: %.2f .. %.2f deg; A-max-down: %s"%(
        ps.min(), ps.max(), round(max([p for p,g in zip(ps,gg) if g=='A'], default=float('nan')),2)))
