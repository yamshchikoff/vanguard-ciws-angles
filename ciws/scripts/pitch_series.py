import cv2, numpy as np, glob, csv

# Build per-frame vertical scene displacement (dy) via phase correlation on two bands,
# integrate to a relative pitch trajectory, anchor at verified level frames (t=300, t=507),
# convert to degrees with f=720 px (960x540 frame, sight = 1.5x of 90-deg base FOV).

files = sorted(glob.glob('frames/ljs1f/*.jpg'))
F0 = 720.0  # px focal assumed for degrees (see caveat in report)

bands = {
    'mid':  (55, 165, 230, 730),   # y0,y1,x0,x1 in 540-row coords
    'low':  (165, 250, 230, 730),
}

series = {k: [] for k in bands}
prev = {k: None for k in bands}
for i, f in enumerate(files):
    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if im is None: continue
    for k,(y0,y1,x0,x1) in bands.items():
        w = im[y0:y1, x0:x1].astype(np.float32)
        if prev[k] is not None:
            (dx,dy), resp = cv2.phaseCorrelate(prev[k], w)
            series[k].append((i, float(dx), float(dy), float(resp)))
        prev[k] = w

# choose the band with better mean confidence after t=225
t0, t1 = 219, 532
def cum_traj(k, tmin, tmax):
    xs=[]; dy_=[]; resp_=[]
    for i,dx,dy,resp in series[k]:
        t=i+1
        if tmin<=t<=tmax:
            xs.append(t); dy_.append(dy); resp_.append(resp)
    dy_=np.array(dy_)
    cum = -np.cumsum(dy_)/F0*180/np.pi   # theta_down relative, deg; scene moving down (+dy) = camera up
    return np.array(xs), cum, np.array(dy_), np.array(resp_)

# anchors (verified level, theta_down=0): t=300 and t=507 (within +-1 deg)
res={}
for k in bands:
    xs,cum,dy_,resp = cum_traj(k, t0, t1)
    res[k]=(xs,cum,dy_,resp)

# use 'mid' band
xs,cum,dy_,resp = res['mid']
anchors=[(300,0.0),(507,0.0)]
# linear detrend per anchor segments: build piecewise offset so that value at anchor = 0
off=np.zeros_like(cum)
# segment boundaries: anchor times
a0=300; a1=507
i0=np.argmin(abs(xs-a0)); i1=np.argmin(abs(xs-a1))
# offsets: t<300 -> align at a0; 300<t<507 -> interpolate offsets between a0 and a1; t>507 -> align at a1
off[:i0]=cum[i0]
if i1>i0:
    off[i0:i1]=cum[i0]+(cum[i1]-cum[i0])*(np.arange(i0,i1)-i0)/(i1-i0)
off[i1:]=cum[i1]
theta=cum-off
# save csv
with open('out/pitch_series.csv','w',newline='') as fh:
    wr=csv.writer(fh); wr.writerow(['t_s','theta_down_deg_f720','dy_px','resp'])
    for t,c,d,r in zip(xs,theta,dy_,resp): wr.writerow([int(t),round(float(c),3),round(d,3),round(r,3)])
print("csv saved: out/pitch_series.csv, n=",len(xs))
print("theta range deg:", round(float(theta.min()),2), "..", round(float(theta.max()),2))
print("value at anchors:", round(float(theta[i0]),3), round(float(theta[i1]),3))
# quick stats: moments below 0
below=np.where(theta< -0.4)[0]
print("samples with theta < -0.4 deg:", len(below))
if len(below):
    segs=[]; s=below[0]; p=below[0]
    for b in below[1:]:
        if b==p+1: p=b
        else: segs.append((int(xs[s]),int(xs[p]))); s=b; p=b
    segs.append((int(xs[s]),int(xs[p])))
    print("segments below -0.4 deg:", segs[:12])
