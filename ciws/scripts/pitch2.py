import cv2, numpy as np, sys, json

def get_segments(path, crop=(0.07, 0.28, 0.16, 0.16), min_len=55):
    img = cv2.imread(path)
    if img is None: return None
    H, W = img.shape[:2]
    t, b, l, r = crop
    x0, x1 = int(W*l), int(W*(1-r)); y0, y1 = int(H*t), int(H*(1-b))
    roi = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    segs = lsd.detect(gray)[0]
    out=[]
    if segs is not None:
        for s in segs[:,0]:
            xa,ya,xb,yb = s
            if np.hypot(xb-xa, yb-ya) < min_len: continue
            out.append((xa+x0, ya+y0, xb+x0, yb+y0))
    return img, out

def lines_from_segs(segs):
    L=[]
    for Xa,Ya,Xb,Yb in segs:
        a=Yb-Ya; bb=Xa-Xb; c=-(a*Xa+bb*Ya); n=np.hypot(a,bb)
        if n: L.append((a/n,bb/n,c/n))
    return np.array(L)

def ransac_vp(L, thresh=2.5, iters=4000, min_inliers=8):
    N=len(L)
    if N<2: return None
    best=None; rng=np.random.default_rng(7)
    for _ in range(iters):
        i,j = rng.integers(0,N,2)
        if i==j: continue
        v=np.cross(L[i],L[j])
        if abs(v[2])<1e-9: continue
        v=v/v[2]
        d=np.abs(L@v)/np.hypot(L[:,0],L[:,1])
        inl=np.where(d<thresh)[0]
        if best is None or len(inl)>len(best[1]): best=(v,inl)
    if best is None or len(best[1])<min_inliers: return None
    for _ in range(3):
        A=L[best[1]]
        _,_,Vt=np.linalg.svd(A); v=Vt[-1]
        if abs(v[2])<1e-9: return None
        v=v/v[2]
        d=np.abs(L@v)/np.hypot(L[:,0],L[:,1])
        inl=np.where(d<thresh)[0]
        if len(inl)<min_inliers: return None
        best=(v,inl)
    return best

def analyze(path, save=None):
    img, segs = get_segments(path)
    if img is None: return None
    H,W = img.shape[:2]; cx,cy = W/2, H/2
    L = lines_from_segs(segs)
    if len(L) < 12: return None
    vps=[]; remaining=np.arange(len(L))
    for _ in range(8):
        res = ransac_vp(L[remaining])
        if res is None: break
        v,inl_local = res
        inl = remaining[inl_local]
        vps.append((v, inl))
        remaining = np.setdiff1d(remaining, inl)
        if len(remaining) < 8: break
    if len(vps) < 2: return None
    vps.sort(key=lambda vp: -len(vp[1]))
    cands=[]
    for a in range(min(len(vps),6)):
        for b in range(a+1, min(len(vps),6)):
            va, ia = vps[a]; vb, ib = vps[b]
            if abs(va[1]-vb[1]) > 18: continue
            dot = (va[0]-cx)*(vb[0]-cx) + (va[1]-cy)*(vb[1]-cy)
            if -dot <= 0: continue
            f = np.sqrt(-dot)
            if not (700 < f < 6500): continue
            cands.append(dict(f=f, yh=(va[1]+vb[1])/2, na=len(ia), nb=len(ib),
                              va=(va[0],va[1]), vb=(vb[0],vb[1]), ia=ia, ib=ib))
    if not cands: return None
    cands.sort(key=lambda c: -(c['na']+c['nb']))
    fs = [c['f'] for c in cands]
    f_med = float(np.median(fs))
    good = [c for c in cands if abs(c['f']-f_med)/f_med < 0.25]
    yh = float(np.median([c['yh'] for c in good]))
    pitch = float(np.degrees(np.arctan2(cy-yh, f_med)))
    info = dict(path=path, pitch_deg=round(pitch,2), f_px=round(f_med,1),
                horizon_y=round(yh,1), n_pairs=len(good),
                f_spread=round(float(np.std(fs)/f_med),3), n_segs=len(segs))
    if save:
        vis = img.copy()
        cols = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,0,255),(255,255,0)]
        for k,(v,inl) in enumerate(vps[:6]):
            for idx in inl:
                Xa,Ya,Xb,Yb = segs[idx]
                cv2.line(vis,(int(Xa),int(Ya)),(int(Xb),int(Yb)),cols[k%6],3)
            vx,vy=int(v[0]),int(v[1])
            if -3*W<vx<4*W and -3*H<vy<4*H: cv2.circle(vis,(vx,vy),12,cols[k%6],-1)
        # use the best pair's vps uniquely colored
        cgood = good[0]
        cv2.line(vis,(0,int(yh)),(W,int(yh)),(255,0,255),2)
        for v in (cgood['va'], cgood['vb']):
            vx,vy=int(v[0]),int(v[1])
            if -3*W<vx<4*W and -3*H<vy<4*H: cv2.circle(vis,(vx,vy),20,(255,0,255),4)
        cv2.circle(vis,(int(cx),int(cy)),10,(255,255,255),3)
        cv2.putText(vis, f"pitch={pitch:.1f} f={f_med:.0f} yh={yh:.0f}", (30,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0,0,255), 3)
        cv2.imwrite(save, vis)
    return info

if __name__ == '__main__':
    args = sys.argv[1:]
    save = None
    if args and args[0]=='--save': save=args[1]; args=args[2:]
    for p in args:
        r = analyze(p, save=(save+'.jpg') if save else None)
        print(json.dumps(r))
