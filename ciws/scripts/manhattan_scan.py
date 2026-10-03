import cv2, numpy as np, sys, glob, json

def segments(img, crop=(0.10,0.30,0.17,0.17), min_len=45):
    H,W=img.shape[:2]
    t,b,l,r=crop
    x0,x1=int(W*l),int(W*(1-r)); y0,y1=int(H*t),int(H*(1-b))
    roi=img[y0:y1,x0:x1]
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    segs=lsd.detect(gray)[0]
    out=[]
    if segs is not None:
        for xa,ya,xb,yb in segs[:,0]:
            if np.hypot(xb-xa,yb-ya)<min_len: continue
            out.append((xa+x0,ya+y0,xb+x0,yb+y0))
    L=[]
    for Xa,Ya,Xb,Yb in out:
        a=Yb-Ya; b2=Xa-Xb; c=-(a*Xa+b2*Ya); n=np.hypot(a,b2)
        if n: L.append((a/n,b2/n,c/n))
    return np.array(L), out

def ransac_vp(L, thresh=2.5, iters=3000, min_inliers=10, rng=None):
    N=len(L)
    if N<2: return None
    if rng is None: rng=np.random.default_rng(1)
    best=None
    for _ in range(iters):
        i,j=rng.integers(0,N,2)
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

def analyze(path):
    img=cv2.imread(path)
    if img is None: return None
    H,W=img.shape[:2]; cx,cy=W/2,H/2
    L,segs=segments(img)
    if len(L)<10: return None
    vps=[]; rem=np.arange(len(L))
    for _ in range(6):
        res=ransac_vp(L[rem])
        if res is None: break
        v,inl=res; inl=rem[inl]
        vps.append((v,len(inl)))
        rem=np.setdiff1d(rem,inl)
        if len(rem)<8: break
    if len(vps)<3: return None
    # test all triples for mutual orthogonality with common f
    best=None
    for i in range(len(vps)):
        for j in range(i+1,len(vps)):
            for k in range(j+1,len(vps)):
                f2s=[]; ok=True
                for (a,b) in ((i,j),(i,k),(j,k)):
                    va=vps[a][0]; vb=vps[b][0]
                    dot=(va[0]-cx)*(vb[0]-cx)+(va[1]-cy)*(vb[1]-cy)
                    if -dot<=0: ok=False; break
                    f2s.append(-dot)
                if not ok: continue
                fs=[np.sqrt(x) for x in f2s]
                if min(fs)<600 or max(fs)>9000: continue
                spread=(max(fs)-min(fs))/np.mean(fs)
                score=spread
                if spread<0.12:
                    if best is None or score<best[0]:
                        best=(score, fs, [(vps[a][0],len(vps[a][1])) for a in (i,j,k)])
    if best is None: return None
    spread, fs, vlist = best
    return dict(path=path, spread=round(float(spread),3), f=[round(float(x),0) for x in fs],
                vps=[(round(float(v[0]),0),round(float(v[1]),0),n) for v,n in vlist])

if __name__=='__main__':
    files=sorted(glob.glob(sys.argv[1]))
    hits=[]
    for f in files:
        try:
            r=analyze(f)
        except Exception as e:
            continue
        if r: hits.append(r)
    hits.sort(key=lambda r:r['spread'])
    for r in hits[:25]: print(json.dumps(r))
    print("total consistent frames:", len(hits), "/", len(files))
