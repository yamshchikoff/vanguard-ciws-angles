import cv2, numpy as np, glob, csv, sys

F0 = 720.0  # px focal at 960x540 (sight 1.5x of default 90-deg hFOV)

def vps_of(img, th=5.0, min_len=22, min_inl=12, max_vps=6):
    H,W=img.shape[:2]; cx,cy=W/2,H/2
    t,b,l,r=0.10,0.30,0.17,0.17
    x0,x1=int(W*l),int(W*(1-r)); y0,y1=int(H*t),int(H*(1-b))
    roi=img[y0:y1,x0:x1]
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    segs=lsd.detect(gray)[0]
    if segs is None: return [],[],(cx,cy)
    L=[]
    for xa,ya,xb,yb in segs[:,0]:
        if np.hypot(xb-xa,yb-ya)<min_len: continue
        a=ya-yb; b2=xa-xb; c=-(a*xa+b2*ya); nn=np.hypot(a,b2)
        if nn==0: continue
        L.append((a/nn,b2/nn,c/nn))
    L=np.array(L)
    if len(L)<10: return [],[],(cx,cy)
    rem=np.arange(len(L)); vps=[]
    for _ in range(max_vps):
        best=None; rng=np.random.default_rng(3)
        for _ in range(2200):
            i,j=rng.integers(0,len(rem),2)
            if i==j: continue
            v=np.cross(L[rem[i]],L[rem[j]])
            if abs(v[2])<1e-9: continue
            v=v/v[2]
            d=np.abs(L[rem]@v)/np.hypot(L[rem][:,0],L[rem][:,1])
            inl=np.where(d<th)[0]
            if best is None or len(inl)>len(best[1]): best=(v,inl)
        if best is None or len(best[1])<min_inl: break
        for _ in range(3):
            A=L[rem[best[1]]]
            _,_,Vt=np.linalg.svd(A); v=Vt[-1]
            if abs(v[2])<1e-9: break
            v=v/v[2]
            d=np.abs(L[rem]@v)/np.hypot(L[rem][:,0],L[rem][:,1])
            inl=np.where(d<th)[0]
            if len(inl)<min_inl: break
            best=(v,inl)
        v,inl=best
        vps.append((np.array([v[0]+x0,v[1]+y0]),len(inl)))
        rem=np.setdiff1d(rem,rem[inl])
        if len(rem)<min_inl: break
    return vps, segs, (cx,cy)

def frame_pitch(path):
    img=cv2.imread(path)
    if img is None: return None
    vps,_,(cx,cy)=vps_of(img)
    if len(vps)<2: return None
    # find pairs of families with SAME horizon row (|dy|<=15px), different x, finite
    pairs=[]
    for i in range(len(vps)):
        for j in range(i+1,len(vps)):
            (vi,ni),(vj,nj)=vps[i],vps[j]
            if abs(vi[0]-vj[0])<40: continue          # must be different directions
            if abs(vi[1]-vj[1])>15: continue          # same horizon row required
            if abs(vi[1])>6000 or abs(vj[1])>6000: continue
            pairs.append((ni+nj, (vi[1]+vj[1])/2, ni, nj, i, j))
    if not pairs: return None
    pairs.sort(reverse=True)
    best=pairs[0]
    yh=best[1]
    pitch=np.degrees(np.arctan2(cy-yh, F0))
    return dict(cx=cx, cy=cy, yh=float(yh), dy=float(yh-cy), pitch=float(pitch),
                n_agree=len(pairs), n1=int(best[2]), n2=int(best[3]))

if __name__=='__main__':
    pat=sys.argv[1] if len(sys.argv)>1 else 'frames/ljs1f/*.jpg'
    files=sorted(glob.glob(pat))
    rows=[]
    for i,f in enumerate(files):
        t=i+1
        if not (219<=t<=532): continue
        r=frame_pitch(f)
        if r:
            rows.append((t, r['pitch'], r['dy'], r['n_agree'], r['n1'], r['n2']))
    with open('out/pitch_perframe.csv','w',newline='') as fh:
        wr=csv.writer(fh); wr.writerow(['t_s','pitch_down_deg_f720','vp_row_minus_center_px','n_agree_pairs','n1','n2'])
        for r in rows: wr.writerow([r[0]]+[round(float(x),2) for x in r[1:]])
    print("frames with valid line-based pitch:", len(rows), "/ 314")
    if rows:
        ps=np.array([r[1] for r in rows])
        print("pitch deg: min %.2f max %.2f mean %.2f"%(ps.min(),ps.max(),ps.mean()))
