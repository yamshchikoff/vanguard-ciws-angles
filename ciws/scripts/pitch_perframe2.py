import cv2, numpy as np, glob, csv, sys

F0 = 720.0

def families(img, th=5.5, min_len=24, min_inl=10, max_vps=6):
    H,W=img.shape[:2]; cx,cy=W/2,H/2
    t,b,l,r=0.10,0.30,0.17,0.17
    x0,x1=int(W*l),int(W*(1-r)); y0,y1=int(H*t),int(H*(1-b))
    roi=img[y0:y1,x0:x1]
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    segs=lsd.detect(gray)[0]
    if segs is None: return [], (cx,cy)
    L=[]
    for xa,ya,xb,yb in segs[:,0]:
        if np.hypot(xb-xa,yb-ya)<min_len: continue
        a=ya-yb; b2=xa-xb; c=-(a*xa+b2*ya); nn=np.hypot(a,b2)
        if nn==0: continue
        L.append((a/nn,b2/nn,c/nn))
    L=np.array(L)
    if len(L)<10: return [], (cx,cy)
    rem=np.arange(len(L)); vps=[]
    for _ in range(max_vps):
        best=None; rng=np.random.default_rng(3)
        for _ in range(1200):
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
    return vps,(cx,cy)

def frame_measure(path):
    img=cv2.imread(path)
    if img is None: return None
    vps,(cx,cy)=families(img)
    if not vps: return None
    # Grade A: two families, same horizon row, different azimuth
    A=None
    for i in range(len(vps)):
        for j in range(i+1,len(vps)):
            (vi,ni),(vj,nj)=vps[i],vps[j]
            if abs(vi[0]-vj[0])<40: continue
            if abs(vi[1]-vj[1])>15: continue
            if abs(vi[1])>6000 or abs(vj[1])>6000: continue
            if A is None or ni+nj>A[0]:
                A=(ni+nj,(vi[1]+vj[1])/2,ni,nj)
    if A is not None:
        yh=A[1]
        return ('A', float(np.degrees(np.arctan2(cy-yh,F0))), float(yh-cy), A[2], A[3])
    # Grade B: strongest single family with finite VP row
    vps.sort(key=lambda v:-v[1])
    v,n=vps[0]
    if abs(v[1]-cy)<4000 and n>=14:
        return ('B', float(np.degrees(np.arctan2(cy-v[1],F0))), float(v[1]-cy), n, 0)
    return None

if __name__=='__main__':
    import os
    pat = sys.argv[1] if len(sys.argv)>1 else 'frames/ljs1f/*.jpg'
    stride = int(sys.argv[2]) if len(sys.argv)>2 else 1
    fps = float(sys.argv[3]) if len(sys.argv)>3 else 1.0
    out_csv = sys.argv[4] if len(sys.argv)>4 else 'out/pitch_perframe.csv'
    files=sorted(glob.glob(pat))[::stride]
    rows=[]
    for i,f in enumerate(files):
        m=frame_measure(f)
        if m: rows.append((round(219.0+i/fps,2),)+m)
    with open(out_csv,'w',newline='') as fh:
        wr=csv.writer(fh); wr.writerow(['t_s','grade','pitch_down_deg_f720','vp_row_minus_center_px','n1','n2'])
        for r in rows: wr.writerow([r[0],r[1],round(r[2],2),round(r[3],1),r[4],r[5]])
    nA=sum(1 for r in rows if r[1]=='A')
    print("valid frames:",len(rows),"/",len(files)," ; gradeA:",nA," gradeB:",len(rows)-nA)
    pa=[r[2] for r in rows if r[1]=='A']; pb=[r[2] for r in rows if r[1]=='B']
    if pa: print("A pitch: min %.2f max %.2f mean %.2f"%(min(pa),max(pa),sum(pa)/len(pa)))
    if pb: print("B pitch: min %.2f max %.2f mean %.2f"%(min(pb),max(pb),sum(pb)/len(pb)))
