
import cv2, numpy as np
def vps_soft(path, th=5.0, min_inl=14):
    img=cv2.imread(path)
    if img is None: return [], (0,0)
    H,W=img.shape[:2]; cx,cy=W/2,H/2
    t,b,l,r=0.10,0.30,0.17,0.17
    x0,x1=int(W*l),int(W*(1-r)); y0,y1=int(H*t),int(H*(1-b))
    roi=img[y0:y1,x0:x1]
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    segs=lsd.detect(gray)[0][:,0]
    L=[]
    for xa,ya,xb,yb in segs:
        if np.hypot(xb-xa,yb-ya)<22: continue
        a=ya-yb; b2=xa-xb; c=-(a*xa+b2*ya); nn=np.hypot(a,b2)
        if nn==0: continue
        L.append((a/nn,b2/nn,c/nn))
    L=np.array(L)
    if len(L)<10: return [], (cx,cy)
    rem=np.arange(len(L)); vps=[]
    for k in range(6):
        best=None; rng=np.random.default_rng(3)
        for _ in range(2500):
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
        vps.append((np.array([v[0]+x0, v[1]+y0]), len(inl)))
        rem=np.setdiff1d(rem,rem[inl])
        if len(rem)<min_inl: break
    return vps,(cx,cy)
