import cv2, numpy as np, sys, glob, json

def vps_of(img, crop=(0.10,0.30,0.17,0.17), minlen=70, thresh=3.0, min_inl=6, max_vps=7):
    H,W=img.shape[:2]; cx,cy=W/2,H/2
    t,b,l,r=crop
    x0,x1=int(W*l),int(W*(1-r)); y0,y1=int(H*t),int(H*(1-b))
    roi=img[y0:y1,x0:x1]
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(cv2.GaussianBlur(gray,(3,3),0),50,150)
    ys,xs=np.nonzero(edges); P=np.stack([xs,ys],1).astype(float)
    lines=cv2.HoughLinesP(edges,1,np.pi/1440,threshold=40,minLineLength=minlen,maxLineGap=6)
    if lines is None: return [],None
    LL=[]; seg=[]
    for xa,ya,xb,yb in lines[:,0]:
        n=np.hypot(xb-xa,yb-ya)
        if n<minlen: continue
        d=np.array([xb-xa,yb-ya])/n; nrm=np.array([-d[1],d[0]])
        rel=P-np.array([xa,ya]); dist=np.abs(rel@nrm); tt=rel@d
        sel=(dist<2.0)&(tt>-20)&(tt<n+20); R=P[sel]
        if len(R)<40: continue
        mean=R.mean(0); _,_,vt=np.linalg.svd(R-mean); dv=vt[0]
        a=dv[1]; b2=-dv[0]; c=-(a*mean[0]+b2*mean[1]); nn=np.hypot(a,b2)
        LL.append((a/nn,b2/nn,c/nn)); seg.append((xa+x0,ya+y0,xb+x0,yb+y0))
    L=np.array(LL)
    if len(L)<8: return [],None
    rem=np.arange(len(L)); vps=[]
    for _ in range(max_vps):
        best=None; rng=np.random.default_rng(11)
        for _ in range(2500):
            i,j=rng.integers(0,len(rem),2)
            if i==j: continue
            v=np.cross(L[rem[i]],L[rem[j]])
            if abs(v[2])<1e-9: continue
            v=v/v[2]
            d=np.abs(L[rem]@v)/np.hypot(L[rem][:,0],L[rem][:,1])
            inl=np.where(d<thresh)[0]
            if best is None or len(inl)>len(best[1]): best=(v,inl)
        if best is None or len(best[1])<min_inl: break
        for _ in range(3):
            A=L[rem[best[1]]]
            _,_,Vt=np.linalg.svd(A); v=Vt[-1]
            if abs(v[2])<1e-9: break
            v=v/v[2]
            d=np.abs(L[rem]@v)/np.hypot(L[rem][:,0],L[rem][:,1])
            inl=np.where(d<thresh)[0]
            if len(inl)<min_inl: break
            best=(v,inl)
        v,inl=best
        vps.append((v,rem[inl]))
        rem=np.setdiff1d(rem,rem[inl])
        if len(rem)<min_inl: break
    return vps,(cx,cy)

def manhattan(vps, cx, cy):
    out=[]
    for i in range(len(vps)):
        for j in range(i+1,len(vps)):
            for k in range(j+1,len(vps)):
                vs=[vps[a][0] for a in (i,j,k)]
                fs=[]; ok=True
                for (a,b) in ((0,1),(0,2),(1,2)):
                    va=vs[a]; vb=vs[b]
                    dot=(va[0]-cx)*(vb[0]-cx)+(va[1]-cy)*(vb[1]-cy)
                    if -dot<=0: ok=False; break
                    fs.append(float(np.sqrt(-dot)))
                if not ok: continue
                if not (600<min(fs) and max(fs)<7000): continue
                spread=(max(fs)-min(fs))/float(np.mean(fs))
                if spread>0.15: continue
                # one of the three pairs should be the two horizontals: find pair with |dy|<25
                ys_pairs=[(abs(vs[a][1]-vs[b][1]),(a,b)) for (a,b) in ((0,1),(0,2),(1,2))]
                dy,pair=min(ys_pairs)
                if dy>30: continue   # no horizontal pair -> reject
                out.append(dict(idx=(i,j,k), f=[round(x,0) for x in fs], spread=round(spread,3),
                                dy=round(float(dy),1), n=[len(vps[a][1]) for a in (i,j,k)]))
    return out

if __name__=='__main__':
    pat=sys.argv[1]
    files=sorted(glob.glob(pat))
    hits=[]
    for fp in files:
        img=cv2.imread(fp)
        if img is None: continue
        vps,cc=vps_of(img)
        if cc is None or len(vps)<3: continue
        ms=manhattan(vps, cc[0], cc[1])
        for m in ms:
            m['path']=fp; hits.append(m)
    hits.sort(key=lambda m:(m['spread']))
    for m in hits[:20]: print(json.dumps(m))
    print("total hits:", len(hits), "/", len(files))
