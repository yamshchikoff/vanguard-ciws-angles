import sys, numpy as np, cv2
sys.path.insert(0,'scripts')
import cylfit as cf

img=cv2.imread('frames/u_364.jpg'); H,W=img.shape[:2]; cx,cy=W/2,H/2
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
x0,x1,y0,y1=740,945,160,800
sub=gray[y0:y1,x0:x1]
bh=cv2.morphologyEx(sub,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_RECT,(13,3)))
th=cv2.threshold(bh,8,255,cv2.THRESH_BINARY)[1]
n,lab,stats,cent=cv2.connectedComponentsWithStats(th,8)
comps=[]
for i in range(1,n):
    xx,yy,ww,hh,area=stats[i]
    if area<40 or ww<15 or yy<5: continue
    pts=np.stack(np.nonzero(lab==i)[::-1],1).astype(float)
    pts[:,0]+=x0; pts[:,1]+=y0
    comps.append(pts)
comps.sort(key=lambda p: p[:,1].mean())
groups=[]
for p in comps:
    if groups and abs(p[:,1].mean()-groups[-1][-1][:,1].mean())<30:
        groups[-1].append(p)
    else:
        groups.append([p])
print("groups:", [(round(float(np.vstack(g)[:,1].mean())), sum(len(x) for x in g), round(float(np.vstack(g)[:,0].std())) ) for g in groups])
# pick the three widest clean groups in the stack area (y between 250 and 560, xstd>25)
cand=[]
for g in groups:
    P=np.vstack(g); ym=P[:,1].mean(); xs=P[:,0].std()
    if 250<ym<560 and xs>25 and len(P)>150:
        cand.append(P)
print("chosen rings:", [len(P) for P in cand])
rings=cand[:3]
# order: top ring first. Set init heights so top ring has larger h.
rings.sort(key=lambda P: P[:,1].mean())
print("ring y means:", [round(float(P[:,1].mean())) for P in rings])
init=[1440.0, 0.0, -0.082, 0.0, 40.0, 5.6, 3.6, 1.6, 3.5, 3.4, 3.3]
init=init[:5+len(rings)+len(rings)]
p, cost, res = cf.fit(rings, cx, cy, np.array(init,float), iters=30)
f,theta,psi,X0,Y0 = p[0],p[1],p[2],p[3],p[4]
K=len(rings)
print("f=%.0f  theta=%.2f deg  psi=%.2f deg  X0=%.2f Y0=%.2f"%(f, np.degrees(theta), np.degrees(psi), X0, Y0))
print("h=", np.round(p[5:5+K],2), " r=", np.round(p[5+K:5+2*K],2))
print("RMS resid px = %.2f (n=%d)"%(np.sqrt((res**2).mean()), res.size))
vis=img.copy()
for i in range(K):
    model=cf.ring_points(X0,Y0,p[5+i],p[5+K+i],n=200)
    proj,_=cf.project(model,f,theta,psi,cx,cy,X0,Y0,p[5+i],p[5+K+i])
    for u,v in proj:
        ui,vi=int(round(u)),int(round(v))
        if 0<=ui<W and 0<=vi<H: vis[vi,ui]=(0,0,255)
cv2.imwrite('out/cylfit.jpg',cv2.resize(vis,None,fx=0.7,fy=0.7),[cv2.IMWRITE_JPEG_QUALITY,85])
print("saved out/cylfit.jpg")
