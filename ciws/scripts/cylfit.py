import cv2, numpy as np, sys, json

def extract_ring_points(img_path, bands):
    """bands: list of (y_center, halfwidth). Returns list of Nx2 arrays (full-image coords)."""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H,W = gray.shape
    x0,x1,y0,y1 = 740, 945, 160, 800
    sub = gray[y0:y1, x0:x1]
    bh = cv2.morphologyEx(sub, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_RECT,(13,3)))
    th = cv2.threshold(bh, 8, 255, cv2.THRESH_BINARY)[1]
    n,lab,stats,cent = cv2.connectedComponentsWithStats(th,8)
    pts_all=[]
    for i in range(1,n):
        xx,yy,ww,hh,area = stats[i]
        if area<40 or ww<18: continue
        pts = np.stack(np.nonzero(lab==i)[::-1],1).astype(float)
        pts[:,0]+=x0; pts[:,1]+=y0
        pts_all.append(pts)
    rings=[]
    for (yc,hw) in bands:
        sel=[]
        for pts in pts_all:
            my = pts[:,1].mean()
            if abs(my-yc)<hw and pts[:,0].std()>18:
                sel.append(pts)
        if sel:
            P=np.vstack(sel)
            rings.append(P)
        else:
            rings.append(np.zeros((0,2)))
    return rings, img.shape

def project(P, f, theta, psi, cx, cy, X0, Y0, h, r):
    # camera at world origin; world Z up
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(psi), np.sin(psi)
    fwd = np.array([ct*sp, ct*cp, -st])
    right = np.array([cp, -sp, 0.0])
    down = np.cross(fwd, right)
    D = P - np.array([0,0,0])
    z = D@fwd
    u = f*(D@right)/z + cx
    v = f*(D@down)/z + cy
    return np.stack([u,v],1), z

def ring_points(X0,Y0,h,r,n=72):
    t = np.linspace(0,2*np.pi,n,endpoint=False)
    return np.stack([X0+r*np.cos(t), Y0+r*np.sin(t), np.full_like(t,h)],1)

def residuals(params, rings, cx, cy):
    f, theta, psi = params[0], params[1], params[2]
    X0, Y0 = params[3], params[4]
    K = len(rings)
    hs = params[5:5+K]; rs = params[5+K:5+2*K]
    res=[]
    for i,P in enumerate(rings):
        if len(P)==0: continue
        model = ring_points(X0,Y0,hs[i],rs[i])
        proj,z = project(model, f, theta, psi, cx, cy, X0, Y0, hs[i], rs[i])
        # nearest neighbor distance from observed points to projected ring
        d = np.sqrt(((P[:,None,:]-proj[None,:,:])**2).sum(-1)).min(1)
        res.append(d)
    return np.concatenate(res) if res else np.zeros(0)

def fit(rings, cx, cy, init, iters=40):
    p = np.array(init, float)
    lam = 1e-3
    r0 = residuals(p, rings, cx, cy); cost = (r0**2).sum()
    for it in range(iters):
        J = []
        base = residuals(p, rings, cx, cy)
        M = base.size
        Jm = np.zeros((M, p.size))
        for k in range(p.size):
            dp = p.copy()
            step = max(1e-6, abs(p[k])*1e-4)
            dp[k]+=step
            rp = residuals(dp, rings, cx, cy)
            Jm[:,k] = (rp-base)/step
        AtlA = Jm.T@Jm + lam*np.eye(p.size)
        g = Jm.T@base
        try:
            stepv = np.linalg.solve(AtlA, -g)
        except np.linalg.LinAlgError:
            break
        p2 = p+stepv
        # keep f, r positive
        if p2[0]<=50 or np.any(p2[5+len(rings):]<=0):
            lam*=10; continue
        cost2 = (residuals(p2, rings, cx, cy)**2).sum()
        if cost2 < cost:
            p = p2; cost = cost2; lam = max(1e-6, lam/3)
        else:
            lam *= 8
            if lam>1e9: break
    return p, cost, residuals(p, rings, cx, cy)

if __name__=='__main__':
    path = sys.argv[1]
    bands = [(233,32),(311,34),(385,36),(459,36),(528,36),(600,40)]
    rings, shape = extract_ring_points(path, bands)
    H,W = shape[:2]; cx,cy = W/2,H/2
    print("ring point counts:", [len(r) for r in rings])
    for i,r in enumerate(rings):
        if len(r): print(f" ring{i}: x {r[:,0].min():.0f}-{r[:,0].max():.0f} y {r[:,1].min():.0f}-{r[:,1].max():.0f}")
