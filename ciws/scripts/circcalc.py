import cv2, numpy as np, sys

def extract_seam_points(gray, x0, x1, band_y, halfw=9):
    """For each column x in [x0,x1], find subpixel y of darkest pixel within band_y±halfw."""
    pts=[]
    H,W = gray.shape
    g = gray.astype(np.float32)
    for x in range(x0, x1):
        y0 = max(0, band_y-halfw); y1 = min(H-1, band_y+halfw)
        col = g[y0:y1+1, x]
        i = int(np.argmin(col))
        if i==0 or i==len(col)-1: continue
        # only accept if it is a real local min (seam) and darker than surroundings
        c = col[i]
        if c > np.median(col) - 3: continue
        # parabolic subpixel
        a,b,cc = col[i-1], col[i], col[i+1]
        denom = (a - 2*b + cc)
        dy = 0.0 if abs(denom)<1e-6 else 0.5*(a - cc)/denom
        pts.append((x, y0 + i + dy))
    return np.array(pts)

def fit_conic(pts):
    x = pts[:,0]; y = pts[:,1]
    # Taubin-ish normalization: build design matrix
    D = np.stack([x*x, x*y, y*y, x, y, np.ones_like(x)], axis=1)
    # normalize columns
    s = np.linalg.norm(D, axis=0); s[s==0]=1
    Dn = D / s
    _,_,Vt = np.linalg.svd(Dn)
    c = Vt[-1] / s
    return c  # [A,B,C,D,E,F]

def solve_fyh(c, cx, cy):
    A,B,C,D,E,F = c
    if abs(B) < 1e-12: return None
    # imag: f*(2A cx + B yh + D) = 0 -> yh = -(2A cx + D)/B
    yh = -(2*A*cx + D)/B
    # real: A(cx^2 - f^2) + B cx yh + C yh^2 + D cx + E yh + F = 0
    rem = A*(cx*cx) + B*cx*yh + C*yh*yh + D*cx + E*yh + F
    # A(cx^2 - f^2) + rem - A cx^2 = 0 -> -A f^2 + rem = 0 -> f^2 = rem/A... careful:
    # A(cx^2 - f^2) + (B cx yh + C yh^2 + D cx + E yh + F) = 0
    rest = B*cx*yh + C*yh*yh + D*cx + E*yh + F
    if abs(A) < 1e-12: return None
    f2 = (A*cx*cx + rest)/A
    if f2 <= 0: return None
    return float(np.sqrt(f2)), float(yh)

def analyze(path, seams, cx, cy, xlo=None, xhi=None, debug=None):
    img = cv2.imread(path); H,W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if xlo is None: xlo = int(W*0.30)
    if xhi is None: xhi = int(W*0.70)
    out=[]
    for i,(band, hw) in enumerate(seams):
        pts = extract_seam_points(gray, xlo, xhi, band, halfw=hw)
        if len(pts) < 60:
            out.append((i, None, None, 0)); continue
        # robust: fit, drop outliers by residual, refit
        c = fit_conic(pts)
        for _ in range(3):
            x=pts[:,0]; y=pts[:,1]
            r = np.abs(c[0]*x*x + c[1]*x*y + c[2]*y*y + c[3]*x + c[4]*y + c[5])
            r = r / np.hypot(2*c[0]*x + c[1]*y + c[3], c[1]*x + 2*c[2]*y + c[4])
            keep = r < max(0.35, np.percentile(r, 80))
            if keep.sum() < 60: break
            pts = pts[keep]
            c = fit_conic(pts)
        res = solve_fyh(c, cx, cy)
        if res is None:
            out.append((i, None, None, len(pts))); continue
        f, yh = res
        out.append((i, f, yh, len(pts)))
        if debug is not None:
            for (px,py) in pts:
                cv2.circle(debug, (int(px), int(py)), 1, (0,0,255), -1)
    return out

if __name__ == '__main__':
    path = sys.argv[1]
    img = cv2.imread(path); H,W = img.shape[:2]; cx,cy = W/2,H/2
    # seams: (y_band, halfwidth)
    seams = [(210,22),(296,26),(368,28),(444,30),(514,30),(600,34)]
    dbg = img.copy()
    res = analyze(path, seams, cx, cy, debug=dbg)
    for i,f,yh,n in res:
        if f: print(f"seam{i}: f={f:.0f} px, y_h={yh:.1f}, pts={n}, pitch={np.degrees(np.arctan2(cy-yh,f)):+.2f} deg")
        else: print(f"seam{i}: fail, pts={n}")
    cv2.imwrite('/tmp/seams_dbg.jpg', cv2.resize(dbg,None,fx=0.75,fy=0.75))
