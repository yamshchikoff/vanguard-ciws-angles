import cv2, numpy as np, sys, json

def ransac_vp(L, thresh=3.0, iters=4000, min_inliers=8):
    N = len(L)
    if N < 2: return None
    best = None
    rng = np.random.default_rng(12345)
    for _ in range(iters):
        i, j = rng.integers(0, N, 2)
        if i == j: continue
        v = np.cross(L[i], L[j])
        if abs(v[2]) < 1e-9: continue
        v = v / v[2]
        d = np.abs(L @ v) / np.hypot(L[:, 0], L[:, 1])
        inl = np.where(d < thresh)[0]
        if best is None or len(inl) > len(best[1]):
            best = (v, inl)
    if best is None or len(best[1]) < min_inliers: return None
    # refit with SVD on inliers
    for _ in range(3):
        A = L[best[1]]
        _, _, Vt = np.linalg.svd(A)
        v = Vt[-1]
        if abs(v[2]) < 1e-9: return None
        v = v / v[2]
        d = np.abs(L @ v) / np.hypot(L[:, 0], L[:, 1])
        inl = np.where(d < thresh)[0]
        if len(inl) < min_inliers: return None
        best = (v, inl)
    return best

def estimate(path, crop=(0.09, 0.24, 0.17, 0.17), save=None, verbose=False):
    img = cv2.imread(path)
    if img is None: return None
    H, W = img.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    t, b, l, r = crop
    x0, x1 = int(W * l), int(W * (1 - r))
    y0, y1 = int(H * t), int(H * (1 - b))
    roi = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=45,
                            minLineLength=40, maxLineGap=5)
    if lines is None: return None
    segs = []
    L = []
    for xa, ya, xb, yb in lines[:, 0]:
        Xa, Ya = xa + x0, ya + y0
        Xb, Yb = xb + x0, yb + y0
        a = Yb - Ya; bb = Xa - Xb; c = -(a * Xa + bb * Ya)
        n = np.hypot(a, bb)
        if n == 0: continue
        L.append((a / n, bb / n, c / n))
        segs.append((Xa, Ya, Xb, Yb))
    L = np.array(L); segs = np.array(segs)
    if len(L) < 12: return None
    vps = []
    remaining = np.arange(len(L))
    for _ in range(5):
        res = ransac_vp(L[remaining])
        if res is None: break
        v, inl_local = res
        inl = remaining[inl_local]
        vps.append((v, inl))
        remaining = np.setdiff1d(remaining, inl)
        if len(remaining) < 6: break
    if len(vps) < 2: return None
    # vertical VP = the one closest to x=cx (roll=0 assumption)
    vert = min(vps, key=lambda vp: abs(vp[0][0] - cx))
    horiz = [vp for vp in vps if vp is not vert]
    horiz.sort(key=lambda vp: -len(vp[1]))
    if len(horiz) < 2: return None
    # focal from orthogonality: (v_vert-c).(v_h-c) = -f^2
    fs = []
    for vh, _ in horiz[:3]:
        dot = (vert[0][0] - cx) * (vh[0] - cx) + (vert[0][1] - cy) * (vh[1] - cy)
        if -dot > 0:
            fs.append(np.sqrt(-dot))
    if not fs: return None
    f = float(np.median(fs))
    if not (0.4 * W < f < 6 * W): return None
    # horizon = line through the two best horizontal VPs
    v1 = horiz[0][0]; v2 = horiz[1][0]
    hl = np.cross([v1[0], v1[1], 1.0], [v2[0], v2[1], 1.0])
    if abs(hl[1]) < 1e-9: return None
    hl = hl / np.hypot(hl[0], hl[1])
    yh = -(hl[0] * cx + hl[2]) / hl[1]
    pitch = np.degrees(np.arctan2(cy - yh, f))
    info = dict(path=str(path), pitch_deg=round(float(pitch), 2), f_px=round(f, 1),
                n_lines=int(len(L)), vps=[(round(v[0],1), round(v[1],1), int(len(i))) for v, i in vps],
                horizon_y_at_cx=round(float(yh), 1),
                conf=dict(nv=len(vert[1]), nh1=len(horiz[0][1]), nh2=len(horiz[1][1])))
    if save:
        for k, (v, inl) in enumerate(vps):
            color = (0, 0, 255) if v is vert[0] else (0, 255, 0)
            for idx in inl:
                Xa, Ya, Xb, Yb = segs[idx]
                cv2.line(img, (int(Xa), int(Ya)), (int(Xb), int(Yb)), color, 2)
            cv2.circle(img, (int(v[0]), int(v[1])), 6, color, -1)
        # draw horizon
        xs = [0, W]
        ys = [-(hl[0] * x + hl[2]) / hl[1] for x in xs]
        cv2.line(img, (int(xs[0]), int(ys[0])), (int(xs[1]), int(ys[1])), (255, 0, 255), 2)
        cv2.circle(img, (int(cx), int(cy)), 8, (255, 255, 255), 2)
        cv2.imwrite(save, img)
    if verbose: print(json.dumps(info))
    return info

if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(json.dumps(estimate(p)))
