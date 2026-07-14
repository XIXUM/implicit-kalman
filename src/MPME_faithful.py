"""
Faithful M-PME — affine, windowed, coarse-to-fine phase-based motion estimation.

Reimplementation of:
  M.Z. Li, Z.T. Yan, G. Liu, Z. Mao, "Large amplitude motion estimation via
  multi-scale phase-based video processing", Mech. Syst. Signal Process. 253
  (2026) 114301. doi:10.1016/j.ymssp.2026.114301

Faithful to the paper's core (vs. the simplified per-pixel translation in
MultiScalePhaseFlow.py):
  - 4 Gabor directions theta = 0,45,90,135, wavelength lam=30 (their Sec. 2/4).
  - Confidence C (their Eq. 7) + directional-consistency mask (Eq. 8).
  - Phase motion constraint c = C*(1/2 d/dx(phi2+phi1), 1/2 d/dy(phi2+phi1),
    phi2-phi1)  (Eq. 12-13).
  - AFFINE 6-parameter motion fit over a window via the 7x7 normal matrix Q
    (Eq. 14-23), solved per pixel. Affine = translation + rotation + scale/shear,
    so scale and rotation are modeled exactly.
  - Gaussian pyramid, coarse-to-fine warp superposition (Sec. 3.1, 3.3, Fig. 4):
    v_l = 2*v_{l-1} + a_l.
"""
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
from scipy.ndimage import convolve1d

EPS = 1e-9
DIRS = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def gabor_kernel(theta, lam=30.0, gamma=1.0, b=1.0, psi=0.0):
    # sigma from bandwidth b (paper Sec. 2, their sigma formula)
    sigma = (1 / np.pi) * np.sqrt(np.log(2) / 2) * ((2 ** b + 1) / (2 ** b - 1)) * lam
    ext = int(np.ceil(3 * sigma))
    y, x = np.mgrid[-ext:ext + 1, -ext:ext + 1]
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    env = np.exp(-(xr ** 2 + gamma ** 2 * yr ** 2) / (2 * sigma ** 2))
    return env * np.exp(1j * (2 * np.pi * xr / lam + psi))


def gabor_response(img, theta, lam):
    k = gabor_kernel(theta, lam)
    return (fftconvolve(img, k.real, mode="same")
            + 1j * fftconvolve(img, k.imag, mode="same"))


def phase_grad(q):
    # d/dx, d/dy of arg(q) without unwrap: Im(grad q * conj(q))/|q|^2
    gy, gx = np.gradient(q)
    denom = np.abs(q) ** 2 + EPS
    return np.imag(gx * np.conj(q)) / denom, np.imag(gy * np.conj(q)) / denom


def gaussian_pyramid(img, levels):
    pyr = [img]
    for _ in range(levels - 1):
        pyr.append(ndimage.gaussian_filter(pyr[-1], 0.5)[::2, ::2])
    return pyr  # [0]=finest


def warp(img, vx, vy):
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    pad = int(np.ceil(max(1.0, np.abs(vx).max(), np.abs(vy).max()))) + 2
    p = np.pad(img, pad, mode="reflect", reflect_type="odd")
    out = ndimage.map_coordinates(p, [(yy - vy).ravel() + pad, (xx - vx).ravel() + pad],
                                  order=1, mode="nearest")
    return out.reshape(H, W)


def _moment_key(m1, m2):
    cnt = {}
    for m in (m1, m2):
        if m != "1":
            cnt[m] = cnt.get(m, 0) + 1
    if not cnt:
        return "1"
    if "dx" in cnt and "dy" in cnt:
        return "dxdy"
    if cnt.get("dx") == 2:
        return "dx2"
    if cnt.get("dy") == 2:
        return "dy2"
    return "dx" if "dx" in cnt else "dy"


_ORDER = {"x": 0, "y": 1, "t": 2}
_CC = ["x", "x", "x", "y", "y", "y", "t"]           # c-component of each u-entry
_MONO = ["dx", "dy", "1", "dx", "dy", "1", "1"]     # window-relative monomial


def single_scale_affine(a, b, lam, win):
    """One-scale residual affine motion, Farnebaeck-style with WINDOW-RELATIVE
    coordinates so the normal matrix stays well conditioned. Displacement at each
    pixel is the constant term of the locally fitted affine (a2, a5)."""
    H, W = a.shape
    r = win // 2
    t = np.arange(-r, r + 1, dtype=float)
    ones = np.ones(win)

    # accumulate the six confidence-weighted c-products over the four directions
    P = {"xx": 0.0, "xy": 0.0, "xt": 0.0, "yy": 0.0, "yt": 0.0, "tt": 0.0}
    for th in DIRS:
        q1 = gabor_response(a, th, lam)
        q2 = gabor_response(b, th, lam)
        p1x, p1y = phase_grad(q1)
        p2x, p2y = phase_grad(q2)
        cx = 0.5 * (p1x + p2x)
        cy = 0.5 * (p1y + p2y)
        ct = np.angle(q2 * np.conj(q1))                      # phi2 - phi1, wrapped
        m1 = np.abs(q1) ** 2; m2 = np.abs(q2) ** 2
        C = (m1 * m2) / ((m1 + m2) ** 1.5 + EPS)             # confidence (Eq. 7)
        C = C * ((cx * np.cos(th) + cy * np.sin(th)) > 0)    # directional mask (Eq. 8)
        cx *= C; cy *= C; ct *= C                            # c = C*(...)  (Eq. 13)
        P["xx"] += cx * cx; P["xy"] += cx * cy; P["xt"] += cx * ct
        P["yy"] += cy * cy; P["yt"] += cy * ct; P["tt"] += ct * ct

    # window moments of each P-field: sum over window of P * {1,dx,dy,dx2,dxdy,dy2}
    def mom(f, ky, kx):
        return convolve1d(convolve1d(f, kx, axis=1, mode="nearest"),
                          ky, axis=0, mode="nearest")

    def moments(f):
        return {"1": mom(f, ones, ones), "dx": mom(f, ones, t), "dy": mom(f, t, ones),
                "dx2": mom(f, ones, t * t), "dxdy": mom(f, t, t), "dy2": mom(f, t * t, ones)}

    S = {k: moments(v) for k, v in P.items()}

    def ck(i, j):
        pair = sorted((_CC[i], _CC[j]), key=_ORDER.get)
        return pair[0] + pair[1]

    M = np.zeros((7, 7, H, W))
    for i in range(7):
        for j in range(i, 7):
            M[i, j] = M[j, i] = S[ck(i, j)][_moment_key(_MONO[i], _MONO[j])]

    A = np.moveaxis(M[0:6, 0:6], [0, 1], [2, 3])             # (H,W,6,6)
    rhs = -np.moveaxis(M[0:6, 6], 0, 2)                       # (H,W,6)
    A = A + np.eye(6) * (1e-4 * np.trace(A, axis1=2, axis2=3)[..., None, None] / 6 + EPS)
    par = np.linalg.solve(A, rhs[..., None])[..., 0]         # (H,W,6)

    lim = lam / 2                                            # per-scale residual < lambda/2
    return np.clip(par[..., 2], -lim, lim), np.clip(par[..., 5], -lim, lim)


def faithful_mpme(a, b, levels=4, lam=30.0, win=21, refine=2):
    """Coarse-to-fine affine phase flow. Returns (U, V) at full resolution."""
    pa = gaussian_pyramid(a, levels)
    pb = gaussian_pyramid(b, levels)

    U = np.zeros_like(pa[-1]); V = np.zeros_like(pa[-1])
    for lvl in range(levels - 1, -1, -1):
        Hl, Wl = pa[lvl].shape
        if U.shape != (Hl, Wl):
            from skimage.transform import resize
            sy, sx = Hl / U.shape[0], Wl / U.shape[1]
            U = resize(U, (Hl, Wl), order=1, preserve_range=True) * sx
            V = resize(V, (Hl, Wl), order=1, preserve_range=True) * sy
        for _ in range(refine):
            aw = warp(pa[lvl], U, V)
            du, dv = single_scale_affine(aw, pb[lvl], lam, win)
            U = U + du; V = V + dv
    return U, V


if __name__ == "__main__":
    # quick self-test on a synthetic scaling
    import os
    from scipy import ndimage as ndi
    rng = np.random.default_rng(7)
    N = 256
    img = np.zeros((N, N))
    for s, amp in [(2, 1.4), (4, 1.8), (8, 2.2), (16, 2.6)]:
        img += amp * ndi.gaussian_filter(rng.standard_normal((N, N)), s)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    c = (N - 1) / 2; s = 1.15
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    b = ndi.map_coordinates(img, [(c + (yy - c) / s), (c + (xx - c) / s)], order=3)
    gu = xx - (c + (xx - c) / s); gv = yy - (c + (yy - c) / s)
    U, V = faithful_mpme(img, b)
    m = np.zeros((N, N), bool); m[30:226, 30:226] = True
    e = np.sqrt((U - gu) ** 2 + (V - gv) ** 2)
    en = np.sqrt((-U - gu) ** 2 + (-V - gv) ** 2)
    e = e if e[m].mean() <= en[m].mean() else en
    print(f"faithful M-PME  scale=1.15  mean EPE {e[m].mean():.3f}px  "
          f"median {np.median(e[m]):.3f}px  maxGT {np.sqrt(gu**2+gv**2)[m].max():.1f}px")
