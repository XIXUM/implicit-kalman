"""
Non-affine benchmark: perspective Z-dolly over a scene with varying depth.

Where the scaling benchmark was globally affine (a fronto-parallel plane under a
Z-dolly is exactly a scaling), a real scene is not. Here depth VARIES across the
image: a near plane in front of a far background. Under a forward camera step the
optical flow is  (x,y) * tz / (Z - tz)  — depth-dependent, with a DISCONTINUITY at
the object boundary. That violates the affine-per-window assumption every method
here relies on.

We measure two things:
  1. endpoint error (EPE) of the recovered flow, overall and in a band around the
     depth boundary,
  2. the DEPTH reconstructed from each method's flow vs. ground-truth depth —
     the quantity 3D reconstruction actually needs.

Ground truth is exact (constructed from the depth map + known camera step).
"""
import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PhaseFlowLocalFreq import phase_flow           # noqa: E402
from MultiScalePhaseFlow import mpme                 # noqa: E402
from MPME_faithful import faithful_mpme              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 256
CX = (N - 1) / 2.0
YY, XX = np.mgrid[0:N, 0:N].astype(float)
RX, RY = XX - CX, YY - CX
R = np.sqrt(RX ** 2 + RY ** 2)
EPS = 1e-6
ACC = "#12507e"; SIG = "#c0392b"; MID = "#e08a1e"


def texture(seed=7):
    rng = np.random.default_rng(seed)
    img = np.zeros((N, N))
    for s, a in [(2, 1.4), (4, 1.8), (8, 2.2), (16, 2.6)]:
        img += a * ndimage.gaussian_filter(rng.standard_normal((N, N)), s)
    return (img - img.min()) / (img.max() - img.min()) * 255


def scene(tz=0.30, z_far=10.0, z_near=4.0):
    """Two-plane depth map + Z-dolly. Returns A, B, gt_u, gt_v, depth."""
    a = texture()
    Z = np.full((N, N), z_far)
    Z[70:180, 55:150] = z_near                       # near object (off-centre)
    gu = RX * tz / (Z - tz)
    gv = RY * tz / (Z - tz)
    b = ndimage.map_coordinates(a, [YY - gv, XX - gu], order=3, mode="reflect")
    return a, b, gu, gv, Z, tz


def depth_from_flow(u, v, tz):
    """Invert flow = r*tz/(Z-tz):  Z = tz*(r + |flow|)/|flow|. FoE (r~0) unobservable."""
    fm = np.sqrt(u ** 2 + v ** 2)
    return tz * (R + fm) / (fm + EPS)


def align(U, V, gu, gv, m):
    ep = np.sqrt((U - gu) ** 2 + (V - gv) ** 2)
    en = np.sqrt((-U - gu) ** 2 + (-V - gv) ** 2)
    return (-U, -V, en) if en[m].mean() < ep[m].mean() else (U, V, ep)


METHODS = [
    ("Fleet (single-scale)", SIG, lambda a, b: phase_flow(a, b)[:2]),
    ("Reimpl (no paper)", MID, lambda a, b: mpme(a, b)[:2]),
    ("M-PME (affine, paper)", ACC, lambda a, b: faithful_mpme(a, b, levels=4, lam=12, win=25)),
]


def main():
    matplotlib.use("Agg")
    a, b, gu, gv, Z, tz = scene()

    valid = np.zeros((N, N), bool); valid[35:221, 35:221] = True
    valid &= (R > 12)                                # drop the focus-of-expansion core
    # boundary band: pixels near the depth discontinuity
    edge = ndimage.binary_dilation(Z < 6, iterations=4) & ~ndimage.binary_erosion(Z < 6, iterations=4)
    band = edge & valid

    gt_depth = depth_from_flow(gu, gv, tz)
    fig, ax = plt.subplots(2, 4, figsize=(16, 8.4))
    plt.subplots_adjust(left=0.03, right=0.99, top=0.93, bottom=0.03, wspace=0.12, hspace=0.14)

    def show_flow(axp, U, V, title, color):
        mag = np.sqrt(U ** 2 + V ** 2)
        axp.imshow(mag, cmap="hot", vmin=0, vmax=np.percentile(np.sqrt(gu**2+gv**2)[valid], 99))
        st = N // 12
        axp.quiver(XX[::st, ::st], YY[::st, ::st], U[::st, ::st], V[::st, ::st],
                   color="#66e0ff", angles="xy", scale_units="xy", scale=0.4, width=0.006)
        axp.set_title(title, fontsize=10, color=color); axp.axis("off")

    def show_depth(axp, D, title, color):
        axp.imshow(np.clip(D, 2, 13), cmap="viridis_r"); axp.set_title(title, fontsize=10, color=color)
        axp.axis("off")

    show_flow(ax[0, 0], gu, gv, "Ground-truth flow", "#333")
    show_depth(ax[1, 0], gt_depth, "Ground-truth depth", "#333")
    print("non-affine benchmark (Z-dolly, two-plane depth discontinuity):")
    print(f"  {'method':24s}  EPE all   EPE@edge   depth-RMSE")
    for k, (name, color, fn) in enumerate(METHODS):
        U, V = fn(a, b)
        U, V, e = align(U, V, gu, gv, valid)
        D = depth_from_flow(U, V, tz)
        d_rmse = np.sqrt(np.mean((np.clip(D, 2, 13) - np.clip(gt_depth, 2, 13))[valid] ** 2))
        show_flow(ax[0, k + 1], U, V, f"{name}\nEPE {np.median(e[valid]):.2f}px (edge {np.median(e[band]):.2f})", color)
        show_depth(ax[1, k + 1], D, f"depth from {name.split()[0]}", color)
        print(f"  {name:24s}  {np.median(e[valid]):6.2f}   {np.median(e[band]):7.2f}   {d_rmse:8.2f}")

    fig.suptitle("Perspective Z-dolly over varying depth — flow (top) and reconstructed depth (bottom)",
                 fontsize=13, y=0.985)
    p = os.path.join(HERE, "benchmark_nonaffine.png")
    fig.savefig(p, dpi=95); plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    main()
