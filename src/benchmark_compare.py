"""
Three-way benchmark of phase-based motion estimators against ground truth.

Methods, in order of increasing fidelity:
  1. Fleet  — single-scale Fleet-Jepson phase flow            (PhaseFlowLocalFreq)
  2. Reimpl — multi-scale, per-pixel translation, NO paper    (MultiScalePhaseFlow)
  3. M-PME  — faithful affine multi-scale, from the 2026 paper (MPME_faithful)

GT-by-construction: warp a broadband texture by an exactly known scaling, so the
true per-pixel flow is known. We sweep the scale and measure endpoint error (EPE),
and render vector fields for visual comparison.

Outputs (src/):
  benchmark_epe_3methods.png    — median EPE vs. displacement, 3 curves
  benchmark_vectorfields.png    — GT + 3 methods x 3 motion levels, vector fields
  benchmark_cover_hero.png      — wide GT | Fleet | M-PME strip (title eye-catcher)
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
CROP = 40
FAITHFUL = dict(levels=4, lam=12, win=25)

ACC = "#12507e"; SIG = "#c0392b"; MID = "#e08a1e"


def texture(seed=7):
    rng = np.random.default_rng(seed)
    img = np.zeros((N, N))
    for s, a in [(2, 1.4), (4, 1.8), (8, 2.2), (16, 2.6)]:
        img += a * ndimage.gaussian_filter(rng.standard_normal((N, N)), s)
    return (img - img.min()) / (img.max() - img.min()) * 255


def make_pair(a, scale):
    c = (N - 1) / 2.0
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    sy = c + (yy - c) / scale
    sx = c + (xx - c) / scale
    b = ndimage.map_coordinates(a, [sy, sx], order=3, mode="nearest")
    return b, xx - sx, yy - sy                       # B, gt_u, gt_v


def mask():
    m = np.zeros((N, N), bool)
    m[CROP:N - CROP, CROP:N - CROP] = True
    return m


M = mask()


def align(U, V, gu, gv):
    """Sign-align a method's field to GT (removes convention differences)."""
    ep = np.sqrt((U - gu) ** 2 + (V - gv) ** 2)
    en = np.sqrt((-U - gu) ** 2 + (-V - gv) ** 2)
    if en[M].mean() < ep[M].mean():
        return -U, -V, en
    return U, V, ep


METHODS = [
    ("Fleet (single-scale)", SIG, lambda a, b: phase_flow(a, b)[:2]),
    ("Reimpl (no paper)", MID, lambda a, b: mpme(a, b)[:2]),
    ("M-PME (faithful, paper)", ACC, lambda a, b: faithful_mpme(a, b, **FAITHFUL)),
]


def run(fn, a, b, gu, gv):
    U, V = fn(a, b)
    return align(U, V, gu, gv)


# --------------------------------------------------------------------- EPE sweep
def epe_sweep(a):
    scales = [1.03, 1.06, 1.10, 1.15, 1.20, 1.25, 1.30]
    disp, med = [], {name: [] for name, _, _ in METHODS}
    for sc in scales:
        b, gu, gv = make_pair(a, sc)
        disp.append(float(np.sqrt(gu ** 2 + gv ** 2)[M].max()))
        for name, _, fn in METHODS:
            _, _, e = run(fn, a, b, gu, gv)
            med[name].append(float(np.median(e[M])))
        print(f"  scale {sc:.2f}  maxDisp {disp[-1]:5.1f}px  " +
              "  ".join(f"{n.split()[0]}={med[n][-1]:.2f}" for n, _, _ in METHODS))
    return np.array(disp), med


def plot_epe(disp, med):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for name, color, _ in METHODS:
        ax.plot(disp, med[name], "o-", color=color, lw=2, ms=6, label=name)
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("max ground-truth displacement (px)")
    ax.set_ylabel("median endpoint error (px)")
    ax.set_title("Scaling benchmark — median EPE vs. displacement (lower is better)")
    ax.set_yscale("log"); ax.grid(alpha=0.35, which="both"); ax.legend()
    fig.tight_layout()
    p = os.path.join(HERE, "benchmark_epe_3methods.png")
    fig.savefig(p, dpi=100); plt.close(fig); print("saved", p)


# --------------------------------------------------------------------- vector fields
def quiver_panel(ax, U, V, title, color_title):
    mag = np.sqrt(U ** 2 + V ** 2)
    ax.imshow(mag, cmap="hot", vmin=0, vmax=max(mag[M].max(), 1e-3))
    step = N // 11
    yy, xx = np.mgrid[0:N:step, 0:N:step]
    ax.quiver(xx, yy, U[::step, ::step], V[::step, ::step], color="#66e0ff",
              angles="xy", scale_units="xy", scale=0.35, width=0.006)
    ax.set_title(title, fontsize=10, color=color_title)
    ax.set_xticks([]); ax.set_yticks([])


def plot_vectorfields(a):
    show_scales = [1.06, 1.15, 1.25]
    cols = ["Ground truth"] + [n for n, _, _ in METHODS]
    colors = ["#333"] + [c for _, c, _ in METHODS]
    fig, axes = plt.subplots(len(show_scales), 4, figsize=(15, 11.2))
    for r, sc in enumerate(show_scales):
        b, gu, gv = make_pair(a, sc)
        maxd = np.sqrt(gu ** 2 + gv ** 2)[M].max()
        quiver_panel(axes[r, 0], gu, gv, f"Ground truth", colors[0])
        axes[r, 0].set_ylabel(f"scale {sc}\nmax {maxd:.0f}px", fontsize=10)
        for k, (name, color, fn) in enumerate(METHODS):
            U, V, e = run(fn, a, b, gu, gv)
            quiver_panel(axes[r, k + 1], U, V,
                         f"{name}\nmed EPE {np.median(e[M]):.2f}px", color)
    fig.suptitle("Recovered displacement fields — scaling, coarse (top) to large (bottom)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    p = os.path.join(HERE, "benchmark_vectorfields.png")
    fig.savefig(p, dpi=95); plt.close(fig); print("saved", p)


def plot_cover_hero(a):
    sc = 1.22
    b, gu, gv = make_pair(a, sc)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    quiver_panel(axes[0], gu, gv, "Ground truth", "#333")
    fU, fV, fe = run(METHODS[0][2], a, b, gu, gv)
    quiver_panel(axes[1], fU, fV, f"Fleet single-scale — breaks (EPE {np.median(fe[M]):.1f}px)", SIG)
    mU, mV, me = run(METHODS[2][2], a, b, gu, gv)
    quiver_panel(axes[2], mU, mV, f"Affine M-PME — clean (EPE {np.median(me[M]):.2f}px)", ACC)
    fig.tight_layout()
    p = os.path.join(HERE, "benchmark_cover_hero.png")
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig); print("saved", p)


if __name__ == "__main__":
    matplotlib.use("Agg")
    a = texture()
    print("EPE sweep:")
    disp, med = epe_sweep(a)
    plot_epe(disp, med)
    plot_vectorfields(a)
    plot_cover_hero(a)
    print("\nfinal median EPE (px):")
    for name, _, _ in METHODS:
        print(f"  {name:26s}: " + " ".join(f"{v:6.2f}" for v in med[name]))
