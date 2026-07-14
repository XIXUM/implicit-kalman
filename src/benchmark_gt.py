"""
Ground-truth benchmark for the phase-flow tools.

GT-by-construction: take a source image, warp it by an EXACTLY known transform
(scale / translation / rotation), which gives a pixel-exact ground-truth flow
field. Run both methods, measure endpoint error (EPE) against the GT, and sweep
the transform magnitude to expose the lambda/2 cliff.

This is the citable evidence layer: real per-pixel error metrics, not eyeballing.
The same GT mechanism accepts any real source image (--source path); a swap-in
Middlebury/Sintel .flo loader can be added later for external comparison.

Usage:
    python benchmark_gt.py            # synthetic broadband texture, scale sweep
    python benchmark_gt.py --source myimage.png
"""
import os, sys, argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PhaseFlowLocalFreq import phase_flow            # noqa: E402
from MultiScalePhaseFlow import mpme                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-9


# --------------------------------------------------------------------- GT construction
def broadband_texture(N=256, seed=7):
    """Deterministic dense multi-scale texture (phase info at every pixel)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((N, N))
    for sigma, amp in [(1, 1.0), (2, 1.4), (4, 1.8), (8, 2.2), (16, 2.6)]:
        img += amp * ndimage.gaussian_filter(rng.standard_normal((N, N)), sigma)
    img -= img.min(); img /= img.max()
    return img * 255.0


def warp_source(a, src_y, src_x):
    N = a.shape[0]
    pad = 4
    p = np.pad(a, pad, mode="reflect", reflect_type="odd")
    out = ndimage.map_coordinates(p, [src_y.ravel() + pad, src_x.ravel() + pad],
                                  order=3, mode="nearest")
    return out.reshape(N, N)


def make_pair(a, kind="scale", amount=1.1):
    """Return (B, gt_u, gt_v) with pixel-exact GT flow (A->B displacement)."""
    N = a.shape[0]
    yy, xx = np.mgrid[0:N, 0:N].astype(float)
    c = (N - 1) / 2.0
    if kind == "scale":
        s = amount
        src_x = c + (xx - c) / s
        src_y = c + (yy - c) / s
    elif kind == "translate":
        src_x = xx - amount
        src_y = yy - amount
    elif kind == "rotate":
        th = np.deg2rad(amount)
        ct, st = np.cos(th), np.sin(th)
        src_x = c + ct * (xx - c) + st * (yy - c)
        src_y = c - st * (xx - c) + ct * (yy - c)
    else:
        raise ValueError(kind)
    b = warp_source(a, src_y, src_x)
    gt_u = xx - src_x          # displacement that maps A -> B
    gt_v = yy - src_y
    return b, gt_u, gt_v


# --------------------------------------------------------------------- metrics
def epe(u, v, gt_u, gt_v, mask):
    """Endpoint error, sign-aligned (removes convention differences)."""
    e_pos = np.sqrt((u - gt_u) ** 2 + (v - gt_v) ** 2)
    e_neg = np.sqrt((-u - gt_u) ** 2 + (-v - gt_v) ** 2)
    e = e_pos if e_pos[mask].mean() <= e_neg[mask].mean() else e_neg
    return e, float(np.mean(e[mask])), float(np.median(e[mask]))


def central_mask(N, crop=0.12):
    m = np.zeros((N, N), bool)
    c = int(N * crop)
    m[c:N - c, c:N - c] = True
    return m


def run_single_scale(a, b):
    U, V, _ = phase_flow(a, b)
    return U, V


def run_multiscale(a, b):
    U, V, _ = mpme(a, b)
    return U, V


# --------------------------------------------------------------------- benchmark
def sweep(a, kind, amounts, mask):
    rows = []
    for amt in amounts:
        b, gu, gv = make_pair(a, kind, amt)
        max_disp = float(np.sqrt(gu ** 2 + gv ** 2)[mask].max())
        _, ss_mean, ss_med = epe(*run_single_scale(a, b), gu, gv, mask)
        _, ms_mean, ms_med = epe(*run_multiscale(a, b), gu, gv, mask)
        rows.append((amt, max_disp, ss_mean, ss_med, ms_mean, ms_med))
        print(f"  {kind} {amt:6.3f}  maxDisp {max_disp:6.2f}px | "
              f"single EPE {ss_mean:6.2f} | multi EPE {ms_mean:6.2f}")
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="optional source image path")
    ap.add_argument("--kind", default="scale", choices=["scale", "translate", "rotate"])
    args = ap.parse_args()
    matplotlib.use("Agg") if "--show" not in sys.argv else None

    if args.source:
        from PIL import Image
        a = np.asarray(Image.open(args.source).convert("L")).astype(float)
    else:
        a = broadband_texture(256)
    N = a.shape[0]
    mask = central_mask(N)

    if args.kind == "scale":
        amounts = [1.0, 1.03, 1.06, 1.10, 1.15, 1.20, 1.30]
    elif args.kind == "translate":
        amounts = [0, 1, 2, 4, 6, 9, 13]
    else:
        amounts = [0, 1, 2, 4, 6, 9, 13]

    print(f"GT benchmark: kind={args.kind}, source={'synthetic' if not args.source else args.source}")
    data = sweep(a, args.kind, amounts, mask)

    # operating point for the per-pixel panels: middle of the sweep
    op = amounts[len(amounts) // 2 + 1]
    b, gu, gv = make_pair(a, args.kind, op)
    ssU, ssV = run_single_scale(a, b); ss_e, _, _ = epe(ssU, ssV, gu, gv, mask)
    msU, msV = run_multiscale(a, b);  ms_e, _, _ = epe(msU, msV, gu, gv, mask)
    gt_mag = np.sqrt(gu ** 2 + gv ** 2)

    # ---- figure
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.24)

    axc = fig.add_subplot(gs[0, :])
    axc.plot(data[:, 1], data[:, 2], "o-", color="#c0392b", label="single-scale (Fleet-Jepson) mean EPE")
    axc.plot(data[:, 1], data[:, 4], "s-", color="#0f3d6e", label="multi-scale (M-PME) mean EPE")
    axc.axhline(1.0, color="k", lw=0.6, ls=":")
    axc.set_xlabel("max GT displacement in crop (px)")
    axc.set_ylabel("mean endpoint error (px)")
    axc.set_title(f"EPE vs displacement — {args.kind} sweep (lower is better)")
    axc.legend(); axc.grid(alpha=0.4)

    a2 = fig.add_subplot(gs[1, 0]); a2.imshow(a, cmap="gray"); a2.set_title("A (source)"); a2.axis("off")
    a3 = fig.add_subplot(gs[1, 1]); im3 = a3.imshow(gt_mag, cmap="viridis")
    a3.set_title(f"GT flow magnitude ({args.kind}={op})"); a3.axis("off")
    fig.colorbar(im3, ax=a3, fraction=0.046)
    a4 = fig.add_subplot(gs[1, 2])
    vmax = float(np.percentile(np.concatenate([ss_e[mask], ms_e[mask]]), 95))
    a4.imshow(np.hstack([ss_e, ms_e]), cmap="magma", vmax=vmax)
    a4.set_title("EPE map: single-scale | multi-scale"); a4.axis("off")

    out = os.path.join(HERE, f"benchmark_gt_{args.kind}.png")
    fig.savefig(out, dpi=90, bbox_inches="tight")
    print(f"\nsummary ({args.kind}):")
    print(f"  single-scale mean EPE range: {data[:,2].min():.2f} - {data[:,2].max():.2f} px")
    print(f"  multi-scale  mean EPE range: {data[:,4].min():.2f} - {data[:,4].max():.2f} px")
    print(f"saved {out}")
    if "--show" in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
