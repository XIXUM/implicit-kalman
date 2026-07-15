"""
Window-size ablation for the affine M-PME estimator.

Claim under test: the affine window is the span over which ONE motion is assumed
to hold. Widening it should IMPROVE the aggregate error (more samples, better
conditioned fit, smoother field) while DEGRADING the error at a depth
discontinuity, because a wider window averages across more of the boundary.

If both move in opposite directions, then aggregate accuracy is not merely a
weak proxy for correctness -- it is the mechanism by which the edge failure
hides.

Reuses the two-plane Z-dolly scene and masks of benchmark_nonaffine.py, so the
numbers are directly comparable to that benchmark.
"""
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MPME_faithful import faithful_mpme                       # noqa: E402
from benchmark_nonaffine import scene, align, N, R, ACC, SIG  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOWS = [9, 13, 17, 21, 25, 31, 37]


def masks(Z):
    valid = np.zeros((N, N), bool)
    valid[35:221, 35:221] = True
    valid &= (R > 12)                       # drop focus-of-expansion core
    edge = (ndimage.binary_dilation(Z < 6, iterations=4)
            & ~ndimage.binary_erosion(Z < 6, iterations=4))
    return valid, edge & valid


def main():
    matplotlib.use("Agg")
    a, b, gu, gv, Z, tz = scene()
    valid, band = masks(Z)

    rows = []
    print("affine-window ablation (M-PME, levels=4, lam=12), two-plane Z-dolly")
    print(f"  {'win':>4}  {'EPE all':>8}  {'EPE@edge':>9}  {'edge/all':>8}")
    for w in WINDOWS:
        U, V = faithful_mpme(a, b, levels=4, lam=12, win=w)
        U, V, e = align(U, V, gu, gv, valid)
        agg = float(np.median(e[valid]))
        edg = float(np.median(e[band]))
        rows.append((w, agg, edg))
        print(f"  {w:4d}  {agg:8.3f}  {edg:9.3f}  {edg / max(agg, 1e-9):8.2f}")

    ws = [r[0] for r in rows]
    aggs = [r[1] for r in rows]
    edges = [r[2] for r in rows]

    # Twin axes: the two curves live on different scales, and the SHAPE of each
    # is the finding (aggregate is U-shaped, edge is monotone). A shared linear
    # axis flattens the aggregate into a line and hides exactly that.
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax2 = ax1.twinx()
    l1, = ax1.plot(ws, aggs, "o-", color=ACC, lw=2, label="median EPE, all (aggregate)")
    l2, = ax2.plot(ws, edges, "s-", color=SIG, lw=2, label="median EPE, depth edge")

    wbest = ws[int(np.argmin(aggs))]
    ax1.axvline(wbest, color="#666", ls="--", lw=1)
    ax1.annotate(f"aggregate optimum (w={wbest}):\nedge already degraded",
                 xy=(wbest, min(aggs)), xytext=(wbest + 1.5, min(aggs) + 0.018),
                 fontsize=8.5, color="#444")

    ax1.set_xlabel("affine window size (px)")
    ax1.set_ylabel("median EPE, all (px)", color=ACC)
    ax2.set_ylabel("median EPE, depth edge (px)", color=SIG)
    ax1.tick_params(axis="y", labelcolor=ACC)
    ax2.tick_params(axis="y", labelcolor=SIG)
    ax1.set_title("The window that minimizes aggregate error does not preserve the edge")
    ax1.grid(alpha=0.3)
    ax1.legend(handles=[l1, l2], loc="upper center", fontsize=9)
    fig.tight_layout()
    p = os.path.join(HERE, "benchmark_window_ablation.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print("saved", p)

    # machine-readable, so the paper table cannot drift from the measurement
    csv = os.path.join(HERE, "benchmark_window_ablation.csv")
    with open(csv, "w") as f:
        f.write("window_px,epe_all_px,epe_edge_px\n")
        for w, agg, edg in rows:
            f.write(f"{w},{agg:.4f},{edg:.4f}\n")
    print("saved", csv)


if __name__ == "__main__":
    main()
