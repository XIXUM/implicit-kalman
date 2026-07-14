"""
M-PME — Multi-scale Phase-based Motion Estimation (large-amplitude).

SIMPLIFIED reimplementation of the coarse-to-fine idea in
  M.Z. Li, Z.T. Yan, G. Liu, Z. Mao, "Large amplitude motion estimation via
  multi-scale phase-based video processing", Mech. Syst. Signal Process. 253
  (2026) 114301. doi:10.1016/j.ymssp.2026.114301

IMPORTANT — this is NOT faithful to the paper. The real M-PME fits an AFFINE
motion model (6 params) over a window via a Farnebaeck-style integral-image least
squares (their Eq. 15-23), with the Eq. 7 confidence and Eq. 8 directional mask.
This file instead fits a per-pixel 2-param translation, which is why it degrades
on scaling. Because scale/rotation ARE affine, the faithful method is expected to
handle them far better than this simplification does. A faithful port is TODO.

Idea: a single phase-based estimate is limited to |delta| < lambda/2 (phase wraps
past +-pi). A large displacement is decomposed across a Gaussian pyramid: at the
coarsest level the motion is downsampled by 2^L and therefore small enough to
measure without wrapping. The estimate is upsampled, used to warp the next finer
level, and only the (small) residual is measured there. Summing across levels
("motion field fusion") recovers the full large-amplitude field. No unwrapping.

This is the benchmark to compare ImplicitKalman against.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import fft, ndimage
from skimage.transform import resize

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-9


# ----------------------------------------------------------------------------- helpers
def load_gray(name, shape=None):
    img = np.asarray(Image.open(os.path.join(HERE, name)).convert("L")).astype(float)
    if shape is not None and img.shape != shape:
        img = resize(img, shape, order=1, preserve_range=True, anti_aliasing=True)
    return img


def instantaneous_freq(R):
    """k = grad(arg R) via Im(grad R * conj(R)/|R|^2). No unwrap. (kx along x/cols)."""
    gy, gx = np.gradient(R)
    denom = np.abs(R) ** 2 + EPS
    return np.imag(gx * np.conj(R)) / denom, np.imag(gy * np.conj(R)) / denom


def gaussian_pyramid(img, levels):
    """pyr[0] = finest (original), pyr[-1] = coarsest."""
    pyr = [img]
    cur = img
    for _ in range(levels - 1):
        cur = ndimage.gaussian_filter(cur, 1.0)[::2, ::2]
        pyr.append(cur)
    return pyr


def warp(img, U, V):
    """Sample img at (row - V, col - U); odd-reflect padding."""
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W]
    pad = int(np.ceil(max(1.0, np.abs(U).max(), np.abs(V).max()))) + 2
    p = np.pad(img, pad, mode="reflect", reflect_type="odd")
    out = ndimage.map_coordinates(
        p, [(yy - V).ravel() + pad, (xx - U).ravel() + pad], order=1, mode="nearest")
    return out.reshape(H, W)


def oriented_filters(H, W, wavelength, n_ori, bw=0.7, ang_power=2.0):
    """Single-lobe oriented bandpass filters (fftshift'd) tuned to `wavelength` px."""
    cy, cx = H // 2, W // 2
    vv, uu = np.mgrid[0:H, 0:W]
    uu = uu - cx; vv = vv - cy
    radius = np.sqrt(uu ** 2 + vv ** 2)
    ang = np.arctan2(vv, uu)

    r0 = min(H, W) / wavelength                     # target radius in FFT bins
    log_r = np.log(np.where(radius > 0, radius, 1))
    radial = np.cos(np.clip((log_r - np.log(max(r0, 1))) / bw, -np.pi / 2, np.pi / 2)) ** 2
    k_nom = 2 * np.pi * r0 / min(H, W)              # rad/px

    filts, thetas = [], []
    for j in range(n_ori):
        th = np.pi * j / n_ori
        lobe = np.clip(np.cos(ang - th), 0, 1) ** ang_power
        filts.append(radial * lobe)
        thetas.append(th)
    return filts, thetas, k_nom


def single_scale_flow(a, b, wavelength=6.0, n_ori=6, tau_amp=0.3, tau_freq=0.6,
                      reg=1e-3, med=3):
    """One-level phase-based residual flow (motion must be < lambda/2). LS over
    orientations resolves the aperture problem; unstable phase is masked out."""
    H, W = a.shape
    ffA = fft.fftshift(fft.fft2(a))
    ffB = fft.fftshift(fft.fft2(b))
    filts, thetas, k_nom = oriented_filters(H, W, wavelength, n_ori)

    Mxx = np.zeros((H, W)); Mxy = np.zeros((H, W)); Myy = np.zeros((H, W))
    bx = np.zeros((H, W)); by = np.zeros((H, W)); n_valid = np.zeros((H, W))

    for filt, th in zip(filts, thetas):
        RA = fft.ifft2(fft.ifftshift(ffA * filt))
        RB = fft.ifft2(fft.ifftshift(ffB * filt))
        dphi = np.angle(RB * np.conj(RA))                 # in (-pi, pi]

        kax, kay = instantaneous_freq(RA)
        kbx, kby = instantaneous_freq(RB)
        kx = 0.5 * (kax + kbx); ky = 0.5 * (kay + kby)
        ct, st = np.cos(th), np.sin(th)
        kproj = kx * ct + ky * st
        k_safe = np.where(np.abs(kproj) < 0.05, np.nan, kproj)
        # shift theorem: dphi = -k.d, so displacement component d_n = -dphi/kproj
        s = -dphi / k_safe                                 # component displacement (px)

        rho = np.abs(RA) * np.abs(RB)
        amp_ok = rho > tau_amp * np.median(rho)
        freq_ok = np.abs(np.abs(kproj) - k_nom) < tau_freq * k_nom
        valid = amp_ok & freq_ok & np.isfinite(s)
        w = np.where(valid, rho, 0.0)
        s = np.nan_to_num(s)

        Mxx += w * ct * ct; Mxy += w * ct * st; Myy += w * st * st
        bx += w * s * ct; by += w * s * st
        n_valid += valid.astype(float)

    det = Mxx * Myy - Mxy ** 2 + reg
    du = (Myy * bx - Mxy * by) / det
    dv = (Mxx * by - Mxy * bx) / det
    du = ndimage.median_filter(du, size=med)
    dv = ndimage.median_filter(dv, size=med)
    half = wavelength / 2.0
    return np.clip(du, -half, half), np.clip(dv, -half, half), n_valid


# ----------------------------------------------------------------------------- M-PME
def mpme(a, b, levels=5, wavelength=6.0, n_ori=6, refine=2):
    """Coarse-to-fine multi-scale phase flow. Returns (U, V) at full resolution."""
    pa = gaussian_pyramid(a, levels)
    pb = gaussian_pyramid(b, levels)

    U = np.zeros_like(pa[-1]); V = np.zeros_like(pa[-1])     # start at coarsest
    per_level = []
    for lvl in range(levels - 1, -1, -1):                    # coarse -> fine
        Hl, Wl = pa[lvl].shape
        if U.shape != (Hl, Wl):                              # upsample estimate x2
            sy, sx = Hl / U.shape[0], Wl / U.shape[1]
            U = resize(U, (Hl, Wl), order=1, preserve_range=True) * sx
            V = resize(V, (Hl, Wl), order=1, preserve_range=True) * sy

        for _ in range(refine):                              # a few warps per level
            a_w = warp(pa[lvl], U, V)
            du, dv, nvalid = single_scale_flow(a_w, pb[lvl], wavelength, n_ori)
            U = U + du; V = V + dv
        per_level.append((lvl, float(np.sqrt(U ** 2 + V ** 2).mean())))

    return U, V, per_level


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    a = load_gray("relief0.png")
    b = load_gray("relief0_sc1.png", shape=a.shape)
    H, W = a.shape

    U, V, per_level = mpme(a, b)
    mag = np.sqrt(U ** 2 + V ** 2)

    # validation: warp A by the estimated flow, compare to B (endpoint quality)
    a_rec = warp(a, U, V)
    fg = ndimage.gaussian_filter(np.abs(np.gradient(a)[0]) + np.abs(np.gradient(a)[1]), 3)
    fg = fg > 0.08 * fg.max()
    err0 = np.abs(a - b)[fg]
    err1 = np.abs(a_rec - b)[fg]
    rmse0, rmse1 = np.sqrt((err0 ** 2).mean()), np.sqrt((err1 ** 2).mean())

    # divergence ~ local scale (s-1)
    dVy, _ = np.gradient(V); _, dUx = np.gradient(U)
    div = dUx + dVy
    s_est = 1.0 + 0.5 * div[H // 4:3 * H // 4, W // 4:3 * W // 4].mean()

    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    plt.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.05, wspace=0.2, hspace=0.22)

    ax[0, 0].imshow(a, cmap="gray"); ax[0, 0].set_title("A (relief0)")
    ax[0, 1].imshow(b, cmap="gray"); ax[0, 1].set_title("B (relief0_sc1)")

    step = max(H // 10, 1)
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    im = ax[0, 2].imshow(mag, cmap="hot")
    ax[0, 2].quiver(xx, yy, U[::step, ::step], V[::step, ::step],
                    color="cyan", angles="xy", scale_units="xy", scale=0.3, width=0.004)
    ax[0, 2].set_title("M-PME: UV magnitude + vector field")
    fig.colorbar(im, ax=ax[0, 2], fraction=0.046)

    ax[1, 0].plot(U[H // 2, :], label="U (horizontal slice)")
    ax[1, 0].plot(V[:, W // 2], label="V (vertical slice)")
    ax[1, 0].axhline(0, color="k", lw=0.5); ax[1, 0].axvline(W // 2, color="k", lw=0.5, ls=":")
    ax[1, 0].set_title("UV slices (expect linear ramp thru center)")
    ax[1, 0].legend(); ax[1, 0].grid()

    im5 = ax[1, 1].imshow(div, cmap="coolwarm", vmin=-0.4, vmax=0.4)
    ax[1, 1].set_title(f"divergence ~ 2*(s-1)   [s_est={s_est:.3f}]")
    fig.colorbar(im5, ax=ax[1, 1], fraction=0.046)

    im6 = ax[1, 2].imshow(np.abs(a_rec - b), cmap="magma")
    ax[1, 2].set_title(f"warp(A)->B residual   RMSE {rmse0:.1f}->{rmse1:.1f}")
    fig.colorbar(im6, ax=ax[1, 2], fraction=0.046)

    out = os.path.join(HERE, "mpme_result.png")
    fig.savefig(out, dpi=90)
    print(f"levels mean|flow| coarse->fine: {[f'L{l}:{m:.2f}' for l, m in per_level]}")
    print(f"UV magnitude: max={mag.max():.2f}px  mean={mag.mean():.2f}px")
    print(f"estimated global scale s ~ {s_est:.4f}")
    print(f"warp residual RMSE (foreground): {rmse0:.2f} -> {rmse1:.2f}  "
          f"({100*(1-rmse1/max(rmse0,EPS)):.0f}% reduction)")
    print(f"saved {out}")
    plt.show()
