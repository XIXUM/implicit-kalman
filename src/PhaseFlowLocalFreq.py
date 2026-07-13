"""
Phase-based flow with Fleet & Jepson corrections.

Two changes vs. ComplexPyramidFlow2D that the phase-method literature calls for:

  1. Local frequency, not nominal band center.
     Displacement along a filter orientation is d = Delta_phi / k, where k is the
     *measured* instantaneous frequency k = |grad phi| — NOT the filter's nominal
     center frequency scaleFt[i]. Measured via Im(grad R * conj(R) / |R|^2), which
     needs no phase unwrapping.

  2. Singularity rejection (stability of phase information).
     Phase is unreliable where the response amplitude rho -> 0 (phase singularities)
     or where the measured instantaneous frequency departs from the filter tuning.
     Such pixels are down-weighted instead of trusted.

All orientations are combined into a full 2D (U,V) vector per pixel by weighted
least squares (resolves the aperture problem), coarse-to-fine with image rewarping.

Refs: Fleet & Jepson 1990/1993, Jepson & Fleet 1991, Barron/Fleet/Beauchemin 1994.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import fft, ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-9


# ----------------------------------------------------------------------------- helpers
def load_gray(name, shape=None):
    img = np.asarray(Image.open(os.path.join(HERE, name)).convert("L")).astype(float)
    if shape is not None and img.shape != shape:
        from skimage.transform import resize
        img = resize(img, shape, order=1, preserve_range=True, anti_aliasing=True)
    return img


def instantaneous_freq(R):
    """Phase gradient k = grad(arg R) via Im(grad R * conj(R) / |R|^2). No unwrap.
    Returns (kx, ky) in rad/px, kx along columns (x), ky along rows (y)."""
    gy, gx = np.gradient(R)                      # d/drow, d/dcol
    denom = np.abs(R) ** 2 + EPS
    kx = np.imag(gx * np.conj(R)) / denom
    ky = np.imag(gy * np.conj(R)) / denom
    return kx, ky


def build_filters(N, n_oct, n_ori, ang_power=2.0):
    """Frequency-domain single-lobe directional bandpass filters (fftshift'd).
    Returns radial[i], angular[j], theta[j], k_nom[i] (rad/px)."""
    cx = N // 2
    u = np.arange(N) - cx
    uu, vv = np.meshgrid(u, u)                   # uu: x/col, vv: y/row
    radius = np.sqrt(uu ** 2 + vv ** 2)
    ang = np.arctan2(vv, uu)                      # polar angle in frequency plane

    half = N // 2
    log_half = np.log(half)
    radial, k_nom = [], []
    log_r = np.log(np.where(radius > 0, radius, 1))
    for i in range(n_oct):
        center = i * log_half / max(n_oct - 1, 1)
        bw = log_half / max(n_oct - 1, 1)
        radial.append(np.cos(np.clip((log_r - center) / bw * (np.pi / 2),
                                     -np.pi / 2, np.pi / 2)) ** 2)
        r_i = max(half ** (i / max(n_oct - 1, 1)), 1.0)   # center radius, cycles/img
        k_nom.append(2 * np.pi * r_i / N)                 # rad/px

    theta, angular = [], []
    for j in range(n_ori):
        th = np.pi * j / n_ori                    # orientations over [0, pi)
        theta.append(th)
        # single lobe aligned with direction th (analytic signal -> real phase)
        proj = np.cos(ang - th)
        angular.append(np.clip(proj, 0, 1) ** ang_power)
    return radial, angular, theta, k_nom


def warp_image(img, U, V):
    """Sample img at (row - V, col - U); odd-reflect pad to avoid boundary aliasing."""
    N = img.shape[0]
    yy, xx = np.mgrid[0:N, 0:N]
    pad = int(np.ceil(max(1.0, np.abs(U).max(), np.abs(V).max()))) + 2
    padded = np.pad(img, pad, mode="reflect", reflect_type="odd")
    out = ndimage.map_coordinates(
        padded, [(yy - V).ravel() + pad, (xx - U).ravel() + pad],
        order=1, mode="nearest")
    return out.reshape(N, N)


# ----------------------------------------------------------------------------- core
def foreground_mask(a, thresh_frac=0.08):
    """True where image A has local structure (edges); flat background -> False.
    Phase has no meaning in flat regions, so flow there must not be trusted."""
    gy, gx = np.gradient(a)
    struct = ndimage.gaussian_filter(np.sqrt(gx ** 2 + gy ** 2), 4)
    return struct > thresh_frac * struct.max()


def phase_flow(a, b, n_oct=8, n_ori=8, oct_lo=2, oct_hi=6,
               tau_amp=0.4, tau_freq=0.5, reg=1e-3,
               min_orients=2, med=5, energy_frac=0.02):
    """Coarse-to-fine phase-based flow. Returns (U, V) pixel displacement maps and
    a diagnostics dict for plotting."""
    N = a.shape[0]
    radial, angular, theta, k_nom = build_filters(N, n_oct, n_ori)
    ffB = fft.fftshift(fft.fft2(b))
    fg = foreground_mask(a)

    # auto-skip octaves that carry no real energy in A (checkerboard has energy
    # only near its harmonic frequencies) — trusting empty bands injects noise.
    ffA0 = fft.fftshift(fft.fft2(a))
    band_e = np.array([np.sum(np.abs(ffA0) ** 2 * radial[i]) for i in range(n_oct)])
    keep = band_e > energy_frac * band_e.max()

    U = np.zeros((N, N))
    V = np.zeros((N, N))
    diag = {"kmap": None, "conf": None, "rho": None, "oct_used": [],
            "fg": fg, "band_e": band_e, "keep": keep}

    for i in range(oct_lo, oct_hi + 1):
        if not keep[i]:
            continue
        a_w = warp_image(a, U, V)                 # rewarp A by current estimate
        ffA = fft.fftshift(fft.fft2(a_w))

        # per-pixel normal-equation accumulators
        Mxx = np.zeros((N, N)); Mxy = np.zeros((N, N)); Myy = np.zeros((N, N))
        bx = np.zeros((N, N));  by = np.zeros((N, N))
        rho_sum = np.zeros((N, N)); n_valid = np.zeros((N, N))
        kmag_acc = np.zeros((N, N)); kmag_w = np.zeros((N, N))

        for j in range(n_ori):
            filt = radial[i] * angular[j]
            if filt.max() < 0.05:
                continue
            RA = fft.ifft2(fft.ifftshift(ffA * filt))
            RB = fft.ifft2(fft.ifftshift(ffB * filt))

            dphi = np.angle(RB * np.conj(RA))     # phase shift A->B in [-pi, pi]

            # measured instantaneous frequency (average of the two responses)
            kax, kay = instantaneous_freq(RA)
            kbx, kby = instantaneous_freq(RB)
            kx = 0.5 * (kax + kbx); ky = 0.5 * (kay + kby)
            ct, st = np.cos(theta[j]), np.sin(theta[j])
            kproj = kx * ct + ky * st             # local freq along orientation

            # component displacement along n_j = Delta_phi / k_local  (pixels)
            k_safe = np.where(np.abs(kproj) < 0.05, np.nan, kproj)
            s = dphi / k_safe

            # --- Fleet-Jepson stability: amplitude + frequency-consistency ---
            rho = np.abs(RA) * np.abs(RB)
            amp_ok = rho > tau_amp * np.median(rho[fg]) if np.any(fg) else rho > 0
            freq_ok = np.abs(np.abs(kproj) - k_nom[i]) < tau_freq * k_nom[i]
            valid = amp_ok & freq_ok & np.isfinite(s) & fg
            w = np.where(valid, rho, 0.0)

            s = np.nan_to_num(s)
            Mxx += w * ct * ct; Mxy += w * ct * st; Myy += w * st * st
            bx += w * s * ct;   by += w * s * st
            rho_sum += rho; n_valid += valid.astype(float)
            kmag_acc += w * np.sqrt(kx ** 2 + ky ** 2); kmag_w += w

        # solve 2x2 per pixel (regularized)
        det = Mxx * Myy - Mxy ** 2 + reg
        du = (Myy * bx - Mxy * by) / det
        dv = (Mxx * by - Mxy * bx) / det

        # robustify: spatial median removes isolated singularity outliers,
        # clamp to +-half wavelength (the max a single band can measure).
        du = ndimage.median_filter(du, size=med)
        dv = ndimage.median_filter(dv, size=med)
        half_wave = np.pi / max(k_nom[i], EPS)
        du = np.clip(du, -half_wave, half_wave)
        dv = np.clip(dv, -half_wave, half_wave)

        # only update where enough orientations agreed and we are on structure
        supp = (n_valid >= min_orients) & fg
        U = U + np.where(supp, du, 0.0)
        V = V + np.where(supp, dv, 0.0)

        diag["oct_used"].append(i)
        diag["kmap"] = kmag_acc / (kmag_w + EPS)          # last octave local freq
        diag["conf"] = n_valid / n_ori                     # fraction of valid orients
        diag["rho"] = rho_sum / n_ori

    # fill the untrusted background by smoothing from trusted foreground
    U = np.where(fg, U, 0.0)
    V = np.where(fg, V, 0.0)
    return U, V, diag


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    a = load_gray("relief0.png")
    b = load_gray("relief0_sc1.png", shape=a.shape)
    N = a.shape[0]

    U, V, diag = phase_flow(a, b)

    # Confidence-weighted smoothing (normalized convolution): phase flow is only
    # defined at textured edges, so propagate those sparse measurements into the
    # flat regions weighted by confidence. Gives a coherent field for display.
    conf = np.where(diag["fg"], np.maximum(diag["conf"], 1e-3), 0.0)
    sig = 12
    Us = ndimage.gaussian_filter(U * conf, sig) / (ndimage.gaussian_filter(conf, sig) + EPS)
    Vs = ndimage.gaussian_filter(V * conf, sig) / (ndimage.gaussian_filter(conf, sig) + EPS)
    mag = np.sqrt(Us ** 2 + Vs ** 2)

    # divergence ~ local scale (s-1): dU/dx + dV/dy  (on the smoothed field)
    dVy, _ = np.gradient(Vs)
    _, dUx = np.gradient(Us)
    div = dUx + dVy

    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    plt.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.05, wspace=0.2, hspace=0.2)

    # 1) Image A
    ax[0, 0].imshow(a, cmap="gray"); ax[0, 0].set_title("A (relief0)")

    # 2) Image B
    ax[0, 1].imshow(b, cmap="gray"); ax[0, 1].set_title("B (relief0_sc1)")

    # 3) UV magnitude map + vector field overlay (step = N/10)
    step = max(N // 10, 1)
    yy, xx = np.mgrid[0:N:step, 0:N:step]
    axm = ax[0, 2]
    im = axm.imshow(mag, cmap="hot")
    axm.quiver(xx, yy, Us[::step, ::step], Vs[::step, ::step],
               color="cyan", angles="xy", scale_units="xy", scale=0.3, width=0.004)
    axm.set_title("UV magnitude + vector field (conf-smoothed)")
    fig.colorbar(im, ax=axm, fraction=0.046)

    # 4) UV center slices — expect a linear ramp through 0 at center for pure scaling
    ax[1, 0].plot(Us[N // 2, :], label="U (horizontal slice)")
    ax[1, 0].plot(Vs[:, N // 2], label="V (vertical slice)")
    ax[1, 0].axhline(0, color="k", lw=0.5)
    ax[1, 0].axvline(N // 2, color="k", lw=0.5, ls=":")
    ax[1, 0].set_title("UV slices, smoothed (expect linear ramp thru center)")
    ax[1, 0].legend(); ax[1, 0].grid()

    # 5) measured local frequency (last octave) — singularities show as spikes/holes
    im5 = ax[1, 1].imshow(diag["kmap"], cmap="viridis")
    ax[1, 1].set_title("measured local freq |grad phi| (last oct)")
    fig.colorbar(im5, ax=ax[1, 1], fraction=0.046)

    # 6) confidence: fraction of orientations that passed the stability mask
    im6 = ax[1, 2].imshow(diag["conf"], cmap="magma", vmin=0, vmax=1)
    ax[1, 2].set_title("phase confidence (valid orientations)")
    fig.colorbar(im6, ax=ax[1, 2], fraction=0.046)

    out = os.path.join(HERE, "phaseflow_result.png")
    fig.savefig(out, dpi=90)
    print(f"octaves used: {diag['oct_used']}")
    print(f"UV magnitude: max={mag.max():.3f}px  mean={mag.mean():.3f}px")
    print(f"mean divergence (~2*(s-1)) over center 50%: "
          f"{div[N//4:3*N//4, N//4:3*N//4].mean():.4f}")
    print(f"saved {out}")
    plt.show()
