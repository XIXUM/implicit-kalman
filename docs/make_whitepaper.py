"""Render the benchmark whitepaper 'The Affine Ceiling' as a PDF.

Reproducible: regenerates the scaling (affine) and depth (non-affine) benchmark
figures, then composes docs/the_affine_ceiling_whitepaper.pdf.

Usage:
    python docs/make_whitepaper.py            # regenerate figures + build
    python docs/make_whitepaper.py --no-regen # build only
"""
import os, sys, subprocess, textwrap, datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")
OUT = os.path.join(HERE, "the_affine_ceiling_whitepaper.pdf")
HERO = os.path.join(SRC, "benchmark_nonaffine_hero.png")
EPE = os.path.join(SRC, "benchmark_epe_3methods.png")
NONAFF = os.path.join(SRC, "benchmark_nonaffine.png")

VERSION = "Draft v0.2"
AUTHOR = "Felix Schaller"
ORG = "FelixSchallerCOM"
CONTACT = "inquiry@felixschaller.com"

A4 = (8.27, 11.69)
INK = "#10202f"; ACCENT = "#12507e"; SIGNAL = "#c0392b"; MUTE = "#5b6a78"; LINE = "#d4dbe2"
SERIF = "DejaVu Serif"; SANS = "DejaVu Sans"; MONO = "DejaVu Sans Mono"
plt.rcParams.update({"text.color": INK})


def regenerate():
    env = dict(os.environ, MPLBACKEND="Agg")
    for s in ("benchmark_compare.py", "benchmark_nonaffine.py"):
        print(f"running {s} ...")
        subprocess.run([sys.executable, os.path.join(SRC, s)], cwd=SRC, env=env, check=True)


def page():
    fig = plt.figure(figsize=A4); fig.patch.set_facecolor("white"); return fig


def eyebrow(fig, y, text):
    fig.text(0.09, y, text, fontsize=8.5, color=ACCENT, family=MONO, weight="bold",
             va="top", transform=fig.transFigure)


def head(fig, y, text, size=15):
    fig.text(0.09, y, text, fontsize=size, color=INK, family=SERIF, weight="bold",
             va="top", transform=fig.transFigure)


def body(fig, y, text, size=10.5, x=0.09, width=92, color="#243542", lh=0.0182, fam=SERIF):
    for para in text.split("\n"):
        if not para.strip():
            y -= lh * 0.6; continue
        for ln in (textwrap.wrap(para, width=width) or [""]):
            fig.text(x, y, ln, fontsize=size, va="top", color=color, family=fam,
                     transform=fig.transFigure)
            y -= lh
    return y


def lead(fig, y, label, text, size=10.5, lh=0.0182, width=None):
    """A bold mini-header followed by body text (for Related Work entries)."""
    for ln in textwrap.wrap(label, width=80):
        fig.text(0.09, y, ln, fontsize=size, va="top", color=INK, family=SERIF,
                 weight="bold", transform=fig.transFigure)
        y -= lh
    for ln in textwrap.wrap(text, width=94):
        fig.text(0.09, y, ln, fontsize=size, va="top", color="#243542", family=SERIF,
                 transform=fig.transFigure)
        y -= lh
    return y - lh * 0.45


def bullets(fig, y, items, width=86):
    for it in items:
        fig.text(0.09, y, "—", fontsize=10.5, color=ACCENT, family=SANS, va="top",
                 transform=fig.transFigure)
        for ln in textwrap.wrap(it, width=width):
            fig.text(0.12, y, ln, fontsize=10.5, va="top", color="#243542", family=SERIF,
                     transform=fig.transFigure)
            y -= 0.0182
        y -= 0.006
    return y


def rule(fig, y, x0=0.09, x1=0.91):
    fig.add_artist(plt.Line2D([x0, x1], [y, y], color=LINE, lw=1.0, transform=fig.transFigure))


def image(fig, path, x, y, w):
    img = mpimg.imread(path)
    h = w * (img.shape[0] / img.shape[1]) * (A4[0] / A4[1])
    ax = fig.add_axes([x, y - h, w, h]); ax.imshow(img); ax.axis("off")
    return y - h


def caption(fig, y, text, width=104):
    for ln in textwrap.wrap(text, width=width):
        fig.text(0.09, y, ln, fontsize=8, family=MONO, color=MUTE, va="top")
        y -= 0.0135
    return y


def table(fig, y, rows, col_x, width_box=(0.13, 0.87), size=9.5):
    x0, x1 = width_box
    for r, row in enumerate(rows):
        if r == 0:
            fig.add_artist(plt.Rectangle((x0, y - 0.004), x1 - x0, 0.020, color=ACCENT,
                           transform=fig.transFigure, zorder=0))
        for cx, cell in zip(col_x, row):
            fig.text(cx, y + 0.006, cell, fontsize=size, family=MONO,
                     color="white" if r == 0 else INK,
                     weight="bold" if r == 0 else "normal", va="center", zorder=1)
        y -= 0.023
    return y


def timeline(fig, ly):
    """A horizontal milestone timeline to anchor the Related Work page."""
    fig.text(0.09, ly + 0.045, "TIMELINE", fontsize=8, family=MONO, color=MUTE, weight="bold")
    x0, x1 = 0.17, 0.85
    fig.add_artist(plt.Line2D([x0, x1], [ly, ly], color=ACCENT, lw=1.5, transform=fig.transFigure))
    items = [("1990", "Fleet & Jepson"), ("1991", "singularities"),
             ("2002", "Gautama-VanHulle"), ("2013", "MIT magnif."),
             ("2026", "M-PME · LingBot")]
    for i, (yr, tag) in enumerate(items):
        x = x0 + (x1 - x0) * i / (len(items) - 1)
        fig.add_artist(plt.Line2D([x], [ly], marker="o", ms=7, color=ACCENT,
                       markeredgecolor="white", transform=fig.transFigure))
        fig.text(x, ly + 0.014, yr, fontsize=9, family=MONO, color=INK, weight="bold",
                 ha="center", va="bottom")
        fig.text(x, ly - 0.014, tag, fontsize=7, family=SERIF, color=MUTE, ha="center", va="top")


def build():
    if "--no-regen" not in sys.argv:
        regenerate()

    with PdfPages(OUT) as pdf:
        # ================================================================ cover
        fig = page()
        fig.add_artist(plt.Rectangle((0, 0.845), 1, 0.155, color=ACCENT, transform=fig.transFigure))
        fig.text(0.09, 0.965, "BENCHMARK WHITEPAPER · MOTION ESTIMATION FOR 3D",
                 fontsize=9, color="#bcd8f0", family=MONO, weight="bold")
        fig.text(0.09, 0.925, "The Affine Ceiling", fontsize=31, color="white",
                 family=SERIF, weight="bold")
        fig.text(0.09, 0.877, "Formal motion estimation is sharp on affine motion — "
                 "and blind at depth.", fontsize=12, color="#e6f0fa", family=SERIF)

        yb = image(fig, HERO, 0.07, 0.825, 0.86)
        caption(fig, yb - 0.006,
                "Depth reconstructed from each method's optical flow under a perspective Z-dolly. "
                "Ground truth is crisp; every phase-based method — including the affine state of "
                "the art — blurs, frays, or destroys the depth boundary. That boundary is what 3D "
                "reconstruction is made of.")

        y = yb - 0.05
        fig.text(0.09, y, "ABSTRACT", fontsize=8, family=MONO, color=ACCENT, weight="bold")
        y -= 0.024
        y = body(fig, y,
            "Motion estimation is the front end of 3D reconstruction. We benchmark how well formal, "
            "non-probabilistic phase-based methods recover it, against pixel-exact ground truth, "
            "across three implementations of increasing fidelity — from single-scale Fleet-Jepson "
            "to a faithful port of the 2026 affine multi-scale method (M-PME). The result splits "
            "cleanly: affine motion (translation, rotation, scale) is solved to sub-pixel accuracy "
            "at large magnitude, overturning the folklore that formal methods only handle small "
            "motion; but on the non-affine, per-pixel, discontinuous motion of a real 3D scene, "
            "every method blurs or destroys the depth boundary reconstruction depends on. The "
            "ceiling is not motion size — it is the affine assumption.", lh=0.0178, width=94)

        # key findings
        ky = 0.415
        fig.text(0.09, ky, "KEY FINDINGS", fontsize=8, family=MONO, color=ACCENT, weight="bold")
        ky -= 0.026
        bullets(fig, ky, [
            "Affine motion (scale, rotation) is solved to sub-pixel accuracy, even at large magnitude.",
            "Non-affine depth motion: every method blurs or destroys the depth boundary.",
            "The ceiling is the affine assumption, not the size of the motion.",
        ])

        # author / version box
        by = 0.205
        fig.add_artist(plt.Rectangle((0.09, by - 0.10), 0.82, 0.10, facecolor="#f4f6f9",
                       edgecolor=LINE, transform=fig.transFigure))
        cells = [("AUTHOR", AUTHOR), ("ORGANISATION", ORG),
                 ("VERSION", VERSION + "  ·  " + datetime.date.today().isoformat()),
                 ("CONTACT", CONTACT), ("STATUS", "Working note — pre-publication"),
                 ("SOURCE", "reproducible · src/benchmark_*.py")]
        for i, (lab, val) in enumerate(cells):
            cx = 0.11 + (i % 2) * 0.41
            cy = by - 0.018 - (i // 2) * 0.03
            fig.text(cx, cy, lab, fontsize=6.8, family=MONO, color=ACCENT, weight="bold", va="top")
            fig.text(cx, cy - 0.013, val, fontsize=9, family=SERIF, color=INK, va="top")

        fig.text(0.09, 0.075, "ImplicitKalman — working note   ·   " + VERSION
                 + "   ·   Generated " + datetime.date.today().isoformat(),
                 fontsize=8, family=MONO, color=MUTE)
        pdf.savefig(fig); plt.close(fig)

        # ================================================================ related work
        fig = page()
        eyebrow(fig, 0.95, "BACKGROUND")
        head(fig, 0.925, "Related work: three decades of phase-based motion"); y = 0.885
        y = body(fig, y,
            "The idea that local phase encodes motion is old and deep. This note stands on it; the "
            "credit belongs to the people below, and so does the map of where it breaks.",
            lh=0.0178, width=94)
        y -= 0.006
        y = lead(fig, y, "Fleet & Jepson, 1990 — component velocity.",
            "They showed that the output phase of a band-pass (Gabor) filter is far more stable "
            "under changes in contrast, scale and illumination than its amplitude, and that the "
            "temporal evolution of that phase yields a component of image velocity — the projection "
            "of motion onto the filter's orientation. Combining orientations resolves the aperture "
            "problem. This is the theoretical spine every method here still rests on [1].",
            width=94)
        y = lead(fig, y, "Jepson & Fleet, 1991; Fleet & Jepson, 1993 — where it fails.",
            "The same authors mapped the failure modes with unusual honesty: phase singularities — "
            "points in scale-space where the response amplitude vanishes and phase becomes "
            "undefined and wildly unstable — and a stability criterion to detect and discard them. "
            "This is why every serious implementation, ours included, carries a confidence measure "
            "and a singularity mask [2][3].", width=94)
        y = lead(fig, y, "Gautama & Van Hulle, 2002 — dense, real-time phase flow.",
            "Recursive temporal filtering and a formalized phase-constancy constraint turned phase "
            "into a dense optical-flow method — the direct ancestor of the windowed least-squares "
            "fit in the 2026 method benchmarked here [5].", width=94)
        y = lead(fig, y, "Wadhwa, Rubinstein, Durand & Freeman (MIT), 2013 — motion magnification.",
            "The most visible modern descendant. Built on the complex steerable pyramid, it "
            "manipulates the local phase of each sub-band to MAGNIFY tiny, invisible motions — a "
            "pulse in a wrist, the sway of a building — without ever computing an explicit flow "
            "field. Beautiful, widely used work. It is also the clearest statement of the wall: the "
            "phase-motion relation is only linear for small displacements, so magnification is "
            "bounded — they derive the limit explicitly — and pushing past a fraction of the "
            "sub-band wavelength produces exactly the aliasing the theory predicts. Their follow-ups "
            "(the Riesz pyramid for speed; later large-motion variants) trade quality for range but "
            "do not remove the wall [6].", width=94)
        y -= 0.004
        y = body(fig, y,
            "The hurdles, in one list. Every method above meets the same four: (1) phase wrapping "
            "beyond +-pi — the lambda/2 ceiling; (2) singularities where amplitude vanishes; "
            "(3) temporal aliasing from the filter's frequency tuning; and (4) the aperture "
            "problem. The 2026 M-PME method is the current best answer to (1) — a Gaussian pyramid "
            "that keeps each level's residual under the wall — while inheriting the classical "
            "answers to (2)-(4) [7]. What none of them addresses is motion that is not locally "
            "affine. That is where this note begins.", lh=0.0178, width=94)
        timeline(fig, 0.16)
        pdf.savefig(fig); plt.close(fig)

        # ================================================================ stakes + limit + methods
        fig = page()
        eyebrow(fig, 0.95, "01 — THE STAKES")
        head(fig, 0.925, "Two ways to estimate motion"); y = 0.885
        y = body(fig, y,
            "Learned monocular models are advancing fast. Robbyant's LingBot-Map (Ant Group), for "
            "one, reconstructs 3D scenes from a single RGB camera in real time — streaming, "
            "end-to-end, ~20 FPS, open-source [8]. Impressive engineering, and formal hand-derived "
            "algorithms do not currently keep up on raw capability. But it is a trained neural "
            "network: probabilistic by construction — no error bound, no repeatable output, no "
            "provable worst case. For a photo app, fine. Under functional-safety standards — "
            "ISO 26262, DO-178C, IEC 62304 — it is disqualifying.", lh=0.0178)
        y = body(fig, y,
            "This work takes the opposite stance: formal, non-probabilistic estimation, derived "
            "from signal theory rather than trained from data. Every output is reproducible and "
            "analyzable. The open question is not whether it can be trusted — it is whether it is "
            "accurate enough. Specifically: accurate enough for 3D.", lh=0.0178)
        y -= 0.012
        eyebrow(fig, y, "02 — THE SINGLE-SCALE LIMIT"); y -= 0.026
        head(fig, y, "A limit you can derive on paper"); y -= 0.038
        y = body(fig, y,
            "Phase-based estimation reads displacement from the phase shift of a band-pass filter. "
            "The Fourier shift theorem makes that shift linear in the displacement:", lh=0.0178)
        y -= 0.006
        fig.add_artist(plt.Rectangle((0.09, y - 0.050), 0.82, 0.054, facecolor="#f2f5f8",
                       edgecolor=LINE, transform=fig.transFigure))
        fig.add_artist(plt.Line2D([0.09, 0.09], [y - 0.050, y + 0.004], color=ACCENT, lw=3,
                       transform=fig.transFigure))
        fig.text(0.12, y - 0.023, r"$\Delta\varphi = 2\pi\,k\,\delta$", fontsize=14,
                 family=SANS, va="center")
        fig.text(0.34, y - 0.023, r"$\longrightarrow\quad \delta_{\max}=\lambda/2$",
                 fontsize=14, color=SIGNAL, va="center")
        fig.text(0.59, y - 0.023, "phase is unique only in (-pi, pi];\npast half a wavelength "
                 "it wraps", fontsize=8.5, color=MUTE, family=SERIF, va="center")
        y -= 0.070
        y = body(fig, y,
            "Beyond half a filter wavelength the phase aliases and the answer is silently wrong "
            "[1][3]. A single-scale method therefore has a hard ceiling — and it is small. But it "
            "is not fundamental: a multi-scale pyramid pushes it away.", lh=0.0178)
        y -= 0.012
        eyebrow(fig, y, "03 — THREE IMPLEMENTATIONS"); y -= 0.026
        head(fig, y, "From a hard ceiling to none"); y -= 0.036
        y = bullets(fig, y, [
            "Fleet (single-scale). One octave band, measured local frequency, singularity masking. "
            "Bounded to delta < lambda/2 — our reference for the limit.",
            "Reimpl (no paper). Multi-scale Gaussian pyramid, per-pixel translation, rewarping — "
            "built before reading the source paper.",
            "M-PME (faithful). The 2026 method of Li et al. [7]: a Gaussian pyramid with an AFFINE "
            "6-parameter motion model fit over a window (Farnebaeck-style), confidence weighting, "
            "four Gabor directions, coarse-to-fine. Scale and rotation ARE affine — it models them "
            "exactly.",
        ])
        y -= 0.008
        y = lead(fig, y, "How the pyramid beats the wall.",
            "Downsampling halves the motion. At a coarse enough level even a large displacement "
            "drops under lambda/2 and can be read without wrapping; warping the finer level by that "
            "estimate leaves only a small residual, read the same way, and so on down to full "
            "resolution. The wall never disappears — each level simply stays below it. This is the "
            "whole trick, and it is why 'formal methods only do small motion' has not been true for "
            "years.")
        y = body(fig, y,
            "The next two sections put the three head to head against pixel-exact ground truth: "
            "first on pure affine motion (a scaling), then on the non-affine motion of a real 3D "
            "scene. The gap between those two results is the subject of this note.", lh=0.0178)
        pdf.savefig(fig); plt.close(fig)

        # ================================================================ affine result
        fig = page()
        eyebrow(fig, 0.95, "04 — AFFINE MOTION IS SOLVED")
        head(fig, 0.925, "Sub-pixel on large scaling"); y = 0.885
        y = body(fig, y,
            "We warp a broadband texture by an exactly known scaling — a fronto-parallel plane "
            "under a Z-dolly is precisely a scaling — and sweep the magnitude, measuring the median "
            "endpoint error (EPE) against the pixel-exact ground-truth flow.", lh=0.0178)
        yb = image(fig, EPE, 0.14, 0.79, 0.72)
        caption(fig, yb - 0.008,
                "Fig. 1  Median EPE vs. displacement (log scale). Single-scale fails past ~11 px; "
                "both multi-scale methods stay sub-pixel; the affine M-PME is cleanest.")
        y = yb - 0.05
        y = table(fig, y, [
            ("max displacement", "Fleet", "Reimpl", "M-PME"),
            ("4 px", "0.41", "0.11", "0.03"),
            ("11 px", "1.91", "0.20", "0.12"),
            ("16 px", "8.93", "0.27", "0.25"),
            ("25 px", "14.11", "0.93", "0.66"),
            ("29 px", "16.82", "1.57", "1.14"),
        ], col_x=(0.15, 0.45, 0.62, 0.78))
        fig.text(0.15, y + 0.004, "Table 1  median EPE (px) — lower is better.", fontsize=8,
                 family=MONO, color=MUTE, va="top")
        y -= 0.03
        y = body(fig, y,
            "The intuition that 'formal methods only do small motion' is false. A multi-scale, "
            "affine, coarse-to-fine method solves large scaling and rotation to sub-pixel accuracy "
            "— because scale and rotation are exactly affine, and the affine model represents them "
            "with no residual. Single-scale, by contrast, degrades linearly with displacement, "
            "exactly as the lambda/2 wall predicts. The ceiling is not motion size. It is the "
            "affine assumption — and real scenes break it.", lh=0.0178)
        pdf.savefig(fig); plt.close(fig)

        # ================================================================ non-affine wall
        fig = page()
        eyebrow(fig, 0.95, "05 — THE NON-AFFINE WALL")
        head(fig, 0.925, "Real depth is not affine — and it shows"); y = 0.885
        y = body(fig, y,
            "A real scene is not a single plane. Move the camera along Z over a near object in "
            "front of a far background, and the optical flow is depth-dependent, with a "
            "DISCONTINUITY at the object's edge. That violates the affine-per-window assumption "
            "every method here relies on. Ground truth — flow AND depth — is exact, built from the "
            "depth map and the known camera step; from each method's recovered flow we invert the "
            "same geometry back to a depth map.", lh=0.0178)
        yb = image(fig, NONAFF, 0.07, 0.785, 0.86)
        caption(fig, yb - 0.008,
                "Fig. 2  Perspective Z-dolly over a two-plane scene. Top: recovered flow. Bottom: "
                "depth reconstructed from that flow. GT depth is a crisp step; Fleet returns noise, "
                "the reimplementation frays, the affine M-PME blurs the boundary away.")
        y = yb - 0.05
        y = bullets(fig, y, [
            "On the smooth, locally-affine plane regions all methods do fine — median EPE "
            "0.13-0.70 px.",
            "At the depth edge they break: EPE jumps 4-9x. The affine M-PME is WORSE at the edge "
            "(1.17 px) than the simpler reimplementation (0.52 px) — its large window averages "
            "across the discontinuity, the very place 3D structure lives.",
            "The reconstructed depth is unusable: noise (Fleet), ragged edges (reimpl), or a "
            "blurred step where 3D needs a crisp one (M-PME).",
        ])
        y -= 0.006
        y = body(fig, y,
            "This is the failure that matters. A depth map with soft, wandering boundaries yields a "
            "3D model with soft, wandering surfaces — walls that bleed into floors, objects fused "
            "with their background. For a robot deciding where the table ends, or a vehicle deciding "
            "where the curb is, a blurred boundary is not a small error: it is the wrong answer at "
            "the one place where accuracy is non-negotiable.", lh=0.0178)
        pdf.savefig(fig); plt.close(fig)

        # ================================================================ gap + outlook + refs
        fig = page()
        eyebrow(fig, 0.95, "06 — THE GAP")
        head(fig, 0.925, "Sharp on affine. Blind at depth."); y = 0.885
        fig.add_artist(plt.Line2D([0.09, 0.09], [0.80, 0.878], color=ACCENT, lw=3,
                       transform=fig.transFigure))
        fig.text(0.12, 0.86, "Every method here is sharp on affine motion and blind\n"
                 "at the depth boundary. For 3D reconstruction, that\n"
                 "boundary is the whole point.",
                 fontsize=13.5, family=SERIF, weight="bold", color=INK, va="top")
        y = 0.775
        y = body(fig, y,
            "The two halves of this benchmark bracket the state of formal motion estimation. On "
            "affine motion — translation, rotation, scale — it is genuinely solved, to sub-pixel "
            "accuracy, at large magnitude. On the non-affine, per-pixel, discontinuous motion of a "
            "real 3D scene, it produces flow that looks plausible in aggregate but destroys exactly "
            "the depth edges reconstruction depends on.", lh=0.0178)
        y = body(fig, y,
            "That is the gap. The probabilistic models that handle real scenes cannot be certified; "
            "the formal methods that can be certified cannot yet resolve depth. Neither is usable "
            "for safety-critical 3D as it stands.", lh=0.0178)
        y -= 0.006
        fig.text(0.09, y, "Outlook", fontsize=12, weight="bold", family=SERIF); y -= 0.028
        y = body(fig, y,
            "We are building a formal, non-probabilistic method designed to stay sharp exactly "
            "where these blur — crisp at the depth boundary, accurate enough per pixel to drive 3D "
            "reconstruction. This note is the baseline it will be measured against.",
            color=ACCENT, lh=0.0178)

        y -= 0.016
        fig.add_artist(plt.Rectangle((0.09, y - 0.072), 0.82, 0.074, facecolor="#eef4fa",
                       edgecolor=LINE, transform=fig.transFigure))
        fig.text(0.105, y - 0.012, "METHOD NOTE", fontsize=7.5, family=MONO, color=ACCENT,
                 weight="bold", va="top")
        body(fig, y - 0.03,
            "Controlled synthetic scenes with pixel-exact ground truth (broadband texture; scaling "
            "and two-plane Z-dolly). EPE is the median over a central crop, excluding the "
            "focus-of-expansion core. M-PME is our faithful reimplementation of the described "
            "method. All figures reproducible from source.",
            size=8.5, x=0.105, width=96, color="#243542", lh=0.0155)

        y -= 0.10
        rule(fig, y); y -= 0.022
        fig.text(0.09, y, "REFERENCES", fontsize=8, family=MONO, color=MUTE, weight="bold")
        y -= 0.026
        refs = [
            "D. J. Fleet & A. D. Jepson. Computation of component image velocity from local phase "
            "information. Int. J. Computer Vision 5(1), 1990.",
            "A. D. Jepson & D. J. Fleet. Phase singularities in scale-space. Image and Vision "
            "Computing 9(5), 1991.",
            "D. J. Fleet & A. D. Jepson. Stability of phase information. IEEE Trans. PAMI 15(12), 1993.",
            "J. L. Barron, D. J. Fleet & S. S. Beauchemin. Performance of optical flow techniques. "
            "Int. J. Computer Vision 12(1), 1994.",
            "T. Gautama & M. M. Van Hulle. A phase-based approach to the estimation of the optical "
            "flow field. IEEE Trans. Neural Networks 13(5), 2002.",
            "N. Wadhwa, M. Rubinstein, F. Durand & W. T. Freeman. Phase-based video motion "
            "processing. ACM Trans. Graphics (SIGGRAPH), 2013.",
            "M. Z. Li, Z. T. Yan, G. Liu & Z. Mao. Large amplitude motion estimation via multi-scale "
            "phase-based video processing. Mech. Syst. Signal Process. 253 (2026) 114301.",
            "Robbyant (Ant Group). LingBot-Map: streaming 3D reconstruction from monocular RGB "
            "video. 2026. arXiv:2604.14141; github.com/Robbyant/lingbot-map.",
        ]
        for i, r in enumerate(refs, 1):
            fig.text(0.09, y, f"[{i}]", fontsize=7.5, family=MONO, color=ACCENT, va="top")
            for ln in textwrap.wrap(r, width=108):
                fig.text(0.12, y, ln, fontsize=7.5, family=MONO, color=MUTE, va="top")
                y -= 0.0135
            y -= 0.003
        pdf.savefig(fig); plt.close(fig)

    print("wrote", OUT)


if __name__ == "__main__":
    build()
