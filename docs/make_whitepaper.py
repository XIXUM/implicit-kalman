"""Render the benchmark whitepaper 'The Affine Ceiling' as a PDF.

Reproducible: regenerates the three-way ground-truth benchmark figures
(src/benchmark_compare.py), then composes
docs/the_affine_ceiling_whitepaper.pdf.

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
HERO = os.path.join(SRC, "benchmark_cover_hero.png")
EPE = os.path.join(SRC, "benchmark_epe_3methods.png")
VEC = os.path.join(SRC, "benchmark_vectorfields.png")

A4 = (8.27, 11.69)
INK = "#10202f"; ACCENT = "#12507e"; SIGNAL = "#c0392b"; MUTE = "#5b6a78"; LINE = "#d4dbe2"
SERIF = "DejaVu Serif"; SANS = "DejaVu Sans"; MONO = "DejaVu Sans Mono"
plt.rcParams.update({"text.color": INK})


def regenerate():
    env = dict(os.environ, MPLBACKEND="Agg")
    print("running benchmark_compare.py ...")
    subprocess.run([sys.executable, os.path.join(SRC, "benchmark_compare.py")],
                   cwd=SRC, env=env, check=True)


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
    h_over_w = img.shape[0] / img.shape[1]
    page_ratio = A4[0] / A4[1]                 # width/height of the page in inches
    h = w * h_over_w * page_ratio              # figure-fraction height preserving aspect
    ax = fig.add_axes([x, y - h, w, h]); ax.imshow(img); ax.axis("off")
    return y - h


def caption(fig, y, text, width=104):
    for ln in textwrap.wrap(text, width=width):
        fig.text(0.09, y, ln, fontsize=8, family=MONO, color=MUTE, va="top")
        y -= 0.0135
    return y


def build():
    if "--no-regen" not in sys.argv:
        regenerate()

    with PdfPages(OUT) as pdf:
        # ---------------------------------------------------------- cover
        fig = page()
        fig.add_artist(plt.Rectangle((0, 0.80), 1, 0.20, color=ACCENT, transform=fig.transFigure))
        fig.text(0.09, 0.955, "BENCHMARK WHITEPAPER · DETERMINISTIC PERCEPTION",
                 fontsize=9, color="#bcd8f0", family=MONO, weight="bold")
        fig.text(0.09, 0.90, "The Affine Ceiling", fontsize=33, color="white",
                 family=SERIF, weight="bold")
        fig.text(0.09, 0.845, "What deterministic motion estimation solves — and what it doesn't.",
                 fontsize=12.5, color="#e6f0fa", family=SERIF)

        # eye-catcher hero, high on the page
        yb = image(fig, HERO, 0.09, 0.78, 0.82)
        caption(fig, yb - 0.006,
                "A 22% scaling. Left: ground-truth radial flow. Middle: single-scale phase "
                "flow collapses. Right: affine multi-scale (M-PME) recovers it to 0.55 px "
                "median error.")

        y = yb - 0.06
        y = body(fig, y,
            "Phase-based motion estimation is deterministic and certifiable — the kind of "
            "algorithm you can ship into a safety-critical system. The open question is not "
            "whether it is trustworthy, but how far it reaches. We measured it against "
            "pixel-exact ground truth, across three implementations of increasing fidelity.",
            size=11.5, lh=0.0205, width=80)
        y -= 0.015
        rule(fig, y); y -= 0.028
        y = body(fig, y,
            "The finding overturns the intuitive story. A single-scale estimate is capped at "
            "half a wavelength and fails on large motion. But a multi-scale, affine, "
            "coarse-to-fine method — the 2026 state of the art — solves large scaling and "
            "rotation to SUB-PIXEL accuracy. The real ceiling is not motion size. It is the "
            "affine assumption itself.", size=11, lh=0.02, width=82)

        fig.text(0.09, 0.075, "ImplicitKalman — working note   ·   Generated "
                 + datetime.date.today().isoformat()
                 + "   ·   reproducible from src/benchmark_compare.py",
                 fontsize=8, family=MONO, color=MUTE)
        pdf.savefig(fig); plt.close(fig)

        # ---------------------------------------------------------- page 2
        fig = page()
        eyebrow(fig, 0.95, "01 — THE CONSTRAINT")
        head(fig, 0.925, "Why determinism is the real constraint"); y = 0.885
        y = body(fig, y,
            "Vehicles, surgical robots and aircraft live under functional-safety standards — "
            "ISO 26262, DO-178C, IEC 62304 — that demand a provable worst case. A learned "
            "depth or optical-flow network offers neither a bound nor a repeatable output: the "
            "same scene in different light returns different numbers. It is probabilistic by "
            "construction — acceptable for a photo app, disqualifying for a braking decision.")
        y = body(fig, y,
            "Classical signal processing has the opposite profile: every output is reproducible "
            "and analyzable. The only question that matters is how accurate it is, and over what "
            "range of motion.")
        y -= 0.02
        eyebrow(fig, y, "02 — THE SINGLE-SCALE LIMIT"); y -= 0.026
        head(fig, y, "A limit you can derive on paper"); y -= 0.04
        y = body(fig, y,
            "Phase-based estimation reads displacement from the phase shift of a band-pass "
            "filter. The Fourier shift theorem makes that shift linear in the displacement:")
        y -= 0.006
        fig.add_artist(plt.Rectangle((0.09, y - 0.052), 0.82, 0.056, facecolor="#f2f5f8",
                       edgecolor=LINE, transform=fig.transFigure))
        fig.add_artist(plt.Line2D([0.09, 0.09], [y - 0.052, y + 0.004], color=ACCENT, lw=3,
                       transform=fig.transFigure))
        fig.text(0.12, y - 0.024, r"$\Delta\varphi = 2\pi\,k\,\delta$", fontsize=15,
                 family=SANS, va="center")
        fig.text(0.34, y - 0.024, r"$\longrightarrow\quad \delta_{\max}=\lambda/2$",
                 fontsize=15, color=SIGNAL, va="center")
        fig.text(0.60, y - 0.024, "phase is unique only in (-pi, pi];\npast half a wavelength "
                 "it wraps", fontsize=8.5, color=MUTE, family=SERIF, va="center")
        y -= 0.075
        y = body(fig, y,
            "Beyond half a filter wavelength the phase aliases and the answer is silently "
            "wrong. Fleet & Jepson characterized this in the early 1990s, down to which "
            "measurements to trust and which to reject [1][2]. A SINGLE-scale method therefore "
            "has a hard ceiling — and it is small. That ceiling is real. The question is "
            "whether it is fundamental. It is not.")
        y -= 0.015
        eyebrow(fig, y, "03 — THREE IMPLEMENTATIONS"); y -= 0.026
        head(fig, y, "From a hard ceiling to none"); y -= 0.04
        y = bullets(fig, y, [
            "Fleet (single-scale). One octave band, measured local frequency, singularity "
            "masking. Bounded to delta < lambda/2 — our reference for the limit.",
            "Reimpl (no paper). A multi-scale Gaussian pyramid with per-pixel translation and "
            "rewarping, built before reading the source paper. Extends the range, but leaves "
            "boundary and periphery outliers.",
            "M-PME (faithful). The 2026 method of Li et al. [5]: a Gaussian pyramid with an "
            "AFFINE 6-parameter motion model fit over a window (Farnebaeck-style), confidence "
            "weighting, four Gabor directions, coarse-to-fine superposition. Scale and rotation "
            "ARE affine — so this models them exactly.",
        ])
        pdf.savefig(fig); plt.close(fig)

        # ---------------------------------------------------------- page 3 (measurement)
        fig = page()
        eyebrow(fig, 0.95, "04 — THE MEASUREMENT")
        head(fig, 0.925, "Endpoint error against pixel-exact ground truth"); y = 0.885
        y = body(fig, y,
            "We warp a broadband texture by an exactly known scaling, giving a pixel-exact "
            "ground-truth flow, then sweep the scale and measure median endpoint error (EPE).",
            lh=0.018)
        yb = image(fig, EPE, 0.13, 0.83, 0.74)
        caption(fig, yb - 0.008,
                "Fig. 1  Median EPE vs. displacement (log scale). Single-scale fails past "
                "~11 px; both multi-scale methods stay sub-pixel; the affine M-PME is cleanest.")

        yb2 = yb - 0.05
        y = body(fig, yb2, "", lh=0.0)
        # results table
        rows = [("max displacement", "Fleet", "Reimpl", "M-PME"),
                ("7 px", "0.86", "0.14", "0.08"),
                ("16 px", "8.93", "0.27", "0.25"),
                ("25 px", "14.11", "0.93", "0.66")]
        ty = yb2
        for r, row in enumerate(rows):
            if r == 0:
                fig.add_artist(plt.Rectangle((0.13, ty - 0.004), 0.74, 0.020, color=ACCENT,
                               transform=fig.transFigure, zorder=0))
            for cx, cell in zip((0.14, 0.44, 0.60, 0.74), row):
                fig.text(cx, ty + 0.006, cell, fontsize=9.5, family=MONO,
                         color="white" if r == 0 else INK,
                         weight="bold" if r == 0 else "normal", va="center", zorder=1)
            ty -= 0.023
        fig.text(0.14, ty + 0.004, "median EPE in px — lower is better", fontsize=8,
                 family=MONO, color=MUTE, va="top")
        pdf.savefig(fig); plt.close(fig)

        # ---------------------------------------------------------- page 4 (vector fields)
        fig = page()
        eyebrow(fig, 0.95, "04 — THE MEASUREMENT (CONT.)")
        head(fig, 0.925, "The recovered fields, side by side"); y = 0.885
        y = body(fig, y,
            "Ground truth against all three methods, at increasing scale (top to bottom). "
            "Single-scale disintegrates; the per-pixel reimplementation is close in the median "
            "but throws visible outlier vectors; the affine M-PME tracks the radial field "
            "cleanly.", lh=0.018)
        yb = image(fig, VEC, 0.06, 0.80, 0.88)
        caption(fig, yb - 0.008,
                "Fig. 2  Displacement fields. Columns: ground truth, Fleet single-scale, "
                "per-pixel reimplementation, affine M-PME. Rows: scale 1.06 / 1.15 / 1.25.")
        pdf.savefig(fig); plt.close(fig)

        # ---------------------------------------------------------- page 5 (the real ceiling)
        fig = page()
        eyebrow(fig, 0.95, "05 — THE REAL CEILING")
        head(fig, 0.925, "Affine is solved. Real 3D is not."); y = 0.885
        fig.add_artist(plt.Line2D([0.09, 0.09], [0.80, 0.878], color=ACCENT, lw=3,
                       transform=fig.transFigure))
        fig.text(0.12, 0.86, "Deterministic phase flow solves large affine motion —\n"
                 "translation, rotation, scale — to sub-pixel accuracy.\n"
                 "The ceiling is the affine assumption, not the motion size.",
                 fontsize=13.5, family=SERIF, weight="bold", color=INK, va="top")
        y = 0.775
        y = body(fig, y,
            "Credit where due: the multi-scale, affine, coarse-to-fine approach genuinely "
            "removes the small-motion ceiling. Li et al. track a rotating turbine blade through "
            "hundreds of pixels of motion at sub-pixel error, and our faithful reimplementation "
            "reproduces sub-pixel accuracy on large scaling.")
        y = body(fig, y,
            "But M-PME assumes the motion is AFFINE within each window. Their only tests are a "
            "single rigid object on a black background — the easiest possible affine case. Dense "
            "2D->3D reconstruction of a real scene is not affine: it has per-pixel depth "
            "parallax, occlusion boundaries where motion is discontinuous, and independently "
            "moving objects that share a window. That is where an affine-per-window model must "
            "break — and it is exactly what depth reconstruction depends on.")
        y = body(fig, y,
            "So the frontier moves. Not 'deterministic can only do small motion' — that is "
            "false. The open problem is deterministic, certifiable, PER-PIXEL, NON-AFFINE motion "
            "for real 3D scenes. That is what ImplicitKalman is for.", color=ACCENT)

        y -= 0.02
        fig.add_artist(plt.Rectangle((0.09, y - 0.088), 0.82, 0.09, facecolor="#eef4fa",
                       edgecolor=LINE, transform=fig.transFigure))
        fig.text(0.105, y - 0.012, "METHOD NOTE", fontsize=7.5, family=MONO, color=ACCENT,
                 weight="bold", va="top")
        body(fig, y - 0.03,
            "Single controlled family: synthetic broadband texture warped by an exact scaling "
            "(pixel-exact GT). EPE reported as the median over a central crop (robust to "
            "boundary-window artefacts), matching the high-confidence evaluation the paper uses. "
            "M-PME is our faithful reimplementation of the described method. All figures "
            "reproducible from source.",
            size=8.5, x=0.105, width=96, color="#243542", lh=0.0155)

        y -= 0.12
        rule(fig, y); y -= 0.022
        fig.text(0.09, y, "REFERENCES", fontsize=8, family=MONO, color=MUTE, weight="bold")
        y -= 0.028
        refs = [
            "D. J. Fleet & A. D. Jepson. Computation of component image velocity from local "
            "phase information. Int. J. Computer Vision 5(1), 1990.",
            "D. J. Fleet & A. D. Jepson. Stability of phase information. IEEE Trans. PAMI "
            "15(12), 1993.",
            "J. L. Barron, D. J. Fleet & S. S. Beauchemin. Performance of optical flow "
            "techniques. Int. J. Computer Vision 12(1), 1994.",
            "N. Wadhwa, M. Rubinstein, F. Durand & W. T. Freeman. Phase-based video motion "
            "processing. ACM Trans. Graphics (SIGGRAPH), 2013.",
            "M. Z. Li, Z. T. Yan, G. Liu & Z. Mao. Large amplitude motion estimation via "
            "multi-scale phase-based video processing. Mech. Syst. Signal Process. 253 "
            "(2026) 114301. doi:10.1016/j.ymssp.2026.114301.",
        ]
        for i, r in enumerate(refs, 1):
            lines = textwrap.wrap(r, width=104)
            fig.text(0.09, y, f"[{i}]", fontsize=8, family=MONO, color=ACCENT, va="top")
            for ln in lines:
                fig.text(0.115, y, ln, fontsize=8, family=MONO, color=MUTE, va="top")
                y -= 0.0145
            y -= 0.004
        pdf.savefig(fig); plt.close(fig)

    print("wrote", OUT)


if __name__ == "__main__":
    build()
