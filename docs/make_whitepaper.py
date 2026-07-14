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


def build():
    if "--no-regen" not in sys.argv:
        regenerate()

    with PdfPages(OUT) as pdf:
        # ---------------------------------------------------------- cover
        fig = page()
        fig.add_artist(plt.Rectangle((0, 0.82), 1, 0.18, color=ACCENT, transform=fig.transFigure))
        fig.text(0.09, 0.958, "BENCHMARK WHITEPAPER · MOTION ESTIMATION FOR 3D",
                 fontsize=9, color="#bcd8f0", family=MONO, weight="bold")
        fig.text(0.09, 0.905, "The Affine Ceiling", fontsize=33, color="white",
                 family=SERIF, weight="bold")
        fig.text(0.09, 0.855, "Formal motion estimation is sharp on affine motion — "
                 "and blind at depth.", fontsize=12.5, color="#e6f0fa", family=SERIF)

        yb = image(fig, HERO, 0.07, 0.80, 0.86)
        caption(fig, yb - 0.006,
                "Depth reconstructed from each method's optical flow under a perspective Z-dolly. "
                "Ground truth is crisp; every phase-based method — including the affine state of "
                "the art — blurs, frays, or destroys the depth boundary. That boundary is what 3D "
                "reconstruction is made of.")

        y = yb - 0.055
        y = body(fig, y,
            "Motion estimation is the front end of 3D reconstruction. This note measures how well "
            "formal, non-probabilistic phase-based methods actually recover it — against pixel-exact "
            "ground truth, across three implementations. The result splits cleanly in two: they "
            "solve affine motion to sub-pixel accuracy, and they cannot recover usable depth.",
            size=11.5, lh=0.0205, width=82)

        fig.text(0.09, 0.075, "ImplicitKalman — working note   ·   Generated "
                 + datetime.date.today().isoformat()
                 + "   ·   reproducible from src/benchmark_*.py",
                 fontsize=8, family=MONO, color=MUTE)
        pdf.savefig(fig); plt.close(fig)

        # ---------------------------------------------------------- page 2 (stakes + physics)
        fig = page()
        eyebrow(fig, 0.95, "01 — THE STAKES")
        head(fig, 0.925, "Two ways to estimate motion"); y = 0.885
        y = body(fig, y,
            "Learned monocular models — the neural depth and matching networks now driving the "
            "field — are remarkably capable. But they are probabilistic by construction: no error "
            "bound, no repeatable output, no provable worst case. The same scene in different light "
            "returns different numbers. For a photo app, fine. For a system under functional-safety "
            "standards — ISO 26262, DO-178C, IEC 62304 — it is disqualifying.")
        y = body(fig, y,
            "This work takes the opposite stance: formal, non-probabilistic estimation, derived "
            "from signal theory rather than trained from data. Every output is reproducible and "
            "analyzable. The open question is not whether it can be trusted — it is whether it is "
            "accurate enough. Specifically: accurate enough for 3D.")
        y -= 0.02
        eyebrow(fig, y, "02 — THE SINGLE-SCALE LIMIT"); y -= 0.026
        head(fig, y, "A limit you can derive on paper"); y -= 0.04
        y = body(fig, y,
            "Phase-based estimation reads displacement from the phase shift of a band-pass filter. "
            "The Fourier shift theorem makes that shift linear in the displacement:")
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
            "Beyond half a filter wavelength the phase aliases and the answer is silently wrong. "
            "Fleet & Jepson characterized this in the early 1990s [1][2]. A single-scale method "
            "therefore has a hard ceiling — and it is small. The question is whether that ceiling "
            "is fundamental. It is not: a multi-scale pyramid pushes it away, as we show next.")
        pdf.savefig(fig); plt.close(fig)

        # ---------------------------------------------------------- page 3 (methods + affine)
        fig = page()
        eyebrow(fig, 0.95, "03 — THREE IMPLEMENTATIONS")
        head(fig, 0.925, "From a hard ceiling to none"); y = 0.885
        y = bullets(fig, y, [
            "Fleet (single-scale). One octave band, measured local frequency, singularity masking. "
            "Bounded to delta < lambda/2 — our reference for the limit.",
            "Reimpl (no paper). Multi-scale Gaussian pyramid, per-pixel translation, rewarping. "
            "Built before reading the source paper.",
            "M-PME (faithful). The 2026 method of Li et al. [5]: a Gaussian pyramid with an AFFINE "
            "6-parameter motion model fit over a window (Farnebaeck-style), confidence weighting, "
            "four Gabor directions, coarse-to-fine. Scale and rotation ARE affine — it models them "
            "exactly.",
        ])
        y -= 0.005
        eyebrow(fig, y, "04 — AFFINE MOTION IS SOLVED"); y -= 0.026
        head(fig, y, "Sub-pixel on large scaling"); y -= 0.038
        y = body(fig, y,
            "We warp a broadband texture by an exactly known scaling (a fronto-parallel plane under "
            "a Z-dolly is precisely a scaling) and sweep the magnitude. Single-scale fails past "
            "~11 px; both multi-scale methods stay sub-pixel; the affine M-PME is cleanest.",
            lh=0.018)
        yb = image(fig, EPE, 0.15, 0.53, 0.70)
        caption(fig, yb - 0.008,
                "Fig. 1  Median EPE vs. displacement (log scale). Affine motion is solved to "
                "sub-pixel accuracy across the range.")
        y = yb - 0.05
        y = body(fig, y,
            "So the intuitive story — 'formal methods only do small motion' — is false. A "
            "multi-scale, affine, coarse-to-fine method solves large scaling and rotation. The "
            "ceiling is not motion size. It is the affine assumption itself — and real scenes break "
            "it.", color=INK)
        pdf.savefig(fig); plt.close(fig)

        # ---------------------------------------------------------- page 4 (non-affine, the crux)
        fig = page()
        eyebrow(fig, 0.95, "05 — THE NON-AFFINE WALL")
        head(fig, 0.925, "Real depth is not affine — and it shows"); y = 0.885
        y = body(fig, y,
            "A real scene is not a single plane. Move the camera along Z over a near object in "
            "front of a far background, and the optical flow is depth-dependent, with a "
            "DISCONTINUITY at the object's edge. That violates the affine-per-window assumption "
            "every method here relies on. Ground truth (flow AND depth) is exact, built from the "
            "depth map and the known camera step.", lh=0.018)
        yb = image(fig, NONAFF, 0.07, 0.79, 0.86)
        caption(fig, yb - 0.008,
                "Fig. 2  Perspective Z-dolly over a two-plane scene. Top: recovered flow. Bottom: "
                "depth reconstructed from that flow. GT depth is a crisp step; Fleet returns noise, "
                "the reimplementation frays, the affine M-PME blurs the boundary away.")
        y = yb - 0.055
        y = bullets(fig, y, [
            "On the smooth (locally-affine) plane regions all methods do fine — median EPE 0.13-0.70 px.",
            "At the depth edge they break: EPE jumps 4-9x. The affine M-PME is WORSE at the edge "
            "(1.17 px) than the simpler reimplementation (0.52 px) — its large window averages "
            "across the discontinuity.",
            "The reconstructed depth is unusable: noise, ragged edges, or a blurred step where 3D "
            "needs a crisp one.",
        ])
        pdf.savefig(fig); plt.close(fig)

        # ---------------------------------------------------------- page 5 (gap + outlook)
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
            "the depth edges that reconstruction depends on.")
        y = body(fig, y,
            "That is the gap. The probabilistic models that handle real scenes cannot be certified; "
            "the formal methods that can be certified cannot yet resolve depth. Neither is usable "
            "for safety-critical 3D as it stands.")
        y -= 0.006
        fig.text(0.09, y, "Outlook", fontsize=12, weight="bold", family=SERIF); y -= 0.03
        y = body(fig, y,
            "We are building a formal, non-probabilistic method designed to stay sharp exactly where "
            "these blur — crisp at the depth boundary, accurate enough per pixel to drive 3D "
            "reconstruction. This note is the baseline it will be measured against.", color=ACCENT)

        y -= 0.02
        fig.add_artist(plt.Rectangle((0.09, y - 0.076), 0.82, 0.078, facecolor="#eef4fa",
                       edgecolor=LINE, transform=fig.transFigure))
        fig.text(0.105, y - 0.012, "METHOD NOTE", fontsize=7.5, family=MONO, color=ACCENT,
                 weight="bold", va="top")
        body(fig, y - 0.03,
            "Controlled synthetic scenes with pixel-exact ground truth (broadband texture; scaling "
            "and two-plane Z-dolly). EPE is the median over a central crop, excluding the "
            "focus-of-expansion core. M-PME is our faithful reimplementation of the described "
            "method. All figures reproducible from source.",
            size=8.5, x=0.105, width=96, color="#243542", lh=0.0155)

        y -= 0.105
        rule(fig, y); y -= 0.022
        fig.text(0.09, y, "REFERENCES", fontsize=8, family=MONO, color=MUTE, weight="bold")
        y -= 0.028
        refs = [
            "D. J. Fleet & A. D. Jepson. Computation of component image velocity from local phase "
            "information. Int. J. Computer Vision 5(1), 1990.",
            "D. J. Fleet & A. D. Jepson. Stability of phase information. IEEE Trans. PAMI 15(12), 1993.",
            "J. L. Barron, D. J. Fleet & S. S. Beauchemin. Performance of optical flow techniques. "
            "Int. J. Computer Vision 12(1), 1994.",
            "N. Wadhwa, M. Rubinstein, F. Durand & W. T. Freeman. Phase-based video motion "
            "processing. ACM Trans. Graphics (SIGGRAPH), 2013.",
            "M. Z. Li, Z. T. Yan, G. Liu & Z. Mao. Large amplitude motion estimation via multi-scale "
            "phase-based video processing. Mech. Syst. Signal Process. 253 (2026) 114301.",
        ]
        for i, r in enumerate(refs, 1):
            fig.text(0.09, y, f"[{i}]", fontsize=8, family=MONO, color=ACCENT, va="top")
            for ln in textwrap.wrap(r, width=104):
                fig.text(0.115, y, ln, fontsize=8, family=MONO, color=MUTE, va="top")
                y -= 0.0145
            y -= 0.004
        pdf.savefig(fig); plt.close(fig)

    print("wrote", OUT)


if __name__ == "__main__":
    build()
