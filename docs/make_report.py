"""Generate the documented PDF benchmark report for the phase-flow tools.

Reproducible one-shot: regenerates both diagnostic PNGs by running the two flow
scripts (headless), then composes docs/benchmark_phase_based_flow.pdf.

Usage:
    python docs/make_report.py
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
OUT = os.path.join(HERE, "benchmark_phase_based_flow.pdf")

PHASEFLOW_PNG = os.path.join(SRC, "phaseflow_result.png")
MPME_PNG = os.path.join(SRC, "mpme_result.png")


def regenerate_pngs():
    """Re-run the two benchmark tools headless so the figures match current code."""
    env = dict(os.environ, MPLBACKEND="Agg")
    for script in ("PhaseFlowLocalFreq.py", "MultiScalePhaseFlow.py"):
        print(f"running {script} ...")
        subprocess.run([sys.executable, os.path.join(SRC, script)],
                       cwd=SRC, env=env, check=True)


A4 = (8.27, 11.69)
INK = "#1a1a2e"; ACCENT = "#0f3d6e"; MUTE = "#555"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})


def new_page():
    fig = plt.figure(figsize=A4)
    fig.patch.set_facecolor("white")
    return fig


def header(fig, kicker, title):
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.945, 0.945], color=ACCENT, lw=3,
                              transform=fig.transFigure))
    fig.text(0.08, 0.955, kicker, fontsize=9, color=ACCENT, weight="bold",
             va="bottom", transform=fig.transFigure)
    fig.text(0.08, 0.90, title, fontsize=17, weight="bold", va="top",
             transform=fig.transFigure)


def body(fig, y, text, size=9.5, x=0.08, width=96, color=INK, lh=0.0175, mono=False):
    fam = "DejaVu Sans Mono" if mono else "DejaVu Sans"
    for para in text.split("\n"):
        if para.strip() == "":
            y -= lh * 0.6; continue
        for ln in (textwrap.wrap(para, width=width) or [""]):
            fig.text(x, y, ln, fontsize=size, va="top", color=color, family=fam,
                     transform=fig.transFigure)
            y -= lh
    return y


def bullet(fig, y, items, size=9.5, x=0.08, width=90):
    for it in items:
        fig.text(x, y, "•", fontsize=size, va="top", color=ACCENT,
                 transform=fig.transFigure)
        for ln in textwrap.wrap(it, width=width):
            fig.text(x + 0.02, y, ln, fontsize=size, va="top",
                     transform=fig.transFigure)
            y -= 0.0175
        y -= 0.004
    return y


def figure_image(fig, path, y_top, y_bot, caption):
    img = mpimg.imread(path)
    ax = fig.add_axes([0.08, y_bot, 0.84, y_top - y_bot])
    ax.imshow(img); ax.axis("off")
    fig.text(0.08, y_bot - 0.012, caption, fontsize=8, style="italic", color=MUTE,
             va="top", transform=fig.transFigure)


def table(fig, y, rows, col_x=(0.08, 0.55, 0.78), size=9.5, head=True):
    for r, row in enumerate(rows):
        wt = "bold" if (head and r == 0) else "normal"
        bg = ACCENT if (head and r == 0) else None
        c = "white" if (head and r == 0) else INK
        if bg:
            fig.add_artist(plt.Rectangle((0.08, y - 0.004), 0.84, 0.020,
                           color=bg, transform=fig.transFigure, zorder=0))
        for cx, cell in zip(col_x, row):
            fig.text(cx + 0.005, y + 0.006, cell, fontsize=size, weight=wt, color=c,
                     va="center", transform=fig.transFigure, zorder=1)
        y -= 0.022
    return y


def build_pdf():
    with PdfPages(OUT) as pdf:
        # ------------------------------------------------------------ page 1 (cover)
        fig = new_page()
        fig.add_artist(plt.Rectangle((0, 0.72), 1, 0.28, color=ACCENT,
                                     transform=fig.transFigure))
        fig.text(0.08, 0.90, "ImplicitKalman — Working Note", fontsize=11,
                 color="#cfe3ff", weight="bold")
        fig.text(0.08, 0.845, "Phase-Based Motion Estimation:", fontsize=23,
                 color="white", weight="bold")
        fig.text(0.08, 0.80, "Benchmark of Two Reference Methods", fontsize=23,
                 color="white", weight="bold")
        fig.text(0.08, 0.745, "Single-scale Fleet–Jepson  vs.  multi-scale coarse-to-fine (M-PME)",
                 fontsize=11, color="#cfe3ff")

        y = 0.66
        y = body(fig, y,
            "Purpose. Establish baseline benchmarks for dense phase-based displacement "
            "estimation, to be compared against the ImplicitKalman approach. Two standalone "
            "tools are evaluated on the same input pair (relief0 -> relief0_sc1), both "
            "producing an identical six-panel diagnostic layout.", size=10.5, lh=0.019)

        y -= 0.025
        fig.text(0.08, y, "Key finding", fontsize=13, weight="bold"); y -= 0.035
        y = body(fig, y,
            "A single-scale phase estimate is fundamentally bounded to displacements below "
            "half a filter wavelength (the +-pi phase-wrap limit). It fails on the large / "
            "peripheral displacements produced by scaling. Decomposing the motion across a "
            "Gaussian pyramid (coarse-to-fine with rewarping) removes that ceiling: the "
            "wrap limit then applies only to the small per-level residual. On the test pair "
            "the multi-scale method reduces the warp reconstruction error by 97%.", lh=0.019)

        y -= 0.02
        y = table(fig, y, [
            ["Method", "Warp residual RMSE", "Verdict"],
            ["Single-scale (Fleet-Jepson)", "chaotic / aliased", "insufficient"],
            ["Multi-scale (M-PME)", "121.4 -> 4.11 px  (-97%)", "converges"],
        ])

        y -= 0.02
        y = body(fig, y,
            "Caveat for 2D->3D reconstruction: even the converging multi-scale field is too "
            "coarse for depth reconstruction. Phase information exists only at textured edges "
            "and is interpolated across flat regions, so the field is not sub-pixel smooth. "
            "This is the gap ImplicitKalman aims to close.", color=MUTE, lh=0.019)

        fig.text(0.08, 0.06, "Generated " + datetime.date.today().isoformat()
                 + "   •   src/PhaseFlowLocalFreq.py, src/MultiScalePhaseFlow.py",
                 fontsize=8, color=MUTE)
        pdf.savefig(fig); plt.close(fig)

        # ------------------------------------------------------------ page 2 (theory)
        fig = new_page()
        header(fig, "BACKGROUND", "Why phase encodes motion — and where it breaks")
        y = 0.86
        y = body(fig, y,
            "Fourier shift theorem. A local displacement delta shifts the phase of a "
            "complex band-pass (Gabor / steerable) response linearly:", lh=0.019)
        y -= 0.016
        fig.text(0.12, y, r"$\Delta\varphi \;=\; 2\pi\,k\,\delta$", fontsize=15, va="top")
        y -= 0.045
        y = body(fig, y,
            "where k is the local spatial frequency (cycles/px). Inverting gives the "
            "displacement from the measured phase difference:", lh=0.019)
        y -= 0.016
        fig.text(0.12, y, r"$\delta \;=\; -\,\Delta\varphi \,/\, k_{\mathrm{local}}$",
                 fontsize=15, va="top")
        y -= 0.045
        y = body(fig, y,
            "Crucial detail (Fleet & Jepson): k must be the MEASURED instantaneous frequency "
            "k = grad(phi), estimated via Im(grad R * conj(R) / |R|^2), not the filter's "
            "nominal centre frequency. Using the nominal value biases every octave differently.",
            lh=0.019)
        y -= 0.02
        fig.text(0.08, y, "The hard limit", fontsize=13, weight="bold"); y -= 0.04
        y = body(fig, y,
            "Phase is unique only in (-pi, pi]. Wrapping occurs once |Delta phi| > pi, i.e.",
            lh=0.019)
        y -= 0.016
        fig.text(0.12, y, r"$\delta_{\max} \;=\; \dfrac{1}{2k} \;=\; \dfrac{\lambda}{2}$"
                 r"  per sub-band", fontsize=15, va="top")
        y -= 0.05
        y = body(fig, y,
            "Scaling is the worst case: displacement grows with radius, delta(r) = (s-1) r. "
            "It is tiny at the centre (works) and large at the periphery (aliases) — exactly "
            "the mixed failure seen below.", lh=0.019)
        y -= 0.014
        fig.text(0.08, y, "Two stabilising ideas used in both tools", fontsize=13,
                 weight="bold"); y -= 0.034
        y = bullet(fig, y, [
            "Singularity rejection: where amplitude rho = |R| -> 0 the phase is undefined and "
            "its gradient diverges; such pixels are masked (Fleet & Jepson, Stability of Phase).",
            "Aperture problem: per-orientation component displacements are fused into a full 2D "
            "(U,V) vector by weighted least squares over all orientations.",
        ])
        y -= 0.008
        fig.text(0.08, y, "Resolution (multi-scale)", fontsize=13, weight="bold"); y -= 0.034
        y = body(fig, y,
            "A Gaussian pyramid decomposes a large motion into small per-level sub-motions: at "
            "level L the motion is downsampled by 2^L until it fits under lambda/2. The estimate "
            "is upsampled, used to rewarp the next finer level, and only the small residual is "
            "measured there (no unwrapping). Summing across levels reconstructs the full field.",
            lh=0.019)
        pdf.savefig(fig); plt.close(fig)

        # ------------------------------------------------------------ page 3 (method A)
        fig = new_page()
        header(fig, "METHOD A — src/PhaseFlowLocalFreq.py",
               "Single-scale Fleet–Jepson phase flow")
        y = 0.87
        y = body(fig, y,
            "One octave band per orientation, measured local frequency as denominator, "
            "amplitude + frequency-consistency masking, least-squares fusion over orientations, "
            "coarse-to-fine only within one FFT (no image pyramid). Confidence-weighted smoothing "
            "propagates edge measurements into flat regions for display.", lh=0.018)
        figure_image(fig, PHASEFLOW_PNG, 0.80, 0.40,
                     "Fig. A1  A, B (checkerboards); UV magnitude + vector field; UV centre "
                     "slices; measured local frequency; phase confidence.")
        y = 0.35
        fig.text(0.08, y, "Observations", fontsize=12, weight="bold"); y -= 0.028
        y = bullet(fig, y, [
            "Local-frequency map is clean (~1.0) and the confidence map correctly lights up "
            "the edges — the Fleet-Jepson core works.",
            "Auto-selected octaves 2-4 (checkerboard fundamental); higher bands carry no energy.",
            "Displacement max ~21 px, mean ~11 px; estimated divergence -> s ~ 1.16.",
            "UV slices are chaotic away from the centre: the peripheral displacement exceeds "
            "lambda/2 and aliases. This is the documented failure mode, not an implementation bug.",
        ])
        pdf.savefig(fig); plt.close(fig)

        # ------------------------------------------------------------ page 4 (method B)
        fig = new_page()
        header(fig, "METHOD B — src/MultiScalePhaseFlow.py",
               "Multi-scale coarse-to-fine (M-PME)")
        y = 0.87
        y = body(fig, y,
            "Gaussian pyramid (5 levels), coarse-to-fine with rewarping and per-level residual "
            "refinement, no phase unwrapping, motion-field fusion across scales. Reimplementation "
            "of the method described in 'Large amplitude motion estimation via multi-scale "
            "phase-based video processing' (Mech. Syst. Signal Process. 2026), not the "
            "original paywalled code.", lh=0.018)
        figure_image(fig, MPME_PNG, 0.80, 0.40,
                     "Fig. B1  A, B; UV magnitude + vector field; UV centre slices; divergence "
                     "(local scale); warp(A)->B residual.")
        y = 0.35
        fig.text(0.08, y, "Results", fontsize=12, weight="bold"); y -= 0.026
        y = table(fig, y, [
            ["Quantity", "Value", ""],
            ["Warp residual RMSE (fg)", "121.4 -> 4.11 px", "(-97%)"],
            ["mean|flow| per level", "0.55 / 1.10 / 2.19 / 4.38 / 8.76", "doubles"],
            ["max / mean displacement", "19.6 / 8.8 px", ""],
        ], col_x=(0.08, 0.45, 0.80), size=9)
        y -= 0.006
        y = bullet(fig, y, [
            "Converges: warp(A, flow) reconstructs B to ~4 px RMSE (thin edge residual only).",
            "mean|flow| doubles cleanly per level — the expected coarse-to-fine signature.",
            "Field is strongly vertical-dominant (V slice +-17 px, U slice +-1 px). Since the warp "
            "reconstructs B, this is real: relief0_sc1 is NOT an isotropic scale of relief0 but "
            "anisotropic (vertical scale/shift). A single scalar s~0.915 is therefore misleading.",
        ])
        pdf.savefig(fig); plt.close(fig)

        # ------------------------------------------------------------ page 5 (conclusion)
        fig = new_page()
        header(fig, "CONCLUSIONS", "Benchmark summary & next steps")
        y = 0.86
        y = table(fig, y, [
            ["Aspect", "Single-scale (A)", "Multi-scale (B)"],
            ["Large motion", "aliases (>lambda/2)", "handled"],
            ["Warp residual", "not usable", "121 -> 4 px"],
            ["UV slices", "chaotic", "approx. linear ramp"],
            ["Cost", "1 FFT set", "pyramid x refine"],
        ], col_x=(0.08, 0.42, 0.68))
        y -= 0.02
        fig.text(0.08, y, "Takeaways", fontsize=13, weight="bold"); y -= 0.032
        y = bullet(fig, y, [
            "The pyramid argument holds: coarse-to-fine contextualisation overcomes the lambda/2 "
            "ceiling, because the wrap limit only ever applies to the small per-level residual.",
            "Sign matters: from dphi = -k*delta, the displacement is -dphi/k; a wrong sign makes "
            "the warp diverge (residual grew to 175 px before the fix).",
            "The test data is anisotropic — clarify what relief0_sc1 actually is before using it "
            "as a scalar-scale ground truth.",
        ])
        y -= 0.01
        fig.text(0.08, y, "Why this is still not enough for 2D->3D", fontsize=13,
                 weight="bold"); y -= 0.032
        y = body(fig, y,
            "The 4 px residual is global reconstruction energy, not per-pixel depth accuracy. "
            "Phase flow is defined only on textured edges and interpolated elsewhere, so the "
            "field lacks the sub-pixel smoothness a depth reconstruction needs. Closing that gap "
            "is the motivation for ImplicitKalman.", lh=0.019)
        y -= 0.014
        fig.text(0.08, y, "References", fontsize=13, weight="bold"); y -= 0.03
        y = body(fig, y,
            "Fleet & Jepson 1990, Computation of component image velocity from local phase "
            "information, IJCV 5(1).\n"
            "Fleet & Jepson 1993, Stability of phase information, IEEE PAMI 15(12).\n"
            "Barron, Fleet & Beauchemin 1994, Performance of optical flow techniques, IJCV 12(1).\n"
            "Wadhwa et al. 2013, Phase-based video motion processing, ACM TOG (SIGGRAPH).\n"
            "Large amplitude motion estimation via multi-scale phase-based video processing, "
            "Mech. Syst. Signal Process. 2026 (S0888327026004589).",
            size=8.5, color=MUTE, lh=0.016)
        pdf.savefig(fig); plt.close(fig)

    print("wrote", OUT)


if __name__ == "__main__":
    if "--no-regen" not in sys.argv:
        regenerate_pngs()
    build_pdf()
