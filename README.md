# implicit-kalman — Featureless Motion Estimation

A deterministic, featureless motion estimation algorithm inspired by the MIT CSAIL paper on phase-based motion magnification.

**Reference paper:** [Phase-based Video Motion Processing (Wadhwa et al., SIGGRAPH 2013)](https://people.csail.mit.edu/nwadhwa/phase-video/phase-video.pdf)

---

## Motivation

### The problem with feature-based motion estimation

All mainstream motion estimation algorithms (optical flow, SLAM, visual odometry, etc.) are ultimately rooted in the same paradigm: find a visually distinct feature in the image — a corner, a blob, a high-contrast region — and track it across frames using a Kalman filter or similar estimator.

This approach has three fundamental weaknesses:

1. **Motion is only detectable where contrast exists.** Uniform regions produce no trackable features, so motion there is interpolated or ignored entirely.

2. **Features are observed in isolation.** Each tracked point knows nothing about its neighbours. There is no shared context, no understanding of the scene structure.

3. **Redundant patterns cause aliasing.** This is the critical failure case: imagine a checkerboard pattern. When the image shifts by approximately one checker-width between frames, a feature-based tracker can mistake a neighbouring square for the original one. The real feature has moved on; the algorithm thinks nothing happened. The denser and more regular the texture, the worse this failure becomes.

This third problem is not an edge case — it is fundamental to any algorithm that does not account for the spatial context around each feature.

---

## The approach

### Phase-based, coarse-to-fine motion estimation

Instead of tracking local features, we decompose the image into a **complex steerable pyramid** — a multi-scale, multi-orientation representation in the frequency domain using 2D FFT.

The key insight from phase-video theory is: **a spatial shift in an image manifests as a phase shift in its frequency-domain representation.** The phase shift is proportional to the displacement and the spatial frequency of the band.

We extend this beyond simple motion magnification (as in the original paper and implementations like [gonzaiq/pb-Motion-Magnification](https://github.com/gonzaiq/pb-Motion-Magnification)) to **quantitative motion estimation** — recovering an actual pixel-displacement vector field from the phase differences between two frames.

#### Coarse-to-fine contextualization

The pyramid is processed from the coarsest scale to the finest:

1. **Coarse level:** Estimate the large-scale, global motion from low-frequency components. At this level, redundant fine-grained textures are invisible — there is no aliasing.
2. **Each finer level:** Rather than re-estimating motion from scratch, we only measure the *residual* phase shift relative to the motion already predicted from coarser levels. This is the contextualization step — finer features are interpreted in the context of the coarser motion already understood.
3. **Pixel level:** The accumulated residuals across all pyramid levels yield the final sub-pixel displacement for each image location.

This approach is **fully deterministic** — no neural network, no probabilistic model, no risk of silent false positives. The output is a 2D array of 2D displacement vectors describing exactly how much each region of image A shifted to reach image B.

---

## Current status

### What works

- Complex steerable pyramid construction and reconstruction (`src/phase-video/ComplexSteerablePyramid.py`)
- Phase-difference extraction per pyramid band
- Coarse-to-fine vector map accumulation (`src/ComplexPyramidFlow2D.py`)
- A working vector map was produced for a **pure translation** (100% pixel shift without scaling)

### Current blocker

**Scaling (zoom) in the displacement field is not yet handled correctly.**

When the motion includes a scale component (e.g. a zooming camera or an object moving in depth), two things happen simultaneously:
- The phase of each frequency band shifts
- The *position* of energy in the frequency domain also shifts (higher/lower spatial frequencies gain or lose energy)

The current implementation handles phase shift but not this frequency-domain repositioning. As a result, scaling is partially recovered at coarse resolutions but degrades and diverges at finer scales. The core challenge is disentangling phase-shift-from-translation from phase-shift-from-scaling within each pyramid band.

---

## Repository structure

```
src/
  phase-video/
    ComplexSteerablePyramid.py   # Pyramid build/reconstruct (reference implementation)
  ComplexPyramidFlow2D.py        # Main motion estimation algorithm (work in progress)
```

---

## Related work

- [Phase-based Video Motion Processing — MIT CSAIL](https://people.csail.mit.edu/nwadhwa/phase-video/phase-video.pdf)
- [gonzaiq/pb-Motion-Magnification](https://github.com/gonzaiq/pb-Motion-Magnification) — Python port of motion magnification (amplification only, no metric displacement output)
