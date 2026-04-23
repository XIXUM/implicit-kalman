# ComplexPyramidFlow2D — Algorithm Reference

This document describes the internal mechanics of `src/ComplexPyramidFlow2D.py`, the main motion estimation algorithm.

---

## Core principle: phase shift encodes spatial shift

A spatial translation of `d` pixels in an image manifests as a phase shift in its 2D FFT representation:

```
Δφ = 2π · d · f
```

where `f` is the spatial frequency of the band. The algorithm inverts this: it measures Δφ per frequency band and converts it back to `d` in pixel units.

---

## Filter bank design

The image is decomposed using two separable filter families applied in the frequency domain (after `fftshift`).

### Angular filter (`anglularFilt`)

Each filter is a cosine-squared ramp centered at a different angular offset:

```python
rRamp = np.cos(np.clip((angularMatr + aStep) * aPr, -π/2, π/2)) ** 2
```

- `angularMatr` is the polar-angle coordinate of each frequency-domain pixel (computed from the FFT-shifted origin at the array center).
- `aPr = 8` sets the angular bandwidth: each filter covers a ±π/16 lobe.
- `faPr = 32` filters are created, evenly tiling the full 2π circle.

**Known issue:** the ramp is not modulo-continuous at 0°/360°. The wraparound creates a discontinuity in that direction instead of a smooth cosine rolloff.

### Radial filter (`radialFilt`)

Each filter is a cosine-squared ramp centered at a different radial offset:

```python
np.cos(np.clip(np.pi * (2 * radialMatr / step - i), -π/2, π/2)) ** 2
```

- `radialMatr` is the distance from the FFT center.
- Each successive filter covers a higher spatial-frequency annulus, stepping by `step = halfX / rO` in radius.

### Combined bandpass filter

```python
pyrFilt = radialFilt[i] * anglularFilt[k]
```

The product of one radial and one angular filter creates a **directional bandpass** — a conic weight map in frequency space that selects energy at a specific scale and orientation. This is the equivalent of one sub-band in a complex steerable pyramid.

---

## Coarse-to-fine estimation pipeline

`pyramidFlow` iterates from the lowest-frequency radial band (`i=0`) to the highest (`i=rO-1`).

### Octave 0 — baseline phase capture

```python
iwC.append(np.angle(iwB * iwA.conj() / np.abs(iwA * iwB)))
```

- `iwA`, `iwB` are the spatial-domain complex signals after inverse FFT through the combined filter.
- `iwB * iwA.conj()` is the **conjugate product** — its angle is the phase difference Δφ between B and A in this band.
- Dividing by the magnitude normalises to the unit circle, making the result purely a phase angle map.
- `iwC[j]` stores the accumulated phase-displacement estimate per angular direction `j`.

### Octave i>0 — residual phase measurement

At each finer scale:

1. **Warp A** by the UV displacement accumulated from coarser bands:

   ```python
   iwAs = shiftMatrixInt(iwA, iwV, iwU, mx, my)
   ```

   This pre-compensates the already-estimated motion so that the finer band only sees the **residual** displacement it couldn't resolve at coarser scales.

2. **Measure residual phase**:

   ```python
   oiwC = iwB * iwAs.conj()
   ```

3. **Unwrap and scale**:

   ```python
   offsC = gradientfix(oiwCu, 1 / scaleFt, ss, cc, mx, my)
   ```

   `gradientfix` attempts to unwrap the phase field (see below) and scale the result by `1/scaleFt` to convert radians into pixel displacement.

4. **Correct for drift** between the current estimate and the coarse running total, then accumulate:

   ```python
   shift = np.mean(iwC[j]) - np.mean(offsC)
   iwC[j] += offsC + shift
   ```

### UV accumulation

After all angular directions at one octave are processed, the per-direction phase accumulators are projected onto the U and V axes using the known `sin`/`cos` of each direction's angle, then averaged across directions. The final UV map is scaled to pixel units:

```python
vuMap[0] = iwV * hh   # V in pixels
vuMap[1] = iwU * ww   # U in pixels
```

---

## Phase unwrapping: `gradientfix`

Each pyramid band only has a phase range of ±π. A displacement larger than half a spatial-frequency cycle wraps around (aliasing within the band). `gradientfix` addresses this:

1. If the phase field's total range is below π·0.95, no wrapping occurred — return `dD * scale` directly.
2. Otherwise, compute the spatial **gradient** of the phase field, then reconstruct the phase by **cumulative summation** along the filter direction (`linSum`).
3. Gradient steps larger than `π` are phase wraps — they are suppressed before cumsum.
4. To disambiguate near-π steps, three rotated versions of the phase are examined in parallel (using `otDeg = exp(i·2π/3)`, a 120° rotation). The gradient with the smallest magnitude is taken as the true gradient, reducing false wrap detections.

**Why it fails at fine scales:** at high spatial frequencies, a single pixel displacement produces a phase shift much larger than π, so the entire field is aliased. `gradientfix` can only suppress individual wrap steps, not recover a field that wraps multiple times per pixel.

---

## Why wavelets help — and why they are not enough on their own

The Gabor filter approach (inverse FFT of a bandpass-filtered spectrum) IS already a local, spatially-aware measurement — it is mathematically equivalent to a complex wavelet transform. The local phase of `iwA` at each pixel is the local phase of the filtered texture at that spatial location. So the lack of spatial localization is NOT the fundamental problem.

The fundamental problem for scaling is different from translation:

| Motion type | Effect on spectrum |
|-------------|-------------------|
| Translation | Phase shifts; magnitude unchanged; energy stays in same bands |
| Scaling | Phase shifts AND energy redistributes to different frequency bands |

When the algorithm warps only the Gabor **filter response** (`shiftMatrixInt`), it cancels the translational phase shift within each band but leaves the energy redistribution untouched. At fine scales, the energy in A and B no longer overlaps in the same band, so the conjugate product no longer measures a meaningful phase difference.

## The image-level warp fix

The solution is to warp the **original image A** (in pixel space) by the current coarse UV estimate, then recompute the FFT of the warped image. Because the warp operates on the full image before frequency decomposition, it cancels both:
1. The spatial shift (same as shiftMatrixInt)
2. The frequency redistribution (because the warped image's spectrum has its energy at the correct bands for comparison with B)

```python
# At each octave > 0:
a_warped = ndimage.map_coordinates(img_a, [my - disp_y, mx - disp_x], order=1)
ffA_work = fft.fftshift(fft.fft2(a_warped))
# Then extract fine-scale response from ffA_work instead of ffA
```

This turns the coarse-to-fine into a genuine iterative refinement: each octave refines the UV estimate, warps the image to cancel the refined motion, and the next octave only sees the residual.

---

## `otDeg` — 120° probe constant

```python
otDeg = np.array(-0.5 + 1j * np.sqrt(0.75))
```

This is `exp(i·2π/3)` — rotation by exactly 120°. Used in `gradientfix` and `angleGradient` to produce two phase-rotated copies of the signal at +120° (`sotDeg`) and −120° (`sotDeg.conj()`). Together with the unrotated original, these three cover the full circle at equal spacing, so at least one is always far from any ±π wraparound.

---

## Debug visualisation (`__main__`)

The test harness at the bottom of the file builds a parallel, more-instrumented version of the algorithm and produces three figure windows:

| Figure | Content |
|--------|---------|
| `fig2` (2×5) | FFT magnitude plots, cross-spectrum, radial filter shapes, UV vector map |
| `fig1` (3×5) | Per-octave 1D slices along the test direction — phase evolution of `iwA`, `iwB`, `iwA2`, `iwD`, `sD` |
| `fig`  (4×7) | Per-octave 2D phase maps (`sD[i]`) and side-by-side comparison of warped A vs B |

The `sect` variable selects which angular sector is used as the test direction. `rr, cl = centerLine(128, 128, angle, shape)` generates pixel indices along the selected direction through the array centre for 1D plotting.

---

## Key variables quick reference

| Variable | Meaning |
|----------|---------|
| `ffA`, `ffB` | 2D FFT (fftshifted) of input images A and B |
| `anglularFilt[k]` | Angular bandpass weight map for direction k |
| `radialFilt[i]` | Radial bandpass weight map for octave i |
| `pyrFilt` | `radialFilt[i] * anglularFilt[k]` — combined directional bandpass |
| `iwA`, `iwB` | Inverse-FFT of filtered spectra — complex spatial signal in this band |
| `iwAs` | `iwA` warped by current UV estimate — for residual measurement |
| `iwC[j]` | Accumulated phase displacement per angular direction j |
| `iwV`, `iwU` | V and U displacement maps (in phase-radians before final scaling) |
| `scaleFt` | Phase-to-pixel scale factor for octave i |
| `otDeg` | `exp(i·2π/3)` — 120° complex rotation for phase-wrap disambiguation |
| `sD[i]` | Accumulated phase angle map at octave i (debug) |
| `aiwD` | Running product of per-octave phase corrections (debug test harness) |
