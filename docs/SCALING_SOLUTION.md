# Scaling in Phase-Based Motion Estimation — Problem & Solution

This document describes the specific problem that prevented `ComplexPyramidFlow2D` from correctly estimating motion when the input contains a scale component (zoom), and the fix that resolved it.

---

## The test case

Two 256×256 images:

- `src/relief0.png` — a checkerboard occupying pixels 37–217 on a yellow background
- `src/relief0_sc.png` — the same checkerboard scaled by ≈ 0.86 (now occupying pixels 50–205), centred on pixel 128

The expected UV displacement field is an **inverse pyramid**:
- Zero at the centre (128, 128)
- Vectors pointing radially inward
- Magnitude rising linearly with distance from centre
- Maximum magnitude ≈ 18 pixels at the corners

For a scaling by factor `s` centred at `c`:

```
d(x, y) = ((x - cx) · (s - 1), (y - cy) · (s - 1))
```

---

## The problem

### Symptom

The algorithm produced a correct vector field for pure translation, but for scaling the result was partially correct at coarse octaves and diverged completely at finer octaves.

### First, what wasn't the cause

A tempting hypothesis: "Fourier decomposition is global, not local — we need wavelets for local motion detection."

This is half-right and half-wrong:
- **Right** in general principle: global FFT phase corresponds to average displacement, not local.
- **Wrong for this code**: `iwA = ifft2(ifftshift(ffA * pyrFilt))` is mathematically equivalent to a **complex Gabor wavelet transform**. The resulting `iwA` is a spatially-localized complex field whose local phase at each pixel encodes the local position of the filtered pattern. So the code was already doing wavelet-style local analysis.

The real problem was elsewhere.

### Root cause: spectral energy redistribution under scaling

Translation and scaling affect the Fourier spectrum in fundamentally different ways:

| Motion | Phase | Magnitude / spectral position |
|--------|-------|------------------------------|
| Translation by d | shifts by `Δφ = 2π · f · d` | unchanged — energy stays at same frequencies |
| Scaling by factor s (centred) | shifts spatially | energy migrates: content at frequency `f` in A appears at frequency `f/s` in B |

The coarse-to-fine pipeline uses `shiftMatrixInt` to warp the **Gabor filter response** `iwA` by the UV estimate from coarser octaves. For pure translation this works perfectly: the warped `iwA` matches `iwB` in phase, residual ≈ 0, and the next octave refines normally.

For scaling it fails because:
1. `shiftMatrixInt` warps only the complex spatial samples of one band. It cannot move spectral energy between bands.
2. At octave i, the filter selects a narrow frequency annulus in A's spectrum and in B's spectrum.
3. In A, the checker's fundamental harmonic sits in one band. After scaling by 0.86, the same harmonic in B sits at a frequency 1/0.86 ≈ 1.16× higher — often **in a different band**.
4. Comparing `iwA` at frequency f with `iwB` at frequency f gives a phase difference contaminated by the frequency mismatch — it is no longer a clean motion signal.
5. The contamination grows with octave: narrower bands at fine scales mean less overlap between where A's energy is and where B's energy is. The conjugate product `iwB * iwA.conj()` drifts away from a valid displacement map.

The `gradientfix` phase-unwrapping then tries to interpret this contaminated phase as displacement, and the result diverges.

---

## The solution: warp the image, not just the filter response

Instead of warping the complex filter response between frequency bands, warp the **original image** in the pixel domain at each coarse-to-fine step, then recompute its FFT for the next octave.

```python
# At each octave i > 0:
disp_y = iwV * hh   # current UV estimate in pixels
disp_x = iwU * ww
a_warped = ndimage.map_coordinates(img_a, [my - disp_y, mx - disp_x],
                                    order=1, mode='nearest')
ffA_work = fft.fftshift(fft.fft2(a_warped))
# Then extract iwA from ffA_work at octave i
```

### Why this fixes both effects

Warping the image in pixel space with `map_coordinates` applies the estimated displacement field directly to the pixels. The warped image:
- Has the checker's pattern at (approximately) the same position as in B → cancels the translational phase shift
- Has the checker's pattern at (approximately) the same size as in B → its FFT puts energy in the same bands as B's FFT → cancels the spectral redistribution

The fine-scale residual after warping contains only the estimation error from the coarser octaves, which is small and centred near zero. `gradientfix` can unwrap the small residual reliably, and the octave adds a refinement instead of drifting.

### Why filter-response warping could never have worked

`shiftMatrixInt` operates on `iwA`, which is a narrowband complex signal. Its spectrum is concentrated in one frequency band. A pixel-space warp of this narrowband signal:
- Shifts the spatial envelope of the signal (good — this is the motion compensation)
- Cannot move its spectrum to a different band

But scaling *requires* moving the spectrum to a different band. So `shiftMatrixInt` is structurally incapable of compensating for scaling, regardless of how accurate the UV estimate is.

---

## Supporting changes

Two smaller changes accompanied the main fix:

**Log-scale radial filters.** The original radial filter bank was linearly spaced, which meant coarse octaves covered near-DC frequencies with no useful signal content for the checker. Switching to logarithmic spacing (each band doubles the centre frequency) places bands where signal energy actually lives and gives finer bands narrower fractional bandwidth, matching the harmonic structure of textured scenes.

**`scaleFt` from log spacing.** The phase-to-pixel conversion factor `scaleFt[i]` is now the log-scale band centre, giving the correct per-octave scaling when `gradientfix` converts radians to pixels.

---

## Result

With image-level warping enabled, the UV map for the scaling test case is the expected linear ramp pointing radially toward the centre. The 1D slices through the centre row and column show straight lines with the correct slope, and the magnitude plot shows the expected inverse-pyramid shape.

---

## Take-aways

1. **"Wavelets are better than Fourier for local motion"** is the right intuition in general, but the code was already using wavelets (Gabor transform). The bug was in what got warped between octaves, not in what got decomposed.
2. **Translation-only assumptions embedded deep in a pipeline** (warping filter responses) silently break for non-translational motion. The fix is to shift the compensation to a stage that can represent the motion type — pixel-space image warping.
3. **Coarse-to-fine iterative refinement works** as long as each step actually cancels what the coarser step estimated. If a step leaves residual motion of a type the next step cannot measure, the pipeline diverges.

---

## Related files

- [src/ComplexPyramidFlow2D.py](../src/ComplexPyramidFlow2D.py) — implementation
- [docs/ALGORITHM.md](ALGORITHM.md) — overall algorithm reference
- [README.md](../README.md) — project motivation
