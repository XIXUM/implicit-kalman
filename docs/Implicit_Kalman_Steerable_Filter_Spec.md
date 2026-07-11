# Implicit Kalman: Steerable Filter and Phase-Based Estimator Spec

Context document for configuring the complex steerable pyramid and the phase-based motion estimator. Written as design rationale plus concrete rules so an assistant can tune the filter bank correctly rather than guess. The physics rationale is included on purpose, because several parameter choices only make sense once the underlying constraint is explicit.

---

## 1. Goal and paradigm

Estimate dense, per-pixel motion from monocular video by tracking the phase of complex, orientation-selective, multi-scale filter responses, not by tracking features. The method is deterministic and training-free. Its intended advantage over learned reconstruction models is verifiability: where the data is underdetermined, the estimator surfaces low confidence instead of inventing a plausible value.

Core lineage: phase-based optical flow (Fleet and Jepson), phase-based video motion (MIT CSAIL), complex steerable pyramids (Simoncelli and Freeman).

## 2. Filter bank design (complex steerable pyramid)

- Use a **complex** steerable pyramid so each subband gives amplitude and phase. Phase carries local motion; amplitude carries confidence.
- Each coefficient is indexed by (location, scale, orientation). Within one (scale, orientation) band, the residual local motion is approximately a translation, i.e. a pure phase shift. This is the "easy" regime; keep the estimator working inside it.
- Orientation count: a single orientation only yields the motion component **along** that orientation (aperture problem). Recover the full 2D vector by combining several orientations at the same location and scale. Start with 4 orientations, allow 6 to 8 where 2D recovery is unstable.
- Keep the filters quadrature (Hilbert pairs) so phase is well defined and monotonic in displacement.

## 3. Pyramid depth and the anti-aliasing role of scale (critical)

Coarse-to-fine is **not** primarily a speed trick, it is the anti-aliasing strategy. Rationale:

- Phase is 2*pi periodic. If the displacement exceeds half the wavelength of the band, the phase **wraps** and the estimator reports a small wrong displacement instead of the large true one. This is the temporal Nyquist limit, the same mechanism as the wagon-wheel effect.
- Coarse bands have long wavelength, hence a **large unambiguous range**. Estimate the large motion on coarse scales where no wrapping occurs, warp, then measure only the small residual delta on finer scales, which now sits safely below the wrap limit.

**Design rule for octave count.** Choose the number of octaves so the coarsest band's half-wavelength exceeds the maximum expected displacement per frame.

- Let `d_max` = maximum expected motion per frame in pixels (from the fastest camera or scene motion).
- Finest band half-wavelength is on the order of 1 pixel of displacement before wrap.
- Each coarser octave doubles the unambiguous range. Required octaves `N` such that `2^N >= d_max`, i.e. `N >= log2(d_max)`.
- Example: `d_max = 32 px` needs `N >= 5` octaves. Add one octave of margin.

Expose `d_max` and `N` as explicit parameters. Do not hardcode pyramid depth independent of the expected motion budget.

## 4. Phase unwrapping and warping loop

- Process coarse to fine. At each level: estimate residual phase shift, convert to displacement, warp the finer level by the accumulated flow, then estimate the next residual.
- Never unwrap phase globally in a single band. Rely on the hierarchy to keep every per-level residual inside the (-pi, pi) unambiguous interval.
- Track accumulated displacement in a float flow field, warp with a high-quality interpolator (bicubic or better) to avoid injecting resampling artifacts that masquerade as motion.

## 5. Observability and the confidence mask (do not skip)

The estimator is only well conditioned where the signal supports it. Produce a per-pixel confidence, not just a vector.

- **Amplitude gate:** low subband amplitude means no reliable phase. Downweight.
- **Structure requirement:** the method needs a **non-linear** intensity profile, not features. A strictly linear gradient is unobservable **along** the gradient direction (translation along a linear ramp is indistinguishable from a brightness change). Only curvature, the second derivative, makes it observable. Estimate local gradient curvature and downweight the flow component along low-curvature directions.
- **Aperture handling:** where only one orientation has signal, mark the vector as 1D (normal flow only) rather than reporting a fabricated 2D vector.
- Output a validity/confidence channel alongside u, v. This confidence channel is the "surfacing" mechanism: it is how the estimator reports non-derivability instead of guessing.

## 6. Preprocessing: lens distortion and vignette

- **Lens distortion** is a purely geometric remapping, static and camera-fixed. Undistort once with the calibrated warp before building the pyramid. Pixel intensities are unchanged, only positions move.
- **Vignette** is a static, spatially varying **multiplicative** intensity falloff, `I_observed = I_scene * v(x)`. For the rough case, compensate with an inverse flat-field map (`I_corrected = I_observed / v(x)`) before the pyramid. This is sufficient for current needs.
  - **Refinement, not a blocker (parked):** because vignette is multiplicative, not additive, a derivative filter does not simply ignore it. By the product rule, `d/dx (I*v) = I'*v + I*v'`, and the second term is a spurious gradient largest where the vignette is steep, i.e. in the corners. This biases phase velocity radially at the frame edges if left uncorrected. The inverse flat-field division removes it, at a small noise-amplification cost in the darkened corners, so downweight confidence slightly at strongly de-vignetted edges. Treat as a special case to revisit, not a current priority.
- Do distortion and vignette correction **before** the pyramid, in this order: undistort, then flat-field.

## 7. Scale and rotation (harder than translation)

- Pure translation is trivial (shift theorem: a phase ramp, magnitude invariant).
- Scale and rotation are **not** in the phase, they deform the frequency carrier: rotation rotates the spectrum by the same angle, scale by factor `a` compresses it by `1/a`.
- Do **not** fight scale and rotation in the global spectrum. Make scale and orientation explicit sampling axes of the pyramid instead. Then within a (scale, orientation) band the residual is again a translation, and scale/rotation appear as motion **across** bands, a flow through the scale-orientation volume.
- Fourier-Mellin / log-polar is the classic global alternative (rotation and scale become translations on the log-polar magnitude, recovered by phase correlation). Note its weaknesses if used: it discards phase and works on magnitude only, and the log-polar resampling injects its own aliasing near the origin and at high frequencies. The pyramid, cross-band approach is preferred here precisely to avoid that.

## 8. Depth chain (downstream, for correctness of expectations)

The dense vector field is not depth yet. Two hard steps sit between them:

- **Rotation vs translation split.** The flow at each pixel is the sum of a translational component proportional to `1/Z` (this carries depth, the parallax) and a rotational component independent of `Z` (Longuet-Higgins and Prazdny). The rotational part carries no depth and must be removed before any depth is computed. This is the genuinely hard step, not the lens correction. A useful calibration trick: a pure rotation about the projection center with zero Z-translation yields a depth-free, purely rotational reference field, usable to isolate distortion and to characterize the rotational flow model.
- **Scale ambiguity.** Even after removing rotation, monocular flow gives depth only up to a global scale, because the translational term is the product of `1/Z` and the camera translation magnitude. An external cue is required (IMU, known baseline, known object size). Human vision solves this with two eyes: a fixed known baseline gives simultaneous parallax without ego-motion, which is exactly the missing external scale built in.

## 9. Motion blur as the integrated motion field (optional module, same theory)

- Linear motion blur is a convolution of the sharp image with a line PSF whose direction is the motion direction and whose length is speed times exposure. In the Fourier domain this line PSF creates parallel zero-lines perpendicular to motion, spacing inversely proportional to blur length.
- Steerable filters detect this directional zero pattern naturally, so **PSF estimation** (direction and length) is deterministic and well conditioned.
- **Deconvolution (the inversion) is ill-posed** at the spectral nulls, where information is destroyed, so recovery needs regularization. Keep "deterministic" for the PSF estimation claim, not for full restoration.
- Key unifying point: a per-pixel motion field is, up to exposure time, the same latent quantity as a per-pixel PSF map. The dense flow field **is** the spatially varying PSF map. Motion blur is the exposure-integrated form of the same motion signal the estimator samples over time. Spatially varying blur invalidates global Fourier analysis, so window locally, which puts this back into the same pyramid.

## 10. Known hard limit (document, do not pretend to solve)

Specular reflection breaks brightness constancy: the highlight is view-dependent and moves with the camera, not the surface. It is not separable in a single view. Only its inconsistent motion relative to the diffuse substrate over a longer sequence separates it, and inferring shape from highlight motion needs still more context. Mark specular regions via brightness-constancy residual over the sequence and downweight them, rather than trusting their flow.

## 11. Parameter summary to expose

- `d_max`: max expected displacement per frame (pixels). Drives octave count.
- `N_octaves`: `>= log2(d_max)` plus one octave margin.
- `N_orientations`: 4 default, 6 to 8 where 2D recovery is unstable.
- `amp_threshold`: subband amplitude gate for confidence.
- `curvature_threshold`: minimum gradient curvature for observability along a direction.
- `interp`: bicubic or better for warping.
- Preprocessing toggles: `undistort` (geometric), `flatfield` (vignette division), applied in that order before the pyramid.
- Output channels: `u`, `v`, `confidence`, `is_1D_normal_flow`.
