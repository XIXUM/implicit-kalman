# Critique and comparison: verifiability vs. the affine ceiling

A critical appraisal of ImplicitKalman's central claim (a formal, verifiable, dense
phase-based motion estimator that surfaces non-derivability instead of guessing), set
against its own "Affine Ceiling" benchmark and the wider motion-estimation landscape.
Written as a working critique, not an endorsement: the goal is to locate exactly where the
verifiability promise holds, where it does not, and what would make it hold at the one
place that matters for 3D.

Sources read: `README.md`, `docs/Implicit_Kalman_Steerable_Filter_Spec.md`,
`docs/ALGORITHM.md`, `docs/SCALING_SOLUTION.md`, `docs/the_affine_ceiling_whitepaper.pdf`,
`src/benchmark_window_ablation.py`.

## 1. What ImplicitKalman actually claims (steelman)

The design is coherent and the claims are narrower than they first read:

- **Featureless, dense.** A complex steerable pyramid measures per-pixel phase; a spatial
  shift is a phase shift (`dphi = 2*pi*f*d`). No corner/blob tracking, so no feature
  aliasing on repetitive texture and no empty regions where features are absent.
- **Deterministic and training-free.** Every output is reproducible and analyzable; the
  stated advantage over learned dense reconstruction is not accuracy but **verifiability**:
  where the data is underdetermined, the estimator is meant to report low confidence rather
  than invent a plausible value (`Spec` sec. 1, 5).
- **Coarse-to-fine as anti-aliasing, not speed.** The pyramid keeps every per-level residual
  inside `(-pi, pi)`, so large motion is read on coarse (long-wavelength) bands and only the
  small residual is read finer. This is the multi-scale answer to the `lambda/2` wall.
- **An observability/confidence channel** gates the output: amplitude gate, a curvature
  requirement (a linear ramp is unobservable along its gradient), and aperture handling
  (mark 1D normal flow instead of fabricating a 2D vector).

The Affine Ceiling whitepaper then makes the honest, and genuinely useful, empirical point:
multi-scale phase estimation is **not** limited by motion size (affine motion, translation,
rotation, scale, is solved to sub-pixel accuracy even at large magnitude, M-PME 29 px ->
1.14 px). The real wall is the **affine assumption at a depth discontinuity**: on the
non-affine, per-pixel, discontinuous flow of a real scene, every method here blurs the depth
boundary, and the windowed least-squares affine fit is worst because its window averages
across the discontinuity.

## 2. The landscape it sits in

Three camps, and IK's position in each:

- **Feature-based (optical flow, SLAM, VO).** IK's README critique (aperture, no shared
  context, aliasing on repetitive texture) is fair against classical KLT-style trackers, but
  it is a partial strawman against the modern field: dense learned flow (RAFT and successors)
  and coarse-to-fine pyramids already defeat the checkerboard-aliasing example. IK's real
  differentiator over this camp is **determinism plus a confidence channel**, not
  anti-aliasing per se, which learned and classical multi-scale methods also have.
- **Learned dense reconstruction (DUSt3R, MASt3R, VGGT, LingBot-Map, Generative Image
  Dynamics, LaGSplat).** This is the sharpest contrast and IK's strongest ground: these are
  probabilistic by construction, with no error bound and no repeatable worst case, which is
  disqualifying under ISO 26262 / DO-178C / IEC 62304. The whitepaper's "the tell, nobody
  trusts the prior alone" (2025-26 work bolting classical geometry back onto learned priors)
  is a strong, well-aimed observation.
- **Classical phase (Fleet & Jepson, MIT motion magnification, Simoncelli-Freeman
  pyramids).** IK is a direct descendant that extends magnification to **metric** displacement.
  It therefore inherits the whole classical toolkit, and the whole classical failure set:
  phase singularities, the `lambda/2` ceiling (pushed, not removed, by the pyramid), the
  aperture problem, and, per the whitepaper, the affine ceiling.

## 3. The central critique: the confidence mask has a blind spot exactly at the boundary

This is the crux, and it is where the two in-house documents (the verifiability claim in the
Spec and the failure identified in the whitepaper) are in unacknowledged tension.

The verifiability promise rests on the confidence mask surfacing "non-derivability instead of
guessing." But look at **what kind** of failure the mask detects. Every gate in Spec sec. 5
is a **signal-observability** gate: low amplitude, low curvature, single-orientation aperture.
These all catch the case *there is not enough signal here to estimate motion*.

The affine-ceiling failure is a **different species**. At a depth boundary the signal is
strong (a crisp, high-contrast edge, plenty of amplitude and curvature, multiple
orientations), so every observability gate reports **high** confidence. Yet the estimate is
wrong: the windowed/affine fit straddles two depths and returns a blurred, confident value.
The fit residual can even be *low* (a smooth affine field fits a blurred edge well), so a
residual gate would not flag it either. In the whitepaper's own numbers the affine M-PME is
*worse* at the edge (1.17 px) than the simpler per-pixel reimplementation (0.52 px), and it
gets there with a well-conditioned, low-residual fit.

**So the "it won't lie, it will say low confidence" promise fails precisely where 3D needs it
most.** The mask surfaces signal gaps; it does not surface **model mismatch** (a single
affine/translation model imposed across a motion discontinuity). At the depth boundary the
estimator is well-conditioned by its own criteria and confidently wrong. That is not a
verifiable failure, it is a silent one, the exact property IK was built to avoid.

This is the honest core of the critique. It does not refute the project; it says the
verifiability guarantee currently covers the interior of surfaces and not their boundaries,
and the boundary is what a 3D reconstruction is made of (the whitepaper says as much).

## 4. Constructive: the missing signal is already in the ablation

The fix is latent in the repo. `benchmark_window_ablation.py` sweeps the affine fit window on
the two-plane Z-dolly. At a depth boundary the estimate is **window-scale dependent** by
construction: a small window is sharp but noisy, a large window is smooth but blurred across
the edge. On a locally-affine interior the estimate is window-**invariant**. That disagreement
across window scales (or across pyramid levels) is a direct, deterministic **model-mismatch
detector**: where the multi-scale estimates diverge beyond their own noise floor, a single
affine model does not hold, i.e. a motion/depth discontinuity is present.

Adding window-scale (or cross-level) disagreement as a fourth confidence gate, alongside
amplitude, curvature, and aperture, would let the mask surface the depth boundary as
low-confidence, restoring the "surfaces non-derivability instead of guessing" promise at the
one place it currently breaks. It stays fully deterministic and needs no new machinery, only
the ablation the project already runs, promoted from an offline benchmark to a per-pixel
confidence signal.

Corollary (cross-project): once the boundary is flagged rather than blurred, the natural move
is to **source the boundary elsewhere and keep IK for the interior**. A LiDAR twin (or a
stereo baseline) gives a crisp depth edge; IK gives certifiable sub-pixel flow on the smooth
interior where it is strong. Each covers the other's blind spot. This also resolves the
monocular scale ambiguity below.

## 5. Secondary critiques (balanced)

- **Monocular scale ambiguity is fundamental, and acknowledged (Spec sec. 8).** Even after a
  correct rotation/translation split (Longuet-Higgins and Prazdny), monocular flow gives depth
  only up to a global scale. So "dense metric 3D from a single camera" is out of reach without
  an external cue (IMU, known baseline, known size). The certifiability argument therefore
  buys a verifiable **flow field**, not yet a verifiable metric **depth**.
- **Verifiability of flow is not verifiability of depth.** The depth chain (rotation vs.
  translation split, then scale) adds hard steps where error compounds. A confidence channel
  on `u, v` does not automatically propagate to a confidence on `Z`. The rotational-flow
  removal in particular is called out as "the genuinely hard step" and is where a small flow
  error becomes a large depth error near the epipole. The end-to-end verifiability claim needs
  the confidence to be carried through that chain, which is not yet shown.
- **Latency vs. the learned camp.** A dense complex steerable pyramid with a per-octave
  full-image FFT and warp is not obviously real-time, while the learned contrast (VGGT,
  sub-second feed-forward over hundreds of views; LingBot at ~20 FPS) is fast. For the
  safety-critical target this is the real trade the whitepaper implies but does not quantify:
  certifiable-but-slower vs. fast-but-uncertifiable. A latency/throughput number belongs next
  to the accuracy numbers.
- **Maturity and known defects.** Scaling was only recently made to work (image-level warp,
  `SCALING_SOLUTION.md`), and `ALGORITHM.md` lists live issues: the angular filter is not
  modulo-continuous at 0/360 degrees (a seam in one direction), and `gradientfix` cannot
  recover a field that wraps multiple times per pixel at fine scales. These are interior-flow
  robustness gaps, separate from, and prior to, the boundary critique above.
- **The anti-aliasing novelty is modest.** Coarse-to-fine multi-scale is shared with the
  learned and classical flow camps. The defensible novelty is the **combination**:
  deterministic dense phase flow + an explicit observability channel. The critique in sec. 3
  is that this channel is currently incomplete, not that it is unoriginal.

## 6. Comparison at a glance

| Dimension | ImplicitKalman (formal phase) | Learned dense (VGGT/DUSt3R/LingBot) | Classical features (SLAM/KLT) |
|---|---|---|---|
| Output | dense `u,v` + confidence | dense depth/pointmap | sparse 6-DoF + keypoints |
| Certifiable / bounded | yes, by design (flow) | no (probabilistic) | yes (geometry), but sparse |
| Large affine motion | sub-pixel (multi-scale) | good | good |
| Depth boundary | **blurs; not yet surfaced** | blurs, plausibly hallucinated | n/a (sparse) |
| Failure mode | silent at boundary (sec. 3), else low-confidence | silent, no bound | silent aliasing on repetitive texture |
| Monocular metric scale | ambiguous (needs cue) | learned prior (unbounded) | ambiguous (needs cue) |
| Latency | unquantified, likely high | low (feed-forward) | low |
| Maturity | early (scaling just fixed) | productized | mature |

## 7. Verdict

ImplicitKalman's real differentiator is not accuracy, it is **failure-surfacing**: a
deterministic dense estimator that is meant to know where it cannot know. Its own benchmark
correctly identifies the unsolved core (the depth boundary, not motion size). The sharpest
criticism is internal and precise: the current confidence mask surfaces
**signal-observability** gaps but not **affine-model mismatch**, so at the depth boundary the
estimator is confidently wrong, which is the one failure the project exists to prevent. The
good news is that the cure is already on hand (window-scale / cross-level disagreement as a
model-mismatch gate, straight out of the existing window ablation), and pairing IK's certified
interior flow with a boundary source (LiDAR twin or stereo) turns the remaining hard problems
(boundary sharpness, monocular scale) into someone else's solved ones.

Positioned honestly, IK is a **certifiable interior-flow front-end with an incomplete
confidence model**, not a standalone dense-3D solution. Complete the confidence model at the
boundary, quantify the latency, and carry the confidence through the depth chain, and the
verifiability claim becomes true where it currently is not.
