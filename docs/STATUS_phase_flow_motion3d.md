# Status / handoff: phase-based flow -> 3D -> mover detection (2026-09)

Where we stand after a research session spanning ImplicitKalman (the deterministic phase-based
estimator + the Affine Ceiling whitepaper) and the digital-twin motion-detection work. Written
so a following thread knows exactly what is proven, what is open, and what the decision is.
All numbers are GT-by-construction or validated against LiDAR / nuScenes boxes. No claims
beyond what was measured.

## The big picture / goal
Detect moving objects (esp. people partly anchored: feet planted, torso moving) from camera
video, by: dense phase-based optical flow -> remove known/estimated ego motion -> reconstruct
3D of the static scene (photogrammetry) -> whatever does not fit the rigid model is an
independent mover. Seeded from the MIT CSAIL line (Eulerian/phase magnification -> Interactive
Dynamic Video) and ImplicitKalman's deterministic, verifiable phase estimator.

## PROVEN (with numbers)
- **Coarse-to-fine idea works** (fresh ~150-line impl; the repo flagship `ComplexPyramidFlow2D`
  is separately broken, but the idea is sound). Scaling law: recoverable range ~= k*N
  (k ~ 0.2 global / ~0.1 dense) from an 8 px coarsest floor; scales with resolution
  (256 px -> ~45/35 px, 512 px -> ~96/55 px). Single-scale wall is lambda/2 ~ 2.5 px.
  The iFFT->warp->FFT detour cancels scaling's spectral shift.
- **Motion -> 3D chain is FEASIBLE, validated vs LiDAR**: known ego-pose (metric, resolves
  monocular scale) + dense flow -> per-pixel depth via known-pose triangulation. Geometry
  verified (perfect GT flow -> 0% depth error). With a good flow front-end (RAFT), camera depth
  matches LiDAR to **3% median (2% periphery, 96% within 20%)**. The research question
  "camera motion -> 3D?" is answered YES.
- **Mover detection works for the observable regime**: ego-compensated flow residual (observed
  flow minus LiDAR-predicted static flow) cleanly flags a transverse near mover (crossing car
  at 19 m -> residual 10.5 px vs 1.8 px background, 6x). Parked cars stay low = the statue/
  mover discriminator holds.

## OPEN / UNSOLVED (honest)
- **The deterministic estimator: coarse->fine CHECKER GATE now PASSED (2026-09).** The gate we
  defined - reliable octave-to-octave offset induction on the simple checker patterns
  (`ImplicitKalman/src/relief0*.png`) - is met by our fresh image-warp c2f: translation 0.002 px
  (global), scaling 0.357 px, rotation 0.154 px (dense median EPE), monotone per-octave
  convergence, NO fine-octave corruption; actual folder pairs relief0->offs 100% / ->ro 98%
  post-warp explained. So the octave induction is reliable; the user's remembered failure was
  the OLD frequency-domain-Gabor-warp ComplexPyramidFlow2D (which could not handle the spectrum
  shift - exactly the bug SCALING_SOLUTION.md fixed via image warping, which c2f does natively).
  The estimator's FOUNDATION is sound. Validation script:
  `digital-twin-motion/experiments/c2f_flow/runs/checker_coarse_to_fine.py`.
  - **Remaining gap is REAL-IMAGE robustness, not the pyramid induction:** on real nuScenes the
    flow is range-limited (under-estimated 5.9 px where the true motion was 17.6 px -> Phase-1
    depth failed with our own flow, 114% err vs RAFT 3%), blobby in flat regions (phase
    singularity / aperture, not fixable by smoothing), and shows a per-octave +-pi wrap quadrant
    break at the range limit. That is the next work when the estimator is resumed.
- **AI models (RAFT) prove a formal solution EXISTS** (they achieve consistent global dense
  flow) but the formal solution is not yet known. The phase approach (MIT line) is the right
  track; the pyramid-induction step is the specific gap.
- **Mover-detection has two fundamental blind spots** (any image-flow ego-residual method):
  (1) noise floor - far/slow movers (50 m pedestrian at 1.2 m/s -> ~1 px, sub-pixel,
  undetectable); (2) epipolar/radial degeneracy - a LEADING vehicle moving in the ego's own
  direction has its independent motion along its epipolar line -> ~indistinguishable from
  static (car at 28 m, 4.5 m/s -> residual 0.9 px). The leading vehicle (a key case) is not
  recoverable from camera flow alone. Plus false positives at near static (parked) edges.
- **Therefore camera + LiDAR fusion is necessary, not optional**: the camera residual sees
  TRANSVERSE motion, LiDAR sees the RADIAL leading vehicle directly. Complementary blind spots.

## DECISION (for following threads)
1. **Deterministic phase estimator: checker gate PASSED (foundation sound).** Still parked
   relative to real-image deployment; when resumed, the next work is real-image robustness
   (range on large near-object motion, flat-region aperture, edge/occlusion, compression),
   NOT the octave induction (which is verified reliable).
2. **Continue Phase 2 (mover detection) consistently**, using RAFT flow as the front-end
   stand-in (it is a validation reference, NOT the certifiable target; swap in our own flow
   once its range is reliable). Next Phase-2 work: spatial coherence to kill edge FPs, restrict
   to the detectable regime, and fuse with the LiDAR twin for the radial-mover blind spot.

## Where things live
- Prototypes + results: `digital-twin-motion/experiments/c2f_flow/` (c2f_flow.py, smoke_test.py,
  nuscenes_camera_flow.py, compare_raft.py, phase1_*.py, phase2_mover_detection.py, runs/),
  branch `feature/motion-detection-idv`.
- ImplicitKalman: `docs/the_affine_ceiling_whitepaper.pdf`,
  `docs/CRITIQUE_verifiability_vs_affine_ceiling.md`, this status note. The flagship
  `src/ComplexPyramidFlow2D.py` is broken (documented in the critique); the working benchmark
  estimators are `PhaseFlowLocalFreq` / `MultiScalePhaseFlow` / `MPME_faithful`.
- Cross-thread memory pointer: the assistant memory `implicitkalman-affine-ceiling` carries the
  full running record and is the primary handoff across sessions.
