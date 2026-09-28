# State of the Art — Radar & Working Insights

**Purpose.** A running log of external work on our radar (camera-only / monocular
3D, phase-based motion, state estimation) and the insights and contradictions we
intend to implement. This note is connective tissue only:

- **Theory** lives in the *Gödel's Incompleteness Theorem and the Power of Context*
  paper (AAAI variant, `…/WhitePapers/Gödel+Kontext/main.tex`).
- **Benchmark evidence** lives in `docs/the_affine_ceiling_whitepaper.pdf`.
- Deliberately kept **out** of the Affine Ceiling whitepaper, which stays focused
  on the affine-vs-dense benchmark. The decidability / Kalman connection is the
  AAAI paper's job, not the whitepaper's (decision 2026-09-11).

---

## 1. Radar — external work we are tracking

| Work | What it is | Camp | Relevance to us |
|---|---|---|---|
| ORB-SLAM3 (Campos et al., 2021) | feature-based visual/-inertial SLAM | geometric, **sparse** | localization is solved; sparse points + 6DoF pose; monocular scale ambiguity |
| NVIDIA cuVSLAM / PyCuVSLAM (2025) | GPU-accelerated visual SLAM (mono/stereo/RGB-D), Isaac ROS; Python bindings | geometric, **sparse** | fast + deployable, but geometric consistency ≠ semantic ground → no false-positive protection (loop-closure aliasing, dynamic scenes). See issue #4 |
| Main Street Autonomy — Pose Engine (2026, commercial) | targetless camera-only 6DoF localization, deployed (farm tractors) | geometric, **sparse** | market signal: camera-only is real in safety-adjacent field. Our edge = certifiability, not novelty of "camera-only" |
| Monocular VSLAM explainer (LinkedIn, 2026) | textbook sparse pipeline (features → triangulation → VO → loop closure → pose-graph) | geometric, **sparse** | not new; reinforces "sparse geometry = solved" |
| DUSt3R → MASt3R → VGGT (NAVER et al., 2024–25) | learned dense pointmaps / metric depth, feed-forward | learned, **dense** | the dense land-grab; impressive, **probabilistic** |
| LingBot-Map (Robbyant / Ant Group, 2026) | productized streaming dense monocular 3D | learned, **dense** | same, deployed & real-time; still learned/probabilistic |
| TrackEverything (CMU & Meta, 2026; arXiv 2609.30222) | learned dense 3D point tracking in world coords, long-horizon (1000+ frames), voxel-de-dup; eval on TAPVid-3D | learned, **dense / 4D tracking** | closest learned SOTA to our dense-motion / mover-detection goal → **benchmark candidate** (adopt TAPVid-3D). 40 GB GPU, learned/probabilistic; *not* Kalman — the "trajectory refiner" is a learned module, Kalman only in spirit. Part of a 2025–26 wave (Track4World, St4RTrack, Multi-View 3D PT) |
| Kalman-filter framing (D. Kumawat, LinkedIn, 2026) | predict → measure → update state estimation under uncertainty | geometric, **sparse** | the recognized discipline for sparse state; the bridge to our name (see §2) |

---

## 2. The unifying insight (why these belong on one radar)

The popular axis — **probabilistic vs. deterministic** — is the wrong axis. The
axis that actually separates "works / certifiable" from "hallucinates" is
**context-free vs. contextful**, exactly as the Gödel/Context paper argues:

- A context-free system (Turing's δ "consults nothing"; an LLM's token window) has
  no internal access to its own semantics → undecidable / hallucinates. Supplying
  context (a referent, a "for") returns decidability at the price of
  non-effectiveness — the paper's **Context Theorem**.
- **Learned dense 3D = structural context without semantic context.** DUSt3R /
  VGGT / LingBot hold a learned manifold (structure) but no referent (ground), so
  they emit depth that is *structurally plausible and ontologically unanchored*.
  **Our blurred depth boundary in the non-affine benchmark IS that hallucination,
  made empirical** — the "0 K teacup" of Fig. hero: nothing in pure structure
  forbids the wrong edge.
- **The Kalman filter makes the Context Theorem visible.** Its innovation term
  compares the measurement against a *model-predicted* one — it *consults the
  model*, the exact opposite of Turing's context-free δ. It converges and stays
  bounded because it carries a referent (a state model, a "for"), **not** because
  it is deterministic (it is Bayesian). A probabilistic system *with* context is
  fine; a context-free system is not.
- **ImplicitKalman = the Context Theorem, constructive, for dense reconstruction.**
  Supply the physical/geometric context (projection geometry, conservation) as the
  variance-bounding structure, so dense per-pixel estimation becomes
  decidable / bounded / certifiable where the context-free (learned) route
  hallucinates. The Kalman filter did this for *sparse* state; we extend it to
  *dense*.

**Reviewer caveat (for the AAAI paper):** the Kalman filter is an *illustration*
of "bounded context ⇒ solvable" and belongs in the resolution/engineering section
— it is **not** evidence that context defeats undecidability (linear-Gaussian
estimation was never undecidable). Use it as an intuition pump and as the
historically recognized precedent ImplicitKalman inherits, not as a decidability
claim.

---

## 3. What we intend to implement / carry forward

- **Ground the dense estimator in an explicit referent.** Projection geometry +
  conservation as the variance-bounding context, analogous to the KF's state
  model. This is the mechanism the learned methods lack.
- **Make the depth boundary the certifiable locus.** Where learned methods
  hallucinate (blur), the grounded method must stay crisp; the non-affine
  benchmark's edge-EPE is the measurable success criterion.
- **Reframe the pitch axis** (marketing/positioning only, not the whitepaper body):
  *grounded/contextful vs. context-free*, not *deterministic vs. probabilistic*.
  This lets us embrace the Kalman heritage instead of distancing from it, and
  removes the "but the Kalman filter is probabilistic" objection.
- **Watch** MSA / LingBot as deployment proof that camera-only is viable; the
  differentiator to hold onto is certifiability (grounded, not learned).
- **Adopt TAPVid-3D as the external dense-motion benchmark.** It is the arena of
  the world-centric dense-3D-tracking wave (TrackEverything, Track4World,
  St4RTrack, …), which is exactly our "phase flow → 3D → mover detection" target.
  TrackEverything is the learned SOTA to measure against — the point is not to beat
  its accuracy but to contrast it on certifiability and cost (40 GB GPU, learned).

---

## 4. Cross-references

- Theory: *Gödel's Incompleteness Theorem and the Power of Context*, AAAI variant
  (`main.tex`) — Hilbert cage; Context Theorem; structural vs. semantic context;
  "halting, re-read semantically."
- Evidence: `docs/the_affine_ceiling_whitepaper.pdf` — affine solved / dense
  hallucinated; the non-affine depth benchmark; references [1]–[13].

---

## 5. Outreach — drafted LinkedIn comment (on the Kalman-filter post, < 1000 chars)

> Great framing. The deepest part hides in your line "how much should I trust this
> measurement?" — the filter can only ask that because it carries a model of what
> it's estimating. Its innovation term compares the measurement against a
> *predicted* one: it consults context.
>
> That's the quiet contrast with the halting problem. A Turing rule consults
> nothing outside its symbol — decidable because it's about nothing, and for the
> same reason it can't decide when a computation is "done." The Kalman filter
> converges precisely because it isn't context-free: it knows what its estimate is
> *for*.
>
> Uncertainty isn't the point — context is. Probabilistic or not, a system with a
> referent converges; a context-free one drifts. Same reason today's monocular-3D
> nets look sharp yet smear the depth edge: structure without ground.
