# LinkedIn / Social Media Drafts

Drafted comments and replies for state-of-the-art / competitor posts on our radar.
Standalone talking points, ready to paste — **not scheduled**. Character counts noted
(LinkedIn comments read best under ~1000 chars).

Cross-references:
- `docs/STATE_OF_THE_ART_radar.md` — radar & the context-free-vs-contextful framing.
- `docs/post_message.md` — whitepaper-launch teasers (LinkedIn / X / one-liner).

---

## 1. Comment — Kalman-filter explainer (Deepak Kumawat)

**Context.** A generic "predict → measure → update" Kalman-filter explainer.
**Angle.** Decidability: the Kalman filter compares against a *model*; Turing's rule
against *nothing*. Uncertainty isn't the axis — context is. *(818 chars)*

> Great framing. The deepest part hides in your line "how much should I trust this
> measurement?" — the filter can only ask that because it carries a model of what
> it's estimating. Its innovation term compares the measurement against a
> *predicted* one: it consults context.
>
> That's the quiet contrast with the halting problem. A Turing rule consults nothing
> outside its symbol — decidable because it's about nothing, and for the same reason
> it can't decide when a computation is "done." The Kalman filter converges precisely
> because it isn't context-free: it knows what its estimate is *for*.
>
> Uncertainty isn't the point — context is. Probabilistic or not, a system with a
> referent converges; a context-free one drifts. Same reason today's monocular-3D
> nets look sharp yet smear the depth edge: structure without ground.

---

## 2. Comment — PyCuVSLAM / cuVSLAM (Ali Pahlevani)

**Context.** NVIDIA GPU-accelerated visual SLAM, "stable tracking / consistent
trajectory" on TUM RGB-D.
**Angle.** Geometric consistency ≠ semantic ground → no false-positive protection.
See backlog issue #4. *(786 chars)*

> Nice work — cuVSLAM's speed is genuinely impressive.
>
> One thing worth naming: "consistent trajectory" is internal *geometric*
> consistency, not certified correctness. A pose graph optimizes for
> self-consistency — it has a geometric model but no semantic referent, so it can't
> tell a true loop closure from perceptual aliasing, or a moving object from a static
> landmark. The map can be perfectly consistent and still wrong.
>
> That's the gap we keep circling in our work on context and decidability: structural
> context (geometry, statistics) without semantic ground yields confident,
> self-consistent, false estimates — the SLAM cousin of model hallucination. TUM
> RGB-D is benign; the interesting question is adversarial input.
>
> Not a knock on cuVSLAM — just where the next hard problem lives.

### 2a. Sub-reply (under our own comment) — paper reference

Adds the concrete dense-side paper. **Needs a public URL** for `[link]` (whitepaper
is not yet hosted publicly). *(313 chars)*

> Where we took this, for anyone curious: our note "The Affine Ceiling" benchmarks
> the dense side against pixel-exact ground truth. Even the affine state of the art
> recovers motion cleanly yet smears the depth boundary — the same "structurally
> consistent, semantically ungrounded" failure, made measurable. → [link]

---

## Not posted (competitor tracking only)

- **TrackEverything** (CMU & Meta, arXiv 2609.30222) — logged in the radar as a
  dense-motion benchmark candidate; no public comment planned.
