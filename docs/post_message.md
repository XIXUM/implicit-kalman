# Post message — "The Affine Ceiling" whitepaper

Teaser copy to accompany the PDF whitepaper
(`the_affine_ceiling_whitepaper.pdf`) as an attachment. Picks up the
"Forget LIDAR" / LingBot-Map buzz. Tone: credit the AI, then the safety caveat —
honest that formal methods can't currently match learned models on capability.
The "we're building the fix" is a quiet undertone, not the headline.

---

## Long version (LinkedIn)

"Forget LIDAR" is making the rounds — Robbyant's LingBot-Map reconstructs 3D from
a single camera, live, ~20 FPS, open-source. It's genuinely impressive, and it's
part of a wave of learned models that have gotten very good, very fast. Formal,
hand-derived algorithms don't currently keep up on raw capability. Credit where
it's due.

It's also a neural network. Probabilistic by construction: no error bound, no
repeatable output, no provable worst case. For your phone, perfect. For a car's
braking decision under ISO 26262 — or a surgical robot under IEC 62304 — that's a
hard stop. Not a knock on the model; a property of the approach.

So we asked the uncomfortable question: how good is the *formal*, non-probabilistic
alternative — the kind you can actually certify? We benchmarked it against
pixel-exact ground truth.

The honest answer, in one whitepaper:
- On affine motion (scale, rotation), formal phase-based methods are sub-pixel
  sharp — the old intuition that "classical only does small motion" is simply wrong.
- On real depth — a camera moving over a scene with near and far surfaces — every
  one of them blurs or destroys the depth boundary. Unusable for 3D.

So today you pick one: capable but uncertifiable, or certifiable but not accurate
enough. Nobody ships both. That gap is the whole point — and the problem we spend
our days on.

Full benchmark, the math, and reproducible code in the attached whitepaper. 📎

#ComputerVision #FunctionalSafety #3DReconstruction #ADAS #Perception

---

## Short version (X / Bluesky)

"Forget LIDAR" — LingBot-Map does live monocular 3D, and it's impressive. It's also
a neural net: probabilistic, uncertifiable for safety-critical use. So how good is
the formal alternative you *can* certify? We measured it: sub-pixel on affine
motion, but it destroys the depth boundary. Nobody ships capable + certifiable yet.
Whitepaper attached. 📎

---

## One-liner (email subject / DM)

Learned 3D models are impressive — and uncertifiable. We benchmarked the formal
alternative against ground truth: sharp on motion, blind at depth. Whitepaper inside.
