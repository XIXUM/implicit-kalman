# implicit-kalman
It's Featureless Tracking Algorithm based on the MIT research paper by
[Link to Paper](https://people.csail.mit.edu/nwadhwa/phase-video/phase-video.pdf)

nowadays motion tracking algorithms track motion by features which is a local contrast in the Image that is localized and followed according to the principle introduced by kalman
the problem in these approaches is, that it can:

1. only track motion where there is a contrast in the image
2. it does not concern the context where the feature is located in
3. the problem: if there is redundance in the image feature track can get lost

which causes confusion for the algorithm which of the redundant feature to follow

this algorithm instead makes use of image pyramids to resolve motion over different resolution layers so redundance is no longer an issue.
it also does not locate features stochastically but decomposes the image into frequencies with a 2D Fast Fourier Transform (FFT) and then calculates the phase shift over a time sequence

a full documentation will be available here soon.
