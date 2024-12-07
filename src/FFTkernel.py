"""
(c) felixschaller.com 2023 this kernel is intended to do Fast Fourier Transform Convolution on Image Maps
currently just supports single channel
"""

from scipy import datasets
import numpy as np
from scipy import signal

class FFTkernel2D:

    input = None
    kernel = None

    def __init__(self, input, kernel):
        self.input = input
        self.kernel = kernel
        self.kx, self.ky = kernel.shape[:2]
        self.w, self.h = input.shape[:2]


    def convolute(self):
        # half kernel size
        hkx = self.kx // 2
        hky = self.ky // 2
        mask = np.pad(np.ones((self.w,self.h)).astype(np.float32), [(hkx, hky), (hkx, hky)])

        # convolute input and mask
        fftc = signal.fftconvolve(self.input, self.kernel)
        fftm = signal.fftconvolve(mask, self.kernel)
        fftm_shape = fftm.shape[:2]
        normalized = fftc / fftm[hkx:(fftm_shape[0]-hkx), hky:(fftm_shape[1]-hky)]
        return normalized[hkx:(self.w+hkx), hky:(self.h+hky)]