"""
Tests the Kernels with samples and plots it
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pyrtools as pt
from FFTkernel import FFTkernel2D

if __name__ == '__main__':

    a = np.asarray(Image.open("testIm3_A.png").convert('L'))

    pyr = pt.pyramids.LaplacianPyramid(a)

    fig, ((ax_1, ax_2, ax_4, ax_8, ax_16), (ax_32, ax_64, ax_128, ax_256, ax_512)) = plt.subplots(2, 5, figsize=(15, 6))

    ax_1.imshow(a, cmap='gray', vmin=0, vmax=255)
    ax_1.set_title('Original')

    #b512 = fftc.convolute()
    # ax_512.imshow(fftc512.convolute(), cmap='gray', vmin=0, vmax=255)
    # ax_512.set_title('Convoluted 512x512')
    #
    # ax_256.imshow(fftc256.convolute(), cmap='gray', vmin=0, vmax=255)
    # ax_256.set_title('Convoluted 256x256')
    #
    # ax_128.imshow(fftc128.convolute(), cmap='gray', vmin=0, vmax=255)
    # ax_128.set_title('Convoluted 128x128')
    #
    # ax_64.imshow(fftc64.convolute(), cmap='gray', vmin=0, vmax=255)
    # ax_64.set_title('Convoluted 64x64')
    #
    # ax_32.imshow(fftc32.convolute(), cmap='gray', vmin=0, vmax=255)
    # ax_32.set_title('Convoluted 32x32')
    #
    # ax_16.imshow(fftc16.convolute(), cmap='gray', vmin=0, vmax=255)
    # ax_16.set_title('Convoluted 16x16')
    #
    # ax_8.imshow(fftc8.convolute(), cmap='gray', vmin=0, vmax=255)
    # ax_8.set_title('Convoluted 8x8')
    #
    # ax_4.imshow(fftc4.convolute(), cmap='gray', vmin=0, vmax=255)
    # ax_4.set_title('Convoluted 4x4')
    #
    # ax_2.imshow(fftc2.convolute(), cmap='gray', vmin=0, vmax=255)
    # ax_2.set_title('Convoluted 2x2')

    fig.show()
    plt.waitforbuttonpress()