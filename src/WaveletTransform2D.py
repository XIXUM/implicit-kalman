import math

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pywt
from scipy import fft, signal

if __name__ == '__main__':

    ### --- load Testimages

    a = np.asarray(Image.open("images/xy_256x256.001.png").convert('L'))
    b = np.asarray(Image.open("images/xy_256x256.002.png").convert('L'))

    wp = pywt.WaveletPacket2D(data=a, wavelet='db1', mode='symmetric')

    maxLV = wp.maxlevel

    fig, ax = plt.subplots(2, 5, figsize=(15, 10), squeeze=True, sharex=False, sharey=False)

    ax[0, 0].imshow(wp['a'].data)
    ax[0, 1].imshow(wp['h'].data)
    ax[0, 2].imshow(wp['v'].data)
    ax[0, 3].imshow(wp['d'].data)
    ax[0, 4].imshow(wp['aaah'].data)

    ax[1, 0].imshow(wp['aaav'].data)
    ax[1, 1].imshow(wp['aaad'].data)
    ax[1, 2].imshow(wp['aaha'].data)
    ax[1, 3].imshow(wp['aahv'].data)
    ax[1, 4].imshow(wp['aahd'].data)

    plt.show()
    plt.waitforbuttonpress()