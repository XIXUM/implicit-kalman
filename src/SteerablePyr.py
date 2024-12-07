
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, signal
from PIL import Image

import PhaseBased
from ComplexSteerablePyramid import im2pyr, pyramid_filter, get_filter_coeffs, angular_filter, bandpass_filter, \
    highpass_filter, lowpass_filter

if __name__ == '__main__':

    ### --- load Testimages

    a = np.asarray(Image.open("relief0.png").convert('L'))
    b = np.asarray(Image.open("relief0_offs.png").convert('L'))

    rows, cols = a.shape
    halfX = cols // 2
    halfY = rows // 2
    x = np.linspace(0, 255, 256)
    y = np.linspace(0, 255, 256)
    mx, my = np.meshgrid(x, y)
    filter = np.exp(-1j * ((mx - halfX) * 0.589 + (my - halfY) * 0.589))

    rows, cols = a.shape

    alpha = 75
    D, N, K = 3,2,8
    fl, fh, fs = 0.2, 0.25, 24
    F_length = 1001
    #frames = np.moveaxis(np.dstack((a,b)), -1 ,0)
    frames = np.stack((a,b))

    verbose = True

    F = PhaseBased.get_temporal_filter(fs, fh, fl, F_length)
    rawF = signal.firwin(F_length,[fl,fh],fs=fs,pass_zero=False)


    Ps, Rhs, Rls = im2pyr(frames, D, N, K, verbose=verbose)

    fig, ax = plt.subplots(2, 5, figsize=(15, 10), squeeze=True, sharex=False, sharey=False)

    n = 0
    k = 0
    filt = lambda r, th: pyramid_filter(r,th,n,N,k,K)
    afilt = lambda r, th: angular_filter(r, th, k, K)
    bpfilt = lambda r, th: bandpass_filter(r,th,n,N)
    hpfilt = lambda r, th: highpass_filter(r/np.power(2.,(N-n-1)/N),th)
    lpfilt = lambda r, th: lowpass_filter(r/np.power(2.,(N-n)/N),th)

    # ax[0,0].plot(F)
    # ax[0, 0].plot(rawF)
    ax[0, 1].imshow(np.log(np.abs(Rhs[0])))
    ax[0, 2].imshow(fft.fftshift(get_filter_coeffs(frames.shape[-2],frames.shape[-1],filt,False)).real)
    n = 1
    k = 0
    ffbpFilter = fft.fftshift(get_filter_coeffs(frames.shape[-2],frames.shape[-1],bpfilt,False))
    ffhpFilter = fft.fftshift(get_filter_coeffs(frames.shape[-2], frames.shape[-1], hpfilt, False))
    fflpFilter = fft.fftshift(get_filter_coeffs(frames.shape[-2], frames.shape[-1], lpfilt, False))
    ax[0, 3].imshow(ffhpFilter.real)
    ax[0, 0].plot(fflpFilter[128].real)
    n = 0
    k = 4
    ax[0, 4].imshow(fft.ifft2(get_filter_coeffs(frames.shape[-2],frames.shape[-1],filt,False) * filter).real)
    ax[1, 0].imshow(np.log(np.abs(Ps[0][0][0][0])))
    ax[1, 1].imshow(np.angle(Ps[0][0][0][0]))
    ax[1, 2].imshow(np.log(np.abs(Ps[0][0][2][0])))
    ax[1, 3].imshow(np.angle(Ps[0][0][5][1]))
    ax[1, 4].imshow(np.angle(Ps[0][0][5][0]))

    plt.show()
    plt.waitforbuttonpress()
