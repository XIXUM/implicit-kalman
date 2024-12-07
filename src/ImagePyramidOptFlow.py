import math

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import fourier_shift
from skimage.transform import pyramid_laplacian, pyramid_expand
from skimage import color
from scipy import fft, signal
from skimage.registration import phase_cross_correlation



if __name__ == '__main__':
    #original = data.astronaut()
    #image = color.rgb2gray(original)

    ### --- load Testimages

    a = np.asarray(Image.open("images/xy_256x256.001.png").convert('L'))
    b = np.asarray(Image.open("images/xy_256x256.002.png").convert('L'))

    ### ----

    # if len(image.shape) > 2:
    #     rows, cols, dim = image.shape
    # else:
    #     rows, cols = image.shape

    rows, cols = a.shape

    pyramidA = tuple(pyramid_laplacian(a, downscale=2, channel_axis=None, sigma=1.2, order=2))
    pyramidB = tuple(pyramid_laplacian(b, downscale=2, channel_axis=None, sigma=1.2, order=2))

    composite_image = np.zeros((rows, cols + cols // 2, 3), dtype=np.double)

    composite_image[:rows, :cols, :] = color.gray2rgb(pyramidA[0]*10+0.5)

    i_row = 0
    for p in pyramidA[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = color.gray2rgb(p*10+0.5)
        i_row += n_rows
##########
    selection = 1
##########
    rA = pyramidA[selection]
    rB = pyramidB[selection]

    fig, ax = plt.subplots()

### Window
    x = np.linspace(0, 256, 256)
    y = np.linspace(0, 256, 256)

    window1d = np.abs(np.blackman(15))
    window2d = np.sqrt(np.outer(window1d, window1d))

    mx, my = np.meshgrid(x, y)
    Z = fft.fftshift(fft.fft2(window2d,(y.shape[0], x.shape[0])))

    fig3D, ax3D = plt.subplots(subplot_kw={"projection": "3d"})

    ax3D.plot_surface(mx, my, fft.ifft2(fft.ifftshift(Z)).real)

### create Parameter Plots ...
    fig2, ax2 = plt.subplots(2, 5, figsize=(15, 10), squeeze=True, sharex=False, sharey=False)
    half = rA.shape[0] // 2

    gradA = np.gradient(rA)
    gradB = np.gradient(rB)
    offset = rA - rB
    offsGrad = np.gradient(offset)
    quotA = offset / gradA[1]
    quotB = offset / gradB[1]



    ax2[0, 0].plot(rA[half], label='rA')
    ax2[0, 0].plot(rB[half], label='rB')
    ax2[0, 0].plot(offset[half], label='offs')
    ax2[0, 0].legend()
    ax2[0, 0].grid()
    ax2[0, 0].set_title("resultA-B")

    ax2[0, 1].plot(gradA[1][half], label='gA')
    ax2[0, 1].plot(gradB[1][half], label='gB')
    ax2[0, 1].plot(quotA[half], label='qA')
    ax2[0, 1].plot(quotB[half], label='qB')

    ax2[0, 1].legend()
    ax2[0, 1].grid()
    ax2[0, 1].set_title("gradientA-B")

    ffA = fft.fftshift(fft.fft2(rA, (rA.shape[1]*2, rA.shape[0]*2)))
    ffB = fft.fftshift(fft.fft2(rB, (rB.shape[1]*2, rB.shape[0]*2)))
    ffO = fft.fftshift(fft.fft2(offset, (offset.shape[1] * 2, offset.shape[0] * 2)))
    #ffA = fft.fft2(rA)
    #ffB = fft.fft2(rB)

    #### calculate global shift by cross correlate:
    #shift, error, diffphase = phase_cross_correlation(rA, rB)
    # ecc = signal.correlate2d(rA, rB,  mode='full', boundary='symm')
    #x = range(0, ecc.shape[0])
    #y = range(0, ecc.shape[1])
    #(X, Y) = np.meshgrid(x, y)
    #x_coord = (X * ecc).sum() / ecc.sum().astype("float")
    #y_coord = (Y * ecc).sum() / ecc.sum().astype("float")
    #shift_x = ((ecc.shape[1] - 1) / 2 - x_coord) * (a.shape[1] / (2*rA.shape[1]))
    #shift_y = ((ecc.shape[0] - 1) / 2 - y_coord) * (a.shape[0] / (2*rA.shape[0]))


    #ax2[1,3].imshow(rA)
    #ax2[1,4].imshow(rB)
    #ax2[0, 2].imshow(np.absolute(ffB))
    ax2[0, 2].imshow(np.absolute(ffA))
    ax2[0, 4].set_title("ffA")
    ax2[0, 3].imshow(np.absolute(ffB))
    ax2[0, 4].set_title("ffB")

    ffOffs = (ffB*ffA.conj())/np.absolute(ffB*ffA.conj())

    #ax2[0, 4].imshow(np.absolute(ffA-ffB))
    ax2[0, 4].imshow(180*np.angle(ffOffs)/np.pi)
    ax2[0, 4].set_title("offset-angle")
    ax2[1, 0].set_title("offset-mag")

    ax2[1, 1].imshow(np.angle(ffO))
    ax2[1, 1].set_title("offset")
    #ax2[1, 2].imshow(fft.ifft2(fft.ifftshift((ffOffs))).real)
    ax2[1, 2].set_title("ifft-offset")

    rA = pyramidA[selection-1]
    rB = pyramidB[selection-1]

    newOffs = np.zeros((rA.shape[1]*2, rA.shape[0]*2), dtype=np.complex128)
    newFreq = newOffs.copy()

    o_rows, o_cols = ffOffs.shape[:2]
    no_rows, no_cols = newOffs.shape[:2]
    ffCorr = (ffA * ffB.conj())/np.absolute(ffA*ffB.conj())
    # Similar to ffO: ffFreq = (ffB - ffA)*(-1)
    ffFreq = (ffB - ffA.conj()) / np.exp(-1j * np.angle(ffCorr))

    he, wi = ffCorr.shape[:2]

    test = np.zeros((he, wi), dtype='complex128')
    #test[128 - 7, 128 - 7] = 100 * (np.cos(0) + 1j * np.sin(0))

    test[128 + 1, 128 - 7] = he * wi * (np.cos(1) + 1j * np.sin(1))
    #test[128 + 7, 128 + 7] = 100 * (np.cos(0) + 1j * np.sin(0))
    #test[128 - 7, 128 + 7] = 100 * (np.cos(-1) + 1j * np.sin(-1))
    #test[128 - 14, 128 - 7] = 100 * (np.cos(0) + 1j * np.sin(0))
    #test[128 + 7, 128 - 14] = 100 * (np.cos(1) + 1j * np.sin(1))
    #test[128 + 7, 128 + 14] = 100 * (np.cos(0) + 1j * np.sin(0))
    #test[128 - 14, 128 + 7] = 100 * (np.cos(-1) + 1j * np.sin(-1))
    #test[128 + 14, 128 - 7] = 100 * (np.cos(0) + 1j * np.sin(0))
    #test[128 - 7, 128 - 14] = 100 * (np.cos(1) + 1j * np.sin(1))
    #test[128 - 7, 128 + 14] = 100 * (np.cos(0) + 1j * np.sin(0))
    #test[128 + 14, 128 + 7] = 100 * (np.cos(-1) + 1j * np.sin(-1))


    # ax2[1, 0].imshow(fft.ifft2(fft.ifftshift(ffCorr)).real)
    # ax2[1, 0].imshow(fft.ifft2(fft.ifftshift(test)).real)
    # rng = np.asarray(Image.open("ring.png").convert('L'))
    ax2[1, 0].imshow(fft.ifft2(fft.ifftshift(100*(rng+1j*rng))).real)

    #ax2[1, 0].imshow(180*np.angle((ffA/ffB)/np.absolute(ffB/ffA)*(np.absolute(ffB) > 1e-6))/np.pi)
    #pad with zeros:
    newOffs[no_rows//2-o_rows//2:no_rows//2+o_rows//2,no_cols//2-o_cols//2:no_cols//2+o_cols//2] = ffCorr
    newFreq[no_rows//2-o_rows//2:no_rows//2+o_rows//2,no_cols//2-o_cols//2:no_cols//2+o_cols//2] = ffFreq
    scaling = (pyramidA[selection-1].max() - pyramidA[selection-1].min()) / (pyramidA[selection].max() - pyramidA[selection].min())
    # fffA=fourier_shift(ffA, (15,15))
    ffA = fft.fftshift(fft.fft2(rA, (rA.shape[1]*2, rA.shape[0]*2)))
    ffB = fft.fftshift(fft.fft2(rB, (rB.shape[1]*2, rB.shape[0]*2)))

    ffOffs = (ffA/ffB)*np.absolute(ffB)
    freqSc = (ffOffs.max()-ffOffs.min())/(newFreq.max()-newFreq.min())


    #ax2[1, 2].imshow(fft.ifft2(fft.ifftshift((ffA-(newOffs.real*offsSc + 1j * newOffs.imag*offsSc)))).real)
    ax2[1, 2].imshow(fft.ifft2(fft.ifftshift((ffA*newOffs.conj()+newFreq*freqSc))).real[:rA.shape[1], :rA.shape[0]])
    ax2[1,3].imshow(rA)
    ax2[1,4].imshow(rB)

    ax.imshow(composite_image)
    plt.show()
    plt.waitforbuttonpress()