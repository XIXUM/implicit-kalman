import math

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scipy import fft, signal

def pyramidFlow(ffA, ffB, angularMatr, radialMatr):
    halfPi = np.pi / 2
    rows, cols = ffA.shape
    halfX = cols // 2
    # halfY = rows // 2
    aPr = 4                     # angular recision
    rSc = 32                    # radial scale / eccentricity
    r = (halfX * 0.5) / rSc     # radial filter location
    hSqrt = np.sqrt(0.5)
    sign = [-1, -1, 1, 1]
    vuMap = np.zeros((2,ffA.shape[-2], ffA.shape[-1]))
    shift = []
    for j in range(0,4):
        aStep = halfPi * j * 0.5
        anglularFilt = np.cos(np.clip((angularMatr + aStep) * aPr, -halfPi, halfPi))
        rRange = ((rSc * radialMatr / halfX)) - 1
        radialFilt = np.cos(np.clip(rRange * np.pi, -halfPi, halfPi))

        iwA = fft.ifft2(fft.ifftshift(ffA * radialFilt * anglularFilt))
        iwB = fft.ifft2(fft.ifftshift(ffB * radialFilt * anglularFilt))
        iwC = iwB * iwA.conj()
        phaseSh = np.angle(iwC)

        shift += [sign[j]*(cols * phaseSh) / (2 * np.pi * r)]

    shift[0] += (shift[1] + shift[3])*hSqrt
    shift[2] += -(shift[1] - shift[3])*hSqrt
    vuMap[0] = shift[2] / 2
    vuMap[1] = shift[0] / 2


    return np.moveaxis(vuMap, 0, -1)



if __name__ == '__main__':


    ### --- load Testimages

    a = np.asarray(Image.open("relief0.png").convert('L'))
    b = np.asarray(Image.open("relief0_ro1.png").convert('L'))

    c = np.stack((a,b))

    rows, cols = a.shape

    #pyramidA = tuple(pyramid_laplacian(a, downscale=2, channel_axis=None, sigma=1.2, order=2))
    #pyramidB = tuple(pyramid_laplacian(b, downscale=2, channel_axis=None, sigma=1.2, order=2))


    # fig, ax = plt.subplots()



### create Parameter Plots ...
    fig2, ax2 = plt.subplots(2, 5, figsize=(15, 10), squeeze=True, sharex=False, sharey=False)
    halfX = cols // 2
    halfY = rows // 2

    gradA = np.gradient(a)
    gradB = np.gradient(b)
    offset = b - a
    offsGrad = np.gradient(offset)
    #quotA = offset / gradA[1]
    #quotB = offset / gradB[1]

### Window
    x = np.linspace(0, 255, 256)
    y = np.linspace(0, 255, 256)
    window1d = np.abs(np.blackman(5))
    window2d = np.sqrt(np.outer(window1d, window1d))

    mx, my = np.meshgrid(x, y)
    Z = fft.fftshift(fft.fft2(window2d, (cols, rows)))
    fig3D, ax3D = plt.subplots(subplot_kw={"projection": "3d"})
    ax3D.plot_surface(mx, my, Z.real)
    ax3D.plot_surface(mx, my, Z.imag)

    #rampX = np.aramp()
    #rampX =
    windowSh = np.zeros((cols,rows))
    windowSh[40:40+window2d.shape[0], 40:40+window2d.shape[1]] = window2d
    filter = np.exp(-1j * ((mx-halfX) * 0.589 + (my-halfY) * 0.589))
    Zo = Z * filter
    # Zof = np.roll(Z, (40,40))
    Zof = fft.fftshift(fft.fft2(windowSh))

    # ax2[1, 0].imshow(Zo.real)
    #ax2[1, 1].imshow(fft.ifft2(fft.ifftshift(Zof)).real)
    # ax2[1, 1].imshow(Zo.imag)
    # ax2[1, 1].imshow(np.angle(Z))
    #ax2[1, 2].imshow(np.angle(Z))
    #ax2[1, 2].imshow(Z.real)
    #ax2[1, 3].imshow(fft.ifft2(fft.ifftshift(Zo)).real)
    #ax2[1, 3].imshow(Zo.real)

# Filter

    angleMatr = np.angle((mx-128) + 1j * (my-128))
    dist = np.sqrt((mx-128)**2 + (my-128)**2)
    halfPi = np.pi / 2
    anglularFilt = np.cos(np.clip((angleMatr+halfPi*0.5)*4, -halfPi, halfPi))
    radialFilt = np.cos(np.clip(((32 * np.pi * dist / halfX)) - np.pi, -halfPi, halfPi))

    #ax2[0, 0].plot(a[halfX], label='rA')
    #ax2[0, 0].plot(b[halfX], label='rB')
    #ax2[0, 0].plot(offset[halfX], label='offs')
    ax2[0, 0].legend()
    ax2[0, 0].grid()
    ax2[0, 0].set_title("resultA-B")

    #ax2[0, 1].plot(gradA[1][halfX], label='gA')
    #ax2[0, 1].plot(gradB[1][halfX], label='gB')
    #ax2[0, 1].plot(quotA[halfX], label='qA')
    #ax2[0, 1].plot(quotB[halfX], label='qB')


    ax2[0, 1].set_title("gradientA-B")

    ffA = fft.fftshift(fft.fft2(a))
    ffB = fft.fftshift(fft.fft2(b))
    ffO = fft.fftshift(fft.fft2(offset))
    ffC = fft.fftshift(fft.fftn(c)) #(4,rows, cols)))

    #### calculate global shift by cross correlate:
    #shift, error, diffphase = phase_cross_correlation(rA, rB)
    # ecc = signal.correlate2d(rA, rB,  mode='full', boundary='symm')

    #ax2[0, 2].imshow(fft.ifft2(fft.ifftshift(ffA * filter.conj())).real)
    ax2[0, 2].set_title("ffA")
    #ax2[0, 3].imshow(np.absolute(fft.ifft2(fft.ifftshift((ffA-Zof)))))
    ax2[0, 3].set_title("ffB")
    ax2[0, 3].imshow(ffC[0].imag)

    ffOffs = (ffB*ffA.conj())/np.absolute(ffB*ffA.conj())

    #w1 = np.zeros((rows,cols), dtype='complex128')
    #w2 = w1.copy()
    #w1[124:132, 124:132] = ffC[0, 124:132, 124:132]
    #w2[120:136, 144:160] = ffC[0, 120:136, 144:160]
    #w1 = ffC[0]
    #iw1 = fft.ifft2(fft.ifftshift(w1))
    #iw2 = fft.ifft2(fft.ifftshift(w2))
    iwA = fft.ifft2(fft.ifftshift(ffA * radialFilt * anglularFilt))
    iwB = fft.ifft2(fft.ifftshift(ffB * radialFilt * anglularFilt))
    iwC = iwB*iwA.conj()

    #ax2[0, 1].plot(iwA[halfX].real, label='iwA(r)')
    #ax2[0, 1].plot(iwA[halfX].imag, label='iwA(i)')
    #ax2[0, 1].plot(iwB[halfX].real, label='iwB(r)')
    #ax2[0, 1].plot(iwB[halfX].imag, label='iwB(i)')
    a_iwA = np.angle(iwA)
    a_iwB = np.angle(iwB)
    a_iwC = np.angle(iwC)

    ax2[0, 1].plot(np.gradient(a_iwA[halfX]), label='iwA(a,x)')
    ax2[0, 1].plot(a_iwB[halfX], label='iwB(a,x)')
    ax2[0, 1].plot(a_iwC[halfX], label='iwC(a,x)')

    ax2[0, 1].legend()
    ax2[0, 1].grid()

    ax2[0, 0].plot(np.gradient(a_iwA[:, halfY]), label='iwA(a,y)')
    ax2[0, 0].plot(a_iwB[:, halfY], label='iwB(a,y)')
    ax2[0, 0].plot(a_iwC[:, halfY], label='iwC(a,y)')
    ax2[0, 0].legend()
    ax2[0, 0].grid()
    ax2[0, 0].set_title("resultA-B")


    # pad = 254
    # iwCubic =np.stack((iwA,iwB))
    # iwPk = np.moveaxis(np.pad(iwCubic,((0,pad),(0,0),(0,0)),mode='edge'),0,-1)
    #
    # fl, fh, fs = 0.2, 0.25, 24
    # F_length = 1001
    # F = fft.fft(fft.ifftshift(signal.firwin(F_length, [fl, fh], fs=fs, pass_zero=False)))
    #
    # dPhi_ff = fft.fft(np.angle(iwPk), axis=-1) * np.broadcast_to(F[:256],iwPk.shape)
    # dPhi = fft.ifft(dPhi_ff, axis=-1)

    #ax2[0, 4].imshow(iw1.real)
    #iC0 = fft.ifft2(fft.ifftshift(ffC[0])).real
    iC1 = fft.ifft2(fft.ifftshift(ffC[1])).real
    #diffC = (+iw1 + iC1) / 2
    ax2[0, 4].imshow(180 * np.angle(ffC[1]*ffC[0].conj()) / np.pi)
    #ax2[1, 2].imshow((np.moveaxis(dPhi, -1, 0)[0]).real)
    #ax2[1, 3].imshow(diffC.real)
    ax2[1, 2].imshow(iwB.real)
    ax2[1, 3].imshow(180*np.angle(iwC)/np.pi)
    ax2[0, 2].imshow(iwA.real)
    ax2[0, 4].set_title("offset-angle")
    ax2[1, 0].set_title("offset-mag")

    #ax2[1, 0].imshow(180*np.angle(ffA)/np.pi)
    #ax2[1, 1].imshow(180*np.angle(ffB)/np.pi)
    # ax2[1, 0].imshow(ffC[0].real)
    # ax2[1, 1].imshow(ffC[1].imag)

    ax2[1, 0].imshow(anglularFilt)
    ax2[1, 1].imshow(radialFilt)

    #ax2[1, 4].imshow(np.angle(ffO))
    #ax2[1, 4].imshow(np.log(np.abs(ffA)))

    vuMap = pyramidFlow(ffA, ffB, angleMatr, dist)

    nvec = 20
    step = max(rows // nvec, cols // nvec)

    vy, vx = np.mgrid[:rows:step, :cols:step]
    u_ = vuMap[::step, ::step, 1]
    v_ = vuMap[::step, ::step, 0]

    #ax2[1, 4].imshow(180*np.angle(vuMap[:,:,1] + 1j * vuMap[:,:,0])/np.pi)
    ax2[1, 4].imshow(np.abs(vuMap[:, :, 1] + 1j * vuMap[:, :, 0]))
    ax2[1, 4].quiver(vx, vy, u_, v_, color='r', units='dots',
               angles='xy', scale_units='xy', lw=3)

    ax2[1, 4].set_title("offset")
    #ax2[1, 2].imshow(fft.ifft2(fft.ifftshift((ffOffs))).real)
    ax2[1, 2].set_title("ifft-offset")

    o_rows, o_cols = ffOffs.shape[:2]

    ffCorr = (ffA * ffB.conj())/np.absolute(ffA*ffB.conj())
    #ax2[0, 3].imshow(fft.ifft2(fft.ifftshift(ffA / Z)).real)

    ffFreq = (ffB - ffA.conj()) / np.exp(-1j * np.angle(ffCorr))

    # he, wi = ffCorr.shape[:2]

    plt.show()
    plt.waitforbuttonpress()