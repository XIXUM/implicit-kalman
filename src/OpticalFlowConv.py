"""
Tests the Kernels find local minimas and the offset and plots it
"""

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.tri as tri
import matplotlib.colors as cols
from matplotlib import cm
import numpy as np
from PIL import Image
from scipy import datasets, ndimage, spatial, fft
from scipy.signal import blackman
from FFTkernel import FFTkernel2D

def zcr(x):
    xzeros = np.expand_dims(np.zeros(x[0].shape[1]), axis =1)
    yzeros = np.expand_dims(np.zeros(x[0].shape[0]), axis=0)

    xdiff = np.append(np.diff(np.sign(x[1]), axis=1), xzeros, axis=1)+np.append(np.diff(np.sign(x[1]), axis=0), yzeros, axis=0)
    ydiff = np.append(np.diff(np.sign(x[0]), axis=0), yzeros,axis=0)+np.append(np.diff(np.sign(x[0]), axis=1), xzeros, axis=1)
    # np.where(np.diff(np.sign(gradA[0]), axis=0) * np.diff(np.sign(gradA[1]), axis=1) != 0)[0]
    return np.argwhere(xdiff+ydiff != 0)

def zcr1d(x):
    # yzeros = np.expand_dims(np.zeros(x.shape[0]), axis=0)

    xdiff = np.append(np.diff(np.sign(x)), [0])
    return np.argwhere(xdiff != 0)


def normZCR(x):
    xMin = np.min(x)
    xMax = np.max(x)
    xn = (2 * x - (xMax + xMin)) / (xMax - xMin)
    return zcr1d(xn)

def createTrackers(ra, rb, ga, gb):
    height = ra.shape[1]
    width = ra.shape[0]

    for i in range(0, width):
        allXA = np.array([])
        allXB = np.array([])
        allYA = np.array([])
        allYB = np.array([])
        lineA = ga[1][i]
        lineB = gb[1][i]
        ptxA = normZCR(lineA)
        ptxB = normZCR(lineB)
        ptxA = thresholdRem(ptxA, ra, i, 0.05)
        ptxB = thresholdRem(ptxB, rb, i, 0.05)
        if ptxA.shape[0] > 0:
            allXA = np.append(allXA, np.hstack(ptxA, np.linspace(i,0, ptxA.shape[0])))
        if ptxB.shape[0] > 0:
            allXB = np.append(allXB, np.hstack(ptxB, np.linspace(i, 0, ptxB.shape[0])))

        if ptxA.shape[0] > ptxB.shape[0]:
            unifyPts(ptxA, ptxB, i)
        elif ptxA.shape[0] < ptxB.shape[0]:
            unifyPts(ptxB, ptxA, i)

    # for j in range(0, height):
    #     colA = ga[0][:,i]
    #     colB = ga[0][:,i]
    #     ptyA = normZCR(colA)
    #     ptyA = normZCR(colB)
    #
    #     if ptyA.shape[0] > 0:
    #         allYA = np.append(allYA, np.hstack(ptyA, np.linspace(i, 0, ptyA.shape[0])))
    #     if ptyB.shape[0] > 0:
    #         allYB = np.append(allYB, np.hstack(ptyB, np.linspace(i, 0, ptyB.shape[0])))


def unifyPts(arr, ref, index, thr, axis='x'):
    pass

def thresholdRem(pts, ref, i, axis = 'x'):
    return pts


def laplacianPyramid(kernels):

    last = kernels[str(resSteps[0])].convolute()
    results = {str(resSteps[0]): last}
    for i in range(1, len(resSteps)):
        new = kernels[str(resSteps[i])].convolute()
        results[str(resSteps[i])] = new - last
        last = new

    return results


if __name__ == '__main__':

    # ------------Step 1:
    # load kernel map for gaussian convolution
    kImage = Image.open("gauss512.tif")

    # ------------Step 2: gaussian image Pyramid
    # create gaussian image pyramid
    resSteps = [256, 128, 64, 32, 16, 8, 4, 2 ]

    layer ='256'

    kernelMaps = { str(r) : np.asarray(kImage.resize((r,r))) for r in resSteps}

    # ------------ Step 2.5: load test images A and B
    # ac = datasets.ascent()
    a = np.asarray(Image.open("testIm3_A.png").convert('L'))
    b = np.asarray(Image.open("testIm3_B.png").convert('L'))

    kernelsA = {str(r): FFTkernel2D(a, kernelMaps[str(r)]) for r in resSteps}
    kernelsB = {str(r): FFTkernel2D(b, kernelMaps[str(r)]) for r in resSteps}

    # ------------ Step 3: Convolute:
    # create gaussian image pyramid in buffer
    #resultsA = {str(r): kernelsA[str(r)].convolute() for r in resSteps}
    #resultsB = {str(r): kernelsB[str(r)].convolute() for r in resSteps}

    resultsA = laplacianPyramid(kernelsA)
    resultsB = laplacianPyramid(kernelsB)

    #prepare plots
    fig, ((ax_1, ax_2, ax_4, ax_8, ax_16), (ax_32, ax_64, ax_128, ax_256, ax_512)) = plt.subplots(2, 5, figsize=(15, 6), squeeze=True, sharex=False,
                       sharey=False)
    fig3D, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # ----------- Step 4: analyze phase shift for Resolution 256 (lower resolution make no sense when kernel > input)
    ax_1.imshow(resultsA[layer], cmap='gray')
    ax_1.set_title('OriginalA')

    # create gradient of luminance of gaussian result for phase shift
    gradA = np.gradient(resultsA[layer])
    gradB = np.gradient(resultsB[layer])

    offsetA = resultsA[layer] - resultsB[layer]

    gradientOffs = np.gradient(offsetA)
    gradientOffsImg = np.dstack((gradientOffs[0],gradientOffs[1]))


    #b512 = fftc.convolute()
    #ax_512.imshow(resultsB['256'], cmap='gray', vmax=255, vmin=0)
    ax_512.set_title('ConvolutedB 256x256')



    #imshow(cols.hsv_to_rgb(
    #    np.dstack((np.absolute(np.angle(fftc) / np.pi + 1) / 2, np.ones(fftc.shape), abs_fftc / abs_fftc.max()))))

    #ax_256.imshow(offsetA, cmap='cool')


    mx = np.arange(0, 256, 1)
    my = np.arange(0, 256, 1)
    mx, my =  np.meshgrid(mx, my)

    #ax.plot_surface(mx, my, offsetA)
    ax.plot_surface(mx, my, np.sqrt(gradA[1]**2 + gradA[0]**2))
    ax.plot_surface(mx, my, np.sqrt(gradB[1]**2 + gradB[0]**2))
    ax_256.set_title('delta')

    #gradA = np.gradient(resultsA['256'])

    gradImA = np.dstack((gradA[0], gradA[1]))
    gradImB = np.dstack((gradB[0], gradB[1]))
    #grad_abs = np.absolute(gradImA)



    # gradDiffIm = (gradImA.real - gradImB.real)+1j*(gradImA.imag - gradImB.imag)
    gradDiffIm = gradImA - gradImB
    # gradA2u = np.gradient(gradImA.real)
    # gradA2v = np.gradient(gradImA.imag)
    gradA2 = np.gradient(gradImA)
    gradDiff2 = np.gradient(gradDiffIm)


    # gradA2uvImg = gradA2u[1] + 1j * gradA2v[0]
    gradA2Img = np.dstack((gradA2[0], gradA2[1], gradA2[2]))
    gradDiff2Img = np.dstack((gradDiff2[0], gradDiff2[1], gradDiff2[2]))
    ax_128.imshow(np.sqrt(gradDiffIm[:,:,0]**2 + gradDiffIm[:,:,1]**2), cmap='gray')
    #ax_256.imshow(np.absolute((gradA2Img[:, :, 2]  + 1j * gradA2Img[:, :, 3]) # * (gradA2Img[:, :, 2] + 1j*gradA2Img[:, :, 3])     # + gradA2Img[:, :, 4] ** 2 + gradA2Img[:, :, 5] ** 2
     #               ),
      #            cmap='gray')

   # ax_256.imshow(np.sqrt(gradA2Img[:, :, 0]**2  +  gradA2Img[:, :, 1]**2 + gradA2Img[:, :, 2]**2  +  gradA2Img[:, :, 3]**2), cmap = 'gray')
    ax_256.imshow((np.absolute((gradA2Img[:,:,1] + 1j * gradA2Img[:,:,0]+gradA2Img[:,:,2] + 1j * gradA2Img[:,:,3])))/2)

    ax_128.set_title('GradientA Magnitude 256x256')

    ax_64.set_title('FFT A')
    ax_64.legend()


    ax_32.imshow(np.angle(gradientOffsImg[:,:,1] + 1j * gradientOffsImg[:,:,0], deg = True))
    ax_32.set_title('Gradient-Angle')

    nvec = 20  # Number of vectors to be displayed along each image dimension
    nl, nc = a.shape
    step = max(nl // nvec, nc // nvec)

    y, x = np.mgrid[:nl:step, :nc:step]
    np.sqrt(gradDiffIm[:, :, 0] ** 2 + gradDiffIm[:, :, 1] ** 2)
    u_ = gradDiffIm[::step, ::step, 1]
    v_ = gradDiffIm[::step, ::step, 0]


    ax_32.quiver(x, y, u_, v_, color='r', units='dots',
               angles='xy', scale_units='xy', lw=3)


    # pxOffs = np.sqrt(gradA[1][128]**2 - offsetA[128]**2)
    ax_8.plot(gradA[1][128], label='gradA')
    ax_8.plot(gradB[1][128], label='gradB')
    ax_8.plot(gradientOffs[1][128], label='gradOffs')
    ax_8.plot(gradDiffIm[128,:, 1], label='1/grOff')
    ax_8.plot(gradA2Img[128, :,0 ]*10, label='1/gr2iReal')
    ax_8.plot(np.absolute(gradA2[1][128]) * 9, label='gr2Abs')
    ax_64.plot(gradA2[0][128,:,1], label='gr2(0)')
    ax_64.plot(gradA2[1][128, :, 1], label='gr2(1)')
    ax_64.plot(gradA2[2][128,:,1], label='gr2(3)')
    ax_64.plot(gradDiff2[0][128, :, 1], label='gd2(1)')
    ax_64.plot(gradDiff2[1][128, :, 1], label='gd2(2)')
    ax_64.plot(gradDiff2[2][128, :, 1], label='gd2(3)')
    ax_8.plot(np.sqrt(gradDiffIm[128,:,0]**2 + gradDiffIm[128,:,1]**2)/np.sqrt(gradA2Img[128,:,0]**2 + gradA2Img[128,:,1]**2 + gradA2Img[128,:,2]**2 + gradA2Img[128,:,3]**2), label='norm')
    ax_64.grid()
    # ax_8.imshow(gradB[1], cmap='gray')
    ax_8.set_title('gradB - X')
    ax_8.legend()
    ax_8.grid()

    ax_4.plot(resultsA[layer][65], label='a65')
    ax_4.plot(resultsB[layer][65], label='b65')
    ax_4.plot(gradA[1][65], label='gA65')
    ax_4.plot(offsetA[65], label='off56')
    ax_4.plot(resultsA[layer][128], label='a65')
    ax_4.plot(resultsB[layer][128], label='b65')
    ax_4.plot(gradA[1][128], label='gA65')
    ax_4.plot(offsetA[128], label='off56')
    ax_4.set_title('Result B')
    ax_4.legend()
    ax_4.grid()

    gaMin = np.min(gradA[1][128])
    gaMax = np.max(gradA[1][128])
    gbMin = np.min(gradB[1][128])
    gbMax = np.max(gradB[1][128])
    raMin = np.min(resultsA[layer][128])
    raMax = np.max(resultsA[layer][128])
    rbMin = np.min(resultsB[layer][128])
    rbMax = np.max(resultsB[layer][128])

    ra = (2 * resultsA[layer][128]-(raMax+raMin)) / (raMax-raMin)
    rb = (2 * resultsB[layer][128]-(rbMax+rbMin)) / (rbMax-rbMin)
    gdoMin = np.min(gradientOffs[0][128])
    gdoMax = np.max(gradientOffs[0][128])
    ga = (2*gradA[1][128]) / (gaMax-gaMin)
    gb = (2*gradB[1][128]) / (gbMax-gbMin)
    ax_2.plot(ga, label='gradA')
    ax_2.plot(gb, label='gradB')
    ax_2.grid()

    phA = np.arctan2(
        (resultsA[layer][128]-resultsA[layer][128].max())/(resultsA[layer][128].max() - resultsA[layer][128].min()),
        2*gradA[1][128]/(gradA[1].max()-gradA[1].min()))
    phB = np.arctan2(
        (resultsB[layer][128] - resultsB[layer][128].max()) / (resultsB[layer][128].max() - resultsB[layer][128].min()),
        2 * gradB[1][128] / (gradB[1].max() - gradB[1].min()))
    #ax_2.plot(phA, label='phaseA')
    #ax_2.plot(phB, label='phaseB')
    shift = 180*((phB - phA + np.pi)%(2*np.pi) - np.pi)/(np.pi)

    #ax_2.plot(shift, label='phaseB')

    ax_2.plot(gradientOffs[1][128] - gradA[1][128])

    ax_2.plot((gradientOffs[1][128])/(gdoMax-gdoMin), label='gradOffs')
    rarbOffs = ra - rb

    #find zero crossing points of gradient in horizontal line => climax / gradient descent
    ptxA = normZCR(gradA[1][128])
    ptxB = normZCR(gradB[1][128])
    # find zero crossing points of gradient in vertical line
    ptyA = normZCR(gradA[0][:,128])
    ptyB = normZCR(gradB[0][:,128])
    ffa2D = fft.fft2(gradImA)
    ffb2D = fft.fft2(gradImB)

    ax_16.imshow(resultsB[layer], cmap='gray')


    #ax_512.imshow(np.sqrt(gradA2[0][:,:,0]**2 + gradA2[0][:,:,1]**2 + gradA2[1][:,:,0]**2 + gradA2[1][:,:,1]**2), cmap='gray') #, vmin='0', vmax='70')
    ax_512.imshow(2*np.absolute((gradDiffIm[:,:,0] + 1j* gradDiffIm[:,:,1]) / (np.absolute((gradA2Img[:,:,1] + 1j * gradA2Img[:,:,0]+gradA2Img[:,:,2] + 1j * gradA2Img[:,:,3])))),
        cmap='gray', vmin='0', vmax='70')

    # wavelet fourier, find phase shift between gradient a and b
    # N = 256
    # T = 1.0 / 512
    # xx = np.linspace(0.0, N * T, N, endpoint=False)
    # yy = np.sin(50.0 * 2.0 * np.pi * xx) + 0.5 * np.sin(80.0 * 2.0 * np.pi * xx)
    # w = blackman(N)
    # yf = fft.fft(yy)
    # xf = fft.fftfreq(N, T)[:N // 2]
    # ffA =  fft.fft(ga*w)
    # ffAf = fft.fftfreq(N, T)[0:N//2]
    # ffB = fft.fft(gb * w)
    # ffaAngle = 180*np.angle(ffA)/np.pi
    # ffbAngle = 180*np.angle(ffB)/np.pi
    #ffBf = fft.fftfreq(N, T)[0:N // 2]

    # compare gradient curve of line 128 between gradient a and b
    #ax_64.plot(ga, label='ga')
    #ax_64.plot(gb, label='gb')

    # ax_64.plot(rarbOffs, label='offsRaRb')
    ax_2.legend()
    ax_2.set_title("Gradients")
    ax_64.legend()


    fig.show()
    fig3D.show()
    plt.waitforbuttonpress()