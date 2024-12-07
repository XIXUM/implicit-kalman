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

def displacement(res0, res1):

    if (isinstance(res0, np.ndarray) ):
        deltaM = res0-res1
    else:
        deltaM = res1

# Deprecated attempt to find deltas beween points of image A and B
def createDelta(za, zb, xSize, ySize):
    zas = za.shape
    zbs = zb.shape
    zMatr = np.zeros((za.shape[0], zb.shape[0]))
    deltas = []
    for i in range(0,za.shape[0]):
        for j in range(0,zb.shape[0]):
            dis = np.linalg.norm(za[i]-zb[j])
            zMatr[i,j] = dis
    zaIndex = np.zeros((zMatr.shape[0]))
    zaIndex.fill(-1)
    for i in range(zMatr.shape[0]):
        zaIndex = np.argpartition(zMatr[i,:], zb.shape[0]-1)[0]
        deltas.append((za[i], zb[zaIndex], zMatr[i,zaIndex]))
    if zas > zbs:
        zbIndex = np.linspace(0,zMatr.shape[1],num=zMatr.shape[1])
        extendDeltas(deltas, zb, zbIndex, za, zaIndex, ySize)

    elif zbs > zas:
        zbIndex = np.linspace(0, zMatr.shape[1]-1, num=zMatr.shape[1])
        extendDeltas(deltas, za, zaIndex, zb, zbIndex, xSize)

    return deltas


def extendDeltas(deltas, za, zaIndex, zb, zbIndex, zSize):
    rem = np.setdiff1d(zbIndex, zaIndex)
    for el in range(0, rem.shape[0]):
        newEl = None
        if (zb[el, 0] > zb[el, 1]):
            if zb[el, 0] < zSize / 2:
                newEl = np.array([zb[el, 0], 0])
            else:
                newEl = np.array([zSize, zb[el, 0]])
        else:
            if zb[el, 1] < zSize / 2:
                newEl = np.array([0, zb[el, 1]])
            else:
                newEl = np.array([zb[el, 0], zSize])
        # np.append(za, newEl)
        deltas.append((newEl, zb[el], np.linalg.norm(newEl - zb[el])))

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


if __name__ == '__main__':

    # ------------Step 1:
    # load kernel map for gaussian convolution
    kImage = Image.open("gauss512.tif")

    # ------------Step 2: gaussian image Pyramid
    # create gaussian image pyramid
    resSteps = [256, 128, 64, 32, 16, 8, 4, 2 ]

    kernelMaps = { str(r) : np.asarray(kImage.resize((r,r))) for r in resSteps}

    # ------------ Step 2.5: load test images A and B
    # ac = datasets.ascent()
    a = np.asarray(Image.open("relief0.png").convert('L'))
    b = np.asarray(Image.open("relief0_offs.png").convert('L'))

    kernelsA = {str(r): FFTkernel2D(a, kernelMaps[str(r)]) for r in resSteps}
    kernelsB = {str(r): FFTkernel2D(b, kernelMaps[str(r)]) for r in resSteps}

    # ------------ Step 3: Convolute:
    # create gaussian image pyramid in buffer
    resultsA = {str(r): kernelsA[str(r)].convolute() for r in resSteps}
    resultsB = {str(r): kernelsB[str(r)].convolute() for r in resSteps}

    #prepare plots
    fig, ((ax_1, ax_2, ax_4, ax_8, ax_16), (ax_32, ax_64, ax_128, ax_256, ax_512)) = plt.subplots(2, 5, figsize=(15, 6), squeeze=True, sharex=False,
                       sharey=False)

    fig3D, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #X = np.arange(-5, 5, 0.25)
    #Y = np.arange(-5, 5, 0.25)
    #X, Y = np.meshgrid(X, Y)
    #R = np.sqrt(X ** 2 + Y ** 2)
    #Z = np.sin(R)
    #ax.plot_surface(X, Y, Z, vmin=Z.min() * 2, cmap=cm.Blues)
    #fig3D.show()


    # ----------- Step 4: analyze phase shift for Resolution 256 (lower resolution make no sense when kernel > input)
    ax_1.imshow(a, cmap='gray', vmin=0, vmax=255)
    ax_1.set_title('OriginalA')

    # create gradient of luminance of gaussian result for phase shift
    gradA = np.gradient(resultsA['256'])
    gradB = np.gradient(resultsB['256'])

    # 2D zero crossing points of real and imaginary gradient part
    za = zcr(gradA)
    zb = zcr(gradB)

    # find second order gradient from gradient magnitude zero crossing, because first order might be too few points, but zero crossing is too sensitive / noisy
    zza = zcr(np.gradient(np.absolute(gradA[0] + 1j*gradA[1])))
    zzb = zcr(np.gradient(np.absolute(gradB[0] + 1j*gradB[1])))

    #zas = zan.shape[0]
    za = np.column_stack((255 - za[:][:,0], 255 - za[:][:,1]))
    zb = np.column_stack((255 - zb[:][:, 0], 255 - zb[:][:, 1]))
    zza = np.column_stack((255 - zza[:][:, 0], 255 - zza[:][:, 1]))
    zzb = np.column_stack((255 - zzb[:][:, 0], 255 - zzb[:][:, 1]))
    #za = np.reshape(zan, (zas,2))

    # deltas = createDelta(za,zb, 255, 255)

    points = np.array([[0, 0], [0, 255], [255, 0], [255, 255]])
    points = np.append(points, zb, axis=0)
    dTriang = spatial.Delaunay(points)
    triang = tri.Triangulation(points[:,0], points[:,1], dTriang.simplices)
    z = np.array([5,10,30, 15, 3, 12])
    #ax_8.tripcolor(triang, z, cmap='gray', shading='gouraud')
    ax.plot(points[:,0], points[:,1], 'r+', markersize=2, color='grey')
    # ax_8.triplot(triang)
    #plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    #plt.plot(points[:, 0], points[:, 1])
    #plt.show()

    #delta = za - zb

    qX, qY = np.mgrid[0:1:(1/128),0:1:(1/128)]
    qMap = np.sqrt(qX*qY)

    ramp = np.append(qMap,ndimage.rotate(qMap, -90),axis=1)
    ramp = np.append(ramp,ndimage.rotate(ramp, 180),axis=0)

    #displ2D = ramp*(delta[0,0] + 1j*delta[0,1])

    #ax_16.imshow(resultsA['256'], cmap='gray', vmax=255, vmin=0)
    ax_16.set_title('OffsetGradient-Y')

    vx, vy = np.meshgrid(np.linspace(0, 255, 32),
                       np.linspace(0, 255, 32))

    #gradOx = ndimage.zoom(np.real(displ2D), 0.125)
    #gradOy = ndimage.zoom(np.imag(displ2D), 0.125)

    # ax_8.quiver(vx, vy, gradOx*10, gradOy*10)



    offsetA = resultsA['256'] - resultsB['256']
    #gradOffs =  []
    #gradOffs.append(gradA[0] - gradB[0])
    #gradOffs.append(gradA[1] - gradB[1])
    gradientOffs = np.gradient(offsetA)
    gradientOffsImg = gradientOffs[1] + 1j * gradientOffs[0]


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
    ax.plot_surface(mx, my, np.absolute(gradA[1] + 1j* gradA[0]))
    ax.plot_surface(mx, my, np.absolute(gradB[1] + 1j* gradB[0]))
    ax_256.set_title('delta')

    #gradA = np.gradient(resultsA['256'])

    gradImA = gradA[1] + 1j * gradA[0]
    gradImB = gradB[1] + 1j * gradB[0]
    #grad_abs = np.absolute(gradImA)
    ax_256.imshow(gradImB.imag, cmap='gray')

    # ax_128.imshow(grad, vmin = -1, vmax = 1)
    #ax_128.imshow(offsetA/gradA[1], cmap='cool', vmin=-50, vmax=50)
    #ax_128.plot(offsetA[128]/gradA[1][128])
    #ax_128.imshow(np.absolute(gradientOffsImg)*64,cmap='gray')
    # magABs = np.cos(np.arctan(gradientOffs[1])) * offsetA + 1j * np.cos(np.arctan(gradientOffs[0])) * offsetA

    #base = plt.gca().transData
    #rot = transforms.Affine2D().rotate_deg(20)

    # gradDiffIm = (gradImA.real - gradImB.real)+1j*(gradImA.imag - gradImB.imag)
    gradDiffIm = gradImA - gradImB
    gradA2u = np.gradient(gradImA.real)
    gradA2v = np.gradient(gradImA.imag)
    gradA2 = np.gradient(gradImA)
    # gradA2dif = np.gradient(gradDiffIm)
    # gradA2abs = np.gradient(gradImA))

    gradA2uvImg = gradA2u[1] + 1j * gradA2v[0]
    gradA2Img = gradA2[1] + 1j * gradA2[0]
    ax_128.imshow(np.absolute(gradDiffIm), cmap='gray')
    #ax_128.imshow(np.absolute(gradA[1] + 1j* gradA[0]))
    #ax_128.plot(za[:][:,0],za[:][:,1], 'r+')
    #ax_128.plot(zza[:][:, 0], zza[:][:, 1], 'b+')
    # ax_128.text(0.5, 0.5, 'I should be rotated', ha='center', va='center')
    # t = ax_128.get_transform()
    # ax_128.set_transform(t.transform(transforms.Affine2D().rotate_deg(45)))

    #ax_128.imshow(np.angle(np.arctan2(gradA[1], gradB[1])))
    ax_128.set_title('GradientA Magnitude 256x256')

    # ax_64.imshow(cols.hsv_to_rgb(np.dstack((np.absolute(np.angle(gradIm)/np.pi+1)/2,np.ones(gradIm.shape), np.ones(gradIm.shape)))))  # grad_abs/grad_abs.max()))))
    # ax_64.plot(offsetA[128], label='offs128')
    # ax_64.plot(offsetA[60], label='offs113')
    #ax_64.plot(np.cos(np.arctan2(gradA[1][128], gradB[1][128])) * offsetA[128], label='spat')
    # ax_64.plot(np.absolute(resultsA['256'][128] + 1j * resultsB['256'][128]), label='mag')
    ax_64.set_title('FFT A')
    # ax_64.plot(ffA, label='ffA')
    # ax_64.semilogy(ffA[1:N // 2], 2.0 / N * np.abs(ffAf[1:N // 2]), '-b')
    #
    ax_64.legend()


    #ax_32.imshow(np.absolute(gradImB), cmap='gray')
    # ax_32.imshow(cols.hsv_to_rgb(np.dstack((np.absolute(np.angle(gradientOffsImg)/np.pi+1)/2,np.ones(gradientOffsImg.shape), np.ones(gradientOffsImg.shape)))))  # grad_abs/grad_abs.max()))))
    ax_32.imshow(np.angle(gradientOffsImg, deg = True))
    #ax_32.imshow(np.absolute(gradB[1] + 1j* gradB[0]))
    #ax_32.plot(zb[:][:,0],zb[:][:,1], 'r+')
    #ax_32.plot(zzb[:][:, 0], zzb[:][:, 1], 'b+')
    # t = ax_32.get_transform()
    # ax_32.set_transform(t.transform(transforms.Affine2D().rotate_deg(180)))
    ax_32.set_title('Gradient-Angle')

    # pxOffs = np.sqrt(gradA[1][128]**2 - offsetA[128]**2)
    ax_8.plot(gradA[1][128], label='gradA')
    ax_8.plot(gradB[1][128], label='gradB')
    ax_8.plot(gradientOffs[1][128], label='gradOffs')
    ax_8.plot(gradDiffIm[128].real, label='1/grOff')
    ax_8.plot(gradA2Img[128].real*10, label='1/gr2iReal')
    ax_8.plot(np.absolute(gradA2[1][128]) * 9, label='gr2Abs')
    ax_8.plot(gradA2[1][128].real * 8, label='gr2')
    ax_8.plot(gradA2uvImg[128].real * 8, label='gri2uv')
    ax_8.plot(gradDiffIm[128].real/np.absolute(gradA2Img[128]), label='norm')
    # ax_8.imshow(gradB[1], cmap='gray')
    ax_8.set_title('gradB - X')
    ax_8.legend()

    #ax_4.imshow(resultsB['256'], cmap='gray', vmin = 0, vmax = 255)
    # ax_4.plot(offsetA[250], label='off250')
    #ax_4.plot(offsetA[128], label='off128')
    #ax_4.plot(np.angle(gradA[1][128] + 1j * gradB[1][128], deg = True), label='angleX')
    ax_4.plot(resultsA['256'][128], label='a')
    ax_4.plot(resultsB['256'][128], label='b')
    ax_4.plot(offsetA[128], label='off')
    # ax_4.plot(np.angle(np.gradient(gradA[1][128])+1j * np.gradient(gradB[1][128]),deg=True), label='delt')
    ax_4.set_title('Result B')
    ax_4.legend()

    # ax_2.plot(offsetA[250], label='offsetA')
    #ax_2.plot(gradientOffs[0][250], label='resultA')

    # noralize between -1 and 1 per horizontal line
    gaMin = np.min(gradA[1][128])
    gaMax = np.max(gradA[1][128])
    gbMin = np.min(gradB[1][128])
    gbMax = np.max(gradB[1][128])
    raMin = np.min(resultsA['256'][128])
    raMax = np.max(resultsA['256'][128])
    rbMin = np.min(resultsB['256'][128])
    rbMax = np.max(resultsB['256'][128])

    # create normalized gradient between -1 and 1
    ra = (2 * resultsA['256'][128]-(raMax+raMin)) / (raMax-raMin)
    rb = (2 * resultsB['256'][128]-(rbMax+rbMin)) / (rbMax-rbMin)
    gdoMin = np.min(gradientOffs[0][128])
    gdoMax = np.max(gradientOffs[0][128])
    ga = (2*gradA[1][128]- (gaMax+gaMin)) / (gaMax-gaMin)
    gb = (2*gradB[1][128] - (gbMax+gbMin)) / (gbMax-gbMin)
    ax_2.plot(ga, label='gradA')
    ax_2.plot(gb, label='gradB')
    # phA = np.arctan2(ra,ga)
    phA = np.arctan2(
        (resultsA['256'][128]-resultsA['256'][128].max())/(resultsA['256'][128].max() - resultsA['256'][128].min()),
        2*gradA[1][128]/(gradA[1].max()-gradA[1].min()))
    phB = np.arctan2(
        (resultsB['256'][128] - resultsB['256'][128].max()) / (resultsB['256'][128].max() - resultsB['256'][128].min()),
        2 * gradB[1][128] / (gradB[1].max() - gradB[1].min()))
    #phB = np.arctan2(rb,gb)
    ax_2.plot(phA, label='phaseA')
    ax_2.plot(phB, label='phaseB')
    shift = 180*((phB - phA + np.pi)%(2*np.pi) - np.pi)/(np.pi)

    ax_2.plot(shift, label='phaseB')

    ax_2.plot((gradientOffs[1][128])/(gdoMax-gdoMin), label='gradOffs')
    # t = ax_2.get_transform()
    # ax_2.set_transform(t.transform(transforms.Affine2D().rotate_deg(45)))
    rarbOffs = ra - rb

    #find zero crossing points of gradient in horizontal line => climax / gradient descent
    ptxA = normZCR(gradA[1][128])
    ptxB = normZCR(gradB[1][128])
    # find zero crossing points of gradient in vertical line
    ptyA = normZCR(gradA[0][:,128])
    ptyB = normZCR(gradB[0][:,128])
    ffa2D = fft.fft2(gradImA)
    ffb2D = fft.fft2(gradImB)

    ax_16.imshow(b)


    ax_512.imshow(np.sqrt(gradA2[0].imag**2+gradA2[1].real**2+gradA2[0].real**2+gradA2[1].imag**2), cmap='gray') #, vmin='0', vmax='70')

    # wavelet fourier, find phase shift between gradient a and b
    N = 256
    T = 1.0 / 512
    xx = np.linspace(0.0, N * T, N, endpoint=False)
    yy = np.sin(50.0 * 2.0 * np.pi * xx) + 0.5 * np.sin(80.0 * 2.0 * np.pi * xx)
    w = blackman(N)
    yf = fft.fft(yy)
    xf = fft.fftfreq(N, T)[:N // 2]
    ffA =  fft.fft(ga*w)
    ffAf = fft.fftfreq(N, T)[0:N//2]
    ffB = fft.fft(gb * w)
    ffaAngle = 180*np.angle(ffA)/np.pi
    ffbAngle = 180*np.angle(ffB)/np.pi
    #ffBf = fft.fftfreq(N, T)[0:N // 2]

    ax_64.plot(ffAf, 2.0 / N * np.abs(ffA[0:N // 2]))
    ax_64.plot(ffAf, 2.0 / N * np.abs(ffB[0:N // 2]))
    #ax_64.semilogy(ffAf[1:N // 2], 2.0 / N * np.abs(ffA[1:N // 2]), '-b')
    #ax_64.semilogy(ffAf[1:N // 2], 2.0 / N * np.abs(ffB[1:N // 2]), '-r')
    #ax_64.plot(ffA, '-r')
    #ax_64.plot(ffB, '-b')

    # compare gradient curve of line 128 between gradient a and b
    ax_64.plot(ga, label='ga')
    ax_64.plot(gb, label='gb')
    #ax_64.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))



    ax_64.plot(rarbOffs, label='offsRaRb')
    #ax_2.plot(16/(offsetA[128]/(32*(gradA[1][128])/((gaMax-gaMin)))), label='hyperA')
    #ax_2.plot((gradA[1][128]-gaMin)/(gaMax-gaMin)-(gradB[1][128] - gbMin) / (gbMax - gbMin), label='gradB')
    #ax_2.plot(((np.arctan2(gradA[1][128], gradB[1][128]) + np.pi) / (2 * np.pi)) * 360, label='atan')
    #ax_2.plot(np.absolute(gradA[1][128] + 1j * gradB[1][128]), label='mag')
    #ax_2.plot(((np.arctan2(gradA[1][128], gradB[1][128]) + np.pi) / (2 * np.pi)) * 360*np.absolute(gradA[1][128] + 1j * gradB[1][128]), label='norm')
    #ax_2.set_ylim(-25,25)
    ax_2.legend()
    ax_64.legend()

    #ax_2.plot(gradA[0][128], )
    #ax_2.plot(gradientOffs[0][128],)
    #ax_2.plot(gradB[0][128], )
    #ax_2.set_title('Convoluted 2x2')

    fig.show()
    fig3D.show()
    plt.waitforbuttonpress()