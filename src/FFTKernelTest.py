"""
Tests the Kernels with samples and plots it
"""

import matplotlib.pyplot as plt
import matplotlib.colors as cols
from matplotlib import cm
import numpy as np
from PIL import Image
from scipy import datasets, ndimage
from FFTkernel import FFTkernel2D

if __name__ == '__main__':
    kImage = Image.open("gauss512.tif")

    resSteps = [256, 128, 64, 32, 16, 8, 4, 2 ]

    kernelMaps = { str(r) : np.asarray(kImage.resize((r,r))) for r in resSteps}

    ac = datasets.ascent()
    a = np.asarray(Image.open("relief0.png").convert('L'))
    b = np.asarray(Image.open("relief0_sc.png").convert('L'))

    kernelsA = {str(r): FFTkernel2D(a, kernelMaps[str(r)]) for r in resSteps}
    kernelsB = {str(r): FFTkernel2D(b, kernelMaps[str(r)]) for r in resSteps}

    resultsA = {str(r): kernelsA[str(r)].convolute() for r in resSteps}
    resultsB = {str(r): kernelsB[str(r)].convolute() for r in resSteps}


    fig, ((ax_1, ax_2, ax_4, ax_8, ax_16), (ax_32, ax_64, ax_128, ax_256, ax_512)) = plt.subplots(2, 5, figsize=(15, 6))

    fig3D, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #X = np.arange(-5, 5, 0.25)
    #Y = np.arange(-5, 5, 0.25)
    #X, Y = np.meshgrid(X, Y)
    #R = np.sqrt(X ** 2 + Y ** 2)
    #Z = np.sin(R)
    #ax.plot_surface(X, Y, Z, vmin=Z.min() * 2, cmap=cm.Blues)
    #fig3D.show()

    ax_1.imshow(a, cmap='gray', vmin=0, vmax=255)
    ax_1.set_title('OriginalA')

    gradA = np.gradient(resultsA['256'])
    gradB = np.gradient(resultsB['256'])
    offsetA = resultsA['256'] - resultsB['256']
    #gradOffs =  []
    #gradOffs.append(gradA[0] - gradB[0])
    #gradOffs.append(gradA[1] - gradB[1])
    gradientOffs = np.gradient(offsetA)
    gradientOffsImg = gradientOffs[1] + 1j * gradientOffs[0]


    #b512 = fftc.convolute()
    ax_512.imshow(resultsB['256'], cmap='gray', vmax=255, vmin=0)
    ax_512.set_title('ConvolutedB 256x256')



    #imshow(cols.hsv_to_rgb(
    #    np.dstack((np.absolute(np.angle(fftc) / np.pi + 1) / 2, np.ones(fftc.shape), abs_fftc / abs_fftc.max()))))

    ax_256.imshow(offsetA, cmap='cool')

    mx = np.arange(0, 256, 1)
    my = np.arange(0, 256, 1)
    mx, my =  np.meshgrid(mx, my)

    #ax.plot_surface(mx, my, offsetA)
    ax.plot_surface(mx, my, offsetA)
    ax.plot_surface(mx, my, np.absolute(gradientOffsImg)*64)
    ax_256.set_title('delta')

    #gradA = np.gradient(resultsA['256'])

    gradIm = gradA[1] + 1j * gradA[0]
    grad_abs = np.absolute(gradIm)

    # ax_128.imshow(grad, vmin = -1, vmax = 1)
    #ax_128.imshow(offsetA/gradA[1], cmap='cool', vmin=-50, vmax=50)
    #ax_128.plot(offsetA[128]/gradA[1][128])
    #ax_128.imshow(np.absolute(gradientOffsImg)*64,cmap='gray')
    magABs = np.cos(np.arctan(gradientOffs[1])) * offsetA + 1j * np.cos(np.arctan(gradientOffs[0])) * offsetA
    ax_128.imshow(np.absolute(magABs))
    #ax_128.imshow(np.angle(np.arctan2(gradA[1], gradB[1])))
    ax_128.set_title('GradientA Magnitude 256x256')

    # ax_64.imshow(cols.hsv_to_rgb(np.dstack((np.absolute(np.angle(gradIm)/np.pi+1)/2,np.ones(gradIm.shape), np.ones(gradIm.shape)))))  # grad_abs/grad_abs.max()))))
    # ax_64.plot(offsetA[128], label='offs128')
    # ax_64.plot(offsetA[60], label='offs113')
    #ax_64.plot(np.cos(np.arctan2(gradA[1][128], gradB[1][128])) * offsetA[128], label='spat')
    # ax_64.plot(np.absolute(resultsA['256'][128] + 1j * resultsB['256'][128]), label='mag')
    ax_64.set_title('GradientA-Angle 256x256')
    ax_64.legend()


    # ax_32.imshow(gradientOffs[0], cmap='cool')
    # ax_32.imshow(cols.hsv_to_rgb(np.dstack((np.absolute(np.angle(gradientOffsImg)/np.pi+1)/2,np.ones(gradientOffsImg.shape), np.ones(gradientOffsImg.shape)))))  # grad_abs/grad_abs.max()))))
    ax_32.imshow(np.angle(gradientOffsImg, deg = True))
    ax_32.set_title('Gradient-Angle')

    ax_16.imshow(resultsA['256'], cmap='gray', vmax=255, vmin=0)
    ax_16.set_title('OffsetGradient-Y')

    vx, vy = np.meshgrid(np.linspace(0, 255, 32),
                       np.linspace(0, 255, 32))

    gradOx = ndimage.zoom(gradientOffs[1], 0.125)
    gradOy = ndimage.zoom(gradientOffs[0], 0.125)
    #gradOx.resize((32,32), refcheck = False)
    #gradOy.resize((32,32), refcheck = False)

    ax_8.quiver(vx, vy, gradOx*10, gradOy*10)

    # ax_8.imshow(gradB[1], cmap='gray')
    ax_8.set_title('gradB - X')

    #ax_4.imshow(resultsB['256'], cmap='gray', vmin = 0, vmax = 255)
    # ax_4.plot(offsetA[250], label='off250')
    #ax_4.plot(offsetA[128], label='off128')
    #ax_4.plot(np.angle(gradA[1][128] + 1j * gradB[1][128], deg = True), label='angleX')
    ax_4.plot(gradA[1][128], label='gradA0')
    ax_4.plot(gradB[1][128], label='gradB0')
    # ax_4.plot(np.angle(np.gradient(gradA[1][128])+1j * np.gradient(gradB[1][128]),deg=True), label='delt')
    ax_4.set_title('Result B')
    ax_4.legend()

    # ax_2.plot(offsetA[250], label='offsetA')
    #ax_2.plot(gradientOffs[0][250], label='resultA')
    gaMin = np.min(gradA[1][128])
    gaMax = np.max(gradA[1][128])
    gbMin = np.min(gradB[1][128])
    gbMax = np.max(gradB[1][128])
    raMin = np.min(resultsA['256'][128])
    raMax = np.max(resultsA['256'][128])
    rbMin = np.min(resultsB['256'][128])
    rbMax = np.max(resultsB['256'][128])
    ra = (2 * resultsA['256'][128]-(raMax+raMin)) / (raMax-raMin)
    rb = (2 * resultsB['256'][128]-(rbMax+rbMin)) / (rbMax-rbMin)
    gdoMin = np.min(gradientOffs[1][128])
    gdoMax = np.max(gradientOffs[1][128])
    ga = 2 * (gradA[1][128]) / 0.5*(gaMax+gaMin)
    gb = 2 * (gradB[1][128]) / 0.5*(gbMax+gbMin)
    ax_2.plot(ra, label='gradA')
    ax_2.plot(rb, label='gradB')
    ax_2.plot(np.arctan2(ra,rb))
    ax_2.plot((gradientOffs[1][128])/(gdoMax-gdoMin), label='gradOffs')
    rarbOffs = ra - rb
    ax_64.plot(rarbOffs, label='offsRaRb')
    #ax_2.plot(16/(offsetA[128]/(32*(gradA[1][128])/((gaMax-gaMin)))), label='hyperA')
    #ax_2.plot((gradA[1][128]-gaMin)/(gaMax-gaMin)-(gradB[1][128] - gbMin) / (gbMax - gbMin), label='gradB')
    #ax_2.plot(((np.arctan2(gradA[1][128], gradB[1][128]) + np.pi) / (2 * np.pi)) * 360, label='atan')
    #ax_2.plot(np.absolute(gradA[1][128] + 1j * gradB[1][128]), label='mag')
    #ax_2.plot(((np.arctan2(gradA[1][128], gradB[1][128]) + np.pi) / (2 * np.pi)) * 360*np.absolute(gradA[1][128] + 1j * gradB[1][128]), label='norm')
    #ax_2.set_ylim(-25,25)
    ax_2.legend()

    #ax_2.plot(gradA[0][128], )
    #ax_2.plot(gradientOffs[0][128],)
    #ax_2.plot(gradB[0][128], )
    #ax_2.set_title('Convoluted 2x2')

    fig.show()
    fig3D.show()
    plt.waitforbuttonpress()