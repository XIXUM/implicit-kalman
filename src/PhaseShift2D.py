import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scipy import fft

def absmin(a,b):
    return a if np.abs(a) < np.abs(b) else b


def leastSquareAngle(shift, last_sh, wx):
    factor = 0
    if wx > 1:
        offs_global = 0
        for ws in range(wx, 1, -1):
            sc_pos = ws / (ws - 1)
            sc_neg = (ws - 1) / ws
            offs_a = np.angle(shift * np.power(last_sh, sc_pos).conj())
            offs_b = np.angle(shift * np.power(last_sh, sc_neg).conj())
            offs_local = absmin(offs_a, offs_b)
            if offs_global == 0:
                offs_global = offs_local
            else:
                min = absmin(offs_global, offs_local)
                if min != offs_global:
                    factor = wx/ws
                    offs_global = min

    else:
        offs_global = np.angle(shift)


    return offs_global, factor


def clockCount(a, la, w_off):
    if np.abs(a) > 0.2 and a - la > 1.3:
        w_off += 1
    if np.abs(a) > 0.2 and a - la < -1.3:
        w_off -= 1
    return w_off

if __name__ == '__main__':
    #original = data.astronaut()
    #image = color.rgb2gray(original)

    ### --- load Testimages

    a = np.asarray(Image.open("relief0.png").convert('L'))
    b = np.asarray(Image.open("relief0_offs.png").convert('L'))

    fig, ax = plt.subplots(2, 5, figsize=(15, 10), squeeze=True, sharex=False, sharey=False)

    ffA = fft.fftshift(fft.fft2(a))
    ffB = fft.fftshift(fft.fft2(b))
    offset = a - b
    ffO = fft.fftshift(fft.fft2(offset))
    ffOffs = (ffB * ffA.conj()) / np.absolute(ffB * ffA.conj())

    ffAngle =  np.angle(ffOffs)

    iH, iW = a.shape[:2]
    hH = iH // 2
    hW = iW // 2

    ax[0, 0].imshow(a)
    ax[0, 0].set_title("imA")
    ax[0, 1].imshow(b)
    ax[0, 1].set_title("imB")

    scanlinex = np.zeros((a.shape[1],hW))
    scanliney = np.zeros((a.shape[0], hH))

    vecMap = np.zeros((iH,iW,2))

    lshpy = np.zeros(hW, dtype='complex128')
    lshny = np.zeros(hW, dtype='complex128')

    for wy in range(0,hH):

        lshp = 0
        lshn = 0
        fshift = 0

        wx_off_pos = 0
        wx_off_neg = 0
        la_pos = 0
        la_neg = 0
        fn = 0
        fp = 0

        if wy > 0:
            pos_Ty = hH / wy
            neg_Ty = hH / -wy
        else:
            pos_Ty = 0
            neg_Ty = 0

        for wx in range(0,hW):
            #w_posX += 1
            #w_negX -= 1
            if wx > 0:
                pos_Tx = hW / wx
                neg_Tx = hW / -wx
            else:
                pos_Tx = 0
                neg_Tx = 0

            sh_pos = ffOffs[hH + wy, hW + wx]
            sh_neg = ffOffs[hH - wy, hW - wx]

            offs_px, fpx = leastSquareAngle(sh_pos, lshp, wx)
            offs_nx, fnx = leastSquareAngle(sh_neg, lshn, wx)
            offs_py, fpy = leastSquareAngle(sh_pos, lshpy[wx], wy)
            offs_ny, fny = leastSquareAngle(sh_neg, lshny[wx], wy)

            scanlinex[wy,wx] -= 0.5 * offs_px * pos_Tx / np.pi
            scanlinex[wy,wx] -= 0.5 * offs_nx * neg_Tx / np.pi
            scanliney[wy,wx] -= 0.5 * offs_py * pos_Ty / np.pi
            scanliney[wy,wx] -= 0.5 * offs_ny * neg_Ty / np.pi
            lshp = sh_pos
            lshn = sh_neg
            lshpy[wx] = sh_pos
            lshny[wx] = sh_neg


        # scanline[x] = sample / 2

    #ax[0, 2].plot(scanline)
    ax[0, 2].imshow(scanlinex)
    ax[0, 2].set_title("x-128-shift")
    ax[0, 3].imshow(scanliney)

    plt.show()
    plt.waitforbuttonpress()