from datetime import timedelta
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import fft, signal
from skimage import draw


def sinTri(alpha):
    """ Triangle Sinus Function"""
    return signal.sawtooth(alpha+np.pi/2, width=0.5)


def cosTri(alpha):
    """ Triangle CoSinus Function"""
    return -signal.sawtooth(alpha, width=0.5) #cos triangle

def line_aa(rowa, cola, rowb, colb, mxRows, mxCols):
    """ create correct weighted anti alias line"""

    def sector_line(col, cola, fwidth, rate, ratio, row, rowa, val, mxWidth, mxHeight):
        mx = max
        for i in range(0, fwidth, np.sign(fwidth)):
            y0 = rowa + i * rate
            y1 = y0 + rate
            ht = ratio / 2

            pos0 = y0 + ht
            fpos0 = pos0 // 1
            mpos0 = pos0 % 1

            pos1 = y1 + ht
            fpos1 = pos1 // 1
            mpos1 = pos1 % 1

            neg0 = y0 - ht
            fneg0 = neg0 // 1
            mneg0 = neg0 % 1

            neg1 = y1 - ht
            fneg1 = neg1 // 1
            mneg1 = neg1 % 1

            pm = mx(mx(fpos0, fpos1), mx(fneg0, fneg1))
            pp0 = int(pm - fpos0)
            pp1 = int(pm - fpos1)
            pn0 = int(pm - fneg0)
            pn1 = int(pm - fneg1)

            aV = [0.0, 0.0, 0.0, 0.0]

            if fpos0 != fpos1:
                if fpos0 > fpos1:
                    a0p = np.abs(mpos0 ** 2 / (2 * rate))
                    a1p = np.abs((1 - mpos1) ** 2 / (2 * rate))
                    aV[pp0] = a0p
                    aV[pp1] = 1 - a1p
                else:
                    a0p = np.abs((1 - mpos0) ** 2 / (2 * rate))
                    a1p = np.abs(mpos1 ** 2 / (2 * rate))
                    aV[pp0] = 1 - a0p
                    aV[pp1] = a1p
            else:
                a0p = (mpos0 + mpos1) / 2
                aV[pp0] = a0p

            if fneg0 != fneg1:
                if fneg0 < fneg1:
                    a0n = np.abs((1 - mneg0) ** 2 / (2 * rate))
                    a1n = np.abs(mneg1 ** 2 / (2 * rate))
                    if (pp0 == pn1):
                        aV[pn1] -= a1n
                    else:
                        aV[pn1] = 1 - a1n
                    aV[pn0] = a0n
                else:
                    a0n = np.abs(mneg0 ** 2 / (2 * rate))
                    a1n = np.abs((1 - mneg1) ** 2 / (2 * rate))
                    if (pp1 == pn0):
                        aV[pn0] -= a0n
                    else:
                        aV[pn0] = 1 - a0n
                    aV[pn1] = a1n
            else:
                a0n = (2 - mneg0 - mneg1) / 2
                aV[pn0] = a0n

            x = int(cola) + i
            if x < mxWidth:
                for j in range(0, 4):
                    y = int(pm - j)
                    if y < mxHeight:
                        row.append(y)
                        col.append(x)
                        val.append(aV[j] / ratio)

    row = []
    col = []
    val = []
    width = colb - cola
    height = rowb - rowa

    fwidth = int(width // 1)
    fheight = int(height // 1)

    length = np.sqrt(width**2 + height**2)
    awidth = np.abs(width)
    aheight = np.abs(height)

    if awidth > aheight:
        rate = height / width
        ratio = length / awidth
        sector_line(col, cola, fwidth, rate, ratio, row, rowa, val, mxCols, mxRows)

    else:
        rate = width / height
        ratio = length / aheight
        sector_line(row, rowa, fheight, rate, ratio, col, cola, val, mxRows, mxCols)

    return np.array(row), np.array(col), np.array(val)


def aaFilter(aStep, aPr, width, height, radialMatr):

    img2 = np.zeros((height, width), dtype='float64')

    halfW = width // 2
    halfH = height // 2
    halfR = np.pi / (2 * aPr)
    start = aStep - halfR
    stop = aStep + halfR
    length = np.sqrt(halfW ** 2 + halfH ** 2)
    step = 1 / length

    for i in np.arange(start, stop, step):
        weight = np.cos((i - aStep) * aPr) ** 2

        yl = halfH - np.sin(i) * length
        xl = halfW + np.cos(i) * length
        if yl < 0 or xl < 0 or yl >= height or xl >= width:
            dy = (yl - halfH)
            dx = (xl - halfW)
            sy = np.sign(dy)
            sx = np.sign(dx)
            ny = yl
            nx = xl
            if abs(dx) > abs(dy):
                rate = dy / dx
                ny = halfW * rate * sx + halfH
                if xl > width:
                    nx = width - 1
                if xl < 0:
                    nx = 0
                if ny < 0:
                    ny = 0
                    nx = halfH / -rate
                if ny > height:
                    ny =  height - 1
                    nx = halfH / rate

            else:
                rate = dx / dy
                nx = halfH * rate * sy + halfW
                if yl > height:
                    ny = height - 1
                if yl < 0:
                    ny = 0
                if nx < 0:
                    nx = 0
                    ny = halfW / -rate
                if nx > width:
                    nx = width - 1
                    ny = halfW / rate

            xl = nx
            yl = ny

        nLen = np.sqrt(halfW ** 2 + halfH ** 2)

        rr, cc, val = line_aa(yl+0.5, xl+0.5, halfH+0.5, halfW+0.5, img2.shape[0], img2.shape[1])
        img2[rr, cc] += val * radialMatr[rr, cc] * weight / nLen

    return img2


def pyramidFlow(ffA, ffB, angularMatr, radialMatr):
    halfPi = np.pi / 2
    rows, cols = ffA.shape
    halfX = cols // 2
    # halfY = rows // 2
    aPr = 32                     # angular precision
                       # radial scale / eccentricity
    jShifts = [1j, -1j, -1]
    aShifts = [-halfPi, halfPi, np.pi]
    vuMap = np.zeros((2, ffA.shape[-2], ffA.shape[-1]))
    octaves = int(np.log2(halfX))
    fH = np.sqrt(2)
    iwCbuf = np.zeros((octaves*2+1, rows, cols), dtype="complex128")
    aCbuf = [] # for Debug
    rMap = np.zeros((rows, cols), dtype='float64')

    for j in range(0,aPr*2):
        rSc = halfX
        r = (halfX * 0.5) / rSc     # radial filter location

        aStep = halfPi * (j * (1/aPr) - 0.5)
        anglularFilt = np.cos(np.clip((angularMatr + aStep) * aPr, -halfPi, halfPi))**2
        langularF = aaFilter(aStep, aPr, rows, cols, radialMatr)
        nRadialMatr = np.clip(32 * radialMatr / np.sqrt(radialMatr.shape[0] ** 2 + radialMatr.shape[1] ** 2), 0, 1)
        bangular = langularF * (1-nRadialMatr) + anglularFilt * nRadialMatr
        #bangular = anglularFilt
        iwC = np.zeros((rows,cols), dtype='complex128')
        split = halfX

        for i in range(0, octaves*2+1):
            rRange = split * radialMatr / halfX
            radialFilt = (np.cos(np.clip((rRange - 1) * np.pi, 0, halfPi)) ** 2 *
                           (1 - np.cos(np.clip((rRange * fH - 1) * np.pi, 0, halfPi)) ** 2))
            split /= fH

            pyrFilt = np.clip(radialFilt * bangular, 1e-10, 1)
            rMap += pyrFilt
            if np.any(pyrFilt > 0.01):
                iwA = fft.ifft2(fft.ifftshift(ffA * pyrFilt))
                iwB = fft.ifft2(fft.ifftshift(ffB * pyrFilt))
                iwCbuf[i] = iwB * iwA.conj() / np.sqrt(np.abs(iwB * iwA))
            iwC += iwCbuf[i]


        aC = np.angle(iwC)
        aCbuf.append(aC)
        ash = 0
        while (aC.max() - aC.min()) > (2 * np.pi * 0.95) and ash < len(aShifts):
            aC = (np.angle(iwC * jShifts[ash]) - aShifts[ash])
            ash += 1
        #if (aC.max() - aC.min()) < (2 * np.pi * 0.95):
        phaseSh = halfX * aC / np.pi
        ss = np.sin(aStep) #* np.cos((aStep - np.pi / 4 )/2 )**2
        cc = -np.cos(aStep)*2 #* np.cos((aStep - np.pi / 4 )/2 )**2
        vuMap[0] += phaseSh * ss
        vuMap[1] += phaseSh * cc


    vuMap /= aPr*2

    return np.moveaxis(vuMap, 0, -1), aCbuf, rMap


def sumAngle(iwC):
    return np.cumsum(
        np.pad((np.abs(np.diff(np.angle(iwC)[:, 128])) > np.pi) * np.sign(np.angle(iwC)[:-1, 128]) * np.pi * 2,
               (1, 0))) + np.angle(iwC)[:, 128]



if __name__ == '__main__':

    mmx = max

### --- load Testimages

    a = np.asarray(Image.open("testIm2_A.png").convert('L'))
    b = np.asarray(Image.open("testIm2_B.png").convert('L'))

    rows, cols = a.shape


### create Parameter Plots ...
    fig2, ax2 = plt.subplots(2, 5, figsize=(15, 10), squeeze=True, sharex=False, sharey=False)
    plt.subplots_adjust(left=0.033, bottom=0.01, right=0.99, top=0.99, wspace=0.12, hspace=0.01)

    halfX = cols // 2
    halfY = rows // 2

# Filter
    x = np.linspace(0, 255, 256)
    y = np.linspace(0, 255, 256)

    mx, my = np.meshgrid(x, y)
    angleMatr = np.angle((mx-128) + 1j * (my-128))
    dist = np.sqrt((mx-128)**2 + (my-128)**2)
    halfPi = np.pi / 2
# angular filter
    anglularFilt = []
    aPr = 32
    for i in range(0,aPr*4):
        aStep = halfPi * (i * (1 / aPr) - 2)
        anglularFilt.append(np.cos(np.clip((angleMatr + aStep) * aPr, -halfPi, halfPi))**2)
    #anglularFilt2 = np.cos(np.clip((angleMatr + halfPi * -1.9062) * 32, -halfPi, halfPi))**2
    #anglularFilt = anglularFilt2
# four octaves radial filter
    splits = [64,48,32,24,16,12,8,6,4,3,2,1.5,1]
    octaves = int(np.log2(halfX))
    fH = np.sqrt(2)
    radialFilt = []
    split = halfX
    for i in range(0,octaves*2+1):
        radialFilt.append((np.cos(np.clip(((split * np.pi * dist / halfX)) - np.pi, 0, halfPi))**2 *
                           (1-np.cos(np.clip(((split * fH * np.pi * dist / halfX)) - np.pi, 0, halfPi))**2)))
        # radialFilt.append(np.clip(((split * np.pi * dist / halfX)) - np.pi, 0, halfPi))
        split /= fH

# create plots
    ax2[0, 0].grid()
    ax2[0, 0].set_title("linear analysis 1")

    ax2[0, 1].set_title("linear analysis 2")

    ffA = fft.fftshift(fft.fft2(a))
    ffB = fft.fftshift(fft.fft2(b))

    ax2[0, 2].set_title("ffA")

    ax2[0, 3].set_title("ffB")

## Apply Complex Pyramid Filter

    iwA = []
    iwB = []
    iwA2 = []
    iwB2 = []
    iwC = []
    iwC2 = []


    for i in range(0,octaves*2+1):
        j = 5
        filter = np.clip(radialFilt[i] * anglularFilt[j],1e-10,1)
        while not np.any(filter > 0.1) and j > 0:
            j -= 1
            filter = np.clip(radialFilt[i] * anglularFilt[j], 1e-10, 1)

        iwA.append(fft.ifft2(fft.ifftshift(ffA * filter)))
        iwB.append(fft.ifft2(fft.ifftshift(ffB * filter)))
        iwC.append(iwB[i] * iwA[i].conj())


    ax2[0, 3].set_title("imageA")
    #ax2[0, 3].imshow(np.clip(radialFilt[5] * anglularFilt,1e-10,1))

    ax2[0, 4].set_title("imageB")

    #ax2[0, 4].imshow(np.clip(radialFilt[5] * anglularFilt2,1e-10,1))


    ax2[0, 1].plot(np.angle(iwA[5][128]), label='iwA2[3](a)')
    ax2[0, 1].plot(np.angle(iwB[5][128]), label='iwB2[3](a)')
    ax2[0, 1].plot(np.angle(iwC[5][128]), label='iwC2[3](a)')

    ax2[0, 1].legend()
    ax2[0, 1].grid()

    sC = []
    #aiwC = iwC[0]
    aiwC = np.zeros(iwC[0].shape, dtype="complex128")
    #sC.append(np.angle(aiwC))

    jShifts = [1j, -1j, -1]
    aShifts = [-halfPi, halfPi, np.pi]

    for i in range(0,octaves*2+1):
        #if np.any(iwC[i] > 0.1):
        aiwC += iwC[i]
        #else:
        #    aiwC += iwC2[i]
        ash = 0
        aC = np.angle(aiwC)
        while (aC.max() - aC.min()) > (2 * np.pi * 0.95) and ash < len(aShifts):
            aC = (np.angle(iwC[i] * jShifts[ash]) - aShifts[ash])
            ash += 1
        sC.append(aC)

    for i in range(0,octaves*2+1):
        ax2[0, 0].plot(sC[i][128], label=f'sc({i})')

    ax2[0, 0].legend()
    ax2[0, 0].grid()
    ax2[0, 0].set_title("recreated slice")

    ax2[1, 2].set_title("phase highest octave")
    ax2[1, 2].imshow(sC[14])

    ax2[0, 2].set_title("fft - imageB")
    ax2[0, 2].imshow(np.log(np.abs(ffB)))

    #for i in range(0, octaves * 2 + 1):
     #   ax2[1, 0].plot(radialFilt[i][128], label=f'r({i})')

    ax2[1, 0].legend()


    sum = np.zeros(256, dtype='float64')
    for i in range(0, octaves * 2 + 1):
        sum += radialFilt[i][128]
    #ax2[1, 0].plot(sum, label=f'sum')


    ax2[1, 0].grid()


    #ax2[1, 1].imshow(np.clip(radialFilt[5] * anglularFilt, 1e-10, 1))
    ax2[1, 1].set_title("phase ImageB")
    ax2[1, 1].imshow(np.angle(ffB))

    start = timer()
    print("Start: ...")
    vuMap, acBuf, pyrFilt = pyramidFlow(ffA, ffB, angleMatr, dist)
    print("... Stop.")
    end = timer()
    print(f"delta:{timedelta(seconds=end-start)}")
    ax2[1, 3].set_title("filter sum")
    ax2[1, 3].imshow(pyrFilt)

    for j in range(2,6):
        ax2[1, 0].plot(acBuf[j][128], label=f'aC({180*(halfPi * (j * (1/32) - 0.5))/np.pi:.1f})')

    #img = np.zeros((100, 100), dtype='float64')

    # for i in range(-3,3):
    #    rr, cc, val = draw.line_aa(10-i, 10+i, 80, 80)
    #    ramp = np.arange(255, 0, -255/val.shape[0])
    #    img[rr, cc] += val * ramp


    ax2[0, 3].imshow(anglularFilt[64])

    img2 = np.zeros((256, 256), dtype='float64')
    # i = 3
    for i in np.arange(40, 50, 0.4775):

        weight = np.sin((i-40)*np.pi/10) ** 2

        phi = i / 180 * np.pi
        yl = 128 - np.sin(phi) * 120
        xl = 128 + np.cos(phi) * 120

        rr, cc, val = line_aa(yl, xl, 128, 128, 256, 256)
        ramp = np.arange(255, 0, -255 / val.shape[0])
        if ramp.shape[0] > val.shape[0]:
            ramp = ramp[:-1]
        if ramp.shape[0] < val.shape[0]:
            ramp = np.pad(ramp, (0, 1))
        # img2[rr, cc] = np.fmax(img2[rr, cc], val * ramp * weight)
        img2[rr, cc] += val * ramp * weight

    #ax2[0, 4].imshow(img2, vmin=0, vmax=255)
    nDist = 2 * dist / np.sqrt(dist.shape[0] ** 2 + dist.shape[1] ** 2)
    ax2[0, 4].imshow(aaFilter(0.0, 32, 256, 256, dist) * (1-nDist) + anglularFilt[64] * nDist)

    ax2[1, 0].legend()
    ax2[1, 0].grid()

    nvec = 20
    step = max(rows // nvec, cols // nvec)

    vy, vx = np.mgrid[:rows:step, :cols:step]
    u_ = vuMap[::step, ::step, 1]

    v_ = vuMap[::step, ::step, 0]
    #ax2[1, 4].imshow(180*np.angle(vuMap[:,:,1] + 1j * vuMap[:,:,0])/np.pi)
    ax2[1, 4].imshow(np.abs(vuMap[:, :, 1] + 1j * vuMap[:, :, 0]))
    ax2[1, 4].quiver(vx, vy, u_, v_, color='r', units='dots',
              angles='xy', scale_units='xy', lw=3)

    ax2[1, 4].set_title("uv-vector map")
    ax2[1, 2].set_title("ifft-offset")

    plt.show()
    plt.waitforbuttonpress()