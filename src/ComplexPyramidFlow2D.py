from datetime import timedelta
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import fft, signal, ndimage, misc
from skimage import draw


def centerLine(cy, cx, phi, shape):
    row = []
    col = []
    dx = np.cos(phi)
    dy = -np.sin(phi)
    if abs(dx) > abs(dy):

        rate = dy / dx
        for i in range(-cx, shape[1] - cx):
            py = i * rate + cy
            if py >= 0 and py < shape[0]:
                row.append(py)
                col.append(cx + i)
    else:
        rate = dx / dy
        for i in range(-cy, shape[0] - cy):
            px = i * rate + cx
            if px >= 0 and px < shape[1]:
                row.append(cy + i)
                col.append(px)

    return np.array(row, dtype='int32'), np.array(col, dtype='int32')

def boundedLine(y1,x1,y2,x2, height, width):
    rr, cc, val = draw.line_aa(y1, x1, y2, x2)
    rrX = np.append(np.argwhere(rr > height-1), np.argwhere(rr < 0))
    ccX = np.append(np.argwhere(cc > width-1), np.argwhere(cc < 0))
    allX = np.append(rrX, ccX)

    rr = np.delete(rr, allX)
    cc = np.delete(cc, allX)
    val = np.delete(val, allX)

    return rr, cc, val

def pyramidFlow(ffA, ffB, angularMatr, radialMatr, my, mx):
    halfPi = np.pi / 2
    rows, cols = ffA.shape
    halfX = cols // 2
    # halfY = rows // 2
    octaves = int(np.log2(halfX))
    aPr = 8 # angular precision
                       # radial scale / eccentricity
    dPi = 2 * np.pi
    hh = rows / dPi
    ww = cols / dPi
    haPr = aPr * 2
    faPr = haPr * 2
    rO = octaves * 2 + 1
    #rO = octaves + 1
    step = halfX / rO

    vuMap = np.zeros((2, ffA.shape[-2], ffA.shape[-1]))

    fH = np.sqrt(2)
    aCbuf = [] # for Debug

    offs = int(1.5 * aPr)

    anglularFilt = []
    for i in range(0,faPr):
        lBuffer = np.zeros(angularMatr.shape, dtype='float64')
        aStep = halfPi * (i * (1 / aPr) - 2)
        rRamp = np.cos(np.clip((angularMatr + aStep) * aPr, -halfPi, halfPi))**2
        anglularFilt.append(rRamp)

    radialFilt = []
    split = halfX
    for i in range(0, rO):
        for i in range(0, octaves + 1):
            radialFilt.append(np.cos(np.clip(np.pi * (2 * radialMatr / step - i), -halfPi, halfPi)) ** 2)
        split /= fH

    iwCf = 1
    iwV = iwU = np.zeros(radialFilt[0].shape)
    # iwV = np.zeros(radialFilt[0].shape)
    iwC = []
    for i in range(0, rO):
        scaleFt = step * (i) + 1

        avg = 0
        for j in range(0,haPr):
            k = j + offs
            aStep = halfPi * (j * (1 / aPr) - 0.5)
            cc, ll, ss = sinCosLen(aStep, hh, ww)

            pyrFilt = radialFilt[i] * anglularFilt[k]

            if np.any(pyrFilt > 0.1):
                avg += 1

                iwA = fft.ifft2(fft.ifftshift(ffA * pyrFilt))
                iwB = fft.ifft2(fft.ifftshift(ffB * pyrFilt))
                if i > 0:
                    iwAs = shiftMatrixInt(iwA, iwV, iwU, mx, my)
                    oiwC = iwB * iwAs.conj()
                    oiwCu = oiwC / (np.abs(oiwC))
                    offsC = gradientfix(oiwCu, 1 / scaleFt, ss, cc, mx, my)
                    debug = np.angle(phaseShiftCompile(oiwCu, 1 / scaleFt))
                    debugDeviation = np.abs(np.mean(debug / offsC) - 1)
                    if  debugDeviation > 0.1:
                        print(debugDeviation)
                    #shift = np.round((np.mean(iwC[j]) * 2 - np.mean(offsC)) / np.pi)
                    shift = np.mean(iwC[j]) - np.mean(offsC)
                    offsCs = offsC + shift
                    #delta = np.angle(np.exp( 1j * offsCs * scaleFt) * oiwCu.conj())
                    iwC[j] +=  offsCs #+ delta / scaleFt # * np.pi / 2
                else:
                    iwC.append(np.angle(iwB * iwA.conj() / np.abs(iwA*iwB)))

                aC = iwC[j]

                if avg > 1:
                    niwV += aC * ss * ll
                    niwU += aC * -cc * ll
                else:
                    niwV = aC * ss * ll
                    niwU = aC * -cc * ll
            else:
                niwU = niwV = np.zeros(radialFilt[0].shape)
                if i < 1:
                    iwC.append(np.zeros(radialFilt[0].shape, dtype="float64"))

        aCbuf.append(np.moveaxis(np.dstack((niwV, niwU)), -1, -0))

        if avg > 0:
            iwV = niwV / avg
            iwU = niwU / avg
        iwCf /= 2

    vuMap[0] = iwV * hh
    vuMap[1] = iwU * ww

    return np.moveaxis(vuMap, 0, -1), aCbuf, anglularFilt

def scale_array(x, new_size):
    min_el = np.min(x)
    max_el = np.max(x)
    y = misc.imresize(x, new_size, mode='L', interp='nearest')
    y = y / 255 * (max_el - min_el) + min_el
    return y

def sumAngle2D(iwC):
    return np.cumsum(
        np.pad((np.abs(np.diff(np.angle(iwC)[:, 128])) > np.pi) * np.sign(np.angle(iwC)[:-1, 128]) * np.pi * 2,
               (1, 0))) + np.angle(iwC)[:, 128]

def sumAngle(iwC):
    aiwC = np.angle(iwC)
    return np.cumsum(
        np.pad(-np.round(np.diff(aiwC) / (2*np.pi)),
               (1, 0))) * 2 * np.pi + aiwC

def sumAngle2D(iwC, ss, cc, my, mx):
    aiwC = np.angle(iwC)
    return offsetAngle2D(aiwC, ss, cc, my, mx, 2) + aiwC

def offsetAngle2D(aiwC, ss, cc, my, mx, split):
    if abs(cc) < 0.001:
        return np.cumsum(
            np.pad(-np.round(np.diff(aiwC, axis = 0) / (split*np.pi)),
                   ((1, 0),(0,0))) , axis = 0) * np.pi * split

    if abs(ss) < 0.001:
        return np.cumsum(
            np.pad(-np.round(np.diff(aiwC, axis=1) / (split * np.pi)),
                   ((0, 0), (1, 0))), axis=1) * np.pi * split

    if abs(ss) < abs(cc):
        cols = iwC.shape[1]
        offsX = np.round((my - cols // 2) * -ss / cc).astype("int32")
        sx = (mx + offsX) % cols
        rsx = (mx - offsX) % cols
        return np.cumsum(
            np.pad((np.abs(np.diff(aiwC[my,sx])) > (np.pi/2)) * np.sign(aiwC[:-1]) * np.pi * 2,
                   (1, 0)))[my,rsx]

    #else:

def shiftMatrixE(iwA, aiwD, height, width, angle, mx, my):

    dPi = 2 * np.pi
    phi = np.angle(aiwD)
    hh = height / dPi
    ww = width / dPi
    cc, ll, ss = sinCosLen(angle, hh, ww)
    iwE = np.exp(1j * ((np.pi*phi*(my*ss+mx*cc) / (rows)))).conj()
    return iwA*iwE

def shiftMatrix(iwA, aiwD, oct, height, width, angle, mx, my, interpolate = True):
    dPi = 2 * np.pi
    hh = height / dPi
    ww = width / dPi

    cc, ll, ss = sinCosLen(angle, hh, ww)

    omy = my + hh * np.abs(ss) * np.angle(aiwD) * ss
    omx = mx + ww * np.abs(cc) * np.angle(aiwD) * cc
    #omx = mx + ww * np.abs(ss) * np.angle(aiwD) * -ss
    ix = np.clip(omx, 0, width - 1).astype(int)
    iy = np.clip(omy, 0, height - 1).astype(int)
    if interpolate:
        ixp = np.clip(omx + 1, 0, width - 1).astype(int)
        iyp = np.clip(omy + 1, 0, height - 1).astype(int)
        domy = omy - np.floor(omy)
        domx = omx - np.floor(omx)
        msy = (omy > 0)
        msx = (omx > 0)

        iwR = (iwA[iy, ix] * (1 - domx) * (1 - domy) * msy * msx + \
              iwA[iy, ixp] * (1 - domy) * domx * msy + \
              iwA[iyp, ix] * domy * (1 - domx) * msx + \
              iwA[iyp, ixp] * domx * domy) #* iwE.conj()
    else:
        iwR = iwA[iy, ix]

    return iwR + 1 * (iwR == 0)

def shiftMatrixA(iwA, aiwD, height, width, angle, mx, my, interpolate = True):
    dPi = 2 * np.pi
    hh = height / dPi
    ww = width / dPi
    cc, ll, ss = sinCosLen(angle, hh, ww)

    omy = my + hh * np.abs(ss) * aiwD * ss
    omx = mx + ww * np.abs(cc) * aiwD * cc

    if interpolate:

        iomx = omx.astype(int)
        iomy = omy.astype(int)

        padArr = ((-min(iomy.min(),0), max(1,max(0, iomy.max())+2-height)), (-min(iomx.min(),0), max(1,max(0, iomx.max())+2-width)))
        iwAp = np.exp(1j * np.pad(np.angle(iwA), padArr,  'reflect', reflect_type='odd'))
        domy = omy - np.floor(omy)
        domx = omx - np.floor(omx)
        iomxp = iomx+padArr[1][0]
        iomyp = iomy+padArr[0][0]

        iwR = (iwAp[iomyp, iomxp] * (1 - domx) * (1 - domy) + \
              iwAp[iomyp, iomxp+1] * (1 - domy) * domx + \
              iwAp[iomyp+1, iomxp] * domy * (1 - domx)  + \
              iwAp[iomyp+1, iomxp+1] * domx * domy)
        iwM = np.pad(np.ones((height, width)),padArr, 'constant', constant_values=((0, 0), (0, 0)))[iomyp, iomxp]
    else:
        ix = np.clip(omx, 0, width - 1).astype(int)
        iy = np.clip(omy, 0, height - 1).astype(int)
        iwR = iwA[iy, ix] #* iwE.conj()
        iwM = (omx < height) * (omx >= 0) * (omy < width) * (omy >= 0)

    return iwR + 1 * (iwR == 0), iwM


def sinCosLen(angle, hh, ww):
    cc = np.cos(angle)
    ss = np.sin(angle)
    if ss == 0:
        ll = 1
    else:
        if ww / hh > abs(cc / ss):
            ll = np.sqrt(1 + (cc / ss) ** 2)
        else:
            ll = np.sqrt(1 + (ss / cc) ** 2)
    return cc, ll, ss


def shiftMatrixInt(iwA, iwV, iwU, mx, my, interpolate = True):
    dPi = 2 * np.pi
    height = iwA.shape[0]
    width = iwA.shape[1]
    hh = height / dPi
    ww = width / dPi

    omy = my - hh * iwV
    omx = mx - ww * iwU

    ix =  np.clip(omx, 0, width - 1).astype(int)
    iy = np.clip(omy, 0, height - 1).astype(int)
    if interpolate:
        ixp = np.clip(omx + 1, 0, width - 1).astype(int)
        iyp = np.clip(omy +1, 0, height - 1).astype(int)
        domy = omy - np.floor(omy)
        domx = omx - np.floor(omx)
        msy = (omy >= 0)
        msx = (omx >= 0)

        iwR = iwA[iy, ix] * (1 - domx) * (1 - domy) * msy * msx  + \
              iwA[iy, ixp] * (1 - domy) * domx * msy + \
              iwA[iyp, ix] * domy * (1 - domx) * msx + \
              iwA[iyp, ixp] * domx * domy
    else:
        iwR = iwA[iy, ix]

    return iwR

def phaseShiftCompile(iwD, scale):
    phi = np.log(iwD).imag
    iwDp = np.power(iwD, scale)
    if (phi.max() - phi.min()) < 1.90 * np.pi:
        return iwDp
    #return gradientScale(iwD, scale)

    phiS = phi * scale

    lim = np.pi * scale

    iwDmU = splitDomain(lim, phiS) / (2 * lim)
    #grad2 = np.gradient(iwDmU)
    iwDm = np.round(iwDmU) # - np.cumsum(grad2[0] * (np.abs(grad2[0]) < lim * 0.9), axis=0) - np.cumsum(grad2[1] * (np.abs(grad2[1]) < lim * 0.9), axis=1))
    iwDm -= iwDm[iwD.shape[0] // 2, iwD.shape[1] // 2]
    #odd = iwDm % 2

    #iwDn = np.power(-iwD, scale)
    #return (1-odd) * iwDp * np.exp(1j * lim * iwDm) + odd * iwDn * np.exp(1j * lim * iwDm).conj()
    return iwDp * np.exp(2j * lim * -iwDm)

def splitDomain(lim, phiS):
    grad = np.gradient(phiS)
    ly = phiS - np.cumsum(grad[0] * (np.abs(grad[0]) < lim * 0.9), axis=0)
    gy = np.gradient(ly)
    return ly - np.cumsum(gy[1] * (np.abs(gy[1]) < lim * 0.9), axis=1)

def gradientScale(iwD, scale):
    ww, hh = iwD.shape
    grad = np.gradient(iwD)
    grad[0] = np.power(grad[0], scale)
    grad[1] = np.power(grad[1], scale)
    result = np.cumsum(grad[0], axis=0) + np.cumsum(grad[1], axis=1)
    return result + (iwD[ww // 2, hh // 2] - result[ww // 2, hh // 2])


def linSum(grad, ss, cc, mx, my):
    #fGrad = np.sqrt((grad[0] * cc) ** 2  + (grad[1] * ss) ** 2) * (np.sign(ss)+ (ss == 0) * 1) * (np.sign(cc) + (cc == 0) * 1)
    fGrad = grad[0] * ss + grad[1] * cc
    if abs(ss) < 0.001:
        return np.cumsum(fGrad, axis = 1) + np.cumsum(grad[0, my, mx][:, 0])[:, np.newaxis]
    if abs(cc) < 0.001:
        return np.cumsum(fGrad, axis = 0) + np.cumsum(grad[1, my, mx][0,:])[:, np.newaxis].T
    if abs(ss) < abs(cc):
        cols = fGrad.shape[1]
        offsX = np.round((my - cols // 2) * -ss / cc).astype("int32")
        sx = (mx + offsX) % cols
        rsx = (mx  - offsX) % cols
        return np.cumsum(fGrad[my, sx], axis=0)[my, rsx] + np.cumsum(grad[1, my, mx][0,:])[:, np.newaxis].T
    else:
        rows = fGrad.shape[0]
        offsY = np.round((mx - rows // 2) * cc / -ss).astype("int32")
        sy = (my + offsY) % rows
        rsy = (my - offsY) % rows
        return np.cumsum(fGrad[sy, mx], axis=1) [rsy, mx] + np.cumsum(grad[0, my, mx][:, 0])[:, np.newaxis]

def angleGradient(iwC, ss, cc, mx, my):
    dD = np.angle(iwC)
    sotDeg = np.power(otDeg, 0.5)
    pD = np.angle(iwC * sotDeg)
    nD = np.angle(iwC * sotDeg.conj())
    # iD = np.angle(sDi)

    gD = np.array(np.gradient(dD))
    gP = np.array(np.gradient(pD))
    gN = np.array(np.gradient(nD))
    # gI = np.gradient(pD)
    sigD = np.sign(gD)
    sigP = np.sign(gP)
    sigN = np.sign(gN)

    gPN = np.abs(gP) < np.abs(gN)
    gDPN = np.abs(gD) < (np.abs(gP) * (1 - gPN) + np.abs(gN) * gPN)

    grad = np.minimum(np.abs(gD), np.minimum(np.abs(gP), np.abs(gN))) * (
                sigD * gDPN + (sigP * (1 - gPN) + sigN * gPN) * (1 - gDPN))

    return grad[0] * ss + grad[1] * cc

def gradientfix(iwD, scale, ss, cc, mx, my):

    threshold =  np.pi*0.95
    dD = np.angle(iwD)
    if dD.max() < threshold and -dD.min() < threshold:
        return dD * scale
    sotDeg = np.power(otDeg,0.2)
    pD = np.angle(iwD*sotDeg)
    nD = np.angle(iwD*sotDeg.conj())
    #iD = np.angle(sDi)

    gD = np.array(np.gradient(dD)) * scale
    gP = np.array(np.gradient(pD)) * scale
    gN = np.array(np.gradient(nD)) * scale
    #gI = np.gradient(pD)
    sigD = np.sign(gD)
    sigP = np.sign(gP)
    sigN = np.sign(gN)

    gPN = np.abs(gP) < np.abs(gN)
    gDPN = np.abs(gD) < (np.abs(gP) * (1 - gPN) + np.abs(gN) * gPN)

    grad = np.minimum(np.abs(gD),np.minimum(np.abs(gP), np.abs(gN)))*(sigD*gDPN + (sigP*(1-gPN) + sigN*gPN) * (1 - gDPN))

    return linSum(grad, ss, cc, mx ,my)

    # return (np.cumsum(grad[0], axis=0) + np.cumsum(grad[1], axis=1)[0, :])

def complexGrad(iwD, ss, cc, mx, my):

    dD = np.angle(iwD)

    nD = np.angle(-iwD)
    #padV = ((1, 0), (0, 0))
    #padU = ((0, 0), (1, 0))

    gD = sharpGrad(dD)
    gN = sharpGrad(nD)

    sigD = np.sign(gD)
    sigN = np.sign(gN)

    gPN = np.abs(gD) < np.abs(gN)
    #gDPN = np.abs(gD) < (np.abs(gP) * (1 - gPN) + np.abs(gN) * gPN)

    grad = np.minimum(np.abs(gD), np.abs(gN))*(sigN*(1-gPN) + sigD*gPN)

    return grad


def sharpGrad(dD):
    padV = ((1, 0), (0, 0))
    padU = ((0, 0), (1, 0))
    gD = np.array([np.pad(np.diff(dD, axis=0), padV, 'reflect', reflect_type='odd'), \
                   np.pad(np.diff(dD, axis=1), padU, 'reflect', reflect_type='odd')])
    return gD


def conjProd(iwA, i):
    if i == 0:
        return iwA[i]

    initA = iwA[0] / np.abs(iwA[0])
    for k in range(1,i):
        initA = (iwA[k] / np.abs(iwA[k])) * initA.conj()

    return initA

if __name__ == '__main__':


### --- load Testimages

    #a = np.asarray(Image.open("Images/xy_256x256.001.png").convert('L'))
    #b = np.asarray(Image.open("Images/xy_256x256.002.png").convert('L'))
    a = np.asarray(Image.open("relief0.png").convert('L'))
    b = np.asarray(Image.open("relief0_sc.png").convert('L'))

    rows, cols = a.shape

### create Parameter Plots ...
    fig2, ax2 = plt.subplots(2, 5, figsize=(15, 10), squeeze=True, sharex=False, sharey=False)
    plt.subplots_adjust(left=0.033, bottom=0.01, right=0.99, top=0.99, wspace=0.12, hspace=0.06)
    fig, ax = plt.subplots(4, 7, figsize=(15, 15), squeeze=True, sharex=False, sharey=False)
    plt.subplots_adjust(left=0.033, bottom=0.01, right=0.99, top=0.99, wspace=0.12, hspace=0.06)
    fig1, ax1 = plt.subplots(3, 5, figsize=(15, 15), squeeze=True, sharex=False, sharey=False)
    plt.subplots_adjust(left=0.033, bottom=0.01, right=0.99, top=0.99, wspace=0.12, hspace=0.06)

    halfX = cols // 2
    halfY = rows // 2

# Filter
    x = np.linspace(0, 255, cols).astype("int32")
    y = np.linspace(0, 255, rows).astype("int32")

    mx, my = np.meshgrid(x, y)
    angleMatr = np.angle((mx-128) + 1j * (my-128))
    dist = np.sqrt((mx-128)**2 + (my-128)**2)
    #dist = np.maximum(np.abs((mx-128)),np.abs((my-128)))
    halfPi = np.pi / 2
# angular filter
    anglularFilt = []
    aPr = 8
    haPr = aPr * 2
    faPr = haPr * 2
    for i in range(0,faPr):
        aStep = halfPi * (i * (1 / aPr) - 2)
        anglularFilt.append(np.cos(np.clip((angleMatr + aStep) * aPr, -halfPi, halfPi))**2)

# four octaves radial filter

    octaves = 16
    fH = 2 #np.sqrt(2)
    #oZwt = np.pi * 0.66666
    otDeg = np.array(-0.5 + 1j * np.sqrt(0.75)).max()
    radialFilt = []
    split = halfX
    step = halfX / octaves
    scaleFt = [np.exp((np.log(cols) / octaves) * oc) for oc in range(0,octaves+1)]



    for i in range(0,octaves+1):
        #radialFilt.append(np.cos(np.clip(np.pi * 2 * (dist / step - i), -halfPi, halfPi))**2)
        radialFilt.append(np.cos(np.clip(np.maximum((np.log(dist) * octaves) / np.log(cols/2),
                                                    (dist > 0)) - i, -1, 1) * np.pi * .5) ** 2)
        scaleFt.append(step * (i) + 1)
        ax2[0, 4].plot(radialFilt[-1][128,:], label=f"{i}")

# create plots
    ax2[0, 0].grid()
    ax2[0, 0].set_title("linear analysis 1")

    ax2[0, 1].set_title("linear analysis 2")

    ffA = fft.fftshift(fft.fft2(a))
    ffB = fft.fftshift(fft.fft2(b))

    #ax2[0, 2].set_title("ffA")
    #ax2[0, 2].imshow(np.log(np.abs(ffA)))
    #ax2[0, 2].imshow(a)

    #ax2[0, 3].set_title("ffB")
    #ax2[1, 2].imshow(np.log(np.abs(ffB)))
    #ax2[1, 2].imshow(b)

## Apply Complex Pyramid Filter

    iwA = []
    iwB = []
    iwF = []
    iwA2 = []

    iwC = []
    iwD = []
    iwC2 = []
    sD = []
    #sC = []

    sE = []
    sEc = []
    fBuf = np.zeros(a.shape)
    iwE = []
    iwE2 = []

    sect = 8
    iwCf = 1

    angle = (sect - (2 * haPr)) * np.pi / (2 * aPr)

    rr, cl = centerLine(128, 128, angle, ffA.shape)

    cs, ll, ss = sinCosLen(angle, rows, cols)

    for i in range(0, octaves * 3):
        iwE.append(np.exp(1j * (2* np.pi * (((i) * (my*ss+mx*cs) / rows)))))

    for i in range(0, octaves + 1):
        iwE2.append(np.exp(1j * (2* np.pi * (((scaleFt[i]) * (my * ss + mx * cs) / rows)))))

    for i in range(0, octaves + 1):

        scOff = np.power(otDeg, 1/ scaleFt[i])
        #angle = np.pi - (sect / haPr) * np.pi


        jp = sect
        jn = sect
        filter =  np.sum(np.stack(radialFilt)[0:i],axis=0) * anglularFilt[sect]
        if radialFilt[i].max() > 0.1:
            if filter.max() < 0.1:
                fp = fn = filter
                while np.all(fp < 0.05) and jp < (aPr*4-1):
                    jp += 1
                    fp = np.sum(np.stack(radialFilt)[0:i],axis=0) * anglularFilt[jp]
                while np.all(fn < 0.05) and jn > 0:
                    jn -= 1
                    fn = np.sum(np.stack(radialFilt)[0:i],axis=0) * anglularFilt[jn]

                dp = jp - sect + 1
                dn = sect - jn + 1
                fsum = (dn + dp)
                filter = (fp * (fsum - dp) + fn * (fsum - dn)) /  (fsum)
        else:
            filter = np.zeros(ffA.shape)

        iwA.append(fft.ifft2(fft.ifftshift(ffA * filter)))
        iwB.append(fft.ifft2(fft.ifftshift(ffB * filter)) )
        #iwF.append(fft.ifft2(fft.ifftshift(ffB * ffA.conj() * filter)))
        iwF.append(filter)

        if i > 0:
            #iwEoff = np.exp(1j * (scaleFt[i]*sC[-1]))

            niwA, mask = shiftMatrixA(iwA[i], sD[-1], rows, cols, np.pi - (sect / haPr) * np.pi, mx, my)
            iwA2.append(niwA)

            #oiwC = iwB[i]*iwE[iwAe[i]]  * (iwA[i]*iwE[iwBe[i]]*sEc[-1]).conj()
            oiwC = iwB[i]  * (iwA[i]).conj()
            oiwC += (oiwC == 0) * 1  # nullsafe
            uoiwC = oiwC / np.abs(oiwC)
            gradC = np.gradient(np.angle(uoiwC))

            #oiwD = iwB[i] * iwE[min(octaves*2,iwAe2[i])] * (iwA2[i] * iwE[min(octaves*2,iwBe2[i])]).conj()

            #oiwD = iwB[i] * iwB[i-1].conj() * iwA2[i].conj() * iwA2[i-1]
            oiwD = iwB[i] * iwA2[i].conj()

            oiwD += (oiwD == 0) * 1 # nullsafe


            iwC.append(uoiwC)
            iwD.append(oiwD / np.abs(oiwD))

            aoffs = -offsetAngle2D(np.angle(iwD[-1]), ss, cs, my, mx, 2)
            #if np.any(aoffs > 0):
                # oiwD = iwB[i] * iwE[2] * iwA2[i].conj()
                # oiwD += (oiwD == 0) * 1  # nullsafe
                # iwD[-1] = oiwD / np.abs(oiwD)
                # aoffs = -offsetAngle2D(np.angle(iwD[-1]), ss, cs, my, mx, 2)

            oavg = float(avg)
            avg = (aoffs.max()+aoffs.min())/2


            offsC = np.power(iwC[-1], 1 / scaleFt[i])

            aiwC = np.angle(iwC[-1])

            old_offsD = offsD
            offsD = iwD[-1] # np.power(iwD[-1], 1 / scaleFt[i]) * np.exp(-1j * aoffs / scaleFt[i])

            offsD = offsD #* old_offsD.conj()
            offsD += (offsD == 0) * 1
            aiwD *= offsD / abs(offsD) #offsD #+ shift


        else:


            avg = 0.0

            oavg = float(avg)

            absC = np.abs(iwA[i] * iwB[i])
            absC += (absC == 0) * 1
            iwC.append(iwB[i] * iwA[i].conj() / absC + (absC == 0) * 1)
            iwD.append(iwC[-1].copy())

            aiwD = offsD = iwD[-1].copy()
            aiwD += (aiwD == 0) * 1
            iwA2.append(iwA[i])

            sE.append(aiwD.copy())
            sEc.append(iwC[-1].copy())
            siwA = iwA[-1].copy()
            siwB = iwB[-1].copy()
        iwCf /= fH
        fBuf += filter

        #sD.append(sumAngle2D(aiwD,ss,cs,my,mx))
        sD.append(np.angle(aiwD))

    ax2[0, 1].grid()

    show_angle = False
    if show_angle:
        ax2[0, 1].plot(np.max((np.log(np.abs(ffA.real) + 1) * anglularFilt[sect])[rr, :], axis=1)
                       * np.sin(np.angle(np.average((ffA * anglularFilt[sect])[rr, :], axis = 1))), label='ffAr', color = (0.0, 0.0, 0.8))
        ax2[0, 1].plot(np.max((np.log(np.abs(ffA.imag) + 1) * anglularFilt[sect])[rr, :], axis=1)
                       * np.cos(np.angle(np.average((ffA.imag * anglularFilt[sect])[rr, :], axis = 1))), label='ffAi', color = (0.5, 0.5, 1.0))
        ax2[0, 1].plot(np.max((np.log(np.abs(ffB.real) + 1) * anglularFilt[sect])[rr, :], axis=1)
                       * np.sin(np.angle(np.average((ffB * anglularFilt[sect])[rr, :], axis = 1))), label='ffBr', color = (0.8, 0.0, 0.0))
        ax2[0, 1].plot(np.max((np.log(np.abs(ffB.imag) + 1) * anglularFilt[sect])[rr, :], axis=1)
                       * np.cos(np.angle(np.average((ffB.imag * anglularFilt[sect])[rr, :], axis = 1))), label='ffBi', color = (1.0, 0.5, 0.5))
    else:
        #ax2[0, 1].plot(np.max((np.log(np.abs(ffA) + 1) * anglularFilt[sect])[rr, :], axis=1), label='ffAr',color=(0.0, 0.0, 0.8))
        #ax2[0, 1].plot(np.max((np.log(np.abs(ffB) + 1) * anglularFilt[sect])[rr, :], axis=1), label='ffBr',color=(0.8, 0.0, 0.0))
        ax2[0, 1].plot(np.log(np.max(ffA[rr, 120:136], axis=1) + 1), label='ffA')
        ax2[0, 1].plot(np.log(np.max(ffB[rr, 120:136], axis=1) + 1), label='ffB')

    ax2[1, 1].plot(np.max((np.log(np.abs((ffB*ffA.conj()).real) + 1) * anglularFilt[sect])[rr, :], axis=1)
                   * np.sin(np.angle(np.average(((ffB*ffA.conj()) * anglularFilt[sect])[rr, :], axis = 1))), label='ffAr', color = (0.0, 0.0, 0.8))
    ax2[1, 1].plot(np.max((np.log(np.abs((ffB*ffA.conj()).imag) + 1) * anglularFilt[sect])[rr, :], axis=1)
                   * np.cos(np.angle(np.average(((ffB*ffA.conj()) * anglularFilt[sect])[rr, :], axis = 1))), label='ffAi', color = (0.5, 0.5, 1.0))
    ax2[1, 1].plot(np.max((np.log(np.abs((ffB*ffA.conj()).imag) + 1) * anglularFilt[sect])[rr, :], axis=1)
                   , label='ffAi', color = (0.8, 0.0, 0.0))
    ax2[1, 1].plot(np.angle(np.average(((ffB*ffA.conj()) * anglularFilt[sect])[rr, :], axis = 1)), label='ffAi', color = (1.0, 0.5, 0.5))

    ax2[1, 4].plot(np.angle(np.max((np.log(np.abs(ffA) + 1) * anglularFilt[sect])[rr, :], axis=1)+ 1j * np.max((np.log(np.abs(ffB) + 1) * anglularFilt[sect])[rr, :], axis=1)))

    ffAB_max = (ffA.max()+ffB.max())/2
    ax2[0, 2].imshow(np.log(np.abs(np.abs(ffAB_max*ffB/ffB.max())-np.abs(ffAB_max*ffA/ffA.max())) + 1)
                     * np.sign(np.abs(ffAB_max*ffB/ffB.max())-np.abs(ffAB_max*ffA/ffA.max())))

    ax2[1, 2].imshow(np.angle(np.abs(ffAB_max*ffB/ffB.max()) + 1j * np.abs(ffAB_max*ffA/ffA.max())))

    rO = octaves + 1
    iwCf = 1

    sDf = []
    sumC = []

    for i in range(0, min(min(len(iwA)-1, rO),15)):

        #ax2[1, 3].plot(sumAngle(iwD[i][rr, cl]), label=f'{i}')
        # ax2[0, 4].plot(sumAngle(naiwD[rr, cl]), label=f'{i}')
        ax2[0, 3].plot(sumAngle((iwA[i] * iwE[i].conj())[rr, cl]) / scaleFt[i], label=f'{i}a')
        ax2[1, 3].plot(sumAngle((iwB[i] * iwE[i].conj())[rr, cl]) / scaleFt[i], label=f'{i}b')


        row = i // 5
        col = i % 5

        #y = np.linspace(0, 255, 256)


        a_sum = True
        if a_sum == True:

            # ax1[row, col].plot(np.max((np.log(np.abs(ffA.real) + 1) * iwF[i])[rr, :], axis=1)
            #                    * np.sin(np.angle(np.average((ffA * iwF[i])[rr, :], axis = 1))), label='ffAr', color = (0.0, 0.0, 0.8))
            # ax1[row, col].plot(np.max((np.log(np.abs(ffA.imag) + 1) * iwF[i])[rr, :], axis=1)
            #                    * np.cos(np.angle(np.average((ffA * iwF[i])[rr, :], axis = 1))), label='ffAi', color = (0.5, 0.5, 1.0))
            # ax1[row, col].plot(np.max((np.log(np.abs(ffB.real) + 1) * iwF[i])[rr, :], axis=1)
            #                    * np.sin(np.angle(np.average((ffB * iwF[i])[rr, :], axis = 1))), label='ffBr', color = (0.8, 0.0, 0.0))
            # ax1[row, col].plot(np.max((np.log(np.abs(ffB.imag) + 1) * iwF[i])[rr, :], axis=1)
            #                    * np.cos(np.angle(np.average((ffB * iwF[i])[rr, :], axis = 1))), label='ffBi', color = (1.0, 0.5, 0.5))

            #ax1[row, col].plot(np.angle((iwA[i]*sEc[i-1])[rr,cl]),label=f'1')
            #ax1[row, col].plot(sumAngle((iwA[i] * np.exp(1j * (scaleFt[i] * sC[i - 1]))* iwE2[i].conj())[rr, cl]), label=f'1')
            #ax1[row, col].plot(sumAngle(iwA[i][rr, cl]) - np.average(sumAngle(iwA[i][rr, cl])[127:129]),label=f'2')
            #ax1[row, col].plot(sumAngle(iwB[i][rr, cl]) - np.average(sumAngle(iwB[i][rr, cl])[127:129]), label = f'3')
            #ax1[row, col].plot(sumAngle(iwF[i][rr, cl]), label=f'4')
            # ax1[row, col].plot(np.angle(iwA[i][rr, cl] * iwE[i][rr, cl].conj()),,label=f'3')
        #ax1[row, col].plot(np.angle((iwA[i]*sEc[max(0,i-1)]*((iwA[max(0,i-1)]*sEc[max(0,i-2)])**2).conj()))[rr, cl], label=f'3')
            # ax1[row, col].plot(np.angle(iwB[i][rr, cl] * iwE[i][rr, cl].conj()),label=f'4')
        #ax1[row, col].plot(np.angle((iwB[i]*iwB[max(0,i-1)].conj()**2))[rr, cl], label=f'4')
            #ax1[row, col].plot(sumAngle(iwA2[i][rr, cl] )- np.average(sumAngle(iwA2[i][rr, cl])[127:129]),label=f'5')
            #ax1[row, col].plot(np.angle(np.power(iwB[i][rr, cl] * iwA2[i][rr, cl].conj(), 1 / (scaleFt[i]))), label=f'6')
            ax1[row, col].plot(sumAngle((iwD[i]*iwE[2])[rr, cl]) - np.average(sumAngle((iwD[i]*iwE[2])[rr, cl])[127:129]) , label=f'8')
                                    #- np.average(np.angle(-iwD[i][127:129, 127]))
                                         #*np.exp(-1j * (np.pi * ((iwAe2[i] - iwBe2[i])/scaleFt[i]) * ((my * ss + mx * cs) / rows - 0.5))))[rr, cl]), label=f'8')
            ax1[row, col].plot(sD[i][rr, cl],label=f'9')

            #ax1[row, col].plot(np.angle(iwE[2])[rr, cl], label=f'10')

            # ax1[row, col].plot(sumAngle(iwE2[i][rr, cl]), label=f'8')
            #ax1[row, col].plot(np.angle((iwB[i] * iwE[iwBe[i]].conj())[rr, cl]), label=f'11')
            #ax1[row, col].plot(np.angle(iwA[i][rr, cl]) + np.angle(iwB[i][rr, cl]), label=f'11')
            if i > 0:
                #if iwBe[i] == 0 :
                ax1[row, col].plot(sumAngle((iwA[i]*iwE[2].conj())[rr, cl]) - np.average(sumAngle((iwA[i]*iwE[2].conj())[rr, cl])[127:129]), label=f'12')
                #ax1[row, col].plot(sumAngle((iwA[i] )[rr, cl]) - np.average(sumAngle((iwA[i])[rr, cl])[127:129]), label=f'14')

                    #ax1[row, col].plot(sumAngle((iwA[i])[rr, cl]), label=f'12')
                    #ax1[row, col].plot(sumAngle((iwA[i] * iwE[i].conj())[rr, cl]), label=f'12')
                #else:
                #    ax1[row, col].plot(sumAngle((-(iwA[i]*conjProd(iwA,i).conj()))[rr, cl]), label=f'12')
                    #ax1[row, col].plot(sumAngle((-iwA[i])[rr, cl]), label=f'12')
                    #ax1[row, col].plot(sumAngle((-iwA[i] * iwE[i].conj())[rr, cl]), label=f'12')

                #if iwAe[i] == 0:
                #    ax1[row, col].plot(np.angle(iwB[i][rr, cl] * iwA[i][rr, cl].conj()), label=f'14')
                #else:
                #    ax1[row, col].plot(np.angle(iwB[i][rr, cl].conj() * -iwA[i][rr, cl]), label=f'14')

                #if iwAe[i] == 0 and iwBe[i] == 0:
                ax1[row, col].plot(sumAngle((iwB[i]*iwE[2].conj())[rr, cl]) - np.average(sumAngle((iwB[i]*iwE[2].conj())[rr, cl])[127:129]), label=f'13')
                    #ax1[row, col].plot(sumAngle((iwB[i])[rr, cl]), label=f'13')
                    #ax1[row, col].plot(sumAngle((iwB[i] * iwE[i].conj())[rr, cl]), label=f'13')
                #else:
                #    ax1[row, col].plot(sumAngle((-(iwB[i]*conjProd(iwB,i).conj()))[rr, cl]), label=f'13')
                    #ax1[row, col].plot(sumAngle((-iwB[i])[rr, cl]), label=f'13')
                    #ax1[row, col].plot(sumAngle((-iwB[i] * iwE[i].conj())[rr, cl]), label=f'13')
                ax1[row, col].plot(sumAngle((iwA2[i]*iwE[2].conj())[rr, cl]) - np.average(sumAngle((iwA2[i]*iwE[2].conj())[rr, cl])[127:129]), label=f'15')

            #ax1[row, col].plot(np.angle(iwE[i][rr, cl]), label=f'15')

            #ax1[row, col].plot(np.angle(iwA2[i][rr, cl]), label=f'14')
            #ax1[row, col].plot(sumAngle(iwC[i][rr, cl]*(iwC[max(0,i-1)][rr, cl]**2)),label=f'8')
            #ax1[row, col].plot(sumAngle(np.power(iwD[i][rr, cl],1/scaleFt[i])),label=f'7')
            #ax1[row, col].plot(sumAngle(iwE[i][rr, cl]),label=f'8')

            #sDf.append(gradientfix(iwD[i], 1 / scaleFt[i], ss, cs, mx, my))

        else:
            #ax1[row, col].plot(np.pi * 2 * i + np.angle(aiwC[rr, cl]), label=f'2')
            #ax1[row, col].plot(np.pi * 2 * i + np.angle(iwA[i][rr, cl] * iwE[rr, cl].conj()), label=f'3')
            #ax1[row, col].plot(np.pi * 2 * i + np.angle(iwB[i][rr, cl] * iwE[rr, cl].conj()), label=f'4')
            #ax1[row, col].plot(np.pi * 2 * i + np.angle(iwA2[i][rr, cl] * iwE[rr, cl].conj()), label=f'5')
            ax1[row, col].plot(np.pi * 2 * 1 + np.angle(iwC[i][rr, cl]), label=f'6')

            ax1[row, col].plot(np.pi * 2 * 1 + np.angle(iwD[i][rr, cl]), label=f'7')
            ax1[row, col].plot(np.pi * 2 * i + np.angle(iwE[i][rr, cl]), label=f'8')
            #ax1[row, col].plot(np.pi * 2 * 2 + np.angle(sC[i][rr, cl]), label=f'9')
            ax1[row, col].plot(np.pi * 2 * 2 + np.angle(sD[i][rr, cl]), label=f'10')
            ax1[row, col].plot(np.pi * 2 * 1 + np.angle((sEc[i])[rr, cl]), label=f'11')
            ax1[row, col].plot(np.pi * 2 * 1 + np.angle((sE[i])[rr, cl]), label=f'12')
            ax1[row, col].plot(np.pi * 2 * 1 + np.angle(
                (iwC[i - 1] * iwC[i])[rr, cl]), label=f'13')


        ax1[row, col].legend()
        ax1[row, col].grid()

    ax2[0, 4].legend()

    for i in range(0, rO):
        ax2[0, 0].plot(sD[i][rr,cl], label=f'sc({i})')
        #ax2[0, 0].plot(np.angle(sC[i][rr, cl]), label=f'sc({i})')

    ax2[0, 0].legend()
    ax2[0, 0].grid()
    ax2[0, 0].set_title("recreated slice")

    ax2[1, 2].set_title("phase highest octave")

    ax2[0, 2].set_title("fft - imageB")


    ax2[1, 0].legend()
    ax2[1, 0].grid()

    ax2[1, 1].set_title("phase ImageB")

    #ax2[1, 1].plot(cl, rr, 'r-')

    ax2[1, 0].imshow(sD[-1],vmin=0,vmax=1)

    start = timer()
    print("Start: ...")
    # vuMap, acBuf, pyrFilt = pyramidFlow(ffA, ffB, angleMatr, dist, my, mx)
    # vuMap, acBuf, pyrFilt = pyramidFlow(ffA, ffB, angleMatr, dist, my, mx)
    print("... Stop.")
    end = timer()
    print(f"delta:{timedelta(seconds=end-start)}")
    ax2[1, 3].set_title("filter sum")


    #ax2[1, 2].plot(cl, rr, 'r-')

# ------ iA / iB
    ax2[0, 3].set_title("imageA")
    ax2[0, 4].set_title("imageB")


    ax2[1, 0].legend()
    ax2[1, 0].grid()

    nvec = 20
    step = max(rows // nvec, cols // nvec)

    vy, vx = np.mgrid[:rows:step, :cols:step]
  #  u_ = vuMap[::step, ::step, 1]

  #  v_ = vuMap[::step, ::step, 0]

 #   vuMag = np.abs(vuMap[:, :, 1] + 1j * vuMap[:, :, 0])

 #   ax2[1, 4].imshow(vuMag)
 #   ax2[1, 4].quiver(vx, vy, u_, v_, color='r', units='dots',
 #             angles='xy', scale_units='xy', lw=3)

  #  rr, cl = centerLine(128, 128, np.pi / 4, ffA.shape)
    #ax2[1, 4].plot(cl, rr, 'r-')
#    for i in range(0, min(10, rO)):
 #       vuM = np.abs(acBuf[i][0] + 1j * acBuf[i][1])
  #      ax2[0, 1].plot(vuM[vuM.shape[0] - rr - 1, vuM.shape[1] - cl - 1])
   #     ax2[0, 1].plot(vuM[128])

    ax2[1, 4].set_title("uv-vector map")
    ax2[1, 2].set_title("ifft-offset")

    aiwC = np.zeros(iwC[0].shape, dtype="complex128")
    for i in range(0, min(14, rO)):
        row = (i) // 7
        col = (i) % 7

        ax[row, col].imshow(sD[i]) #, vmin=-np.pi, vmax=np.pi)

        #ax[row+2, col].imshow(np.angle(iwA[i]) + np.angle(iwB[i]))
        #ax[row + 2, col].imshow(np.angle(iwB[i]) + np.angle(iwA[i].conj()))

        #ax[row+2, col].imshow(iwF[i])

        #if i < 7:
        #     ax[2, col].imshow(np.angle(iwA[i].conj()*iwB[i]))
        #ax[row+2, col].imshow(np.angle(iwB[i])+np.angle(iwA[i]))
            #ax[2, col].imshow(np.angle((iwA[i]*((iwA[max(0,i-1)]*iwC[max(0,i-1)])**2).conj())), vmin=-np.pi, vmax=np.pi)
            #ax[3, col].imshow(np.angle((iwB[i]*((iwB[max(0,i-1)])**2).conj())), vmin=-np.pi, vmax=np.pi)
            #ax[2, col].imshow(np.log(np.abs(ffA.conj()*ffB)))#*iwF[i])

            #ax[3, col].imshow(np.angle(ffA.conj()*ffB))#*iwF[i])

            #ax[2, col].imshow(np.angle(iwA[i] * iwF[i].conj()))
            #ax[3, col].imshow(np.angle(iwB[i] * iwF[i].conj()))
        #ax[row + 2, col].imshow(np.abs(fft.ifft2(fft.ifftshift(np.exp(1j * ((np.pi*np.sqrt(2)*256*(my*np.sin(np.pi/4)+mx*np.cos(np.pi/4)) / (rows))))*np.sum(np.stack(iwF)[0:i],axis=0)))))
        ax[row + 2, col].imshow(np.c_[np.angle(-iwA2[i])[:,0:127],np.angle(-iwB[i])[:,128:255]])
        #ax[row + 2, col].imshow(radialFilt[i+3] * anglularFilt[sect])

    # u_ = acBuf[i][1, ::step, ::step]
    # v_ = acBuf[i][0, ::step, ::step]
    #ax[row, col].quiver(vx, vy, u_, v_, color='r', units='dots',
        #        angles='xy', scale_units='xy', lw=3)
    # ax[row, col].imshow(np.abs(acBuf[i][1] * 1j * acBuf[i][0]))

        #ax[row, col].imshow(np.angle(iwA[i]))
        #ax[row + 2, col].imshow(np.angle(iwB[i]))
        #ax[row+2, col].imshow(np.angle(iwB[i]*iwA[i].conj()))
        #ax[row+2, col].imshow(np.angle(iwB[i]*(iwA[i]*sC[i-1]).conj()))
        #ax[row, col].imshow(np.dstack((ffA.real*radialFilt[i] * anglularFilt[sect],ffA.imag*radialFilt[i] * anglularFilt[sect], np.zeros(ffA.shape) )))
        # if i > 0:
        #     ax[row + 2, col].imshow(np.angle(iwC[i] * sEc[i-1].conj()))
        # else:
        #ax[row + 2, col].imshow(np.angle(iwC[i]), vmin=-np.pi, vmax=np.pi)
        #ax[row + 2, col].imshow(np.angle(iwD[i]), vmin=-np.pi, vmax=np.pi)
        #ax[row + 2, col].imshow(sDf[i], vmin=-np.pi, vmax=np.pi)
        #ax[row + 2, col].imshow(np.dstack((ffB.real*radialFilt[i] * anglularFilt[sect],ffB.imag*radialFilt[i] * anglularFilt[sect],np.zeros(ffA.shape))))
        #ax[row + 2, col].imshow(radialFilt[i] * anglularFilt[sect])
        #if i > 1:
        #    ax[row+2, col].imshow(np.angle(((iwB[i]*iwA[i].conj())/np.abs(iwB[i]*iwA[i]))*np.power(sC[i-1].conj(),(i-1)**2)), vmin=-np.pi, vmax=np.pi)
        #else:
        #    ax[row + 2, col].imshow(np.angle(iwB[i]*iwA[i].conj()), vmin=-np.pi, vmax=np.pi)
        #ax[row + 2, col].imshow(sC[i])
    plt.show()
    print("pass00")
    plt.waitforbuttonpress()
