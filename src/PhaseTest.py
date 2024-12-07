
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, interpolate

if __name__ == '__main__':

    fscale = 1.2
    fshift = 0.2


    fig2, ax2 = plt.subplots(2, 3, figsize=(15, 8), squeeze=False, sharex=False, sharey=False)

    x = np.arange(-np.pi,np.pi,np.pi/100)
    x2 = np.arange(-np.pi*3,np.pi*3,np.pi/100)

    a = np.sin(x)
    b = np.sin(x-fshift)
    c = a-b
    d = np.sin(fscale*x-fshift)
    e = a-d

    rnd = (np.random.default_rng().random((x2.shape[0],))-0.5)*0.25
    noise = rnd + (np.sin(x2*4)+np.sin(x2*8)+np.sin(x2*10)++np.sin(x2*16)+np.sin(x2*20)+np.sin(x2*32))*0.25
    fn = interpolate.interp1d(x2,noise, kind='cubic')

    xa = a + fn(x)
    xb = np.roll(xa,16)
    xbs = b + fn(x - fshift)
    xd = d + fn(fscale*x - fshift)

    ax2[0,0].plot(a)
    ax2[0, 0].plot(b)
    ax2[0, 0].plot(c)
    ax2[0, 0].plot(d)
    ax2[0, 0].plot(e)
    ax2[0,0].grid()

    #ax2[0,1].plot(xa)
    #ax2[0, 1].plot(xb)
    #ax2[0, 1].plot(b)
    ax2[0,1].grid()

    ffa = fft.fftshift(fft.fft(a))
    ffb = fft.fftshift(fft.fft(b))
    ffc = fft.fftshift(fft.fft(c))
    ffd = fft.fftshift(fft.fft(d))
    ffe = fft.fftshift(fft.fft(e))

    #ax2[0, 2].plot(ffa[90:110].real)
    #ax2[0, 2].plot(ffa[90:110].imag)
    ax2[0, 2].plot(xa)
    ax2[0, 2].plot(xb)
    ax2[0, 2].plot(xd)
    ax2[0, 2].grid()

    ax2[1, 0].plot(ffb[90:110].real, label='ffb(r)')
    ax2[1, 0].plot(ffb[90:110].imag, label='ffb(i)')
    ax2[1, 0].plot(ffd[90:110].real, label='ffd(r)')
    ax2[1, 0].plot(ffd[90:110].imag, label='ffd(i)')
    ax2[1, 0].plot(ffe[90:110].real, label='ffe(r)')
    ax2[1, 0].plot(ffe[90:110].imag, label='ffe(i)')
    ax2[1, 0].grid()
    ax2[1, 0].legend()

    ax2[1, 1].plot(np.angle(ffa[90:110]), label='ffa')
    ax2[1, 1].plot(np.angle(ffb[90:110]), label='ffb')
    ax2[1, 1].plot(np.angle(ffd[90:110]), label='ffd')
    ax2[1, 1].grid()


    ffxa = fft.fftshift(fft.fft(xa))
    ffxb = fft.fftshift(fft.fft(xb))
    ffxd = fft.fftshift(fft.fft(xd))
    ffxbs = fft.fftshift(fft.fft(xbs))
    ax2[1, 1].plot(np.angle(ffxb[90:110]), label='ffxb')
    ax2[1, 1].plot(np.angle(ffxd[90:110]), label='ffxd')
    ax2[1, 1].legend()

    #ax2[0, 2].plot(np.absolute(ffxa))
    #ax2[0, 2].plot(np.absolute(ffxb))
    #ax2[0, 2].plot(np.absolute(ffxd))

    ax2[0, 1].plot(np.absolute(ffxa), label='ffxa')
    ax2[0, 1].plot(np.absolute(ffxb), label='ffxb')
    ax2[0, 1].plot(np.absolute(ffxd), label='ffxd-ffxa')
    ax2[0, 1].legend()

    # ax2[1, 2].plot(np.angle((ffa/ffb)*(np.absolute(ffb) > 1e-8))[90:110])
    # ax2[1, 2].plot(np.angle((ffxa/ffxb)*(np.absolute(ffxb) > 1e-8))[90:110])
    #ax2[1, 2].plot(np.angle((ffb * ffa.conj()) / np.absolute(ffb * ffa.conj())), label='ffa/ffb')
    #ax2[1, 2].plot(np.angle((ffd * ffa.conj()) / np.absolute(ffd * ffa.conj())), label='ffa/ffd')
    #ax2[1, 2].plot(np.angle((ffxb * ffxa.conj()) / np.absolute(ffxb * ffxa.conj())), label='ffxa/ffxb')
    ax2[1, 2].plot(np.angle((ffxbs * ffxa.conj()) / np.absolute(ffxbs * ffxa.conj())), label='ffxa/ffxbs')
    ax2[1, 2].plot(np.angle((ffxd * ffxa.conj()) / np.absolute(ffxd * ffxa.conj())), label='ffxd/ffxa')
    #ax2[1, 2].plot(np.angle(ffxbs / ffxa), label='ffxa/ffxbs (div)')
    ax2[1, 2].legend()
    ax2[1, 2].grid()
    xw =  np.arange(-100,100,1)
    ffas = ffa * np.exp(-1j * (xw) * fshift)

    #ax2[0, 1].plot(ffas[90:110].real)
    #ax2[0, 1].plot(ffas[90:110].imag)
    ax2[0, 2].grid()

    plt.show()
    plt.waitforbuttonpress()

