from scipy import signal
from scipy import datasets
from PIL import Image

import numpy as np
from scipy import signal

def pooling(mat,ksize,method='max',pad=(0,0)):
    '''Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize
    mask = np.pad(np.ones((m-2*pad[0], n-2*pad[1])).astype(np.float32), [pad, pad])

    _ceil=lambda x,y: int(np.ceil(x/float(y)))
    mask_rs = None

    if pad != (0,0):
        ny=_ceil(m,ky)
        nx=_ceil(n,kx)
        new_shape = (ny, ky, nx, kx) + mat.shape[2:]
        size=(ny*ky, nx*kx)+mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
        mask_pad=np.full(size,np.nan)
        mask_pad[:m, :n, ...] = mask

        reshaped = mat_pad.reshape(new_shape)
        mask_rs = mask_pad.reshape(new_shape)
        # reshaped = mat_rs / mask_rs
    else:
        ny=m//ky
        nx=n//kx
        mat_pad=mat[:ny*ky, :nx*kx, ...]
        new_shape = (ny, ky, nx, kx) + mat.shape[2:]
        reshaped = mat_pad.reshape(new_shape)

    if method=='max':
        result=np.nanmax(reshaped,axis=(1,3))
    else:
        result=np.nanmean(reshaped,axis=(1,3))

    if pad != (0,0):
        excl = result / np.nanmean(mask_rs,axis=(1,3))
        return excl

    return result


image = Image.open("../../gauss.tif")
arr = np.asarray(image)

a = datasets.ascent()
mask = np.pad(np.ones(a.shape).astype(np.float32),[(64,64),(64,64)])
b = np.array([[ -3-3j, 0-10j,  +3 -3j],
                   [-10+0j, 0+ 0j, +10 +0j],
                   [ -3+3j, 0+10j,  +3 +3j]])
#c = np.empty(262144).reshape([512,512])

## conv = signal.convolve(a, b)
fftc = signal.fftconvolve(a, arr)


import matplotlib.pyplot as plt
# import matplotlib.colors as cols

#lum = cols._vector_magnitude(arr).flatten().reshape([128,128])

fig, ((ax_orig,ay_orig), (ax_mag,ay_mag), (ax_conv,ay_conv)) = plt.subplots(3,2, figsize=(6, 15))
ax_orig.imshow(a)
ay_orig.imshow(a)
ax_orig.set_title('Original')
ay_orig.set_title('Original')
#ax_orig.set_axis_off()

# ax_mag.imshow(np.absolute(fftc*0.1+128))
ax_mag.imshow(np.absolute(fftc/fftc.max()))
ay_mag.imshow(np.absolute(fftc[64:576,64:576]/fftc.max()))
ax_mag.set_title('Convoluted')
ay_mag.set_title('Convoluted Cropped')

#ax_mag.set_axis_off()
pool_fftc = pooling(fftc,(32,32), method='avg', pad=(16,16))

#ax_conv.imshow(cols.hsv_to_rgb(np.dstack((np.absolute(np.angle(fftc)/np.pi+1)/2,np.ones(fftc.shape), abs_fftc/abs_fftc.max()))))
ax_conv.imshow(pool_fftc)
sx, sy = pool_fftc.shape[:2]
ay_conv.imshow(pool_fftc[2:(sx-2),2:(sy-2)])
ax_conv.set_title('Pooled')
ay_conv.set_title('Pooled Masked')

fig.show()
input("Press Enter to continue... ")
pass