"""
(c) 2023 felixschaller.com
extended pooling kernel with padding exclusion option
"""

import numpy as np

class Pooling2Dkernel:
    input = None
    ksize = None

    def __init__(self, input, kernelSize=(0,0)):
        self.input = input
        self.ksize = kernelSize

    def poolOP(self, method='max', pad=(0, 0)):
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

        m, n = self.input.shape[:2]
        ky, kx = self.ksize
        mask = np.pad(np.ones((m - 2 * pad[0], n - 2 * pad[1])).astype(np.float32), [pad, pad])

        _ceil = lambda x, y: int(np.ceil(x / float(y)))
        mask_rs = None

        if pad != (0, 0):
            ny = _ceil(m, ky)
            nx = _ceil(n, kx)
            new_shape = (ny, ky, nx, kx) + mat.shape[2:]
            size = (ny * ky, nx * kx) + mat.shape[2:]
            mat_pad = np.full(size, np.nan)
            mat_pad[:m, :n, ...] = mat
            mask_pad = np.full(size, np.nan)
            mask_pad[:m, :n, ...] = mask

            reshaped = mat_pad.reshape(new_shape)
            mask_rs = mask_pad.reshape(new_shape)
            # reshaped = mat_rs / mask_rs
        else:
            ny = m // ky
            nx = n // kx
            mat_pad = mat[:ny * ky, :nx * kx, ...]
            new_shape = (ny, ky, nx, kx) + mat.shape[2:]
            reshaped = mat_pad.reshape(new_shape)

        if method == 'max':
            result = np.nanmax(reshaped, axis=(1, 3))
        else:
            result = np.nanmean(reshaped, axis=(1, 3))

        if pad != (0, 0):
            excl = result / np.nanmean(mask_rs, axis=(1, 3))
            return excl

        return result