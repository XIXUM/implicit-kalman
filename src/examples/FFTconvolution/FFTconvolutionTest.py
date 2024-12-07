from timeit import default_timer as timer
from datetime import timedelta

import numpy as np
from scipy import signal

a = np.random.randn(10**5)
c = np.frombuffer(np.random.randn(262144)).reshape([512,512])
d = np.array([[ -3, 0,  +3],
                   [-10, 0, +10],
                   [ -3, 0,  +3]])
b = np.random.randn(10**3)

print('Time required for normal discrete convolution:')
start = timer()
resultConv = signal.convolve(c, d)
end = timer()
print(timedelta(seconds=end-start))

print('Time required for FFT convolution:')
start = timer()
resultFFTC = signal.fftconvolve(a, b)
end = timer()
print(timedelta(seconds=end-start))
