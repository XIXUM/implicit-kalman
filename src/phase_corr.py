import numpy as np
from skimage.registration import phase_cross_correlation, optical_flow_tvl1, optical_flow_ilk
from PIL import Image
from matplotlib import pyplot as plt
from scipy import stats

a = np.asarray(Image.open("Images/xy_checker2.0001.png").convert('L'))
b = np.asarray(Image.open("Images/xy_checker2.0002.png").convert('L'))

# pixel precision, subpixel also available
shift, error, diffphase = phase_cross_correlation(a, b)

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(3, 3, 1)
ax2 = plt.subplot(3, 3, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(3, 3, 3)
ax4 = plt.subplot(3, 3, 4)
ax5 = plt.subplot(3, 3, 5)
ax6 = plt.subplot(3, 3, 6)
ax7 = plt.subplot(3, 3, 7)
ax8 = plt.subplot(3, 3, 8)
ax9 = plt.subplot(3, 3, 9)

ax1.imshow(a, cmap='gray')
#ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(b, cmap='gray')
#ax2.set_axis_off()
ax2.set_title('Offset image')

# the output of a cross-correlation
image_product = np.fft.fft2(a) * np.fft.fft2(b).conj()
cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
ax3.imshow(cc_image.real)
#ax3.set_axis_off()
ax3.set_title("Cross-correlation")

print(f'Detected pixel offset (y, x): {shift}')


# # Calculate the upsampled DFT, again to show what the algorithm is doing
# # behind the scenes.  Constants correspond to calculated values in routine.
# # See source code for details.
# cc_image = _upsampled_dft(image_product, 150, 100, (shift*100)+75).conj()
# ax3.imshow(cc_image.real)
# ax3.set_axis_off()
# ax3.set_title("Supersampled XC sub-area")


# print(f'Detected subpixel offset (y, x): {shift}')


# --- Compute the optical flow
v, u = optical_flow_tvl1(a, b, attachment=15, tightness=0.3, num_warp=15, num_iter=15, tol=0.0001, prefilter=True)
# v, u = optical_flow_ilk(a, b, radius=20, num_warp=20, gaussian=True)

# --- Compute flow magnitude
norm = np.sqrt(u ** 2 + v ** 2)

fig2, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
fig3D, (ax3D1, ax3D2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
ax0.imshow(a, cmap='gray')
ax0.set_title("Sequence image sample")
#ax0.set_axis_off()

# --- Quiver plot arguments

nvec = 20  # Number of vectors to be displayed along each image dimension
nl, nc = a.shape
step = max(nl//nvec, nc//nvec)

y, x = np.mgrid[:nl:step, :nc:step]
u_ = u[::step, ::step] 
v_ = v[::step, ::step]

ax1.imshow(norm)
ax1.quiver(x, y, u_, v_, color='r', units='dots',
           angles='xy', scale_units='xy', lw=3)
ax1.set_title("Optical flow magnitude and vector field")
#ax1.set_axis_off()
fig2.tight_layout()

# mx = np.arange(0, a.shape[1], 1)
# my = np.arange(0, a.shape[0], 1)
mx, my = np.meshgrid(np.arange(0, a.shape[1], 1), np.arange(0, a.shape[0], 1))
ax3D1.plot_surface(mx, my,u)
ax3D1.set_title("OpticalFlow-U")
ax3D2.plot_surface(mx, my,v)
ax3D2.set_title("OpticalFlow-V")

#average horizontal shift in v
vL = np.average(v[0])
vH = np.average(v[v.shape[0]-1])
vM = np.average(v[v.shape[0]//2-1])

ax4.plot(v[0], label='v-top')
ax4.plot(v[v.shape[0]-1], label='v-bottom')
ax4.plot(v[v.shape[0]//2-1], label='v-mid')
ax4.legend()
ax4.plot(u[0], label='u-top')
ax4.plot(u[u.shape[0]-1], label='u-bottom')
ax4.plot(u[u.shape[0]//2-1], label='u-mid')
ax4.legend()
vGrad = np.gradient(v)
uGrad = np.gradient(v)

ax5.plot(vGrad[1][0], label='v-top')
ax5.plot(vGrad[1][v.shape[0]-1], label='v-bottom')
ax5.plot(vGrad[1][v.shape[0]//2-1], label='v-mid')
ax5.legend()


ax7.imshow(vGrad[1], vmax=0.12, vmin=-0.12)
ax8.imshow(vGrad[0], vmax=0.12, vmin=-0.12)
ax9.imshow(uGrad[1], vmax=0.12, vmin=-0.12)

# determine via linear regression
hLine = np.arange(0,v.shape[1])
vHres = stats.linregress(hLine, v[v.shape[0]-1])
vStart = vHres.intercept
vEnd = vHres.intercept + vHres.slope * v.shape[1]
vMean =  (vStart+vEnd) / 2
# rLine = np.arange(vStart, vEnd)


fig3D.show()

plt.show()
plt.waitforbuttonpress()