import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.transform import pyramid_laplacian


if __name__ == '__main__':
    image = data.astronaut()
    rows, cols, dim = image.shape
    pyramid = tuple(pyramid_laplacian(image, downscale=2, channel_axis=-1))

    composite_image = np.zeros((rows, cols + cols // 2, 3), dtype=np.double)

    composite_image[:rows, :cols, :] = pyramid[0]*10+0.5

    i_row = 0
    for p in pyramid[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p*10+0.5
        i_row += n_rows

    fig, ax = plt.subplots()
    ax.imshow(composite_image, vmin='-2', vmax='2')
    plt.show()
    plt.waitforbuttonpress()