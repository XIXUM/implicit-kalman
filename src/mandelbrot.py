import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x,y=np.ogrid[-2:1:500j,-1.5:1.5:500j]
    fig2, ax2 = plt.subplots(2, 5, figsize=(15, 10), squeeze=True, sharex=False, sharey=False)

    print('')
    print('Grid set')
    print('')

    c=x + 1j*y
    z=0
    i = 0
    threshold = 2

    for g in range(20):
            print('Iteration number: ',g)
            if((g-1) % 2 == 0):
                row = (i) // 5
                col = (i) % 5
                i += 1
                mask = np.abs(z) < threshold
                ax2[row,col].imshow(mask.copy().T,extent=[-2,1,-1.5,1.5])
                ax2[row, col].set_title(f"Iteration g = {g}")
            z=z**2 + c

    print('')
    print('Plotting using imshow()')
    #plt.imshow(mask.T,extent=[-2,1,-1.5,1.5])

    print('')
    print('plotting done')
    print('')

    plt.gray()

    print('')
    print('Preparing to render')
    print('')

    plt.show()
    plt.waitforbuttonpress()