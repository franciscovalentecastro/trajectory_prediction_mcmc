import numpy as np
import matplotlib.pyplot as plt


# functions to show an images
def imshow(img):
    npimg = img.detach().numpy()
    plt.figure(figsize=(12, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)),
               cmap='hot',
               interpolation='nearest')
    plt.axis('off')

    # Show image
    plt.show()
