import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import imread


def load_img(path):

    img = imread(path)
    #average channels to reduce to one channel
    img = np.mean(img, axis=2)
    #normalization
    img /= np.max(img)

    return img


def threshold(img, t, top=True, bottom=True):

    below = img <= t
    above = img > t

    if bottom:
        img[below] = 0
    if top:
        img[above] = 1

    return img


if __name__ == '__main__':

    img = load_img('data/0.jpg')
    print(img.shape)
    plt.imshow(img)
    plt.show()

    img = threshold(img, 0.5, bottom=False)
    plt.imshow(img)
    plt.show()