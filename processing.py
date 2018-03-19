import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from skimage.transform import resize


def load_img(path):

    img = imread(path)
    #average channels to reduce to one channel
    img = np.mean(img, axis=2)
    #normalization
    img /= np.max(img)
    #inverting
    img = 1 - img

    return img


def threshold(img, t, top=True, bottom=True):

    below = img <= t
    above = img > t

    if bottom:
        img[below] = 0
    if top:
        img[above] = 1

    return img


def rescale(img, size=10):

    # add zeros to make img square
    max_edge = max(img.shape[0], img.shape[1])

    diff_0 = max_edge - img.shape[0]
    diff_1 = max_edge - img.shape[1]
    left = int(np.floor(diff_1 / 2))
    right = int(np.ceil(diff_1 / 2))
    top = int(np.floor(diff_0 / 2))
    bottom = int(np.ceil(diff_0 / 2))
    padded = np.pad(img, ((bottom, top), (left, right)), 'constant', constant_values=0)

    resized = resize(padded, (size, size))

    return resized


if __name__ == '__main__':

    a = np.array([[1, 1], [1, 1], [1, 1]])
    padded = rescale(a)
    print(padded.shape)
