import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from skimage.transform import resize
from skimage.filters import threshold_sauvola as sauvola
import cv2 as cv
from arguments import Arguments


def load_img(path):
    '''loads an image, normalizes and inverts it and returs it as an ndarray'''

    img = imread(path)
    #average channels to reduce to one channel
    img = np.mean(img, axis=2)
    #normalization
    img /= np.max(img)
    #inverting
    img = 1 - img

    return img


def threshold(img, args):
    '''returs a binary image of img with parameters defined in args.'''

    below = img <= args.blob_t
    above = img > args.blob_t

    if args.cut_bottom:
        img[below] = 0
    if args.cut_top:
        img[above] = 1

    return img


def sauvola_threshold(img, args):
    '''returns a binary image resulting from performing sauvola thresholding on img with parameters defined in args.'''

    t = sauvola(img, args.window, args.k, args.r)
    binary = img > t

    return binary


def rescale(img, args):
    '''rescales img to the inout shape of the model defined in args.'''

    size = args.input_shape

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

    args = Arguments()
    img = cv.imread('fotos/ex02.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, None, fx=0.25, fy=0.25)
    gray = np.divide(gray, np.max(gray))

    binary = sauvola_threshold(gray, args)

    plt.imshow(binary)
    plt.show()