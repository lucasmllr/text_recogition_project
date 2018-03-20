import numpy as np


def projection(img, angle):

    # dimensions
    height = img.shape[0]
    width = img.shape[1]
    size = height * width

    # projection plane
    e = np.array([np.cos(angle), np.sin(angle)])

    # pixel coordinates
    C = np.mgrid[:height, :width].reshape((2, size)).T

    # projection
    proj = np.dot(C, e).T

    # bin
