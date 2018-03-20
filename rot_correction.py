import numpy as np
import matplotlib.pyplot as plt
from processing import load_img
from scipy.ndimage import rotate


def correct_rot(img, n_angles=10, n_bins=11, angle=20.):
    '''
    Function to rotate an input image according to the minimum entropy of its projection along the horizontal axis.
    Entropy is evaluated on a grid.

    Args:
        img (ndarray): 2d image array
        n_angles (int): number of projection angles
        n_bins (int): number of bins for projection histograms. Should be odd.
        angle (float): abs of min and max angle (degrees) in whose boundaries n_angles projections are evaluated.
    Returns:
        An ndarray of the rotated image padded with zeros
    '''

    # dimensions
    height = img.shape[0]
    width = img.shape[1]
    size = height * width
    O = np.array([height / 2, width / 2])  # origin
    angle = angle * np.pi / 180

    # bins
    if n_bins % 2 == 0: n_bins += 1  # assure odd number of bins
    bin_width = np.sqrt(height**2 + width**2) / n_bins
    min_bin = -int(n_bins/2)
    max_bin = int(n_bins/2)

    # array of angles
    angles = np.linspace(-angle, angle, n_angles, dtype=np.float32)

    # array of normalized vectors IN projection planes with shape: 2 x n_angles
    e = np.array([np.cos(angles), np.sin(angles)])

    # pixel coordinates relative to origin with shape: size x 2
    C = np.mgrid[:height, :width].reshape((2, size)).T
    C = C - O

    # bin indices through projections
    # bins has shape: n_angles x size
    proj = np.dot(C, e).T
    bins = np.array(np.round(proj / bin_width), dtype=np.int)

    #visualize_bins = bins.reshape(n_angles, height, width)  # nice to visualize projections
    #for i in range(visualize_bins.shape[0]):
    #    plt.imshow(visualize_bins[i])
    #    plt.show()

    # bin counts
    flat_img = img.reshape((height * width))
    counts = np.zeros((n_angles, n_bins))

    for i in range(n_angles):
        for j in range(min_bin, max_bin+1):
            bin_pixels = np.argwhere(bins[i, :] == j).reshape(-1).tolist()
            counts[i, j] = np.sum(flat_img[bin_pixels])

    # normalization
    counts /= np.sum(img)

    # entropy (H = - Sum P * log(P), 0 if P=0)
    A = np.zeros((n_angles, n_bins))
    A[np.nonzero(counts)] = np.log(counts[np.nonzero(counts)])  # logarithms of non-zero normalized bin counts
    H = - np.sum(np.multiply(counts, A), axis=1)

    # turning angle
    angle = angles[np.argmin(H)]
    angle = - angle * 180 / np.pi

    # rotating image
    img = rotate(img, angle)
    proj = counts[np.argmin(H)]

    return img, proj


if __name__ == "__main__":

    img = load_img('data/4.jpg')
    rotated, proj = correct_rot(img, n_angles=50, n_bins=70, angle=10.)

    plt.imshow(img)
    plt.show()

    plt.imshow(rotated)
    plt.grid(True)
    plt.show()