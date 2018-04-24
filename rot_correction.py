import numpy as np
import matplotlib.pyplot as plt
from processing import load_img
from scipy.ndimage import rotate
from arguments import Arguments
from matplotlib2tikz import save


def correct_rot(img, args):
    '''
    Function to rotate an input image according to the minimum entropy of its projection along the horizontal axis.
    Entropy is evaluated on a grid.

    Args:
        img (ndarray): 2d image array
        args (Arguments instance): out of which the following are required
            args.n_angles (int): number of projection angles
            args.n_bins (int): number of bins for projection histograms. Should be odd.
            args.angle (float): abs of min and max angle (degrees) in whose boundaries n_angles projections are evaluated.
    Returns:
        An ndarray of the rotated image padded with zeros
    '''

    # arguments
    angle = args.angle
    n_angles = args.n_angles
    n_bins = args.n_bins

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
            counts[i, j - min_bin] = np.sum(flat_img[bin_pixels])  # smallest bin indx gets array inx zero

    # normalization
    counts /= np.sum(img)

    # entropy (H = - Sum P * log(P), 0 if P=0)
    A = np.zeros((n_angles, n_bins))
    A[np.nonzero(counts)] = np.log(counts[np.nonzero(counts)])  # logarithms of non-zero normalized bin counts
    H = - np.sum(np.multiply(counts, A), axis=1)

    # plotting a histogram
    #x = np.arange(-args.angle, args.angle, 2*args.angle/(len(H)))
    #plt.ylabel('H', fontsize=20)
    #plt.xlabel('Angle / °', fontsize=20)
    #plt.tick_params(direction='in')
    #plt.xticks([-20, 0, 20], fontsize=20)
    #plt.yticks([])
    #plt.plot(x, H)
    #plt.savefig('plots/entropy.pdf', bbox_inches='tight')
    #save('plots/entropy.tex')

    # turning angle
    angle = angles[np.argmin(H)]
    angle = - angle * 180 / np.pi

    # rotating image
    img = rotate(img, angle)

    return img


if __name__ == "__main__":

    args = Arguments()
    args.n_angles = 50
    args.n_bins = 100

    img = load_img('data/5.jpg')
    rotated = correct_rot(img, args)

    plt.imshow(img[0:150, 0:200], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    #plt.savefig('plots/to_be_rotated.pdf', bbox_inches='tight')
    plt.show()

    plt.imshow(rotated[50:200, 50:250], cmap='gray')
    plt.grid(True)
    plt.xticks([])
    plt.yticks([])
    #plt.savefig('plots/rotated.pdf', bbox_inches='tight')
    plt.show()
