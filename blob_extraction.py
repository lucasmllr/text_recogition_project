import numpy as np
from DisjointSet import DisjointSet
import processing
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from copy import deepcopy
import cv2 as cv
from skimage.transform import rescale
from arguments import Arguments
from components import Components
from component_evaluation import eliminate_insiders, filter_neighbors


def find_blobs(img, args):
    '''function performing two dimensional connected component analysis on an image.

    Args:
        img (ndarray): original image to be analyzed
        args (Arguments instance): defined the threshold value to binarze the image

    Returns:
        an instance of the Components class, a stencil containing the final labels of components,
        and a stencil containing the labels before eliminating equivalences
    '''

    # dimensions
    height = img.shape[0]
    width = img.shape[1]

    raw = deepcopy(img)
    img = processing.threshold(img, args)

    # adding column of zeros to prevent left and right most blob
    # form being mistaken as one
    zeros = np.zeros((height, 1))
    img = np.concatenate((img, zeros), axis=1)
    width += 1

    size = height * width
    img = img.reshape(size)
    stencil = np.zeros(size, dtype=int)
    labels = DisjointSet(n_labels=1)

    # first pass
    for i in range(size):

        if img[i] != 0:

            # if a neighboring pixel is labeled the investigated pixel is given the same label
            # Note: when iterating from top left to bottom right indices to the right bottom of investigated
            # pixel cannot be labeled before this pixel
            for j in [i-1, i-width, i-width-1, i-width+1]:

                if j < 0 or j >= size:
                    continue

                if stencil[j] != 0 and stencil[i] == 0: # connection
                    stencil[i] = stencil[j]

                elif stencil[j] != 0 and stencil[j] != stencil[i]: # conflict
                    labels.unite(stencil[i], stencil[j])

                else:  # no connection nor conflict
                    continue

            # if no neighboring pixel is labeled the investigated pixel is give a new label
            if stencil[i] == 0:
                new_label = labels.next()
                stencil[i] = new_label
                labels.add(new_label)

    # uncomment to show labels after first pass
    first_pass = deepcopy(stencil.reshape((height, width)))

    # second pass to eliminate equivalences
    eq = labels.get_equivalents()
    for label in eq.keys():
        stencil[stencil == label] = eq[label]

    # reshaping stencil
    stencil = stencil.reshape((height, width))

    # components
    regions = []
    bboxes = []
    final_labels = labels.final_labels()

    for label in final_labels:

        if label == 0: continue  # background

        pixels = np.argwhere(stencil == label)
        region = (pixels[:, 0], pixels[:, 1])
        regions.append(region)

        x_min = np.min(pixels[:, 0])
        x_max = np.max(pixels[:, 0]) + 1
        y_min = np.min(pixels[:, 1])
        y_max = np.max(pixels[:, 1]) + 1
        width = x_max - x_min
        height = y_max - y_min
        bboxes.append([y_min, x_min, height, width])

    comps = Components(regions, bboxes, raw)

    #eliminate_insiders(comps)
    #filter_neighbors(comps, args)

    return comps, stencil, first_pass


if __name__ == "__main__":

    args = Arguments()

    img = cv.imread('fotos/ex02.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, None, fx=0.25, fy=0.25)
    gray = np.divide(gray, np.max(gray))

    #print('image range:', min(gray), max(gray))
    #plt.imshow(gray, cmap='gray')
    #plt.show()

    comps, _, _ = find_blobs(gray, args)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.imshow(gray, cmap='gray')
    for i, region in enumerate(comps.bboxes()):
        x = region[0]
        y = region[1]
        width = region[2]
        height = region[3]
        rect = Rectangle((x, y), width, height, edgecolor='red', fill=False)
        ax1.add_patch(rect)
    ax1.set_xticks([])
    ax1.set_yticks([])
    #plt.savefig('plots/cvimg_bad_threshold.pdf', bbox_inches='tight')
    plt.show()





