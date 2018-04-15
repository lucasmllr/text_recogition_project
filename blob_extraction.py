import numpy as np
from DisjointSet import DisjointSet
import processing
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from copy import deepcopy
from heapq import heappush, heappop
from skimage.transform import rescale
from arguments import Arguments
from components import Components


def find_blobs(img, args):

    t = args.blob_t

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

    # components
    regions = []
    bboxes = []

    # second pass to eliminate equivalences
    eq = labels.get_equivalents()
    for label in eq.keys():
        stencil[stencil == label] = eq[label]

    # reshaping stencil
    stencil = stencil.reshape((height, width))

    # construct boxes around letters
    final_labels = labels.final_labels()
    boxes = []
    for label in final_labels:
        if label == 0: continue  # background
        pixels = np.argwhere(stencil == label)
        x_min = np.min(pixels[:, 0])
        x_max = np.max(pixels[:, 0]) + 1
        y_min = np.min(pixels[:, 1])
        y_max = np.max(pixels[:, 1]) + 1
        width = x_max - x_min
        height = y_max - y_min
        regions.append(pixels)
        bboxes.append([y_min, x_min, height, width])

    return comps, stencil, first_pass


if __name__ == "__main__":

    args = Arguments()

    #for i in range(args.n):

    img = processing.load_img('plot_data/6.jpg')
    orig = deepcopy(img)
    zeros = np.zeros((img.shape[0], 1))
    orig = np.concatenate((orig, zeros), axis=1)

    #img = rescale(img, 0.1)
    comps, stencil, first_pass = find_blobs(img, args)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(orig)
    for i, box in enumerate(comps.bboxes()):
        rect = Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor='red')
        ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    #fig.savefig('plots/cc_bboxes.pdf', bbox_inches='tight')
    plt.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.imshow(first_pass)
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.show()
    #fig2.savefig('plots/cc_first_pass.pdf', bbox_inches='tight')

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.imshow(stencil)
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.show()
    #fig3.savefig('plots/cc_stencil.pdf', bbox_inches='tight')





