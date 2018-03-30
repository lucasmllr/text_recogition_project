import numpy as np
from DisjointSet import DisjointSet
import processing
import matplotlib.pyplot as plt
from copy import deepcopy
from heapq import heappush, heappop
from arguments import Arguments


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

    if args.documentation:
        plt.imshow(img)
        plt.title('thresholded image with t={}'.format(t))
        plt.show()

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

    first_pass = deepcopy(stencil.reshape((height, width)))

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
        pixels = np.argwhere(stencil == label)
        y_min = np.min(pixels[:, 0])
        y_max = np.max(pixels[:, 0]) + 1
        x_min = np.min(pixels[:, 1])
        x_max = np.max(pixels[:, 1]) + 1
        heappush(boxes, (x_min, x_max, y_min, y_max))

    # extract characters from image in correct order
    chars = []
    bounding_boxes = []
    while boxes:
        box = heappop(boxes)
        chars.append(raw[box[2]:box[3], box[0]:box[1]])
        bounding_boxes.append(box)

    return chars, bounding_boxes, stencil, first_pass


if __name__ == "__main__":

    args = Arguments()
    args.documentation = True

    #for i in range(args.n):

    img = processing.load_img('data/41.jpg')
    orig = deepcopy(img)

    blobs, boxes, stencil, first_pass = find_blobs(img, args)

    fig = plt.figure()
    fig.suptitle('iamge 41')
    o = fig.add_subplot(131)
    f = fig.add_subplot(132)
    t = fig.add_subplot(133)
    o.imshow(orig, cmap='gray')
    o.set_title('original')
    f.imshow(first_pass)
    f.set_title('first pass')
    t.imshow(stencil)
    t.set_title('second pass')
    plt.show()



