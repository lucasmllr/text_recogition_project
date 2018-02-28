import numpy as np
from DisjointSet import DisjointSet
import processing
import matplotlib.pyplot as plt


def find_blobs(img):

    height = img.shape[0]
    width = img.shape[1]
    size = height * width

    img = img.reshape(size)
    stencil = np.zeros(size, dtype=int)
    labels = DisjointSet()

    # first pass
    for i in range(size):

        if img[i] != 0:

            # if a neighboring pixel is labeled the investigated pixel is given the same label
            # Note: when iterating from top left to bottom right indices to the right bottom of investigated
            # pixel cannot be labeled before this pixel
            for j in [i-1, i-width, i-width-1, i-width+1]:

                if j < 0 or j >= size:
                    continue

                if stencil[j] != 0 and stencil[i] == 0: # connected
                    stencil[i] = stencil[j]
                elif stencil[j] != 0 and stencil[j] != stencil[i]: # conflict
                    labels.unite(stencil[i], stencil[j])
                else: # no connection nor conflict
                    continue

            # if no neighboring pixel is labeled the investigated pixel is give a new label
            if stencil[i] == 0:
                new_label = labels.next()
                stencil[i] = new_label
                labels.add(new_label)

    # second pass
    eq = labels.get_equivalents()
    print('equivalent labels:', eq)
    print('parents', labels.parents)
    for label in eq.keys():
        stencil[stencil == label] = eq[label]

    # reshaping stencil
    stencil = stencil.reshape((height, width))

    # Todo: construct boxes around letters

    # Todo: assure right order of letters in case of height difference
    
    return stencil


if __name__ == "__main__":

    img = processing.load_img('data/9.jpg')
    #plt.imshow(img)
    #plt.show()

    img = processing.threshold(img, t=0.52)

    img = find_blobs(img)
    plt.imshow(img)
    plt.show()


