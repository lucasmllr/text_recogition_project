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

                if stencil[j] != 0 and stencil[i] == 0: # connection
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
        x_min = np.min(pixels[:, 0])
        x_max = np.max(pixels[:, 0]) + 1
        y_min = np.min(pixels[:, 1])
        y_max = np.max(pixels[:, 1]) + 1
        boxes.append((x_min, x_max, y_min, y_max))

    # Todo: assure right order of letters in case of height difference

    return stencil, boxes


if __name__ == "__main__":

    img = processing.load_img('data/9.jpg')
    #plt.imshow(img)
    #plt.show()

    img = processing.threshold(img, t=0.52)

    img, boxes = find_blobs(img)
    plt.imshow(img)
    plt.show()

    print(boxes)
    print(boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3])

    box1 = img[boxes[0][0]:boxes[0][1], boxes[0][2]:boxes[0][3]]
    plt.imshow(box1)
    plt.show()



