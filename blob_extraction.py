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
from component_evaluation import eliminate_insiders, filter_neighbors


def find_blobs(raw_img, args):
    '''function performing two dimensional connected component analysis on an image.

    Args:
        img (ndarray): original image to be analyzed
        args (Arguments instance): defined the threshold value to binarze the image

    Returns:
        an instance of the Components class, a stencil containing the final labels of components,
        and a stencil containing the labels before eliminating equivalences
    '''

    # dimensions
    height = raw_img.shape[0]
    width = raw_img.shape[1]

    img = processing.threshold(raw_img, args)

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

    # uncomment to print show labels after first pass
    # first_pass = deepcopy(stencil.reshape((height, width)))

    # second pass to eliminate equivalences
    eq = labels.get_equivalents()
    for label in eq.keys():
        stencil[stencil == label] = eq[label]

    # reshaping stencil
    stencil = stencil.reshape((height, width))
    # SCIPY VARIANT
    #stencil = measure.label(img, background=0)

    # count pixels in blobs, calculate median to filter blobs
    final_labels = np.arange(1, np.max(stencil)+1)
    pixel_counts = []
    for label in final_labels:
        pixel_counts.append(np.sum(stencil == label))
    pixel_counts = np.array(pixel_counts)
    min_allowed_pixels = np.median(pixel_counts[pixel_counts > 0]) / 5  # arbitrary; seems to work well

    # filter final lables and stencil
    final_labels = np.array(final_labels)[pixel_counts >= min_allowed_pixels]
    new_stencil = np.zeros_like(stencil)
    for i, label in enumerate(final_labels):
        new_stencil[stencil == label] = i+1
    stencil = new_stencil

    # construct boxes around letters
    bounding_boxes = get_bboxes(stencil)
    # chars = get_chars_from_boxes(raw, bounding_boxes)
    # extract characters from image in correct order
    #chars = []
    #bounding_boxes = []
    #while boxes:
    #    box = heappop(boxes)
    #    chars.append(raw[box[2]:box[3], box[0]:box[1]])
    #    bounding_boxes.append(box)
    return Components(boxes=bounding_boxes, img=raw_img, stencil=stencil)


def get_bboxes(stencil, labels=None):
    '''
    find the bounding boxes in a stencil of labeled regions.
    Args:
        stencil (ndarray): image with different pixel values for each region
        labels: region labels in the image

    Returns:
        a list of bboxes in the format [x_min, x_max, y_min, y_max]
    '''
    boxes = []
    if labels is None:
        labels = range(1, np.max(stencil)+1)
    for label in labels:
        pixels = np.argwhere(stencil == label)
        # check if blob is present
        if len(pixels) == 0:
            continue
        y_min = np.min(pixels[:, 1])
        y_max = np.max(pixels[:, 1])
        x_min = np.min(pixels[:, 0])
        x_max = np.max(pixels[:, 0])
        #heappush(boxes, (x_min, x_max, y_min, y_max))
        boxes.append((x_min, x_max, y_min, y_max))
    return np.array(boxes, dtype=np.int32)


if __name__ == "__main__":

    args = Arguments()

    img = processing.load_img('data_test/1.jpg')

    scale = 1
    orig = rescale(deepcopy(img), scale)
    img = rescale(img, scale)
    components = find_blobs(img, args)
    boxes = components.bboxes()
    stencil = components.get_stencil()

    print()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(orig, cmap='gray')
    for box in boxes:
        rect = Rectangle((box[2], box[0]), box[3]-box[2], box[1]-box[0], fill=False, edgecolor='red')
        ax.add_patch(rect)
    plt.show()




