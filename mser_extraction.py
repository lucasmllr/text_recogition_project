import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2 as cv
from heapq import heappop, heappush
import processing
from DisjointSet import DisjointSet
from arguments import Arguments
from bbox_evaluation import is_inside


def extract_mser(img, args):
    '''
    Function using mser module from the openCV library to extract maximally stabel extremal regions.
    Additionally MSERs that ly inside others are omitted.

    Args:
        img (openCV image): input image
        args (Arguments opbject): args.min_are, args.max_area and args.delta are needed

    Returns:
        bounding boxes of filtered MSERs
    '''

    # find MSERs
    mser = cv.MSER_create()
    mser.setMinArea(args.min_area)
    mser.setMaxArea(args.max_area)
    mser.setDelta(args.delta)
    _, bboxes = mser.detectRegions(img)

    # filter MSERs

    # ordering boxes by size b/c in DisjointSet the smallest label becomes the equivalence class' representative
    # labels will be indices of boxes_by_size
    heap = []
    for count, box in enumerate(bboxes):
        b = (- box[2] * box[3], count, box)
        heappush(heap, b)
    boxes_by_size = [heappop(heap) for _ in range(len(heap))]

    box_labels = DisjointSet(n_labels=len(boxes_by_size))

    # pairwise check of bounding boxes. once per pair.
    for a in range(len(boxes_by_size)):
        for b in range(a+1, len(boxes_by_size)):
            if is_inside(boxes_by_size[a][2], boxes_by_size[b][2]):
                box_labels.unite(a, b)

    survivors = []
    eq = box_labels.get_equivalents()
    for i in range(len(eq)):
        if eq[i] not in survivors:
            survivors.append(eq[i])

    box_candidates = [boxes_by_size[i][2] for i in survivors]

    # extracting character candidates
    chars = []
    for box in box_candidates:
        x_min = box[0]
        y_min = box[1]
        x_max = x_min + box[2]
        y_max = y_min + box[3]
        chars.append(img[y_min:y_max, x_min:x_max])

    return chars, box_candidates


if __name__ == '__main__':

    args = Arguments()

    img = cv.imread('data/2.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    chars, boxes = extract_mser(gray, args)

    print(boxes)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(gray, cmap='gray')
    for region in boxes:
        #print(region)
        x = region[0]
        y = region[1]
        width = region[2]
        height = region[3]
        rect = Rectangle((x, y), width, height, edgecolor='red', fill=False)
        ax.add_patch(rect)
    plt.show()

    for char in chars:
        plt.imshow(char)
        plt.show()