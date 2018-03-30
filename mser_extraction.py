import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2 as cv
from heapq import heappop, heappush
import processing
from DisjointSet import DisjointSet


def extract_mser(img):

    # find MSERs
    mser = cv.MSER_create()
    mser.setMinArea(5)
    mser.setMaxArea(500)
    mser.setDelta(15)
    _, bboxes = mser.detectRegions(img)

    # filter MSERs

    # ordering boxes by size b/c in DisjointSet the smallest label becomes the equivalence class' representative
    # labels will be indices of boxes_by_size
    heap = []
    for count, box in enumerate(bboxes):
        b = (- box[2] * box[3], count, box)
        heappush(heap, b)
    boxes_by_size = [heappop(heap) for _ in range(len(heap))]
    print(boxes_by_size)

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

    return [boxes_by_size[i][2] for i in survivors]


def is_inside(a, b):
    '''
    helper function to determine wheter bounding box b lies inside a
    '''

    # box coordinates
    a_x = a[0]
    a_y = a[1]
    a_w = a[2]
    a_h = a[3]
    b_x = b[0]
    b_y = b[1]
    b_w = b[2]
    b_h = b[3]

    if a_x <= b_x and a_y <= b_y:  # bottom left
        # top right
        if a_x + a_w >= b_x + b_w:
            if a_y + a_h >= b_y + b_h:
                return True

    return False

if __name__ == '__main__':

    img = cv.imread('data/4.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    regions = extract_mser(gray)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(gray, cmap='gray')
    for region in regions:
        #print(region)
        x = region[0]
        y = region[1]
        width = region[2]
        height = region[3]
        rect = Rectangle((x, y), width, height, edgecolor='red', fill=False)
        ax.add_patch(rect)
    plt.show()