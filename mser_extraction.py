import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2 as cv
from arguments import Arguments
from components import Components
from component_evaluation import eliminate_insiders, filter_neighbors


def extract_mser(img, args):
    '''
    Function using mser module from the openCV library to extract maximally stable extremal regions.
    Additionally MSERs that ly inside others are omitted.

    Args:
        img (openCV image): input image
        args (Arguments object): args.min_area, args.max_area, args.delta, args.invert, args.normalize are needed

    Returns:
        extracted characters and bboxes in the format (x_min, y_min, width, height)
    '''

    # find MSERs
    mser = cv.MSER_create()
    mser.setMinArea(args.min_area)
    mser.setMaxArea(args.max_area)
    mser.setDelta(args.delta)
    msers, bboxes = mser.detectRegions(img)

    components = Components(msers, bboxes, img)

    eliminate_insiders(components)

    filter_neighbors(components, args)

    return components


if __name__ == '__main__':

    args = Arguments()

    img = cv.imread('fotos/ex02.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    scaled = cv.resize(gray, None, fx=0.5, fy=0.5)
    print('scaled image:', scaled.shape)

    components = extract_mser(scaled, args)
    boxes = components.bboxes()

    print('#boxes', len(boxes))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(scaled, cmap='gray')
    for i, region in enumerate(boxes):
        x = region[0]
        y = region[1]
        width = region[2]
        height = region[3]
        rect = Rectangle((x, y), width, height, edgecolor='red', fill=False)
        ax.add_patch(rect)
    plt.show()

    chars, pos = components.extract()
    fig = plt.figure()
    for i in range(50):
        ax = fig.add_subplot(5, 10, i+1)
        ax.imshow(chars[i])
    plt.show()

