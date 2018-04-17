import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2 as cv
from arguments import Arguments
from components import Components
from component_evaluation import eliminate_insiders, filter_neighbors
from copy import deepcopy


def extract_mser(img, args, filter=True):
    '''
    Function using mser module from the openCV library to extract maximally stable extremal regions.
    Additionally MSERs that lie inside others or have no neighbors are eliminated.
    '''

    # find MSERs
    mser = cv.MSER_create()
    mser.setMinArea(args.min_area)
    mser.setMaxArea(args.max_area)
    mser.setDelta(args.delta)
    msers, bboxes = mser.detectRegions(img)

    # reshping region indices to numpy coordinates
    regions = []
    for i in range(len(msers)):
        mser = np.array(msers[i])
        region = (mser[:, 1], mser[:, 0])  # openCV uses corrdinates in opposite order as numpy...
        regions.append(region)

    components = Components(regions, bboxes, img)

    if filter:
        eliminate_insiders(components)
        filter_neighbors(components, args)

    return components


if __name__ == '__main__':

    args = Arguments()

    img = cv.imread('fotos/ex02.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, None, fx=0.5, fy=0.5)

    unfiltered = extract_mser(gray, args, filter=False)
    unfiltered_boxes = unfiltered.bboxes()

    filtered = extract_mser(gray, args)
    filtered_boxes = filtered.bboxes()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.imshow(gray, cmap='gray')
    for i, region in enumerate(unfiltered_boxes):
        x = region[0]
        y = region[1]
        width = region[2]
        height = region[3]
        rect = Rectangle((x, y), width, height, edgecolor='red', fill=False)
        ax1.add_patch(rect)
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.show()
    #plt.savefig('plots/cvimg_mser_unfiltered.pdf', bbox_inches='tight')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.imshow(gray, cmap='gray')
    for i, region in enumerate(filtered_boxes):
        x = region[0]
        y = region[1]
        width = region[2]
        height = region[3]
        rect = Rectangle((x, y), width, height, edgecolor='red', fill=False)
        ax2.add_patch(rect)
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.show()
    #plt.savefig('plots/cvimg_mser_filtered.pdf', bbox_inches='tight')

