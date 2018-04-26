import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2 as cv
from arguments import Arguments
from components import Components
from component_evaluation import eliminate_insiders, filter_neighbors, eliminate_tiny


def extract_mser(img, args):
    '''
    Function using mser module from the openCV library to extract maximally stable extremal regions.
    Additionally MSERs that lie inside others or have no neighbors are eliminated.

    Args:
        img (ndarray): image from which MSERs are to be extracted
        args (Arguments instance):
        filter (Bool): whether to filter the extracted MSERs

    Returns:
        an instance of the Components class
    '''
    # find MSERs
    mser = cv.MSER_create()
    mser.setMinArea(args.min_area)
    mser.setMaxArea(args.max_area)
    mser.setDelta(args.delta)
    msers, bboxes = mser.detectRegions(img)

    #bringing bboxes into the format [x_min, x_max, y_min, y_max]
    bboxes = [[box[1], box[1]+box[3], box[0], box[0]+box[2]] for box in bboxes]
    components = Components(regions=msers, boxes=bboxes, img=img.astype(np.float32)/np.max(img))

    #components.show_img()

    eliminate_insiders(components)

    filter_neighbors(components, args)

    eliminate_tiny(components, args.pixel_threshold_factor)

    return components


if __name__ == '__main__':

    args = Arguments()

    img = cv.imread('data_test/2.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    scaled = cv.resize(gray, None, fx=0.5, fy=0.5)
    print(np.min(scaled))
    print('scaled image:', scaled.shape)

    components = extract_mser(scaled, args)
    components.show_img()

