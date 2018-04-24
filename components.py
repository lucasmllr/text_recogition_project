from heapq import heappush, heappop
import numpy as np
from DisjointSet import DisjointSet


class Component():
    '''a class to hold a single component as found by the mser or blob extraction algorithm.

     Attributes:
        region (tuple of lists): as accepted by __init__
        bbox (list): bounding box as accepted by __init__
        x (float): upper left x coordinate of bbox (horizontal dimension)
        y (flaot): upper left y coordinate of bbox (vertical dimension directed down)
        w (float): width of bbox (horizontal)
        h (float): height of bbox (vertical)
        A (float): area of bbox
        asp (float): aspect ration of bbox (w/h)
        color (float): mean color, i.e. intensity of all pixels in region
    '''

    def __init__(self, region, bbox, img):
        '''
        function to initialize an object.

        Args:
            region (tuple of lists): in the shape ([y_coordinates], [x_coordinates]) of pixels belongint to the region
                                    note: x is the horizontal coordinate
            bbox (list): in the format [x_min, y_min, height, width] of the bounding box
            img (ndarray): reference to the image the component is contained in
        '''

        self.region = region
        self.bbox = bbox
        self.x = self.bbox[0]
        self.y = self.bbox[1]
        self.w = self.bbox[2]
        self.h = self.bbox[3]
        self.A = self.w * self.h
        self.asp = self.w / self.h
        self.color = np.mean(img[self.region])


class Components():
    '''a class to manage all components found by the mser algorithm.

    Attributes:
        cadidates (list): list of instances of the Component class
        img (ndarray): reference to the image
    '''

    def __init__(self, regions, boxes, img):
        '''
        initializes a components object.

        Args:
            regions (list): of regions as taken by the Component class
            boxes (list): of bboxes as taken by the Component class
            img (ndarray): reference to the image all components are contained in.
        '''
        self.candidates = [Component(r, b, img) for r, b in zip(regions, boxes)]
        self.img = img

    def bboxes(self):
        '''returns a list of bounding boxes in the format [x_min, y_min, width, height].'''
        return [component.bbox for component in self.candidates]

    def regions(self):
        '''returns a list of pixels for each component'''
        return [component.region for component in self.candidates]

    def regions_by_size(self):
        '''returns regions in order of their size'''
        heap = []
        for i, c in enumerate(self.candidates):
            heappush(heap, (-len(c.region), i, c.region))
        return [c[2] for c in heap]

    def extract(self):
        '''extracts the components from the image.

         Returns:
             a list of the extracted parts plus a list of their top left positions
        '''
        chars = []
        pos = []
        for c in self.candidates:
            x_min = c.x
            y_min = c.y
            x_max = c.x + c.w
            y_max = c.y + c.h
            chars.append(self.img[y_min:y_max, x_min:x_max])
            pos.append((c.x, c.y))
        return chars, pos

    def __len__(self):
        return len(self.candidates)