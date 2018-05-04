from heapq import heappush, heappop
import numpy as np
from DisjointSet import DisjointSet
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import processing


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
            bbox (list): in the format [x_min, x_max, y_min, y_max] of the bounding box
            img (ndarray): reference to the image the component is contained in
        '''

        self.region = region
        self.bbox = bbox
        self.x = self.bbox[0]
        self.y = self.bbox[2]
        self.w = self.bbox[1] - bbox[0]
        self.h = self.bbox[3] - bbox[2]
        self.A = self.w * self.h
        self.asp = self.w / self.h
        self.img = img
        self.color = np.mean(self.img[self.region[:, 1], self.region[:, 0]])

    def get_rect(self):
        '''returns a matplotlib Rectange patch for of component's bbox.'''
        return Rectangle((self.y-1, self.x-1), self.h+1, self.w+1, edgecolor='red', fill=False)

    def extract(self, padding=1, min_height=0):
        '''
        extracts section of the component's bbox plus a buffer from the image.
        Args:
            padding: additional padding of the bbox
            min_height:

        Returns:
            an image of the extracted region
        '''
        off = max(0, min_height - (self.w + padding))
        char_img = self.img[max(0, self.x - padding - off):min(self.x + self.w + padding, self.img.shape[0]),
                            max(0, self.y - padding):min(self.y + self.h + padding, self.img.shape[1])]
        # scale image such that char is approximately 1, background is approximately 0
        color = self.color
        bg_color = (np.mean(char_img) * char_img.size - color * len(self.region)) / (char_img.size - len(self.region))
        if color < bg_color:
            char_img = -char_img
        char_img = (char_img - np.min(char_img)) / (np.max(char_img) - np.min(char_img))
        return char_img

class Components():
    '''a class to manage all components found by the mser algorithm.

    Attributes:
        cadidates (list): list of instances of the Component class
        img (ndarray): reference to the image
    '''

    def __init__(self, boxes, img, regions=None, stencil=None, lines=None):
        '''
        initializes a Components instance.

        Args:
            boxes: list of bounding boxes
            img: the image
            regions: list of regions. A region is a list of coordinate tuples.
            stencil: image in shape of img containing region labels as pixel values
            lines:
        '''
        if regions is not None:
            self.chars = [Component(r, b, img) for r, b in zip(regions, boxes)]
            self.stencil = stencil
        else:
            assert stencil is not None
            self.stencil = stencil
            regions = self.regions_from_stencil()
            assert len(regions) == len(boxes)
            self.chars = [Component(r, b, img) for r, b in zip(regions, boxes)]
        self.img = img
        self.lines = lines
        # print(f'Components object initialized with {len(self.bboxes())} components')

    def bboxes(self):
        '''returns a list of bboxes of all stored components.'''
        return [component.bbox for component in self.chars]

    def regions(self):
        '''returns a list of regions of all stored components. A region is a list of coordinate tubles.'''
        return [component.region for component in self.chars]

    def set_lines(self, lines):
        ''' ... '''
        assert sum(len(l) for l in lines) == len(self.chars)
        self.lines = lines

    def extract(self, args=None, use_line_heights=True):
        '''returns a list of rescaled images of the extracted components in the image.

        Args:
            args: Arguments instance
            use_line_heights (bool): whether to use line height as height. if flase the images are rescaled to args.input_shape

        Returns:
            a list of reshaped images of all components
            '''
        if args is None:
            return [c.extract() for c in self.chars]
        else:
            if not use_line_heights:
                return [processing.rescale(c.extract(), args) for c in self.chars]
            else:
                return [c for line in self.extract_lines(args, use_line_heights) for c in line]

    def extract_lines(self, args=None, use_line_heights=True):
        '''
        extracts images for lines of text found in the image.

        Args:
            args: Arguments instance
            use_line_heights:

        Returns:
            a list of image of lines
        '''
        assert self.lines is not None
        if args is None:
            return [[self.chars[i].extract() for i in line] for line in self.lines]
        if use_line_heights:
            line_heights = [max([self.chars[i].w for i in line]) for line in self.lines]
        else:
            line_heights = [0]*len(self.lines)
        return [[processing.rescale(self.chars[i].extract(min_height=line_heights[line_id]), args) for i in line]
                for line_id, line in enumerate(self.lines)]

    def get_spaces(self, threshold_factor=.25):
        '''
        finds space positions based on the horizontal distances between components and a threshold computed from the
        median horizontal spacing.

        Args:
            threshold_factor: factor for median to determine threshold

        Returns:
            list of character positions after which to add spaces
        '''
        spaces = []
        space_threshold = self.median_bbox_width() * threshold_factor
        for line in self.lines:
            current = []
            for i in range(len(line)-1):
                c1 = self.chars[line[i]]
                c2 = self.chars[line[i+1]]
                if c2.y - (c1.y + c1.h) > space_threshold:
                    current.append(i)
            spaces.append(current)
        return spaces

    def regions_from_stencil(self):
        '''
        extracts regions from a self.stencil.

        Returns:
            a list of regions, where a region is a list of coordinate tubles.
        '''
        regions = []
        for i in range(1, np.max(self.stencil) + 1):
            region = np.flip(np.argwhere(self.stencil == i), axis=1)
            regions.append(region)
        return regions

    def generate_stencil(self):
        '''creates a stencil from self.regions.'''
        stencil = np.zeros_like(self.img, dtype=np.int32)
        for i, region in enumerate(self.regions()):
            stencil[region[:, 1], region[:, 0]] = i + 1
        self.stencil = stencil

    def get_stencil(self):
        '''
        creates and returns a stencil from self.regions

        Returns:
            a stencil of the components
        '''
        if self.stencil is None:
            self.generate_stencil()
        return self.stencil

    def median_bbox_width(self):
        '''returns median of compontent's bbox heights.'''
        return np.median([c.h for c in self.chars])

    def show_img(self, axes=None):
        '''shows self.img with bboxes of components in self.chars

        Args:
            axes (matplotlib axes instance): if not None plot is added to this axes instance
            '''
        if axes is None:
            ax = plt.gca()
        else:
            ax = axes
        ax.imshow(self.img, cmap='gray')
        for c in self.chars:
            ax.add_patch(c.get_rect())
        if axes is None:
            plt.show()

    def regions_by_size(self):
        '''returns regions in order of their size'''
        heap = []
        for i, c in enumerate(self.chars):
            heappush(heap, (-len(c.region), i, c.region))
        return [c[2] for c in heap]

    def __len__(self):
        return len(self.chars)
