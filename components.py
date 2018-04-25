from heapq import heappush, heappop
import numpy as np
from DisjointSet import DisjointSet
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import processing


class Component():

    def __init__(self, region, bbox, img):

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
        return Rectangle((self.y-1, self.x-1), self.h+1, self.w+1, edgecolor='red', fill=False)

    def extract(self, padding=1):
        char_img = self.img[max(0, self.x - padding):min(self.x + self.w + padding, self.img.shape[0]),
                            max(0, self.y - padding):min(self.y + self.h + padding, self.img.shape[1])]
        # scale image such that char is approximately 1, background is approximately 0
        color = self.color
        bg_color = (np.mean(char_img) * char_img.size - color * len(self.region)) / (char_img.size - len(self.region))
        if color < bg_color:
            char_img = -char_img
        char_img = (char_img - np.min(char_img)) / (np.max(char_img) - np.min(char_img))
        return char_img

class Components():

    def __init__(self, boxes, img, regions=None, stencil=None, lines=None):
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
        return [component.bbox for component in self.chars]

    def regions(self):
        return [component.region for component in self.chars]

    def set_lines(self, lines):
        assert sum(len(l) for l in lines) == len(self.chars)
        self.lines = lines

    def extract(self, args=None):
        if args is None:
            return [c.extract() for c in self.chars]
        else:
            return [processing.rescale(c.extract(), args) for c in self.chars]

    def extract_lines(self, args=None):
        assert self.lines is not None
        if args is None:
            return [[self.chars[i].extract() for i in line] for line in self.lines]
        else:
            return [[processing.rescale(self.chars[i].extract(), args) for i in line] for line in self.lines]

    def regions_from_stencil(self):
        regions = []
        for i in range(1, np.max(self.stencil) + 1):
            region = np.flip(np.argwhere(self.stencil == i), axis=1)
            regions.append(region)
        return regions

    def generate_stencil(self):
        stencil = np.zeros_like(self.img, dtype=np.int32)
        for i, region in enumerate(self.regions()):
            stencil[region[:, 1], region[:, 0]] = i + 1
        self.stencil = stencil

    def get_stencil(self):
        if self.stencil is None:
            self.generate_stencil()
        return self.stencil

    def show_img(self, axes=None):
        if axes is None:
            ax = plt.gca()
        else:
            ax = axes
        ax.imshow(self.img, cmap='gray')
        for c in self.chars:
            ax.add_patch(c.get_rect())
        if axes is None:
            plt.show()

    def __len__(self):
        return len(self.chars)
