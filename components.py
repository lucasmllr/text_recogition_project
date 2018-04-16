from heapq import heappush, heappop
import numpy as np
from DisjointSet import DisjointSet


class Component():

    def __init__(self, region, bbox, img):

        self.region = region
        self.bbox = bbox
        self.x = self.bbox[0]
        self.y = self.bbox[1]
        self.w = self.bbox[2]
        self.h = self.bbox[3]
        self.A = self.w * self.h
        self.asp = self.w / self.h
        self.color = np.mean(img[self.region[:, 1], self.region[:, 0]])


class Components():

    def __init__(self, regions, boxes, img):
        self.candidates = [Component(r, b, img) for r, b in zip(regions, boxes)]
        self.img = img

    def bboxes(self):
        return [component.bbox for component in self.candidates]

    def regions(self):
        return [component.region for component in self.candidates]

    def extract(self):
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