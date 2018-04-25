import numpy as np
from heapq import heappush, heappop
from DisjointSet import DisjointSet


def by_bbox_size(components):

    heap = []
    for count, comp in enumerate(components.chars):
        b = (- comp.A, count, comp)
        heappush(heap, b)

    return [heappop(heap)[2] for _ in range(len(heap))]


def is_inside(a, b):
    '''
    helper function to determine whether bounding box of component b lies inside that of a
    '''

    if a.x <= b.x and a.y <= b.y:  # bottom left
        # top right
        if a.x + a.w >= b.x + b.w:
            if a.y + a.h >= b.y + b.h:
                return True

    return False


def eliminate_insiders(components):

    by_size = by_bbox_size(components)

    labels = DisjointSet(n_labels=len(by_size))

    # pairwise check of bounding boxes. once per pair.
    for a in range(len(by_size)):
        for b in range(a + 1, len(by_size)):
            if is_inside(by_size[a], by_size[b]):
                labels.unite(a, b)

    survivors = labels.final_labels()

    components.chars = [by_size[i] for i in survivors]

    return


def eliminate_tiny(components, threshold_factor=.2):

    threshold = np.median(np.array([len(c.region) for c in components.chars])) * threshold_factor
    components.chars = [c for c in components.chars if len(c.region) >= threshold]


def filter_neighbors(components, args):

    size = len(components)
    n = np.zeros((size, size), dtype=np.int)

    if not (args.distance or args.color or args.dims):
        raise('No measure for whether components are neighbors is enabled. Check arguments.')

    for i in range(size):
        for j in range(i + 1, size):

            a = components.chars[i]
            b = components.chars[j]

            if args.distance:
                t = 2 * min(max(a.w, a.h), max(b.w, b.h))
                if np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2) > t:
                    continue

            if args.dims:
                if max(a.A, b.A) / min(a.A, b.A) > args.t_A:
                    continue
                if max(a.asp, b.asp) / min(a.asp, b.asp) > args.t_asp:
                    continue

            if args.color:
                if np.absolute(a.color - b.color) > args.t_color:
                    continue

            n[i, j] = 1
            n[j, i] = 1

    n_count = np.sum(n, axis=0)
    components.chars = [components.chars[i] for i in range(size) if n_count[i] != 0]

    return
