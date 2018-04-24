import numpy as np
from heapq import heappush, heappop
from DisjointSet import DisjointSet


def by_bbox_size(components):
    '''returns a list of the components ordered by bbox size'''

    heap = []
    for count, comp in enumerate(components.candidates):
        b = (- comp.A, count, comp)
        heappush(heap, b)

    return [heappop(heap)[2] for _ in range(len(heap))]


def is_inside(a, b):
    '''determines whether the bounding box of component b lies inside that of a

    Args:
        a (Component objects): the outer one
        b (Component objects): the inner one

    Returns:
         Boolean whether bbox of component b lies inside that of component a
'''

    if a.x <= b.x and a.y <= b.y:  # bottom left
        # top right
        if a.x + a.w >= b.x + b.w:
            if a.y + a.h >= b.y + b.h:
                return True

    return False


def eliminate_insiders(components):
    '''eliminates all components whose bounding boxes lie inside of others. The components object is manipulated in place'''

    by_size = by_bbox_size(components)

    labels = DisjointSet(n_labels=len(by_size))

    # pairwise check of bounding boxes. once per pair.
    for a in range(len(by_size)):
        for b in range(a + 1, len(by_size)):
            if is_inside(by_size[a], by_size[b]):
                labels.unite(a, b)

    survivors = labels.final_labels()

    components.candidates = [by_size[i] for i in survivors]

    return


def filter_neighbors(components, args):
    '''eliminates all components that have no neighbors as specified by the corresponding parameters in args.
    The components object is manipulated in place.'''

    size = len(components)
    n = np.zeros((size, size), dtype=np.int)

    if not (args.distance or args.color or args.dims):
        raise('No measure for whether components are neighbors is enabled. Check arguments.')

    for i in range(size):
        for j in range(i + 1, size):

            a = components.candidates[i]
            b = components.candidates[j]

            if args.distance:
                t = args.C_d * min(max(a.w, a.h), max(b.w, b.h))
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
    components.candidates = [components.candidates[i] for i in range(size) if n_count[i] != 0]

    return
