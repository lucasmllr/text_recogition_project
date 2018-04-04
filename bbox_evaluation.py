import numpy as np


def is_inside(a, b):
    '''
    helper function to determine whether bounding box b lies inside a
    '''

    # box coordinates
    a_x, a_y, a_w, a_h = a
    b_x, b_y, b_w, b_h = b

    if a_x <= b_x and a_y <= b_y:  # bottom left
        # top right
        if a_x + a_w >= b_x + b_w:
            if a_y + a_h >= b_y + b_h:
                return True

    return False


def bbox_dims(bboxes):

    widths = np.array([bboxes[i][2] for i in range(len(bboxes))])
    heights = np.array([bboxes[i][3] for i in range(len(bboxes))])
    areas = widths * heights
    aspects = widths / heights

    return {'width':widths, 'height':heights, 'area':areas, 'aspects':aspects}


def distances(bboxes):

    size = len(bboxes)
    dist = np.zeros((size, size))

    for i in range(size):
        for j in range(i + 1, size):

            x_i, y_i = bboxes[i][0], bboxes[i][1]
            x_j, y_j = bboxes[j][0], bboxes[j][1]
            d = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
            dist[i, j] = d
            dist[j, i] = d

    return dist


def neighbors(bboxes, args):

    size = len(bboxes)
    n = np.zeros((size, size), dtype=np.int)
    dist = distances(bboxes)

    if not (args.distance or args.color or args.dims):
        raise('No measure whether components are neighbors is enabled. Check arguments.')

    for i in range(size):
        for j in range(i + 1, size):

            w_i, h_i = bboxes[i][2], bboxes[i][3]
            w_j, h_j = bboxes[j][2], bboxes[j][3]

            if args.distance:

                t = 2 * min(max(w_i, h_i), max(w_j, h_j))

                if dist[i, j] > t:
                    continue

            if args.dims:

                A_i = w_i * h_i
                A_j = w_j * h_j
                r_i = w_i / h_i
                r_j = w_j / h_j

                if max(A_i, A_j) / min(A_i, A_j) > args.t_A:
                    continue

                if max(r_i, r_j) / min(r_i, r_j) > args.t_r:
                    continue

            n[i, j] = 1
            n[j, i] = 1

    n_count = np.sum(n, axis=0)

    return n, n_count
