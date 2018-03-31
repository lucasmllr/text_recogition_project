import numpy as np


def is_inside(a, b):
    '''
    helper function to determine whether bounding box b lies inside a
    '''

    # box coordinates
    a_x = a[0]
    a_y = a[1]
    a_w = a[2]
    a_h = a[3]
    b_x = b[0]
    b_y = b[1]
    b_w = b[2]
    b_h = b[3]

    if a_x <= b_x and a_y <= b_y:  # bottom left
        # top right
        if a_x + a_w >= b_x + b_w:
            if a_y + a_h >= b_y + b_h:
                return True

    return False


def bbox_stats(bboxes):

    widths = np.array([bboxes[i][2] for i in range(len(bboxes))])
    heights = np.array([bboxes[i][3] for i in range(len(bboxes))])
    areas = widths * heights
    aspects = widths / heights

    mean_width = np.mean(widths)
    std_width = np.std(widths)

    mean_height = np.mean(heights)
    std_height = np.std(heights)

    mean_area = np.mean(areas)
    std_area = (np.std(areas))

    mean_aspect = np.mean(aspects)
    std_aspects = np.mean(aspects)

    return { 'width':(mean_width, std_width, widths), 'height':(mean_height, std_height, heights),
            'area':(mean_area, std_area, areas), 'aspects':(mean_aspect, std_aspects, aspects)}