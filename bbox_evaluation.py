

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