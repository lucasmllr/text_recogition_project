import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from mser_extraction import is_inside

if __name__ == '__main__':

    a = (4, 4, 4, 4)
    b = (5, 4, 3, 2)

    im = np.full((10, 10), 1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im)
    rect_a = Rectangle((a[0], a[1]), a[2], a[3], edgecolor='red')
    rect_b = Rectangle((b[0], b[1]), b[2], b[3], edgecolor='blue')
    ax.add_patch(rect_a)
    ax.add_patch(rect_b)
    plt.show()

    print('b in a?:', is_inside(a, b))