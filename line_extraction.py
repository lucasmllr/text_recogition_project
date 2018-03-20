import numpy as np
import matplotlib.pyplot as plt
from rot_correction import correct_rot
from processing import load_img


def extract_lines(img, proj, t):

    # dimensions
    bin_width = img.shape[0] / len(proj)

    # threshold projection
    proj[proj < t] = 0

    # finding lines with 1d blob extraction
    labels = np.zeros(len(proj))
    next_label = 1
    for i in range(len(proj)):

        # blank part of image
        if proj[i] == 0:
            continue

        # special case of first bin
        elif i == 0:
            labels[i] == next_label
            next_label += 1

        # left neighbor is either labeled or not
        else:
            if labels[i - 1] != 0:
                labels[i] = labels[i - 1]
            else:
                labels[i] = next_label
                next_label += 1

    # line coordinates
    boxes = []
    for i in range(1, next_label):
        min_bin = min(np.argwhere(labels == i).reshape(-1))
        max_bin = max(np.argwhere(labels == i).reshape(-1))
        min_pixel = min_bin * bin_width
        max_pixel = max_bin * bin_width
        boxes.append([min_pixel, max_pixel])

    return labels, boxes


if __name__ == "__main__":

    img = load_img('data/4.jpg')
    rotated, proj = correct_rot(img, n_angles=50, n_bins=100, angle=10.)

    x = np.arange(len(proj))
    plt.bar(x, proj)
    plt.show()

    labels, boxes = extract_lines(rotated, proj, t=0.01)
    print(labels)
    print(boxes)

    xx = np.arange(rotated.shape[1])
    plt.imshow(rotated)
    for box in boxes:
        mi = np.full(len(xx), box[0])
        ma = np.full(len(xx), box[1])
        plt.plot(xx, mi)
        plt.plot(xx, ma)
    plt.show()