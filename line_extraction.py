import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from rot_correction import correct_rot
from processing import load_img
from arguments import Arguments


def extract_lines(img, args):

    t = args.line_t

    # dimensions
    height = img.shape[0]
    width = img.shape[1]
    size = height * width

    # projection in horizontal direction
    hor_proj = np.sum(img, axis=1) / np.sum(img)

    # visualization of line separatation
    x = np.arange(height)
    plt.plot(hor_proj[50:200], -x[50:200])
    plt.plot(np.array([t] * height)[50:200], -x[50:200], 'r--')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('plots/rotated_good_yhist.pdf', bbox_inches='tight')

    # threshold projection
    hor_proj[hor_proj < t] = 0

    # finding vertical boundaries of lines with 1d blob extraction
    labels = np.zeros(height)
    next_label = 1
    for i in range(height):

        # blank part of image
        if hor_proj[i] == 0:
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
        mi = min(np.argwhere(labels == i).reshape(-1))
        ma = max(np.argwhere(labels == i).reshape(-1))
        buffer = int(np.ceil((ma - mi) / 10))
        boxes.append([mi - buffer, ma + buffer])

    lines = []
    # find vertical boundaries for each line
    for i, box in enumerate(boxes):
        line = img[box[0]:box[1]]

        # vertical projection
        vert_proj = np.sum(line, axis=0) / np.sum(line)
        vert_proj[vert_proj < t] = 0

        # vertical boundaries
        hor_max = np.max(np.argwhere(vert_proj != 0)) + 1
        hor_min = np.min(np.argwhere(vert_proj != 0))

        boxes[i] += [hor_min, hor_max]
        lines.append(img[box[0]:box[1], box[2]:box[3]])

    # bounding boxes in shape [x_min, y_min, width, height]
    bboxes = [[box[2], box[0], box[3] - box[2], box[1] - box[0]] for box in boxes]

    return lines, bboxes


if __name__ == "__main__":

    args = Arguments()
    args.n_angles = 50
    args.n_bins = 30

    img = load_img('data/5.jpg')
    rotated = correct_rot(img, args)

    lines, bboxes = extract_lines(rotated, args)

    plt.imshow(img[0:150, 0:200], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    #plt.savefig('plots/to_be_rotated.pdf', bbox_inches='tight')
    plt.show()

    o = 50
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(rotated[o:200, o:250], cmap='gray')
    ax.grid(True)
    ax.set_xticks([])
    ax.set_yticks([])
    for b in bboxes:
        ax.add_patch(Rectangle((b[0]-o, b[1]-o), b[2], b[3], edgecolor='red', fill=False))
    plt.savefig('plots/rotated_good.pdf', bbox_inches='tight')
    plt.show()