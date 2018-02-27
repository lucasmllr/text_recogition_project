import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import imread


def load_img(path):

    img = imread(path)
    #average channels to reduce to one channel
    img = np.mean(img, axis=2)
    #normalization
    img /= np.max(img)
    #inverting
    img = 1 - img

    return img


def threshold(img, t, top=True, bottom=True):

    below = img <= t
    above = img > t

    if bottom:
        img[below] = 0
    if top:
        img[above] = 1

    return img


def make_stencil(img):

    height = img.shape[0]
    width = img.shape[1]
    size = height * width

    img = img.reshape(size)
    stencil = np.zeros(size)
    labels = [0]

    for i in range(size):

        if img[i] != 0:

            # if a neighboring pixel is labeled the investigated pixel is given the same label
            # Note: when iterating from top left to bottom right indices to the right bottom of investigated
            # pixel cannot be labeled before this pixel
            for j in [i-1, i-width, i-width-1, i-width+1]:

                if j < 0 or j >= size:
                    continue

                if stencil[j] != 0:
                    stencil[i] = stencil[j]
                    break

            # if no neighboring pixel is labeled the investigated pixel is give a new label
            if stencil[i] == 0:
                new_label = max(labels) + 1
                stencil[i] = new_label
                labels.append(new_label)


    # reshaping stencil
    stencil = stencil.reshape((height, width))

    return stencil



if __name__ == '__main__':

    img = load_img('data/0.jpg')
    plt.imshow(img)
    plt.show()

    img = threshold(img, t=0.5)

    sten = make_stencil(img)
    plt.imshow(sten)
    plt.show()
