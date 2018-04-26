import numpy as np
import processing
import matplotlib.pyplot as plt
from arguments import Arguments
from components import Components
from blob_extraction import get_bboxes, find_blobs

from sklearn.cluster import MeanShift
from scipy.ndimage import rotate
import os


def get_chars_from_boxes(img, boxes, padding=1): # TODO: replace
    chars = []
    for box in boxes:
        chars.append(img[max(0, box[0] - padding):min(box[1] + padding, img.shape[0]),
                     max(0, box[2] - padding):min(box[3] + padding, img.shape[1])])
    return chars


def correction_angle(components, args=None, verbose=False):
    '''
    function to rotation correct a list of bounding boxes of characters, incapsulated in a components object.
    also outputs indices by which the boxes can be grouped into lines.
    :param components: components object to be processed
    :param args: Arguments object for options
    :param verbose: if set to True, debugging information is shown
    :return: rotation angle, labels grouping the boxes into lines
    '''
    boxes = components.bboxes()
    n_blobs = len(boxes)
    if n_blobs == 0:
        print('no blobs found!')
        assert False
    boxes = np.array(boxes)

    # calculate centers and heights of boxes
    centers = np.stack([(boxes[:, 0] + boxes[:, 1]) / 2, (boxes[:, 2] + boxes[:, 3]) / 2], axis=1)
    box_heights = boxes[:, 1] - boxes[:, 0]

    # plt.scatter(centers[:,0], centers[:,1])
    # plt.show()

    # calculate projection of box-centers for range of angles
    angles = np.linspace(-np.pi / 4, np.pi / 4, 100)
    dirs = np.stack([-np.cos(angles), np.sin(angles)], axis=1)
    projected = centers.dot(dirs.T)

    if(verbose):
        plt.scatter(np.arange(len(dirs))[None, :].repeat(len(boxes), axis=0).reshape(-1), projected.reshape(-1))
        plt.show()

    # estimate bandwidth for Mean Shift algorithm
    bandwidth = np.mean(box_heights) / 2
    if verbose:
        print('bandwidth:', bandwidth)

    # test if only one line is present
    stds = np.std(projected, axis=0)
    if np.min(stds) < bandwidth:  # only one line
        angle = angles[np.argmin(stds)]
        labels = np.zeros(n_blobs, dtype=np.int32)

    else:  # multiple lines
        n_clusters = np.empty(len(angles), dtype=np.int32)
        all_cluster_centers = []
        losses = np.empty(len(angles), dtype=np.float32)
        all_labels = []

        for i in range(len(angles)):  # loop over angles to find the best one
            X = projected[:, i].reshape(-1, 1)
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(X)
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_
            loss = np.mean(np.linalg.norm(X - cluster_centers[labels], axis=1)) * len(cluster_centers)
            losses[i] = loss
            n_clusters[i] = len(cluster_centers)
            all_labels.append(labels)
            all_cluster_centers.append(cluster_centers)

        if verbose:
            plt.plot(angles, n_clusters)
            plt.show()
            plt.plot(angles, losses)
            plt.show()

        ind = np.argmin(losses)
        angle = angles[ind]

        # get row-labels of blobs and sort them by y-coordinate (this should be possible a lot easier)
        labels = all_labels[ind]
        cluster_centers = all_cluster_centers[ind]
        order = np.flip(np.argsort(cluster_centers.reshape(-1)), axis=0)
        mapping = np.concatenate([np.where(order == i)[0] for i in range(len(order))])
        labels = np.array(list([mapping[label] for label in labels]))

    return angle, labels


def get_rotation_corrected_blobs(components, angle, labels, args):
    '''
    function to extract rotated char images grouped in lines
    :param components: components object to be rotated
    :param angle: angle to be rotated by
    :param labels: labels to group the chars into lines
    :param args: Arguments object
    :return: lines, a list of lists containing the individual chars of a line as images
    '''
    img = components.img #TODO: remove this
    if img.dtype != np.float64:
        print(img.dtype)
        img = img.astype(np.float64)
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)
        #normalization
        img = (img - np.min(img))/ (np.max(img) - np.min(img))
        img = -img + 1
    print('image:', np.max(img), np.min(img))
    #components.generate_stencil()
    stencil = components.get_stencil()
    rotated_stencil = rotate(stencil, angle*180/np.pi, order=0)
    rotated_image = np.clip(rotate(img, angle*180/np.pi, order=2), 0, 1)

    boxes = get_bboxes(rotated_stencil)

    lines = []
    #print(boxes)
    for i in range(np.max(labels)+1):  # each label corresponds to a line
        line_boxes = boxes[labels == i]
        # sort boxes by x-coord
        line_boxes = line_boxes[np.argsort(line_boxes[:, 2])]

        lines.append(get_chars_from_boxes(rotated_image, line_boxes))

    if args.documentation:
        print(f'{len(lines)} lines in total')
        for i, line in enumerate(lines):
            print(f'line {i}: {len(line)} chars')
        print()

    #for char in lines[0]:
    #    plt.imshow(char)
    #    plt.show()

    return lines


def rotation_correct_and_line_order(components):
    angle, line_labels = correction_angle(components)
    components = rotated_components(components, angle)
    boxes = np.array(components.bboxes())
    lines = []
    for i in range(np.max(line_labels) + 1):  # each label corresponds to a line
        line_boxes = boxes[line_labels == i]
        # sort boxes by x-coord
        line_ids = np.argwhere(line_labels == i)[:, 0][np.argsort(line_boxes[:, 2])]
        lines.append(line_ids)
    components.set_lines(lines)
    return components



def rotated_components(components, angle):
    img = components.img
    stencil = components.get_stencil()
    rotated_stencil = rotate(stencil, angle*180/np.pi, order=0)
    rotated_image = np.clip(rotate(img, angle*180/np.pi, order=2), 0, 1)
    boxes = get_bboxes(rotated_stencil)
    return Components(boxes, rotated_image, stencil=rotated_stencil)


def get_rescaled_chars(components, args=None, separate_lines=False):

    if args is None:
        args = Arguments()
    char_res = args.input_shape

    angle, labels = correction_angle(components, args, False)
    lines = get_rotation_corrected_blobs(components, angle, labels, args)

    rescaled_lines = []
    for line in lines:
        n_chars = len(line)
        chars = np.empty((n_chars, char_res, char_res), dtype=np.float32)
        for i, char in enumerate(line):
            chars[i] = processing.rescale(char, args)
        rescaled_lines.append(chars)

    if separate_lines:
        return rescaled_lines
    else:
        return np.concatenate(rescaled_lines, axis=0)


# function for testing
def show_rotated(load_path, args=None, indices='all'):
    if args is None:
        args = Arguments()
    for i, file in enumerate(os.listdir(load_path)):
        if file.endswith('.jpg') and (indices == 'all' or i in indices):
            file_path = os.path.join(load_path, file)
            img = processing.load_img(file_path)
            components = find_blobs(img, args)
            boxes = components.bboxes()
            stencil = components.get_stencil()
            angle, labels = correction_angle(components, args, False)
            rotated_img = rotate(img, angle * 180 / np.pi)
            plt.imshow(rotated_img)
            plt.grid(True)
            plt.show()
            get_rotation_corrected_blobs(components, angle, labels, args)


if __name__ == "__main__":
    args = Arguments()
    show_rotated('data_test', args, indices=[1,2,3])