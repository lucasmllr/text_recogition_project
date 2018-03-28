import numpy as np
from DisjointSet import DisjointSet
import processing
import matplotlib.pyplot as plt
from copy import deepcopy
from heapq import heappush, heappop
from arguments import Arguments

from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.ndimage import rotate
import os
from skimage import measure

def find_blobs(img, args):

    t = args.blob_t

    # dimensions
    height = img.shape[0]
    width = img.shape[1]

    raw = deepcopy(img)
    img = processing.threshold(img, args)

    stencil = measure.label(img, background=0)

    # count pixels in blobs, calculate median to filter blobs
    final_labels = np.arange(1, np.max(stencil)+1)
    pixel_counts = []
    for label in final_labels:
        pixel_counts.append(np.sum(stencil == label))
    pixel_counts = np.array(pixel_counts)
    min_allowed_pixels = np.median(pixel_counts) / 5  # arbitrary; seems to work well

    # filter final lables and stencil
    final_labels = np.array(final_labels)[pixel_counts >= min_allowed_pixels]
    new_stencil = np.zeros_like(stencil)
    for i, label in enumerate(final_labels):
        new_stencil[stencil == label] = i+1
    stencil = new_stencil


    # construct boxes around letters
    bounding_boxes = get_bboxes(stencil)
    # chars = get_chars_from_boxes(raw, bounding_boxes)
    # extract characters from image in correct order
    #chars = []
    #bounding_boxes = []
    #while boxes:
    #    box = heappop(boxes)
    #    chars.append(raw[box[2]:box[3], box[0]:box[1]])
    #    bounding_boxes.append(box)
    return bounding_boxes, stencil


def get_bboxes(stencil, labels=None):
    boxes = []
    if labels is None:
        labels = range(1, np.max(stencil)+1)
    for label in labels:
        pixels = np.argwhere(stencil == label)
        # check if blob is present
        if len(pixels) == 0:
            continue
        y_min = np.min(pixels[:, 0])
        y_max = np.max(pixels[:, 0]) + 1
        x_min = np.min(pixels[:, 1])
        x_max = np.max(pixels[:, 1]) + 1
        boxes.append((x_min, x_max, y_min, y_max))
    return np.array(boxes, dtype=np.int32)


def get_chars_from_boxes(img, boxes, padding=1):
    chars = []
    for box in boxes:
        chars.append(img[max(0, box[2] - padding):min(box[3] + padding, img.shape[0]),
                     max(0, box[0] - padding):min(box[1] + padding, img.shape[1])])
    return chars


def correction_angle(boxes, args, verbose=False):
    n_blobs = len(boxes)

    boxes = np.array(boxes)

    # calculate centers and heights of boxes
    centers = np.stack([(boxes[:, 0] + boxes[:, 1]) / 2, (boxes[:, 2] + boxes[:, 3]) / 2], axis=1)
    box_heights = boxes[:, 3] - boxes[:, 2]

    # plt.scatter(centers[:,0], centers[:,1])
    # plt.show()

    # calculate projection of box-centers for range of angles
    angles = np.linspace(-np.pi / 4, np.pi / 4, 100)
    dirs = np.stack([-np.sin(angles), np.cos(angles)], axis=1)
    projected = centers.dot(dirs.T)

    # plt.scatter(np.arange(len(dirs))[None, :].repeat(len(boxes), axis=0).reshape(-1), projected.reshape(-1))
    # plt.show()

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

        for i in range(len(angles)):
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

        # get row-labels of blobs and sort them by y-coordinate (this should be possible a lot easier, but whatever)
        labels = all_labels[ind]
        cluster_centers = all_cluster_centers[ind]
        order = np.argsort(cluster_centers.reshape(-1))
        mapping = np.concatenate([np.where(order == i)[0] for i in range(len(order))])
        labels = np.array(list([mapping[label] for label in labels]))

    return angle, labels


def get_rotation_corrected_blobs(img, stencil, angle, labels, args):
    rotated_stencil = rotate(stencil, angle*180/np.pi, order=0)
    rotated_image = rotate(img, angle*180/np.pi, order=2)
    boxes = get_bboxes(rotated_stencil)


    lines = []
    for i in range(np.max(labels)+1):  # each label corresponds to a line
        line_boxes = boxes[labels == i]
        # sort boxes by x-coord
        line_boxes = line_boxes[np.argsort(line_boxes[:, 0])]

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


def get_rescaled_chars(img, args=None, separate_lines=False):
    if args is None:
        args = Arguments()
    char_res = args.input_shape
    boxes, stencil = find_blobs(img, args)
    angle, labels = correction_angle(boxes, args, False)
    lines = get_rotation_corrected_blobs(img, stencil, angle, labels, args)

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


def generate_char_data(load_path, args=None):
    if args is None:
        args = Arguments()
    # load alphabet
    alphabet = args.alphabet
    char_dict = args.char_dict
    # load truth
    with open(os.path.join(load_path, 'truth.txt')) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    truth_blocks = []
    current_block = []
    for line in content:
        if line == '':
            truth_blocks.append(current_block)
            current_block = []
        else:
            current_block.append(np.array([char_dict[char] for char in line], dtype=np.int32))
    # process images
    char_images = []
    char_truths = []
    for i in range(args.n):
        file = str(i)+'.jpg'
        if file.endswith('.jpg'):
            file_path = os.path.join(load_path, file)
            img = processing.load_img(file_path)
            chars = get_rescaled_chars(img, args)
            diff = sum(len(line) for line in truth_blocks[i]) - len(chars)
            print(diff)
            if diff == 0:
                char_images.append(chars)
                char_truths.append(np.concatenate(truth_blocks[i]))
                print('image is valid')
    return np.concatenate(char_images, axis=0), np.concatenate(char_truths, axis=0)


def save_char_data(load_path, save_path, args=None):
    if args is None:
        args = Arguments()
    if not(os.path.exists(save_path)):
        os.mkdir(save_path)
    char_images, char_truth = generate_char_data(load_path, args)
    print(f'saving {len(char_truth)} images')
    np.save(os.path.join(save_path, 'images.npy'), char_images)
    np.save(os.path.join(save_path, 'gt.npy'), char_truth)


# function for testing
def show_rotated(load_path, args=None):
    if args is None:
        args = Arguments()
    for i, file in enumerate(os.listdir(load_path)):
        if file.endswith('.jpg') and i==2:
            file_path = os.path.join(load_path, file)
            img = processing.load_img(file_path)
            boxes, stencil = find_blobs(img, args)
            angle, labels = correction_angle(boxes, args, False)
            rotated_img = rotate(img, angle * 180 / np.pi)
            plt.imshow(rotated_img)
            plt.grid(True)
            plt.show()
            get_rotation_corrected_blobs(img, stencil, angle, labels, args)


if __name__ == "__main__":
    save_char_data('data_test', 'char_data')
    #chars = get_rescaled_chars('data_test/5.jpg')
    #for char in chars:
    #    plt.imshow(char)
    #    plt.show()
    #show_rotated('data_test')

    #img = processing.load_img('data_test/5.jpg')
    #plt.imshow(img)
    #plt.show()

    #args = Arguments()
    #blobs, boxes, stencil = find_blobs(img, args)
    #for blob in blobs:
    #    plt.imshow(blob)
    #   plt.show()



