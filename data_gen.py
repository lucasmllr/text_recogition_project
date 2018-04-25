import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
from arguments import Arguments
from bbox_based_rot_correction import rotation_correct_and_line_order
import processing
import blob_extraction
import mser_extraction

def make_string(args):

    lines = random.randint(1, args.max_lines)
    n_char = len(args.alphabet)
    string = ''

    for line in range(lines):
        l = random.randint(args.min_l, args.max_l)
        for i in range(l):
            rand = random.randint(0, n_char - 1)
            string += args.alphabet[rand]
        string += '\n'

    if args.lower_case:
        return string.lower()
    else:
        return string


def make_image(args, string, font=None):

    if font is None:
        font = args.font

    fnt = ImageFont.truetype(font, args.font_size)

    img = Image.new(args.colorspace, args.shape, color='white')
    text = Image.new('L', args.text_box)

    angle = np.random.uniform(- args.max_angle, args.max_angle)
    d = ImageDraw.Draw(text)
    d.text((0, 0), string, font=fnt, fill=255)
    text = text.rotate(angle, expand=True)
    text = ImageOps.colorize(text, black=(255, 255, 255), white=(0, 0, 0))
    if args.pos is not None:
        x_pos = random.randint(0, int(args.shape[0] - args.pos[0] * args.shape[0]))
        y_pos = random.randint(0, int(args.shape[1] - args.pos[1] * args.shape[1]))
    else:
        x_pos = 0
        y_pos = 0
    img.paste(text, (x_pos, y_pos))

    return img


def get_font_files():
    font_files = []
    for path, subdirs, files in os.walk('fonts'):
        for name in files:
            if name.endswith('.ttf'):
                font_files.append(os.path.join(path, name))
    return font_files


def make_data(args):

    if os.path.exists(args.image_path):
        if args.safe_override:
            r = input('Path exists. Do you want to override? Type "y" for yes: \n')
            if r is not 'y':
                return
    else:
        os.makedirs(args.image_path)

    if not args.container:
        f = open('{}/truth.txt'.format(args.image_path), 'w')

        if args.font == 'all':
            font_files = get_font_files()

        for i in range(args.n):
            string = make_string(args)
            if args.font == 'all':
                font = font_files[i % len(font_files)]
            else:
                font = None
            img = make_image(args, string=string, font=font)
            img.save('{}/{}.jpg'.format(args.image_path, str(i)))

            truth = string + '\n'
            f.write(truth)

        f.close()

        return
    else:
        img_container = [None] * args.n
        truth_container = [None] * args.n
        for i in range(args.n):
            string = make_string(args.alphabet, args.min_l, args.max_l, args.max_lines)
            img_container[i] = make_image(args.shape, pos=args.pos, string=string, colorspace='L')
            truth_container[i] = string

        data, target, labels = convertToNumpy(img_container, truth_container)
        np.save('{}/data'.format(args.image_path), data)
        np.save('{}/target'.format(args.image_path), target)
        np.save('{}/label'.format(args.image_path), labels)
        return data, target, labels


def convertToNumpy(data, target):
    num_images = target.__len__()

    labels = target
    target = np.asarray(target)
    _, target = np.unique(target, return_inverse=True)  # convert target to numbers

    data = list(map(np.array, data))
    data = np.array(data) / 255
    # data = np.reshape(data, (data.shape[0], -1))

    return data, target, labels


def generate_char_data(load_path, args=None):
    if args is None:
        args = Arguments()

    if args.method == 'threshold':
        method = blob_extraction.find_blobs
    elif args.method == 'mser':
        method = mser_extraction.extract_mser
    else:
        assert False

    # load alphabet
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
            img = processing.load_img(file_path, args)
            components = method(img, args)
            components = rotation_correct_and_line_order(components)
            chars = components.extract(args)

            # check if number of detected chars conicides with groundtruth
            diff = sum(len(line) for line in truth_blocks[i]) - len(chars)
            print(f'image {i}: #gt chars - #detected chars: {diff}')
            if diff == 0:
                char_images.append(chars)
                char_truths.append(np.concatenate(truth_blocks[i]))
                #print('image is valid')

    return np.concatenate(char_images, axis=0), np.concatenate(char_truths, axis=0)


def save_char_data(args=None):
    if args is None:
        args = Arguments()
    load_path = args.image_path
    save_path = args.train_path
    if not(os.path.exists(save_path)):
        os.mkdir(save_path)
    char_images, char_truth = generate_char_data(load_path, args)
    print(f'saving {len(char_truth)} images')
    np.save(os.path.join(save_path, 'images.npy'), char_images)
    np.save(os.path.join(save_path, 'gt.npy'), char_truth)


if __name__ == '__main__':

    args = Arguments()
    args.n = 10
    args.image_path = 'test_data'
    args.train_path = 'test_data'
    make_data(args)
    save_char_data(args)
