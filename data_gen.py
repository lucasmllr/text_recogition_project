import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
from arguments import Arguments

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


def make_data(args):

    if os.path.exists(args.path):
        r = input('Path exists. Do you want to override? Type "y" for yes: \n')
        if r is not 'y':
            return
    else:
        os.makedirs(args.path)

    if not args.container:
        f = open('{}/truth.txt'.format(args.path), 'w')

        if args.font == 'all':
            font_files = []
            for path, subdirs, files in os.walk('fonts'):
                for name in files:
                    if name.endswith('.ttf'):
                        font_files.append(os.path.join(path, name))

        for i in range(args.n):
            string = make_string(args)
            if args.font == 'all':
                font = font_files[i % len(font_files)]
            else:
                font = None
            img = make_image(args, string=string, font=font)
            img.save('{}/{}.jpg'.format(args.path, str(i)))

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
        np.save('{}/data'.format(args.path), data)
        np.save('{}/target'.format(args.path), target)
        np.save('{}/label'.format(args.path), labels)
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

if __name__ == '__main__':

    args = Arguments()
    #args.font = 'all'
    #args.path = 'data_test'
    make_data(args)
