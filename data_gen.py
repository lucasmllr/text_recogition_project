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

    return string


def make_image(args, string):

    fnt = ImageFont.truetype('fonts/{}'.format(args.font), args.font_size)

    img = Image.new(args.colorspace, args.shape, color='white')
    text = Image.new('L', (250, 250))

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

    '''
    add functionality to assign image to trash when specific conditions are met,
    e.g. most of the character is outside the frameself.
    '''

    return img, string


def make_data(args):

    if os.path.exists(args.path):
        r = input('Path exists. Do you want to override? Type "y" for yes: \n')
        if r is not 'y':
            return
    else:
        os.makedirs(args.path)

    fonts_list = os.listdir('fonts/')

    if args.outputformat is 'image':
        f = open('{}/truth.txt'.format(args.path), 'w')

        for i in range(args.n):
            string = make_string(args)
            if args.font_randomise:
                args.font = np.random.choice(fonts_list)  # pick one of the available fonts
            args.font_size = random.randint(args.font_size_range[0], args.font_size_range[1])
            img, string = make_image(args, string=string)
            img.save('{}/{}.jpg'.format(args.path, str(i)))

            if args.lower_case:
                truth = (string + '\n').lower()
            else:
                truth = string + '\n'
            f.write(truth)

        f.close()

        return
    else:
        img_container = [None] * args.n
        truth_container = [None] * args.n
        for i in range(args.n):
            string = make_string(args)
            if args.font_randomise:
                args.font = np.random.choice(fonts_list)  # pick one of the available fonts
            args.font_size = random.randint(args.font_size_range[0], args.font_size_range[1])
            img_container[i], string = make_image(args, string=string)
            if args.lower_case:
                truth_container[i] = string.lower()
            else:
                truth_container[i] = string

        data, target, labels = convertToNumpy(img_container, truth_container)
        if args.outputformat is 'container':
            np.save('{}/data'.format(path), data)
            np.save('{}/target'.format(path), target)
            np.save('{}/label'.format(path), labels)
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
    args.path = 'data'
    make_data(args)
