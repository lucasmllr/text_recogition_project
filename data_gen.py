import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os


def make_string(alphabet, min_l=5, max_l=10, max_lines=10, lowerCase=False):

    lines = random.randint(1, max_lines)
    n_char = len(alphabet)
    string = ''

    for line in range(lines):
        l = random.randint(min_l, max_l)
        for i in range(l):
            rand = random.randint(0, n_char - 1)
            string += alphabet[rand]
        string += '\n'

    if lowerCase:
        return string.lower()
    else:
        return string


def make_image(shape=(400, 300), pos=(0.8, 0.3), max_angle=0, string='This is text!', font='Arial', colorspace='RGB'):

    fnt = ImageFont.truetype('Library/Fonts/{}.ttf'.format(font), 25)

    img = Image.new(colorspace, shape, color='white')
    text = Image.new('L', (250, 100))

    angle = np.random.uniform(-max_angle, max_angle)
    d = ImageDraw.Draw(text)
    d.text((0, 0), string, font=fnt, fill=255)
    text = text.rotate(angle, expand=True)
    text = ImageOps.colorize(text, black=(255, 255, 255), white=(0, 0, 0))

    x_pos = random.randint(0, int(shape[0] - pos[0] * shape[0]))
    y_pos = random.randint(0, int(shape[1] - pos[1] * shape[1]))
    img.paste(text, (x_pos, y_pos))

    return img


def make_data(n, alphabet, shape, min_l, max_l, max_lines, path, writeContainer=True):

    if os.path.exists(path):
        r = input('Path exists. Do you want to override? Type "y" for yes: \n')
        if r is not 'y':
            return
    else:
        os.makedirs(path)

    if not writeContainer:
        f = open('{}/truth.txt'.format(path), 'w')

        for i in range(n):
            string = make_string(alphabet, min_l, max_l, max_lines)
            img = make_image(shape, string=string)
            img.save('{}/{}.jpg'.format(path, str(i)))

            truth = string + '\n'
            f.write(truth)

        f.close()

        return
    else:
        img_container = [None] * n
        truth_container = [None] * n
        for i in range(n):
            string = make_string(alphabet, min_l, max_l, max_lines)
            img_container[i] = make_image(shape, pos=(0, 0), string=string, colorspace='L')
            truth_container[i] = string

        data, target, labels = convertToNumpy(img_container, truth_container)
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

    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    make_data(10, alphabet, shape=(400, 300), min_l=1, max_l=10, max_lines=2, path='data', writeContainer=False)
