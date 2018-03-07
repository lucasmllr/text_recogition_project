import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw, ImageFont
import os


def make_string(alphabet, l=5):

    n_char = len(alphabet)
    string = ''

    for i in range(l):
        rand = random.randint(0, n_char - 1)
        string += alphabet[rand]

    return string


def make_image(shape=(400, 300), pos=(0.5, 0.5), string='blub blub bla bla. This is text!', font='Arial'):

    x_pos = pos
    img = Image.new('L', shape, color='white')
    d = ImageDraw.Draw(img)
    fnt = ImageFont.truetype('Library/Fonts/{}.ttf'.format(font), 15)

    x_pos = random.randint(0, int(shape[0] - pos[0] * shape[0]))
    y_pos = random.randint(0, int(shape[1] - pos[1] * shape[1]))
    d.text((x_pos, y_pos), string, font=fnt, fill='black')

    return img


def make_data(n, alphabet, shape, length, path, writeContainer=True):

    if os.path.exists(path):
        r = input('Path exists. Do you want to override? Type "y" for yes: \n')
        if r is not 'y':
            return
    else:
        os.makedirs(path)

    if not writeContainer:
        f = open('{}/truth.txt'.format(path), 'w')

        for i in range(n):

            string = make_string(alphabet, l=length)
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
            string = make_string(alphabet, l=length)
            img_container[i] = make_image(shape, string=string)
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

    alphabet = '0123456789'
    img = make_image()
    make_data(100, alphabet, (32, 32), length=1, path='data', writeContainer=False)
