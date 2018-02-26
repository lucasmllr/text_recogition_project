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
    img = Image.new('RGB', shape, color='white')
    d = ImageDraw.Draw(img)
    fnt = ImageFont.truetype('Library/Fonts/{}.ttf'.format(font), 15)

    x_pos = random.randint(0, int(shape[0] - pos[0] * shape[0]))
    y_pos = random.randint(0, int(shape[1] - pos[1] * shape[1]))
    d.text((x_pos, y_pos), string, font=fnt, fill='black')

    return img


def make_data(n, alphabet, shape, length, path):

    if os.path.exists(path):
        raise IOError('path exists')
    else:
        os.makedirs(path)

    f = open('{}/truth.txt'.format(path), 'w')

    for i in range(n):

        string = make_string(alphabet, l=length)
        img = make_image(shape, string=string)
        img.save('{}/{}.jpg'.format(path, str(i)))

        truth = string + '\n'
        f.write(truth)

    f.close()

    return

if __name__ == "__main__":

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWZ0123456789"

    img = make_image()
    make_data(10, alphabet, (100, 50), length=5, path='data')