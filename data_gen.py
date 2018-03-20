import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os


def make_string(alphabet, min_l=5, max_l=10, max_lines=10):

    lines = random.randint(1, max_lines)
    n_char = len(alphabet)
    string = ''

    for line in range(lines):
        l = random.randint(min_l, max_l)
        for i in range(l):
            rand = random.randint(0, n_char - 1)
            string += alphabet[rand]
        string += '\n'

    return string


def make_image(shape=(400, 300), pos=(0.8, 0.3), max_angle=10, string='This is text!', font='Arial'):

    fnt = ImageFont.truetype('Library/Fonts/{}.ttf'.format(font), 25)

    img = Image.new('RGB', shape, color='white')
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


def make_data(n, alphabet, shape, min_l, max_l, max_lines, path):

    if os.path.exists(path):
        raise IOError('path exists')
    else:
        os.makedirs(path)

    f = open('{}/truth.txt'.format(path), 'w')

    for i in range(n):

        string = make_string(alphabet, min_l, max_l, max_lines)
        img = make_image(shape, string=string)
        img.save('{}/{}.jpg'.format(path, str(i)))

        truth = string + '\n'
        f.write(truth)

    f.close()

    return

if __name__ == "__main__":

    alphabet = " ABCDEFGHIJKLMNOPQRSTUVWZ0123456789"

    #for _ in range(15):
    #    img = make_image()
    #    plt.imshow(img)
    #    plt.show()

    make_data(10, alphabet, shape=(400, 300), min_l=5, max_l=10, max_lines=1, path='data')