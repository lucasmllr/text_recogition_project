import matplotlib.pyplot as plt
from arguments import Arguments
from processing import load_img, threshold, rescale
from rot_correction import correct_rot
from line_extraction import extract_lines
import blob_extraction
import model
import numpy as np
import torch
from torch.autograd import Variable


def read(img, args):

    if args.documentation:
        plt.imshow(img)
        plt.title('original image as ndarray')
        plt.show()

    # rotation correction, line and char extraction
    lines = blob_extraction.get_rescaled_chars(img, args, separate_lines=True)

    if args.documentation:
        for i, line in enumerate(lines):
            plt.imshow(np.concatenate(line, axis=1))
            plt.title(f'Line {i}')
            plt.show()

    # feed through NN:
    net = model.ConvolutionalNN()
    net.train(False)
    net.load_state_dict(torch.load('model_weights/weights.pth'))

    result = []
    for line in lines:
        chars = ''
        for char_img in line:
            inp = Variable(torch.FloatTensor(char_img[None, None, :, :]))
            chars += args.int_dict[np.argmax(net(inp).data.numpy())]
        result.append(chars)

    return result


if __name__ == '__main__':

    args = Arguments()
    #args.documentation = True

    img = load_img('data_test/3.jpg')
    result = read(img, args)
    title = ''
    for line in result:
        print(line)
        title += line
        title += ', '
    title = title[:-2]
    plt.title(title)
    plt.imshow(img)
    plt.show()
