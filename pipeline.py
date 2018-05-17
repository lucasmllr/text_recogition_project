import matplotlib.pyplot as plt
from arguments import Arguments
from processing import load_img, threshold, rescale
from rot_correction import correct_rot
from line_extraction import extract_lines
import train_nn
import blob_extraction
import mser_extraction
import bbox_based_rot_correction
import model
import data_gen
import numpy as np
import torch
from torch.autograd import Variable
import os
import ntpath


def read(img, args):
    '''
    'reads' the text on an image
    Args:
        img: image to be read
        args: Arguments object

    Returns:
        predicted text on the image as list of strings
    '''

    # blob extraction
    if args.method == 'threshold':
        components = blob_extraction.find_blobs(img, args)
    elif args.method == 'mser':
        components = mser_extraction.extract_mser(img, args)
    else:
        assert False

    if args.documentation:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xticks([])
        ax.set_yticks([])
        #plt.title('Original Image with detected Bounding-Boxes')
        components.show_img(axes=ax)
        plt.savefig('IBANs/ex/ex5.png', bbox_inches='tight', dpi=150)
        plt.show()

    # rotation correction, line and char extraction
    components = bbox_based_rot_correction.rotation_correct_and_line_order(components)

    #if args.documentation:
    #    plt.figure()
    #    plt.title('Rotation-Corrected Image with detected Bounding-Boxes')
    #    components.show_img()

    lines = components.extract_lines(args)
    spaces = components.get_spaces(args.space_threshold)

    # feed through NN:
    net = model.ConvolutionalNN()
    net.train(False)
    torch.load(os.path.join(args.model_path, 'weights.pth'))
    net.load_state_dict(torch.load(os.path.join(args.model_path, 'weights.pth')))

    result = []
    for i, line in enumerate(lines):
        chars = ''
        for j, char_img in enumerate(line):
            inp = Variable(torch.FloatTensor(char_img[None, None, :, :]))
            chars += args.int_dict[np.argmax(net(inp).data.numpy())]
            if j in spaces[i]:
                chars += ' '
        result.append(chars)

        #if args.documentation:
        #    plt.imshow(np.concatenate(line, axis=1))
        #    plt.title(f'Line {i}, Prediction: {chars}')
        #    plt.show()

    return result


def generate_and_train(args):
    '''
    generates images and extracts individual chars, then trains the nn on the generated data.
    Args:
        args: Arguments object
    '''

    data_gen.make_data(args)
    data_gen.save_char_data(args)
    train_nn.train_and_test(args)


def test_on_images(args):
    '''
    tests a trained model (specified by args.model_path) on the images labeled 1.jpg, 2.jpg.. in args.image_path
    Args:
        args: Arguments object

    Returns:
        predicted lines, ratio of correctly predicted lines,
        ratio of correctly predicted lines not considering case,
        ratio of lines with a correctly predicted amount of characters
    '''

    # get gt
    with open(os.path.join(args.image_path, 'truth.txt')) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    truth_blocks = []
    current_block = []
    for line in content:
        if line == '':
            truth_blocks.append(current_block)
            current_block = []
        else:
            current_block.append(line)

    total_lines = 0
    correct_lines = 0
    lowercase_correct_lines = 0
    n_letters_correct_lines = 0
    for i in range(args.n):
        file = str(i)+'.jpg'
        file_path = os.path.join(args.image_path, file)
        img = load_img(file_path, args)
        pred_lines = read(img, args)
        gt_lines = truth_blocks[i]
        print('prediction: ', pred_lines)
        print('ground truth: ', gt_lines)
        total_lines += len(gt_lines)
        for i in range(min(len(gt_lines), len(pred_lines))):
            correct_lines += (pred_lines[i] == gt_lines[i])
            lowercase_correct_lines += (pred_lines[i].lower() == gt_lines[i].lower())
            n_letters_correct_lines += len(pred_lines[i]) == len(gt_lines[i])
    print(f'Out of {total_lines} lines, {correct_lines} ({100*correct_lines/total_lines:.2f}%) were predicted perfectly, '
          f'and {lowercase_correct_lines} ({100*lowercase_correct_lines/total_lines:.2f}%) when neglecting case. '
          f'in {n_letters_correct_lines} ({100*n_letters_correct_lines/total_lines:.2f}%) cases, at least the right number of letters was predicted.')

    return total_lines, correct_lines/total_lines, lowercase_correct_lines/total_lines, n_letters_correct_lines/total_lines


def generate_distinct_font_data(args, prefix, font_files=None):
    '''
    generate images according to args for a list of different fonts. saves images of different fonts in separate folders
    Args:
        args: Arguments object
        prefix: folder to save generated images in
        font_files: list of paths of font files to use
    '''

    if font_files is None:
        font_files = data_gen.get_font_files()
    for font in font_files:
        args.font = font
        print(ntpath.basename(font))
        args.image_path = os.path.join(prefix, ntpath.basename(font))
        data_gen.make_data(args)


def test_distinct_font_data(args, prefix, font_files=None, output_file='font_results.txt'):
    '''
    tests the model on different font types and saves the result
    Args:
        args: Arguments object
        prefix: folder in which folders of test images are saved
        font_files: list of fonts to use
        output_file: file to save results
    '''

    if font_files is None:
        font_files = data_gen.get_font_files()

    f = open(output_file, 'w')
    for font in font_files:
        print(ntpath.basename(font))
        args.image_path = os.path.join(prefix, ntpath.basename(font))
        total_lines, correct, correct_case_insensitive, correct_n_chars = test_on_images(args)
        f.write(str(total_lines) + ' ' +
                str(correct) + ' ' +
                str(correct_case_insensitive) + ' ' +
                str(correct_n_chars) + ' ' +
                ntpath.basename(font) + '\n')
    f.close()


if __name__ == '__main__':

    args = Arguments()
    args.documentation = True
    args.method = 'mser'
    img = load_img('IBANs/cropped/IMG_0892.jpg', args)
    print(read(img, args))