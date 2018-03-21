import matplotlib.pyplot as plt
from arguments import Arguments
from processing import load_img, threshold, rescale
from rot_correction import correct_rot
from line_extraction import extract_lines
from blob_extraction import find_blobs


def read(img, args):

    if args.documentation:
        plt.imshow(img)
        plt.title('original image as ndarray')
        plt.show()

    # rotation correction
    rotated = correct_rot(img, args)

    if args.documentation:
        plt.imshow(rotated)
        plt.title('corrected for rotation')
        plt.show()

    # line extraction
    lines, _ = extract_lines(rotated, args)

    if args.documentation:
        for i, line in enumerate(lines):
            plt.imshow(line)
            plt.title('line {}'.format(i))
            plt.show()

    # character extraction
    chars = []
    for i, line in enumerate(lines):
        line_chars, boxes = find_blobs(line, args)
        chars.append(line_chars)
        print('character boxes of line {}: {}'.format(i, boxes))

    if args.documentation:
        for i, line in enumerate(chars):
            for j, char in enumerate(line):
                plt.imshow(char)
                plt.title('character {} in line {}'.format(j, i))
                plt.show()

    #Todo: feed chars through the NN

    return chars


if __name__ == '__main__':

    args = Arguments()
    args.documentation = True

    img = load_img('data/6.jpg')
    chars = read(img, args)

