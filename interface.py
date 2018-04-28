from pipeline import read
import os
from arguments import Arguments
from processing import load_img
import pyperclip

def interface():
    '''provides a terminal interface to execute the read function with an image path.
    It prints out the result in the terminal and optionally extracts an IBAN.'''

    path = input('please provide a path to an image containing text relative to this directory [example]\n> ')
    if path=='':
        path = ('IBANs/cropped/IMG_0872.jpg')
    while not os.path.exists(path):
        path = input('This path doesn\'t exist. please provide a valid one [example]\n> ')

    args = Arguments()
    img = load_img(path, args)

    results = read(img, args)
    out = ''
    for line in results:
        out += line + '\n'

    print('\nFrom the provided image the following could be read:\n')
    print(out)

    get_iban = input('Do you want to extract an IBAN? y/n? [y] > ')

    #TODO: extract iban from out

    if get_iban=='y' or get_iban=='':
        pyperclip.copy(out)



if __name__ == '__main__':

    interface()
