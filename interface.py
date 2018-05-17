from pipeline import read
import os
import pyperclip
from arguments import Arguments
from processing import load_img
import detect_iban

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

    if get_iban=='y' or get_iban=='':
        ibans = detect_iban.find_iban(out)
        if len(ibans)==0:
            print('No IBAN could be extracted.')
        elif len(ibans)==1:
            copy = input('found the following IBAN: {} \n Do you want to copy it y/n? [y] > '.format(ibans[0]))
            if copy == '' or copy == 'y':
                pyperclip.copy(ibans[0])
                print('\nCopied {} to the clipboard'.format(ibans[0]))
        else:
            print('\nThe following IBANs were found:')
            for i, iban in enumerate(ibans):
                print(i, ':', iban)
            idx = input('\nIf you want to copy any of these type its number > [0]')
            if idx == '':
                pyperclip.copy(ibans[0])
                print('\nCopied {} to the clipboard'.format(ibans[0]))
            elif int(idx) in range(len(ibans)):
                pyperclip.copy(ibans[int(idx)])
                print('\nCopied {} to the clipboard'.format(ibans[int(idx)]))
            else:
                print('\nThe input was not understood.')

    print('\nexit')


if __name__ == '__main__':

    interface()
