import sys
sys.path.insert(0, './python-stdnum')
from stdnum import iban


def iban_check(iban_string):
    '''
    checks whether a string is an IBAN
    '''
    try:
        iban.validate(iban_string)
        return True
    except:  # this is bad style. change to invalid checksum exception later
        return False


def find_iban(string):
    '''
    finds all ibans in a string after performing preprocessing

    Args:
       string: strnig to find ibans in

    Returns:
        a list of strings of all ibans found
    '''
    string = string.replace(' ', '')
    string = string.replace('\n', '')
    string = preprocess(string)
    ibans = []
    for pos in range(len(string)):
        for l in range(15, 31):
            if pos + l >= len(string):
                break
            candidate = preprocess(string[pos:pos+l])
            if iban_check(candidate):
                ibans.append(candidate)
    return ibans


def preprocess(string):
    '''
    preprocesses a string to find ibans in. Common classification errors defined in errors are corrected.
    '''
    errors = [('O', '0'), ('l', '1'), ('S', '5'), ('a', '4')]
    for touple in errors:
        string = string.replace(touple[0], touple[1])
    return string


if __name__ == '__main__':

    print(iban_check('DE53 3507 0030 0540 7572 00'))
    print(iban_check(preprocess('DES3 3SO7 OO3O O5aO 7572 OO')))