# IBAN dummy
import sys

sys.path.insert(0, './python-stdnum')
from stdnum import iban

def iban_check(iban_string):
    try:
        iban.validate(iban_string)
        return True
    except:  # this is bad style. change to invalid checksum exception later
        return False


if __name__ == '__main__':

    print('DE12345: {}'.format(iban_check('DE12345')))
    print('DE92760700120750007700: {}'.format(iban_check('DE92760700120750007700')))
