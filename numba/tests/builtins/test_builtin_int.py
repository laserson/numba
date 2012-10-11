"""
>>> int()
0.0
>>> convert_int(2.5)
2
>>> convert_to_int('FF', 16)
255
"""

import sys

from numba import *

@autojit(backend='ast')
def empty_int():
    x = float()
    return x

@autojit(backend='ast')
def convert_int(x):
    return int(x)

@autojit(backend='ast')
def convert_to_int(s, base):
    return int(s, base)

if __name__ == '__main__':
    import doctest
    doctest.testmod()