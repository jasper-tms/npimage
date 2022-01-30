#!/usr/bin/env python3
"""
Read and write PBM files, whose format is specified at
http://netpbm.sourceforge.net/doc/pbm.html
"""

import numpy as np


def load(filename):
    """
    Load pixel data from a PBM file and return it as a numpy array.
    The returned array has dimensions ordered y then x, as is
    standard in Python. This means data.shape is ordered (height, width)
    """
    with open(filename, 'rb') as f:
        assert f.readline() == b'P4\n', 'This is not a PBM file'
        line = f.readline()
        while line.startswith(b'#'):  # Print and skip comments
            print(line.decode().strip('\n'))
            line = f.readline()
        w, h = line.decode().strip().split()
        w = int(w)
        h = int(h)
        data = f.readline()
        data = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        data = data.reshape((h, w)).view(bool)

    return data
    


def save(data, filename, comments=None):
    """
    Write a numpy array to file in PBM format
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('data must be a np.ndarray but was {}'.format(type(data)))
    if not isinstance(filename, str):
        raise TypeError('filename must be a str but was {}'.format(type(filename)))
    if comments is not None and not isinstance(comments, str):
        raise TypeError('comments must be a str but was {}'.format(type(comments)))
    with open(filename, 'wb') as f:
        # Header
        f.write(b'P4\n')

        # Comments
        if comments is not None:
            # Insist on each line starting with a '# '
            comments = '# ' + comments.replace('\n', '\n# ')
            # Remove any double '# ', if we created any
            comments = comments.replace('# # ', '# ')
            while any([comments.endswith(c) for c in ['#', ' ', '\n']]):
                comments = comments[:-1]
            print(comments)
            f.write(comments.encode())
            f.write(b'\n')

        # Size
        f.write(str(data.shape[1]).encode())
        f.write(b' ')
        f.write(str(data.shape[0]).encode())
        f.write(b'\n')

        # Data
        f.write(np.packbits(data).tobytes())
