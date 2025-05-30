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
        line = line.decode().strip().split()
        w = int(line[0])
        h = int(line[1])

        # Calculate the number of bytes per row (padded to the next byte boundary)
        row_bytes = (w + 7) // 8
        data = np.frombuffer(f.read(row_bytes * h), dtype=np.uint8)

        # Unpack bits and reshape, then slice to remove padding bits
        data = np.unpackbits(data).reshape((h, row_bytes * 8))[:, :w]

    return data.astype(bool)


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
        # Pad each row to the next byte boundary
        row_bytes = (data.shape[1] + 7) // 8
        padded_data = np.zeros((data.shape[0], row_bytes * 8), dtype=bool)
        padded_data[:, :data.shape[1]] = data
        f.write(np.packbits(padded_data).tobytes())


def predict_file_size(data: np.ndarray) -> int:
    """
    Predict the file size of a PBM file given the image data.

    Parameters
    ----------
    data : np.ndarray
        The image data as a numpy array.

    Returns
    -------
    int
        The predicted file size in bytes.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('data must be a np.ndarray but was {}'.format(type(data)))

    # Header: 'P4\n' (3 bytes)
    header_size = 3

    # Dimensions: '<width> <height>\n'
    dimensions_size = len(f"{data.shape[1]} {data.shape[0]}\n")

    # Data: Each row is padded to the next byte boundary
    row_bytes = (data.shape[1] + 7) // 8
    data_size = row_bytes * data.shape[0]

    return header_size + dimensions_size + data_size
