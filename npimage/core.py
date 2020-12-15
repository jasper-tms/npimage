#!/usr/bin/env python3

import os
import glob

import numpy as np

supported_extensions = [
    'tif', 'tiff', 'jpg', 'jpeg', 'png',
    'nrrd', 'zarr', 'raw', 'vol'
]

def load(filename, dim_order='zyx', **kwargs):
    """
    Open a 2D or 3D image file and return its pixel values as a numpy
    array.  The returned array will have dimensions ordered as zyx (or
    yx for 2D images) as is typical for numpy. Set dim_order='xyz' to
    transpose the image to xyz order (or xy for 2D images).
    """
    if filename[-1] == '/':
        filename = filename[-1]
    extension = filename.split('.')[-1]
    assert extension in supported_extensions, f'Filetype {extension} not supported'

    data = None

    if extension in ['jpg', 'jpeg', 'png']:
        from PIL import Image
        data = np.array(Image.open(filename)) #PIL.Image.open returns zyx order

    if extension in ['tif', 'tiff']:
        import tifffile
        data = tifffile.imread(filename) #tifffile.imread returns zyx order

    if extension == 'nrrd':
        import nrrd  # pip install pynrrd
        # Specify C-style ordering to get zyx ordering from nrrd.read
        # https://pynrrd.readthedocs.io/en/latest/user-guide.html#index-ordering
        data, metadata = nrrd.read(filename, index_order='C')

    if extension in ['raw', 'vol']:
        dtype = kwargs.get('dtype', np.uint8)
        with open(filename, 'rb') as f:
            if shape in kwargs:
                im = np.fromfile(f, dtype=dtype).reshape(*shape)
            else:
                im = np.fromfile(f, dtype=dtype)

    if extension == 'zarr':
        if 'dataset' not in kwargs:
            dataset = filename + '/'
            while (not os.path.exists(os.path.join(dataset, '.zarray')) and
                    len(glob.glob(dataset + '*/')) == 1):
                dataset = glob.glob(dataset + '*/')[0]

            if os.path.exists(os.path.join(dataset, '.zarray')):
                dataset = dataset.replace(filename + '/', '')[:-1]
                print('Found exactly one dataset in this zarr:'
                      ' Opening {dataset}')
            elif len(glob.glob(dataset + '*/')) == 0:
                raise Exception('No valid datasets containing a .zarray file'
                                f' found within {filename}')
            else:
                raise Exception(f'Multiple datasets found within {filename}.'
                                ' Pass a "dataset=" argument to specify the'
                                ' one you want.')

        try:
            import daisy
        except:
            daisy = None

        if daisy:
            data = daisy.open_ds(filename, dataset)[:] #TODO check zyx/xyz order. Pretty sure daisy zarrs are zyx
        else:
            raise NotImplementedError
            import zarr
            data = zarr.open(TODO) #TODO check zyx/xyz order

    if 'xy' in dim_order:
        data = data.T

    if kwargs.get('metadata', False) or kwargs.get('get_metadata', False):
        try:
            return data, metadata
        except:
            print('WARNING: Metadata requested but not found.'
                  ' Returning image only.')
            return data
    else:
        return data

open = load  # Function name alias
read = load  # Function name alias
imread = load  # Function name alias


def save(data, filename, metadata=None):
    if filename[-1] == '/':
        filename = filename[-1]
    extension = filename.split('.')[-1]
    assert extension in supported_extensions, f'Filetype {extension} not supported'

    if extension in ['tif', 'tiff']:
        import tifffile
        tifffile.imsave(filename, data=data)

    if extension in ['jpg', 'jpeg', 'png']:
        # imagej only writes 8-bit jpgs, with nasty clipping if you try to save
        # a 16 or 32 bit image as jpg. PIL probably does too. TODO investigate,
        # and print warning if user tries to save a 16- or more bit array as jpg.
        from PIL import Image  # pip install pillow
        Image.fromarray(data).save(filename)

    if extension == 'nrrd':
        import nrrd  # pip install pynrrd
        # Specify C-style ordering when writing zyx-ordered array using nrrd.write
        # https://pynrrd.readthedocs.io/en/latest/user-guide.html#index-ordering
        nrrd.write(filename, data, header=metadata, index_order='C')

    if extension in ['raw', 'vol']:
        raise NotImplementedError

    if extension == 'zarr':
        raise NotImplementedError

write = save  # Function name alias
to_file = save  # Function name alias


def show(data, mode='PIL', **kwargs):
    """
    Show a 2-dimensional data array as an image, using either
    PIL.Image.fromarray(im).show() or matplotlib.pyplot.imshow(im) depending
    on 'mode'. Uses PIL by default. Set mode='mpl' to use matplotlib.

    kwargs get passed along to Image.fromarray or pyplot.imshow,
    except a few options get parsed by this function:
        colorbar=True     : Display a color bar (mpl only)
        TODO more options
    """
    if isinstance(data, str):
        if os.path.exists(data):
            data = load(data)

    if len(data.shape) == 3 and data.shape[2] == 3:
        # RGB image. Both PIL and matplotlib can handle this.
        pass
    elif len(data.shape) != 2:
        raise ValueError(
            'Input array must have shape (width, height) for grayscale'
            f' or (width, height, 3) for RGB but had dimensions {data.shape}')

    colorbar = False
    if 'colorbar' in kwargs:
        if kwargs['colorbar']:
            colorbar = True
        kwargs.pop('colorbar')

    if mode.lower() in ['pil', 'pillow']:
        from PIL import Image  # pip install pillow
        Image.fromarray(data, **kwargs).show()
    elif mode.lower() in ['mpl', 'matplotlib', 'pyplot']:
        import matplotlib.pyplot as plt  # pip install matplotlib
        plt.imshow(data, **kwargs)
        if colorbar:
            plt.colorbar()
        plt.show()

imshow = show  # Function name alias
