#!/usr/bin/env python3
"""
Functions for reading, writing, and showing images.

Function list:
- load(filename) -> numpy.ndarray
- save(data, filename) -> Saves a numpy array as an nD image file
- save_video(data, filename) -> Saves a 3D numpy array as a video
- show(np_array) -> Displays a numpy array of pixel values as an image
"""

import os
import glob
from builtins import open as builtin_open

import numpy as np

from . import utils

supported_extensions = [
    'tif', 'tiff', 'jpg', 'jpeg', 'png', 'pbm',
    'nrrd', 'zarr', 'raw', 'vol'
]

def load(filename, dim_order='zyx', **kwargs):
    """
    Open a 2D or 3D image file and return its pixel values as a numpy array.

    As is typical for Python, the returned array will by default have
     dimensions ordered as zyx for 3D image volumes, yx for 1-channel 2D
     images, or yxc for multi-channel (RGB/RGBA) 2D images.
    Set dim_order='xy' if you instead want dimensions ordered as
     xyz for 3D image volumes, xy for 1-channel 2D images, or
     xyc for multi-channel 2D images.
    """
    filename = str(filename)
    while filename.endswith('/'):
        filename = filename[:-1]
    extension = filename.split('.')[-1]
    assert extension in supported_extensions, f'Filetype {extension} not supported'

    data = None

    if extension in ['jpg', 'jpeg', 'png']:
        from PIL import Image
        data = np.array(Image.open(filename)) #PIL.Image.open returns zyx order

    if extension in ['tif', 'tiff']:
        import tifffile
        data = tifffile.imread(filename) #tifffile.imread returns zyx order

    if extension == 'pbm':
        from .filetypes import pbm
        data = pbm.load(filename)  # pbm.load returns zyx

    if extension == 'nrrd':
        import nrrd  # pip install pynrrd
        # Specify C-style ordering to get zyx ordering from nrrd.read
        # https://pynrrd.readthedocs.io/en/stable/background/index-ordering.html
        data, metadata = nrrd.read(filename, index_order='C')
        # Metadata is in Fortran order, so flip it to C order
        utils.transpose_metadata(metadata, inplace=True)

    if extension in ['raw', 'vol']:
        dtype = kwargs.get('dtype', np.uint8)
        with builtin_open(filename, 'rb') as f:
            data = np.fromfile(f, dtype=dtype)
        if 'shape' in kwargs:
            data = data.reshape(*kwargs['shape'])

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
        if is_rgb_or_rgba(data):
            # Can't just transpose because if data is a multi-channel 2D
            # image, need the channel axis to stay as the last axis.
            data = data.swapaxes(0, 1)
        else:
            data = data.T

        if extension == 'nrrd':  # TODO check if other formats need this
            utils.transpose_metadata(metadata, inplace=True)

    if any([kwargs.get(key, False) for key in
            ['metadata', 'get_metadata', 'return_metadata']]):
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


def save(data,
         filename,
         overwrite=False,
         dim_order='zyx',
         pixel_size=None,
         unit=None,
         compress=False,
         metadata=None):
    """
    Save a numpy array to file with a file type specified by the
    filename extension.

    As is typical for Python, the input array is assumed to have
     dimensions ordered as zyx for 3D images, yx for 1-channel 2D
     images, or yxc for multi-channel 2D images.
    Set dim_order='xy' if your array is a 3D image in in xyz order,
     a 1-channel 2D image in xy order, or a multi-channel 2D image
     in xyc order.

    Currently `pixel_size`, `unit`, and `compress` are only recognized
    when saving to .nrrd files. For other formats, they are ignored.
    """
    filename = str(filename)
    filename = filename.rstrip('/')
    if os.path.exists(filename) and not overwrite:
        raise FileExistsError(f'File {filename} already exists. '
                              'Set overwrite=True to overwrite.')
    extension = filename.split('.')[-1]
    assert extension in supported_extensions, f'Filetype {extension} not supported'

    if compress and extension != 'nrrd':
        print('WARNING: compress argument is ignored because not saving as '
              '.nrrd. Whether or not compression occurs now will depend on '
              'the format you are saving to.')

    if 'xy' in dim_order:
        if is_rgb_or_rgba(data):
            # Can't just transpose because if data is a multi-channel 2D
            # image, need the channel axis to stay as the last axis.
            data = data.swapaxes(0, 1)
        else:
            data = data.T

    if extension in ['tif', 'tiff']:
        import tifffile
        tifffile.imsave(filename, data=data)

    if extension in ['jpg', 'jpeg', 'png']:
        # imagej only writes 8-bit jpgs, with nasty clipping if you try to save
        # a 16 or 32 bit image as jpg. PIL probably does too. TODO investigate,
        # and print warning if user tries to save a 16- or more bit array as jpg.
        from PIL import Image  # pip install pillow
        Image.fromarray(data).save(filename)

    if extension == 'pbm':
        from .filetypes import pbm
        pbm.save(data, filename, comments=metadata)

    if extension == 'nrrd':
        import nrrd  # pip install pynrrd
        if metadata is None:
            metadata = {}
        else:
            metadata = metadata.copy()
        if compress:
            metadata.update({'encoding': 'gzip'})
        if 'encoding' not in metadata:
            metadata.update({'encoding': 'raw'})
        if pixel_size is not None:
            try:
                iter(pixel_size)
            except TypeError:
                pixel_size = [pixel_size] * data.ndim

            if 'xy' in dim_order:
                pixel_size = np.flip(pixel_size)
            metadata.update({'space directions': np.flip(np.diag(pixel_size),
                                                         axis=-1)})
            if is_rgb_or_rgba(data):
                metadata.update({'space dimension': data.ndim - 1})
            else:
                metadata.update({'space dimension': data.ndim})
        if unit is not None:
            metadata.update({'space units': [unit] * data.ndim})

        # From https://pynrrd.readthedocs.io/en/stable/background/index-ordering.html
        # "C-order is the index order used in Python and many Python libraries
        #  (e.g. NumPy, scikit-image, PIL, OpenCV). pynrrd recommends using
        #  C-order indexing to be consistent with the Python community. However,
        #  as of this time, the default indexing [in the nrrd.write command] is
        #  Fortran-order to maintain backwards compatibility."
        # "All header fields are specified in Fortran order, per the NRRD
        #  specification, regardless of the index order. For example, a
        #  C-ordered array with shape (60, 800, 600) would have a sizes field
        #  of (600, 800, 60)."
        #
        # We expect users of this save function to pass in metadata that has
        # header fields ordered in the same order as their data, so in addition
        # to effectively flipping the order of the data's axes by specifying
        # index_order='C' to the nrrd.write command below, we also need to flip
        # the order of any per-axis metadata fields.
        utils.transpose_metadata(metadata, inplace=True)
        nrrd.write(filename, data, header=metadata, index_order='C')

    if extension in ['raw', 'vol']:
        raise NotImplementedError

    if extension == 'zarr':
        raise NotImplementedError


write = save  # Function name alias
to_file = save  # Function name alias


def save_video(data, filename, overwrite=False, dim_order='yx', time_axis=0,
               framerate=30, crf=23, compression_speed='medium'):
    """
    Save a 3D numpy array as a video, with a specified axis as the time axis.

    Follows the PyAV cookbook section on generating video from numpy arrays:
    https://pyav.basswood-io.com/docs/develop/cookbook/numpy.html#generating-video
    """
    if not data.ndim == 3:
        raise ValueError('Input data must be a 3D numpy array.')
    try:
        import av
    except ImportError:
        raise ImportError('To save videos, you must have PyAV installed. '
                          'You can install it with "pip install av".')
    data = np.moveaxis(data, time_axis, 0)
    if 'xy' in dim_order:
        data = data.swapaxes(1, 2)
    n_frames = data.shape[0]

    if filename.split('.')[-1] not in ['mp4', 'mkv', 'avi', 'mov']:
        filename += '.mp4'
    if os.path.exists(filename) and not overwrite:
        raise FileExistsError(f'File {filename} already exists. '
                              'Set overwrite=True to overwrite.')
    container = av.open(filename, mode='w')

    stream = container.add_stream('libx264', rate=framerate)
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': str(crf), 'preset': compression_speed}
    stream.height = data.shape[1]
    stream.width = data.shape[2]

    for frame_i in range(n_frames):
        frame = av.VideoFrame.from_ndarray(data[frame_i], format='gray')
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()


def show(data, dim_order='yx', mode='PIL', **kwargs):
    """
    Display a numpy array of pixel values as an image. Supported types:
      1-channel (grayscale) : data.shape must be (y, x)
      3-channel (RGB)       : data.shape must be (y, x, 3)
      4-channel (RGBA)  : data.shape must be (y, x, 4)

    If `dim_order` is set to 'xy' (instead of the default 'yx'), then
    swap the y and x above. The channel axis must come last regardless.

    Images will be shown using either `PIL.Image.fromarray(data).show()`
    or `matplotlib.pyplot.imshow(data)` depending on 'mode'. Uses PIL by
    default. Set mode='mpl' to use matplotlib.

    kwargs get passed along to Image.fromarray or pyplot.imshow,
    except a few options get parsed by this function:
        colorbar=True     : Display a color bar (mpl only)
        TODO more options
    """
    if isinstance(data, str):
        if os.path.exists(data):
            data = load(data)

    if not is_rgb_or_rgba(data) and data.ndim != 2:
        m = ('Input array must have shape (y, x) for grayscale, '
            '(y, x, 3) for RGB, or (y, x, 4) for RGBA but had '
            f'shape {data.shape}')
        if 'xy' in dim_order:
            m = m.replace('y, x', 'x, y')
        raise ValueError(m)

    if 'xy' in dim_order:
        # Can't just transpose because if data is a multi-channel 2D
        # image, need the channel axis to stay as the last axis.
        data = data.swapaxes(0, 1)

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


def is_rgb_or_rgba(data):
    """
    Return True if the given numpy array has a shape indicating
    that it's either an RGB or RGBA image.

    data.shape == (i, j, 3)  ->  it's RGB, return True
    data.shape == (i, j, 4)  ->  it's RGBA, return True
    data.shape == anything else -> return False

    """
    if data.ndim == 3 and data.shape[2] in [3, 4]:
        return True
    return False
