#!/usr/bin/env python3
"""
Functions for reading, writing, and showing images.

Function list:
- load(filename) -> numpy.ndarray
- save(data, filename) -> Saves a numpy array as an nD image file
- save_video(data, filename) -> Saves a 3D numpy array as a video
- show(np_array) -> Displays a numpy array of pixel values as an image
"""

from typing import Literal
import os
import glob
from builtins import open as builtin_open

import numpy as np

from . import operations, utils

supported_extensions = [
    'tif', 'tiff', 'jpg', 'jpeg', 'png', 'pbm',
    'nrrd', 'zarr', 'raw', 'vol', 'ng'
]


def load(filename, dim_order='zyx', **kwargs):
    """
    Open a 2D or 3D image file and return its pixel values as a numpy array.

    As is typical for Python, the returned array will by default have
     dimensions ordered as zyx for 3D image volumes, yx for 1-channel 2D
     images, or yxc for multi-channel (RGB/RGBA) 2D images.
    Set dim_order='xy' if you want to reverse the order of the axes.
    """
    filename = os.path.expanduser(str(filename))
    while filename.endswith('/'):
        filename = filename[:-1]
    if 'format' in kwargs:
        extension = kwargs['format'].lower()
    elif '.' in filename:
        extension = filename.split('.')[-1].lower()
    else:
        raise ValueError('Could not determine file format from filename'
                         f' "{filename}". Please specify the file type via'
                         ' the `format` argument, e.g. format="tif"')
    if extension not in supported_extensions:
        raise ValueError(f'File format of "{filename}" not supported/recognized.')

    data = None
    metadata = None

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

    if extension == 'ng':
        from cloudvolume import CloudVolume
        if '://' not in filename:
            filename = 'file://' + filename
        vol = CloudVolume(filename)
        data = np.array(vol[:]).squeeze()
        # CloudVolume is unusual in returning data in Fortran order,
        # so we transpose xyz -> zyx
        data = data.T
        metadata = utils.transpose_metadata(vol.info, inplace=False)

    if extension == 'zarr':
        if '.ome.zarr' in filename:
            import ome_zarr.io.parse_url
            import ome_zarr.reader.Reader
            raise NotImplementedError
            return 'something something'

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
        raise NotImplementedError('Reading Zarr format not yet implemented.')
        #import zarr
        #data = zarr.open(TODO) #Finish this and be sure to check zyx/xyz order

    if data is None:
        raise ValueError(f'Could not read file {filename}. '
                         'Please check the file format and try again.')

    if 'xy' in dim_order:
        data = data.T
        if extension in ['nrrd', 'ng']:
            metadata = utils.transpose_metadata(metadata, inplace=False)

    if any([kwargs.get(key, False) for key in
            ['metadata', 'get_metadata', 'return_metadata']]):
        if metadata is None:
            print('WARNING: Metadata is None.')
        return data, metadata
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
         units=None,
         compress=None,
         compression_level=3,
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

    Currently `pixel_size`, `units`, and `compress` are only recognized
    when saving to .nrrd and .ng files, and ignored otherwise.
    """
    filename = os.path.expanduser(str(filename))
    filename = filename.rstrip('/')
    if os.path.exists(filename) and not overwrite:
        raise FileExistsError(f'File {filename} already exists. '
                              'Set overwrite=True to overwrite.')
    extension = filename.split('.')[-1].lower()
    assert extension in supported_extensions, f'Filetype {extension} not supported'

    if compress is not None and extension not in ['nrrd', 'ng']:
        print('WARNING: compress argument is ignored because not saving as '
              '.nrrd. Whether or not compression occurs now will depend on '
              'the format you are saving to.')

    channel_axis = find_channel_axis(data)
    if 'xy' in dim_order:
        data = data.T
        if hasattr(pixel_size, '__iter__') and not isinstance(pixel_size, str):
            pixel_size = pixel_size[::-1]
        if hasattr(units, '__iter__') and not isinstance(units, str):
            units = units[::-1]
        utils.transpose_metadata(metadata, inplace=True)
        # The spatial axes and associated metadata are now in zyx order.

    if extension in ['tif', 'tiff']:
        import tifffile
        tifffile.imwrite(filename, data=data)

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
        custom_field_map = {}

        if compress is True or compress in ['lossless', 'gzip']:
            metadata.update({'encoding': 'gzip'})
        elif compress is False:
            metadata.update({'encoding': 'raw'})
        elif compress is not None:
            raise ValueError('For .nrrd format, compress must be '
                             'True/"lossless"/"gzip", or False,'
                             f' but was "{compress}"')
        if 'encoding' not in metadata:
            metadata.update({'encoding': 'raw'})

        if isinstance(metadata.get('space directions', None), str):
            custom_field_map['space directions'] = 'string'
        if pixel_size is not None:
            if 'space directions' in metadata:
                raise ValueError('Cannot specify both "space directions" in'
                                 ' metadata and "pixel_size" as an argument.')

            try:
                iter(pixel_size)
                if len(pixel_size) == data.ndim - 1 and channel_axis is not None:
                    pixel_size.insert(channel_axis, np.nan)
            except TypeError:
                if channel_axis is not None:
                    pixel_size = [pixel_size] * (data.ndim - 1)
                    pixel_size.insert(channel_axis, np.nan)
                else:
                    pixel_size = [pixel_size] * data.ndim
            if len(pixel_size) != data.ndim:
                raise ValueError(f'pixel_size has length {len(pixel_size)},'
                                 f' but data has {data.ndim} dimensions.')

            pixel_size_not_none = [size for size in pixel_size
                                   if size is not None and size != np.nan]
            space_directions = np.diag(pixel_size_not_none).astype(float)
            # nrrd.format_optional_matrix() expects an entire row of
            # np.nan for non-spatial dimensions, so insert row(s) for that.
            none_indices = [i for i, size in enumerate(pixel_size)
                            if size is None or size == np.nan]
            space_directions = np.insert(space_directions, none_indices, np.nan, axis=0)
            metadata.update({'space directions': space_directions})

        if 'space dimension' not in metadata and 'space' not in metadata:
            metadata.update({'space dimension': data.ndim})
        if units is not None:
            if hasattr(units, '__iter__') and not isinstance(units, str):
                units = list(units)
            elif 'space dimension' in metadata:
                units = [units] * metadata['space dimension']
            else:
                units = [units] * data.ndim
            metadata.update({'space units': units})

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
        nrrd.write(filename,
                   data,
                   header=metadata,
                   custom_field_map=custom_field_map,
                   compression_level=compression_level,
                   index_order='C')

    if extension in ['raw', 'vol']:
        raise NotImplementedError

    if extension == 'zarr':
        raise NotImplementedError

    if extension == 'ng':
        from cloudvolume import CloudVolume
        from cloudvolume.exceptions import InfoUnavailableError

        # CloudVolume expects data in Fortran order
        data = data.T

        resolution = 1
        if pixel_size is not None:
            resolution = pixel_size
        try:
            iter(resolution)
        except TypeError:
            resolution = [resolution] * data.ndim
        if compress is True or compress in ['lossy', 'jpeg', 'jpg']:
            encoding = 'jpeg'
            gzip = False
        elif compress is None or compress in ['lossless', 'gzip']:
            if compress is None:
                print('Using gzip compression by default for .ng format.')
            encoding = 'raw'
            gzip = True
        elif compress is False:
            encoding = 'raw'
            gzip = False
        else:
            raise ValueError('For .ng format, compress must be '
                             'True/"lossy"/"jpeg"/"jpg", "lossless"/"gzip",'
                             f' or False, but was "{compress}"')

        if not any(filename.startswith(prefix)
                   for prefix in ['file://', 'gs://', 's3://']):
            filename = 'file://' + filename

        try:
            vol = CloudVolume(filename, compress=gzip)
            if (vol.shape[:data.ndim] != data.shape
                    or any(s != 1 for s in vol.shape[data.ndim:])):
                raise ValueError('Dataset already exists at the specified path,'
                                 ' but its shape does not match the given data.')
        except InfoUnavailableError:
            info = CloudVolume.create_new_info(
                num_channels=1,
                layer_type='image',
                data_type=data.dtype,
                encoding=encoding,
                resolution=resolution,
                voxel_offset=[0, 0, 0],
                chunk_size=[64, 64, 64],
                volume_size=data.shape
            )

            vol = CloudVolume(filename, info=info, compress=gzip)
            vol.commit_info()

        vol[:] = data


write = save  # Function name alias
to_file = save  # Function name alias


def save_video(data, filename, time_axis=0, color_axis=None, overwrite=False,
               dim_order='yx', framerate=30, crf=23, compression_speed='medium'):
    """
    Save a 3D numpy array of greyscale values OR a 4D numpy array of RGB values as a video

    Follows the PyAV cookbook section on generating video from numpy arrays:
    https://pyav.basswood-io.com/docs/develop/cookbook/numpy.html#generating-video

    Parameters
    ----------
    data : numpy.ndarray or list of filenames
        A 3D (grayscale) or 4D (RGB) numpy array of pixel values.

    filename : str
        The filename to save the video to.

    time_axis : int, default 0
        The axis of the data array that will be played as time in the video.

    color_axis : int or None, default None
        If not None, specifies the axis of the color channels (e.g., -1 for last axis, 1 for second axis).
        If None, data is assumed to be grayscale.

    overwrite : bool, default False
        Whether to overwrite the file if it already exists.

    dim_order : 'yx' (default) or 'xy'
        The order of the spatial dimensions in the input data.
    """
    try:
        import av
    except ImportError:
        raise ImportError('To save videos, you must have PyAV installed. You can install it with "pip install av".')

    if color_axis is not None:
        if data.ndim != 4:
            raise ValueError('Input data must be a 4D numpy array for color video.')
        # Move time axis to 0, color axis to -1
        data = np.moveaxis(data, time_axis, 0)
        if color_axis != -1:
            data = np.moveaxis(data, color_axis, -1)
        if 'xy' in dim_order:
            data = data.swapaxes(1, 2)
        n_frames = data.shape[0]
        height, width, channels = data.shape[1:]
        if channels != 3:
            raise ValueError('Color video must have 3 channels (RGB).')
    else:
        if data.ndim != 3:
            raise ValueError('Input data must be a 3D numpy array for grayscale video.')
        data = np.moveaxis(data, time_axis, 0)
        if 'xy' in dim_order:
            data = data.swapaxes(1, 2)
        n_frames = data.shape[0]
        height, width = data.shape[1:]

    filename = os.path.expanduser(str(filename))
    if filename.split('.')[-1].lower() not in ['mp4', 'mkv', 'avi', 'mov']:
        filename += '.mp4'
    if os.path.exists(filename) and not overwrite:
        raise FileExistsError(f'File {filename} already exists. '
                              'Set overwrite=True to overwrite.')
    extension = filename.split('.')[-1].lower()
    if extension == 'mp4':
        pad = [[0, 0], [0, 0], [0, 0]]
        if height % 2 != 0:
            pad[1][1] = 1
        if width % 2 != 0:
            pad[2][1] = 1
        if pad != [[0, 0], [0, 0], [0, 0]]:
            data = np.pad(data, pad, mode='edge')

    container = av.open(filename, mode='w')
    stream = container.add_stream('libx264', rate=framerate)
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': str(crf), 'preset': compression_speed}
    stream.height = data.shape[1]
    stream.width = data.shape[2]

    for frame_i in range(n_frames):
        if color_axis is not None:
            frame = av.VideoFrame.from_ndarray(data[frame_i], format='rgb24')
        else:
            frame = av.VideoFrame.from_ndarray(data[frame_i], format='gray')
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()


def show(data,
         dim_order='yx',
         data_type: Literal['image', 'segmentation'] = 'image',
         mode: Literal['PIL', 'mpl'] = 'PIL',
         convert_to_8bit=True,
         channel_axis='guess',
         **kwargs):
    """
    Display a numpy array of pixel values as an image. Supported types:
      1-channel (grayscale) : data.shape must be (y, x)
      3-channel (RGB)       : data.shape must be (y, x, 3)
      4-channel (RGBA)  : data.shape must be (y, x, 4)

    If `dim_order` is set to 'xy' (instead of the default 'yx'), then
    swap the y and x above.

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

    if data_type == 'segmentation':
        data = utils.assign_random_colors(data, seed=kwargs.get('seed', None))

    if (not data.ndim == 2) and not (data.ndim == 3 and find_channel_axis(data) is not None):
        m = ('Data must have shape (y, x) for grayscale, '
             '(y, x, 3) for RGB, or (y, x, 4) for RGBA but had '
             f'shape {data.shape}')
        if 'xy' in dim_order:
            m = m.replace('y, x', 'x, y')
        raise ValueError(m)

    if 'xy' in dim_order:
        data = data.T
    if channel_axis == 'guess':
        channel_axis = find_channel_axis(data)
    if utils.isint(channel_axis) and channel_axis != -1:
        data = np.moveaxis(data, find_channel_axis(data), -1)

    if convert_to_8bit and data.dtype != np.uint8:
        data = operations.to_8bit(data)

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


def find_channel_axis(data, expected_channel_axis=[0, -1]):
    """
    If the given numpy array has a shape suggesting that it has a
    channel (color) axis (that is, any axis with length 2 (2-color),
    3 (RGB), or 4 (RGBA)), return the index of that axis.

    Parameters
    ----------
    data : numpy.ndarray
        The numpy array to check for a channel axis.

    expected_channel_axis : int or list of int, default [0, -1]
        If None, any axis having length 2, 3, or 4 will be considered
        a channel axis.
        If an int, only that axis index will be checked.
        If a list of ints, all axes with those indices will be checked.
        The default value of [0, -1] checks the first and last axes, which is
        almost always where a channel axis will be found.

    Returns
    -------
    int or None
        The index of the channel axis, or None if no channel axis was found.

        If expected_channel_axis is given, the returned value will be one of
        the expected_channel_axis values, or None if no channel axis was found.

        If expected_channel_axis is None, the returned value will be between
        0 and data.ndim - 1, inclusive, or None if no channel axis was found.

        Note that returning 0 means the channel axis was found and is the first
        axis, so be careful not to do a test like `if find_channel_axis(data):`
        because 0 will evaluate to False even though the data has a channel axis.
        Instead write `if find_channel_axis(data) is not None:`
    """
    if isinstance(expected_channel_axis, int):
        expected_channel_axis = [expected_channel_axis]
    if expected_channel_axis is None:
        expected_channel_axis = range(data.ndim)
    for axis in expected_channel_axis:
        if data.shape[axis] in [2, 3, 4]:
            return axis
    return None
