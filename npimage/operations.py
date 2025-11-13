#!/usr/bin/env python3
"""
Functions that manipulate pixel data in an image in some way.

Function list:
- to_8bit: Convert from whatever bit-depth to 8-bit.
- downsample: Reduce the size of an image by a given factor.
- offset: Shift an image by a given distance.
- paste: Paste one image onto another at a given offset.
- overlay: Overlay multiple images onto a canvas at given offsets.
- overlay_two_images: Overlay two images with the second one offset.
"""

from typing import Iterable, Literal, Union, Optional, Tuple, List

import numpy as np
from tqdm import tqdm

from .utils import iround, eq, find_channel_axis


def squeeze_dtype(image: np.ndarray, minimum_bits=1):
    """
    Cast a numpy array to the smallest possible dtype without losing information.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError('image must be a numpy array')
    if minimum_bits > 64:
        raise ValueError('minimum_bits must be <= 64')

    im_has_fractions = not np.all(image == np.floor(image))
    if im_has_fractions:
        candidates = [np.float16, np.float32, np.float64]
        # TODO some logic
        print('WARNING: squeeze_dtype not fully implemented on float values,'
              ' returning original data unchanged.')
        return image
    elif image.min() < 0:
        candidates = [np.int8, np.int16, np.int32, np.int64]
    elif image.max() <= 1 and minimum_bits <= 1:
        return image.astype(bool, copy=False)
    else:
        candidates = [np.uint8, np.uint16, np.uint32, np.uint64]

    image_max = image.max()
    if image_max <= 1 and minimum_bits <= 1:
        return image.astype(bool, copy=False)
    for dtype in candidates:
        info = np.iinfo(dtype)
        if image_max <= info.max and info.bits >= minimum_bits:
            return image.astype(dtype, copy=False)
    raise ValueError('Image has values larger than the maximum representable value')


def to_8bit(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Convert an image to 8-bit.

    See `npimage.cast()` for full description of available arguments.
    """
    return cast(image, 'uint8', **kwargs)


def to_16bit(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Convert an image to 16-bit.

    See `npimage.cast()` for full description of available arguments.
    """
    return cast(image, 'uint16', **kwargs)


def cast(image: np.ndarray,
         output_dtype: Union[str, np.dtype],
         maximize_contrast: bool = False,
         round_before_cast_to_int: bool = True,
         bottom_percentile=0.05,
         top_percentile=99.95,
         bottom_value=None,
         top_value=None) -> np.ndarray:
    """
    Cast an image to a new dtype.

    If maximize_contrast is False, the image is simply clipped to the
    range of the new dtype and cast to that dtype. In this case all
    arguments after maximize_contrast are ignored.

    If maximize_contrast is True, pixel values will be adjusted linearly to
    fill the output_dtype's range. The default adjustment parameters of:
    - bottom_percentile=0.05
    - top_percentile=99.95
    will stretch nearly the entire range of the source pixels to fill the
    entire range of the output data type. (Compared to using percentiles
    of 0 and 100, using these percentiles prevents a few extreme outlier
    pixel values from causing a suboptimal contrast adjustment. These
    percentages are are ~3.3 standard deviations below and above the mean,
    respectively, if the pixel values are approximately normally distributed.)

    bottom_value and top_value are determined by the bottom_percentile and
    top_percentile arguments and the distribution of pixel values in the image.
    However you may set bottom_value and/or top_value directly, in which case
    the corresponding bottom and/or top percentile argument(s) will be ignored.

    Values in the starting image <= bottom_value are mapped to the minimum
    value representable in the new dtype, and values >= top_value are mapped
    to the maximum value representable in the new dtype. Values within the
    range (bottom_value, top_value) will be mapped linearly into the range
    of the new dtype between the minimum and maximum values. More specifically,
    there is a linear remapping step followed by a rounding down step, which
    produces the following two transformations for a number of example values
    (during a cast to uint8; max target value would change for other dtypes):
    - bottom_value or less -> just less than 1 -> 0
    - a value just greater than bottom_value -> just greater than 1 -> 1
    - top_value or larger -> just greater than 255 -> 255
    - a value just less than top_value -> just less than 255 -> 254
    - values between bottom_value and top_value -> linearly mapped to
      values between 1 and 254 -> rounded down to the nearest integer.

    Examples of behavior if you change bottom_percentile/top_percentile:
    - Set bottom_percentile=0 to map only the minimum value in the source
      image to 0 and and top_percentile=100 to map only the maximum value in
      the source image to 255, and all other values to the range [1, 254].
    - Set bottom_percentile=25 to map all of the the lowest 25% of the source
      image's pixel values to 0 and set top_percentile=75 to map all of the
      highest 25% of the source image's pixel values to 255, and values in
      between the 25th and 75th percentiles to the range [1, 254].
    """
    assert isinstance(image, np.ndarray)

    if np.issubdtype(output_dtype, np.integer):
        info = np.iinfo(output_dtype)
    elif np.issubdtype(output_dtype, np.floating):
        info = np.finfo(output_dtype)
    else:
        raise TypeError(f'Unsupported dtype: {output_dtype}')

    if not maximize_contrast:
        if (round_before_cast_to_int
                and np.issubdtype(output_dtype, np.integer)
                and not np.issubdtype(image.dtype, np.integer)):
            image = image.round()
        return np.clip(
            image,
            info.min,
            info.max
        ).astype(output_dtype, copy=False)

    if bottom_value is None or top_value is None:
        percentiles = np.percentile(image, [bottom_percentile, top_percentile])
    if bottom_value is None:
        bottom_value = percentiles[0]
    if top_value is None:
        top_value = percentiles[1]
    if bottom_value == top_value:
        raise ZeroDivisionError('top_value and bottom_value are the same: '
                                '{}'.format(top_value))

    if np.issubdtype(output_dtype, np.integer):
        target_range = (np.iinfo(output_dtype).min + 1 - 1e-5,
                        np.iinfo(output_dtype).max + 1e-5)
    else:
        raise ValueError(
            '`maximize_contrast=True` only supports integer data types.'
            ' You can cast to other dtypes while specifying how to'
            ' adjust the pixel values to maintain contrast by calling'
            ' `npimage.adjust_brightness()` with the target_range set'
            ' to the range you want to map to.')

    return adjust_brightness(image,
                             (bottom_value, top_value),
                             target_range,
                             clip=True,
                             output_dtype=output_dtype)


def adjust_brightness(image: np.ndarray,
                      starting_range: Tuple[float, float],
                      target_range: Tuple[float, float],
                      output_dtype: Optional[Union[str, np.dtype]] = None,
                      clip: bool = False) -> np.ndarray:
    """
    Linearly map an image's pixel values from one range to another,
    thereby adjusting the brightness and contrast of the image.

    By including clip and output_dtype, this can be used to convert
    image volumes to lower bit-depths, e.g. 8-bit.
    See `npimage.cast()` for an example of how to do this.

    Parameters
    ----------
    image : np.ndarray
      The image to adjust.

    starting_range : tuple of 2 float
      The range of pixel values in the input image. Values outside
      this range will be clipped if clip=True.

    target_range : tuple of float
      The range of pixel values to adjust the starting_range to.
      If not provided, output_dtype must be provided instead.

    """
    if target_range is None and output_dtype is None:
        raise ValueError('Must provide either target_range or output_dtype')

    if output_dtype is None:
        output_dtype = image.dtype

    if clip:
        image = np.clip(image, *starting_range)

    bottom_value, top_value = starting_range
    bottom_target, top_target = target_range

    # TODO write a more clever implementation that doesn't require casting
    # to float64, which can take up a lot of memory for large arrays.
    return ((image.astype('float64') - bottom_value)
            / (top_value - bottom_value)
            * (top_target - bottom_target)
            + bottom_target).astype(output_dtype)


def downsample(image: np.ndarray,
               factor: Union[int, Iterable[int]] = 2,
               method: Literal['mean', 'median', 'max', 'min'] = 'mean',
               keep_input_dtype=True,
               verbose=False) -> np.ndarray:
    """
    Downsample an image by a given factor along each axis.

    Parameters
    ----------
    image : np.ndarray
        The image to downsample.

    factor : int or iterable of ints, default 2
        An iterable with length matching the number of axes in the image,
        specifying a downsampling factor along each axis.
        If factor is provided as an int, that int will be used for each axis.
        If the image has a channel axis (RGB/RGBA), it is not necessary to
        specify a factor for that axis and so the 'factor' iterable can be
        one element shorter than the number of axes in the image.

    keep_input_dtype : bool, default True
        If True, the output image will have the same dtype as the input image.
        If False, the output image will have dtype float64 to keep full precision.

    Returns
    -------
    np.ndarray
        The downsampled image.

    Examples
    --------
    Downsample a shape (2, 4) array using default settings of factor=2, keep_input_dtype=True

    >>> downsample(np.array([[1, 2, 3, 4],
    ...                      [5, 6, 7, 8]]))
    array([[4, 6]])

    Note that the output dtype is kept as int, which loses some precision. Compare to:

    >>> downsample(np.array([[1, 2, 3, 4],
    ...                      [5, 6, 7, 8]]),
    ...            keep_input_dtype=False)
    array([[3.5, 5.5]])
    """
    channel_axis = find_channel_axis(image)
    if np.issubdtype(type(factor), np.integer):
        if channel_axis is not None:
            # If RGB/RGBA image, don't downsample the colors axis
            factor = (factor,) * (len(image.shape) - 1)
        else:
            factor = (factor,) * len(image.shape)
    if len(factor) == len(image.shape) - 1 and channel_axis is not None:
        if verbose:
            print('RGB/RGBA image detected - not downsampling channel axis.')
        factor = list(factor)
        factor.insert(channel_axis, 1)
    if any([f > l > 1 for f, l in zip(factor, image.shape)]):
        raise ValueError('Downsampling factor must be <= image size along each axis')

    # In the code below, 'l' is used for the elements of image.shape and 'f' is
    # used for the elements of factor.
    padding = [(0, 0) if l % f == 0
               else (0, f - l % f)
               for l, f in zip(image.shape, factor)]
    # Pad the image so that its axes are a multiple of the downsampling factor
    # This is necessary because the image will be split into blocks of size
    # 'factor' and then the mean of each block will be taken, so the image
    # needs to be a multiple of 'factor' in size so that the blocks are all
    # the same size.
    image = np.pad(image, padding, mode='edge')
    if not all([(l == 1) or (l % f == 0) for l, f in zip(image.shape, factor)]):
        raise RuntimeError('Padding failed: shape should be a multiple of factor')

    # For any dimension with length > 1, split it into blocks of size f, then
    # average over each block.
    temp_shape = []
    for l, f in zip(image.shape, factor):
        if l == 1:
            if f != 1:
                raise ValueError("You can't downsample along"
                                 " an axis of length 1.")
            temp_shape.extend([1, 1])
        else:
            temp_shape.extend([l // f, f])
    axes_to_collapse = tuple(range(1, len(temp_shape), 2))
    collapse_functions = {'mean': np.mean, 'max': np.max,
                          'min': np.min, 'median': np.median}
    try:
        collapse_function = collapse_functions[method]
    except KeyError:
        raise ValueError(f'method must be one of {collapse_functions.keys()}')
    image_downsampled = collapse_function(image.reshape(temp_shape), axis=axes_to_collapse)

    if keep_input_dtype:
        if np.issubdtype(image.dtype, np.integer):
            image_downsampled = iround(image_downsampled, output_dtype=image.dtype)
        else:
            image_downsampled = image_downsampled.astype(image.dtype)

    return image_downsampled


def offset(image: np.ndarray,
           distance: Union[float, Iterable[float]],
           axis: int = None,
           expand_bounds: bool = False,
           fill_empty_with: float = 0,
           keep_input_dtype: bool = True,
           fill_transparent: bool = False,
           verbose: bool = False) -> np.ndarray:
    """
    Offset an image by a given distance.

    'distance' must be an iterable with length matching the number of axes in
    the image, to specify an number of pixels to offset along each axis. If the
    image is rgb or rgba (that is, the final axis has length 3 or 4), it is not
    necessary to specify an offset for that axis and so the 'distance' iterable
    can be one element shorter than the number of axes in the image.

    If edge_mode is set to 'constant', the pixels no longer occupied by the
    original image as a result of the offset will be filled in with
    'fill_empty_value'.

    See also scipy.ndimage.shift, which performs a very similar operation
    """
    try:
        iter(distance)
        if axis is not None:
            raise ValueError('Either give distance as one number and specify'
                             ' axis, or give distance as an iterable. You'
                             ' did a mix of those two.')
    except TypeError:
        distance_iter = [0] * len(image.shape)
        if len(distance_iter) == 1 and axis is None:
            axis = 0
        if axis is None:
            raise ValueError('Must specify axis when giving distance as a'
                             ' single number.')
        distance_iter[axis] = distance
        distance = distance_iter

    if len(image.shape) == len(distance) + 1:
        channel_axis = find_channel_axis(image)
        if channel_axis is not None:
            # Specify no offset along the channels axis, if not specified by user
            distance = list(distance)
            distance.insert(channel_axis, 0)
            distance = tuple(distance)

    if len(image.shape) != len(distance):
        m = (f'distance must have length {len(image.shape)} to specify an'
             ' offset along each axis of the image, but instead had length'
             f' {len(distance)}')
        raise ValueError(m)

    new_dtype = (image.dtype if keep_input_dtype or all(eq(d, int(d)) for d in distance)
                 else np.float64)
    distance_int = [int(d) for d in distance]
    new_shape = (image.shape if not expand_bounds else
                 np.array(image.shape) + np.array([int(max(0, d)) for d in distance]))
    new_image = np.full(new_shape, fill_empty_with, dtype=new_dtype)

    if fill_transparent:
        raise NotImplementedError('fill_transparent not yet implemented')
    # Previous implementation of fill_transparent:
    #if image.shape[-1] == 4 and not fill_transparent:
    #    # If rgba, set alpha channel value to max
    #    # The line below means new_image[:, :, :, ..., :, -1] = 255
    #    new_image[tuple([slice(None, None)] * (len(image.shape)-1) + [-1])] = 255

    source_range = [slice(max(0, -d), min(s, s-d))
                    for d, s in zip(distance_int, new_image.shape)]
    target_range = [slice(max(0, d), min(s, s+d))
                    for d, s in zip(distance_int, new_image.shape)]

    if verbose:
        print('Performing integer offset...')
    # This is the line that does the main operation
    new_image[tuple(target_range)] = image[tuple(source_range)]

    for i, d in enumerate(distance):
        progress_msg = f'Performing subpixel offset along axis {i}' if verbose else None
        if not eq(d, int(d)):
            _offset_subpixel(new_image, d - int(d), i,
                             edge_mode='constant',
                             constant_edge_value=fill_empty_with,
                             keep_input_dtype=True,
                             fill_transparent=fill_transparent,
                             inplace=True,
                             progress_msg=progress_msg)

    return new_image


def _offset_subpixel(image: np.ndarray,
                     distance: float,
                     axis: int,
                     edge_mode: Literal['extend', 'wrap',
                                        'reflect', 'constant'] = 'extend',
                     constant_edge_value: Optional[float] = None,
                     keep_input_dtype: bool = True,
                     fill_transparent: bool = False,
                     inplace: bool = False,
                     progress_msg: Optional[str] = None):
    """
    Offset an image by a fraction of a pixel along a single specified axis.

    If an offset of 0.1 is requested, the output will be 10% of the image
    shifted one pixel upward plus 90% of the original image.
    If an offset of -0.1 is requested, the output will be 10% of the image
    shifted one pixel downward plus 90% of the original image.
    If an offset of 0.5 is requested, the output will be 50% of the image
    shifted one pixel upward plus 50% of the original image.
    etc.

    The pixels no longer occupied by the original image as a result of the
    offset will be filled in with 'edge_fill_value'.
    """
    if inplace and not keep_input_dtype:
        raise ValueError("inplace=True doesn't make sense with keep_input_dtype=False")
    if distance < -1 or distance > 1:
        raise ValueError('subpixel offset distance must be between -1 and 1')
    if abs(distance) < 1e-6:
        return image if inplace else image.copy()
    if edge_mode not in ['extend', 'wrap', 'reflect', 'constant']:
        raise ValueError('edge_mode must be one of "extend", "wrap",'
                         ' "reflect", or "constant"')
    if fill_transparent:
        raise NotImplementedError('fill_transparent not yet implemented')

    if not inplace:
        if keep_input_dtype:
            image = image.copy()
        else:
            image = image.astype('float64')

    abs_distance = abs(distance)
    sign = 1 if distance > 0 else -1
    axis_size = image.shape[axis]
    slicer: List[Union[slice, int]] = [slice(None)] * image.ndim

    # Handle last slice which attempts to pull data from out of bounds
    final_index = 0 if sign > 0 else -1
    slicer[axis] = final_index
    final_slice = tuple(slicer)

    if edge_mode == 'extend':
        edge_data = image[final_slice].copy()
    elif edge_mode == 'wrap':
        slicer[axis] -= sign
        edge_data = image[tuple(slicer)].copy()
    elif edge_mode == 'reflect':
        slicer[axis] += sign
        edge_data = image[tuple(slicer)].copy()
    elif edge_mode == 'constant':
        if constant_edge_value is None:
            raise ValueError('constant_edge_value must be provided when'
                             ' edge_mode is "constant"')
        edge_data = constant_edge_value

    loop_range = range(axis_size - 1, 0, -1) if sign > 0 else range(0, axis_size - 1, 1)

    for i in tqdm(loop_range, desc=progress_msg, disable=not bool(progress_msg)):
        slicer[axis] = i
        current_slice = tuple(slicer)
        slicer[axis] = i - sign
        adjacent_slice = tuple(slicer)

        new_values = (
            (1 - abs_distance) * image[current_slice]
            + abs_distance * image[adjacent_slice]
        )
        if keep_input_dtype and np.issubdtype(image.dtype, np.integer):
            new_values = iround(new_values, output_dtype=image.dtype)
        image[current_slice] = new_values

    image[final_slice] = (
        (1 - abs_distance) * image[final_slice]
        + abs_distance * edge_data
    )

    if not inplace:
        return image


def paste(image: np.ndarray,
          target: np.ndarray,
          offset: Iterable[float],
          subpixel_edge_mode: Literal['extend', 'wrap',
                                      'reflect', 'constant'] = 'extend',
          constant_edge_value: Optional[float] = None,
          verbose: bool = False):
    """
    Paste an image onto another image at a given offset.

    `target` is modified in place. Regions of `image` that would be
    pasted outside the bounds of `target` are ignored.
    """
    try:
        iter(offset)
    except TypeError:
        offset = [offset] * len(image.shape)
    if len(offset) != image.ndim:
        raise ValueError('The length of the offset must match the number of '
                         'dimensions in the image.')
    offset_int = [int(x) for x in offset]
    offset_subpixel = [x - int(x) for x in offset]
    if any(not eq(o, 0) for o in offset_subpixel):
        image = image.copy()
    for i, offset in enumerate(offset_subpixel):
        progress_msg = f'Performing subpixel offset along axis {i}' if verbose else None
        if not eq(offset, 0):
            _offset_subpixel(image, offset, i,
                             edge_mode=subpixel_edge_mode,
                             constant_edge_value=constant_edge_value,
                             inplace=True,
                             progress_msg=progress_msg)

    source_range = [slice(max(0, -o), min(s, t-o))
                    for o, s, t in zip(offset_int, image.shape, target.shape)]
    target_range = [slice(max(0, o), min(t, max(0, o) + min(s, t-o) - max(0, -o)))
                    for o, s, t in zip(offset_int, image.shape, target.shape)]

    target[tuple(target_range)] = image[tuple(source_range)]


def overlay(ims: List[np.ndarray],
            offsets: List[Tuple[float]],
            later_images_on_top=True,
            expand_bounds=False,
            fill_value=0):
    """
    Overlay multiple images / image volumes onto each other, with each
    image offset by a given amount.

    Parameters
    ----------
    ims : list of np.ndarray
        The images or image volumes to overlay.

    offsets : list of tuple of float
        The offsets to apply to each image volume. Each tuple should
        have the same length as the number of dimensions in the
        corresponding images.

    later_images_on_top : bool
        If True, the later images in the list will be drawn on top of
        the earlier images. If False, the earlier images will be drawn
        on top of the later images.

    expand_bounds : bool
        If True, the output image will be large enough to contain all
        of the input images. If False, the output image will be the same
        size as the first input image.
    """
    if not len(ims) == len(offsets):
        raise ValueError('The number of images must match the number of offsets.')
    ndims = ims[0].ndim
    if not all([im.ndim == ndims for im in ims[1:]]):
        raise ValueError('All images must have the same number of dimensions.')
    if not all([len(offset) == ndims for offset in offsets]):
        raise ValueError('All offsets must have the same length as the number'
                         ' of dimensions in the images.')

    if expand_bounds:
        origin = [min([0] + [offset[axis] for offset in offsets])
                  for axis in range(ndims)]
        max_coord = [max([im.shape[axis] + offset[axis]
                          for im, offset in zip(ims, offsets)])
                     for axis in range(ndims)]
        canvas_shape = [max_coord[axis] - origin[axis] for axis in range(ndims)]
        offsets = [tuple(offset[axis] - origin[axis] for axis in range(ndims))
                   for offset in offsets]
        canvas = np.full(canvas_shape, fill_value, dtype=ims[0].dtype)
    else:
        canvas = np.full_like(ims[0], fill_value)

    if not later_images_on_top:
        ims = reversed(ims)
        offsets = reversed(offsets)

    for im, offset in zip(ims, offsets):
        paste(im, canvas, offset)

    return canvas


def overlay_two_images(im1: np.ndarray,
                       im2: np.ndarray,
                       im2_offset: Iterable[float] = 0,
                       later_images_on_top=True,
                       expand_bounds=False,
                       fill_value=0):
    """
    Overlay two images, with the second image offset by the given amount.
    """
    if any(np.issubdtype(type(im2_offset), t) for t in [np.integer, np.floating]):
        im2_offset = [im2_offset] * im2.ndim
    if expand_bounds:
        offsets = ([max(0, -o) for o in im2_offset],
                   [max(0, o) for o in im2_offset])
    else:
        offsets = ([0] * len(im2_offset), im2_offset)
    return overlay([im1, im2],
                   offsets,
                   later_images_on_top=later_images_on_top,
                   expand_bounds=expand_bounds,
                   fill_value=fill_value)


def remove_bleedthrough(im, contaminated_slice, source_slice,
                        leave_saturated_pixels_alone=True):
    """
    Given two channels of multi-channel image, determine how strong the
    bleedthrough was from the source channel to the contaminated channel
    and remove the contamination.
    Saturated pixels are by default not changed, which is reasonable
    when bleedthrough is weak. If bleedthrough is strong, you may want
    to try leave_saturated_pixels_alone=False, though this may
    adjust those pixels more than desired.
    """
    #TODO implement ICA and use it to separate the two independent sources
    raise NotImplementedError


def assign_random_colors(data: np.ndarray,
                         seed: Optional[int] = None,
                         verbose: bool = False,
                         ) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if verbose:
        print('Finding unique IDs...')
    unique_ids, inverse = np.unique(data, return_inverse=True)
    if verbose:
        print('Assigning random colors...')
    colors = rng.integers(0, 255, size=(len(unique_ids), 3), dtype=np.uint8)
    data_colored = colors[inverse].reshape(data.shape + (3,))
    return data_colored
