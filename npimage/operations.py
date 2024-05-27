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

from collections.abc import Iterable
from typing import Literal
import gc

import numpy as np

from .utils import iround, eq, isint


def to_8bit(image: np.ndarray,
            bottom_percentile=0.05,
            top_percentile=99.95,
            bottom_value=None,
            top_value=None) -> np.ndarray:
    """
    Convert an image to 8-bit (uint8) by scaling the image's pixel values so
    that values <= bottom_value are mapped to 0 and values >= top_value are
    mapped to 255. Values within the range (bottom_value, top_value) will be
    mapped linearly into the range [1, 254]. By default, bottom_value and
    top_value are set to percentiles of the source image's pixel values.
    Some examples:
    - Set bottom_percentile=0 to map only the minimum value in the source image
      to 0 and and top_percentile=100 to map only the maximum value in the
      source image to 255, and all other values to the range [1, 254].
    - Set bottom_percentile=25 to map all of the the lowest 25% of the source
      image's pixel values to 0 and set top_percentile=75 to map all of the
      highest 25% of the source image's pixel values to 255.
    The defaults are:
    - bottom_percentile=0.05
    - top_percentile=99.95
    which are ~3.3 standard deviations below and above the mean, respectively,
    if the pixel values are approximately normally distributed. (Compared to
    using percentiles of 0 and 100, using these percentiles lessens the impact
    of a few extreme outlier pixel values in the image.)

    You may however specify bottom_value and/or top_value explicitly yourself,
    in which case bottom_percentile and/or top_percentile will be ignored.

    Algorithm:
    - Clip values < bottom_value or > top_value
      (i.e. replace them with bottom_value or top_value)
    - Map the range [bottom_value, top_value] to
      [just less than 1, just greater than 255]
    - Cast to int (which rounds down)
    From these steps, values will be transformed as follows:
    - bottom_value or less -> just less than 1 -> 0
    - a value just greater than bottom_value -> just greater than 1 -> 1
    - top_value or larger -> just greater than 255 -> 255
    - a value just less than top_value -> just less than 255 -> 254
    - values between bottom_value and top_value -> linearly mapped to
      values between 1 and 254
    """
    assert isinstance(image, np.ndarray)

    if bottom_value is None or top_value is None:
        percentiles = np.percentile(image, [bottom_percentile, top_percentile])
    if bottom_value is None:
        bottom_value = percentiles[0]
    if top_value is None:
        top_value = percentiles[1]

    if bottom_value == top_value:
        raise ZeroDivisionError('top_value and bottom_value are the same: '
                                '{}'.format(top_value))

    image = np.clip(image, bottom_value, top_value)

    bottom_target = 1 - 1e-12
    top_target = 255 + 1e-12
    # TODO some more clever implementation that doesn't require casting to
    # float64, which can take up a lot of memory.
    return ((image.astype('float64') - bottom_value)
            / (top_value - bottom_value)
            * (top_target - bottom_target)
            + bottom_target).astype('uint8')


def downsample(image: np.ndarray,
               factor: [int, Iterable] = 2,
               keep_input_dtype=True) -> np.ndarray:
    """
    Downsample an image by a given factor along each axis.

    Parameters
    ----------
    image : np.ndarray
        The image to downsample.

    factor : int or Iterable
        An iterable with length matching the number of axes in the image,
        specifying a downsampling factor along each axis.
        If factor is provided as an int, that int will be used for each axis.
        If the image is rgb or rgba (that is, the final axis has length 3 or
        4), it is not necessary to specify a factor for that axis and so the
        'factor' iterable can be one element shorter than the number of axes in
        the image.

    keep_input_dtype : bool
        If True, the output image will have the same dtype as the input image.
        If False, the output image will have dtype float64 to keep full precision.
    """
    if isinstance(factor, int):
        if image.shape[-1] in [3, 4]:
            # If RGB/RGBA image, don't downsample the colors axis
            factor = (factor,) * (len(image.shape) - 1)
        else:
            factor = (factor,) * len(image.shape)
    if len(factor) == len(image.shape) - 1 and image.shape[-1] in [3, 4]:
        print('RGB/RGBA image detected - not downsampling last axis.')
        factor = (*factor, 1)
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
    axes_to_avg = tuple(range(1, len(temp_shape), 2))
    image_downsampled = image.reshape(temp_shape).mean(axis=axes_to_avg)

    if keep_input_dtype:
        if np.issubdtype(image.dtype, np.integer):
            image_downsampled = iround(image_downsampled, output_dtype=image.dtype)
        else:
            image_downsampled = image_downsampled.astype(image.dtype)

    return image_downsampled


def offset(image: np.ndarray,
           distance: float or Iterable[float],
           axis: int = None,
           expand_bounds: bool = False,
           edge_mode: Literal['extend', 'wrap',
                              'reflect', 'constant'] = 'extend',
           edge_fill_value=0,
           fill_transparent=False) -> np.ndarray:
    """
    Offset an image by a given distance.

    'distance' must be an iterable with length matching the number of axes in
    the image, to specify an number of pixels to offset along each axis. If the
    image is rgb or rgba (that is, the final axis has length 3 or 4), it is not
    necessary to specify an offset for that axis and so the 'distance' iterable
    can be one element shorter than the number of axes in the image.

    If edge_mode is set to 'constant', the pixels no longer occupied by the
    original image as a result of the offset will be filled in with
    'edge_fill_value'.

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
        if axis is None:
            raise ValueError('Must specify axis when giving distance as a'
                             ' single number.')
        distance_iter[axis] = distance
        distance = distance_iter
    if edge_mode not in ['extend', 'wrap', 'reflect', 'constant']:
        raise ValueError("edge_mode must be one of 'extend', 'wrap', 'reflect', or 'constant'")
    if edge_mode in ['wrap', 'reflect']:
        raise NotImplementedError("edge_mode '{}' not yet implemented".format(edge_mode))

    if len(image.shape) == len(distance) + 1 and image.shape[-1] in [1, 3, 4]:
        # Specify no offset along the channels axis, if not specified by user
        distance = (*distance, 0)

    if len(image.shape) != len(distance):
        m = (f'distance must have length {len(image.shape)} to specify an'
             ' offset along each axis of the image, but instead had length'
             f' {len(distance)}')
        raise ValueError(m)

    if not expand_bounds:
        new_image = np.full_like(image, edge_fill_value)
    else:
        new_shape = np.array(image.shape) + np.array([int(max(0, d)) for d in distance])
        new_image = np.full(new_shape, edge_fill_value, dtype=image.dtype)

    if image.shape[-1] == 4 and not fill_transparent:
        # If rgba, set alpha channel value to max
        # The line below means new_image[:, :, :, ..., :, -1] = 255
        new_image[tuple([slice(None, None)] * (len(image.shape)-1) + [-1])] = 255

    distance_int = [int(x) for x in distance]

    source_range = [slice(max(0, -d), min(s, s-d)) for d, s in zip(distance_int, new_image.shape)]
    target_range = [slice(max(0, d), min(s, s+d)) for d, s in zip(distance_int, new_image.shape)]

    new_image[tuple(target_range)] = image[tuple(source_range)]

    for i, d in enumerate(distance):
        if not eq(d, int(d)):
            _offset_subpixel(new_image, d - int(d), i,
                             edge_fill_value=edge_fill_value,
                             fill_transparent=fill_transparent, inplace=True)

    return new_image


def _offset_subpixel(image: np.ndarray,
                     distance: float,
                     axis: int,
                     edge_mode: Literal['extend', 'wrap',
                                        'reflect', 'constant'] = 'extend',
                     edge_fill_value=0,
                     fill_transparent=False,
                     inplace=False):
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
    assert -1 < distance and distance < 1

    one_pix_offset = [0] * len(image.shape)
    one_pix_offset[axis] = 1 if distance >= 0 else -1
    image_1pix_shifted = offset(image, one_pix_offset,
                                edge_fill_value=edge_fill_value,
                                fill_transparent=fill_transparent)
    distance = abs(distance)

    if np.issubdtype(image.dtype, np.integer):
        # If the input array is an integer type, we should avoid creating a
        # float version of the array during calculations, because float arrays
        # can take up an unacceptable amount of memory. (e.g. If I write lazy
        # code that ends up multiplying a uint8 array by a fractional value
        # like 0.5, a float64 array is created which takes up 8x the amount of
        # memory as the source array. And we need to make two of these!).
        # Instead, we will do a trick of increasing the bit-depth of the source
        # array by 1 byte (or a few bytes, since numpy only works with bit
        # depths that are a power of 2, e.g. uint24 isn't a thing), use that
        # additional range to keep some accuracy during the weighted average
        # calculation, then cast back to the original dtype.

        upcast_dtype = np.dtype(f'{image.dtype.kind}{image.dtype.itemsize * 2}')
        image_upcast = image.astype(upcast_dtype)
        image_1pix_shifted_upcast = image_1pix_shifted.astype(upcast_dtype)

        # We'll use the extra bit of precision to enable us to use integer
        # weights from 0 to 255 instead of float weights from 0.0 to 1.0
        image_weight = int(256 * (1 - distance))
        image_1pix_shifted_weight = int(256 * distance)
        # Adding 127 before dividing by 256 means we round to the nearest
        # integer instead of truncating (which we'd get without the 127)
        image_subpix_shifted = (
            (image_upcast * image_weight) +
            (image_1pix_shifted_upcast * image_1pix_shifted_weight) +
            127
        ) // 256
        del image_upcast, image_1pix_shifted_upcast, image_1pix_shifted
        gc.collect()

        # Now cast back to the original dtype
        image_subpix_shifted = image_subpix_shifted.astype(image.dtype)

    elif np.issubdtype(image.dtype, np.floating):
        image_subpix_shifted = image * (1 - distance) + image_1pix_shifted * distance

    if inplace:
        image[:] = image_subpix_shifted
    else:
        return image_subpix_shifted


def paste(image: np.ndarray,
          target: np.ndarray,
          offset: Iterable[float]):
    """
    Paste an image onto another image at a given offset.

    `target` is modified in place. Regions of `image` that would be
    pasted outside the bounds of `target` are ignored.
    """
    image = image.copy()
    offset_int = [int(x) for x in offset]
    offset_subpixel = [x - int(x) for x in offset]
    for i, offset in enumerate(offset_subpixel):
        if not eq(offset, 0):
            _offset_subpixel(image, offset, i, inplace=True)

    source_range = [slice(max(0, -o), min(s, t-o))
                    for o, s, t in zip(offset_int, image.shape, target.shape)]
    target_range = [slice(max(0, o), min(t, max(0, o) + min(s, t-o) - max(0, -o)))
                    for o, s, t in zip(offset_int, image.shape, target.shape)]

    target[tuple(target_range)] = image[tuple(source_range)]


def overlay(ims: list[np.ndarray],
            offsets: list[tuple[float]],
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
                       offset: Iterable[float] = 0,
                       later_images_on_top=True,
                       expand_bounds=False,
                       fill_value=0):
    """
    Overlay two images, with the second image offset by the given amount.
    """
    if isint(offset):
        offset = [offset] * im1.ndim
    if expand_bounds:
        offsets = ([abs(max(0, -o)) for o in offset], [max(0, o) for o in offset])
    else:
        offsets = ([0] * len(offset), offset)
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


def find_landmark(im1, landmark_bbox,
                  im2, search_bbox=None,
                  rotation_angles=[0]):
    """
    Given a region in a source image, find the region in the target
      image that most resembles the source feature.
    Specify search_bbox to restrict the search to a particular region
      within the target image. Otherwise the whole image is searched.
    By default rotation is not allowed, but specific rotation angles can
      be searched by passing an iterable specifying rotations (in degrees)
      to search, e.g.:
       rotation_range=[-90, 0, 90]
       rotation_range=np.arange(-10, 10.5, 0.5)
       rotation_range=np.linspace(-10, 10, 5)

    See also cv2.matchTemplate, e.g. https://docs.opencv.org/4.5.2/d4/dc6/tutorial_py_template_matching.html
    """
    #TODO implement
    raise NotImplementedError
