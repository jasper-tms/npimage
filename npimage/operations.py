#!/usr/bin/env python3

from collections.abc import Iterable

import numpy as np

from .utils import iround, eq, isint


def to_8bit(image: np.ndarray,
            bottom_percentile=0.4,
            top_percentile=99.6,
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
    - bottom_percentile=0.4 (~= 100*1/256)
    - top_percentile=99.6 (~= 100*255/256)
    to map the bottom 1/256th of the pixel values to 0 and the top 1/256th of
    the pixel values to 255. (Compared to using percentiles of 0 and 100, this
    approach lessens the impact of a few extreme outlier pixel values in the
    image.)

    You may however specify bottom_value and/or top_value explicitly yourself,
    in which case bottom_percentile and/or top_percentile will be ignored.

    Algorithm:
    - Clip values < bottom_value or > top_value
      (i.e. replace them with bottom_value or top_value)
    - Map the range [bottom_value, top_value] to
      [just less than 1, just greater than 255]
    - Cast to int (which rounds down)
    From these two steps, values will be transformed as follows:
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
                raise ValueError("You can't ask for downsampling along"
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
           distance: Iterable,
           fill_value=0,
           fill_transparent=False) -> np.ndarray:
    """
    Offset an image by a given distance.

    'distance' must be an iterable with length matching the number of axes in
    the image, to specify an number of pixels to offset along each axis. If the
    image is rgb or rgba (that is, the final axis has length 3 or 4), it is not
    necessary to specify an offset for that axis and so the 'distance' iterable
    can be one element shorter than the number of axes in the image.

    The pixels no longer occupied by the original image as a result of the
    offset will be filled in with 'fill_value'.

    See also scipy.ndimage.shift, which performs a very similar operation
    """
    if len(image.shape) == len(distance) + 1 and image.shape[-1] in [3, 4]:
        # Specify no offset along the channels axis, if not specified by user
        distance = (*distance, 0)

    if len(image.shape) != len(distance):
        m = (f'distance must have length {len(image.shape)} to specify an'
             ' offset along each axis of the image, but instead had length'
             f' {len(distance)}')
        raise ValueError(m)

    new_image = np.full_like(image, fill_value)  # Blank image filled with fill_value
    if image.shape[-1] == 4 and not fill_transparent:
        # If rgba, set alpha channel value to max
        # The line below means new_image[:, :, :, ..., :, -1] = 255
        new_image[tuple([slice(None, None)] * (len(image.shape)-1) + [-1])] = 255

    distance_int = [int(x) for x in distance]

    source_range = [slice(max(0, -d), min(s, s-d)) for d, s in zip(distance_int, image.shape)]
    target_range = [slice(max(0, d), min(s, s+d)) for d, s in zip(distance_int, image.shape)]

    new_image[tuple(target_range)] = image[tuple(source_range)]

    for i, d in enumerate(distance):
        if not eq(d, int(d)):
            _offset_subpixel(new_image, d - int(d), i, fill_value=fill_value,
                             fill_transparent=fill_transparent, inplace=True)

    return new_image


def _offset_subpixel(image: np.ndarray,
                     distance: float,
                     axis: int,
                     fill_value=0,
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
    offset will be filled in with 'fill_value'.
    """
    assert -1 < distance and distance < 1

    one_pix_offset = [0] * len(image.shape)
    one_pix_offset[axis] = 1 if distance >= 0 else -1
    image_1pix_shifted = offset(image, one_pix_offset, fill_value=fill_value,
                                fill_transparent=fill_transparent)
    distance = abs(distance)

    image_subpix_shifted = image * (1 - distance) + image_1pix_shifted * distance
    if np.issubdtype(image.dtype, np.integer):
        image_subpix_shifted = iround(image_subpix_shifted, output_dtype=image.dtype)

    if inplace:
        image[:] = image_subpix_shifted
    else:
        return image_subpix_shifted


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
