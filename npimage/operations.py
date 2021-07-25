#!/usr/bin/env python3

from collections.abc import Iterable

import numpy as np

from .utils import iround, eq, isint


def offset(image: np.ndarray,
           distance: Iterable,
           fill_value=0,
           fill_transparent=False) -> np.ndarray:
    """
    Offset an image by a given distance.
    'distance' must be an iterable with length matching the number of axes in
    the image, to specify an number of pixels to offset along each axis.
    The pixels no longer occupied by the original image as a result of the
    offset will be filled in with 'fill_value'.

    See also scipy.ndimage.shift, which performs a very similar operation
    """
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
        image_subpix_shifted = iround(image_subpix_shifted)

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
