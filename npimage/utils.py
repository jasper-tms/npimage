#!/usr/bin/env python3
"""
Mostly tiny wrappers around existing numpy functions.

Function list:
- iround (round and cast to int)
- ifloor (round down and cast to int)
- iceil (round up and cast to int)
- eq (test if two objects are equal within a small tolerance)
- isint (test if an object or each element in an iterable object is an int/np.int)
- transpose_metadata (reverse the order of all per-axis metadata values)
"""

from typing import Union, Optional
from collections import OrderedDict
from fractions import Fraction

import numpy as np


def iround(n, output_dtype=int):
    # This rounds to the nearest integer, rounding values ending in .5 up
    # (toward positive infinity)
    return np.floor(n + 0.5).astype(output_dtype)

    # "For values exactly halfway between rounded decimal values, NumPy.round
    # rounds to the nearest even value. Thus 1.5 and 2.5 round to 2.0, -0.5 and
    # 0.5 round to 0.0, etc."
    # I think this convention is potentially problematic for graphics
    # applications, that is, it could break watertightness of shapes that are
    # supposed to be watertight, so I'm not using it here.
    #return np.round(n).astype(int)


def ifloor(n, output_dtype=int):
    return np.floor(n).astype(output_dtype)


def iceil(n, output_dtype=int):
    return np.ceil(n).astype(output_dtype)


def eq(a, b, tolerance=1e-8):
    """
    Test if two objects have an absolute difference less than a small value.
    Useful for testing for "equivalence" of floats which are not be stored to
    infinite precision.
    """
    return abs(a - b) < tolerance


def isint(n):
    """
    Check whether the given variable is an integer (either a built-in
    int or a numpy integer type). If the argument is iterable, return
    a list of booleans indicating whether each element is an integer
    (again either a built-in int or a numpy integer type).

    Notes
    -----
    You might think that isinstance(n, int) is what you want, but that
      has the undesirable behavior of returning True for bools, and
      returning False for numpy integer types.
    To check whether n is a built-in int or np.integer _but not an iterable
      of them_, use `if npimage.isint(n) is True`, or just
      call `if np.issubdtype(type(n), np.integer)` yourself.
    This function does NOT recognize integer types from other packages
      including TensorFlow, PyTorch, or JAX.
    """
    try:
        return [np.issubdtype(type(a), np.integer) for a in n]
    except TypeError:
        return np.issubdtype(type(n), np.integer)


def find_channel_axis(data,
                      possible_channel_axes=[-1, 0],
                      possible_channel_lengths=[2, 3, 4],
                      minimum_image_size=(32, 32)) -> Union[int, None]:
    """
    If the given numpy array has a shape suggesting that it has a
    channel (color) axis (that is, an axis with length 2 (2-color),
    3 (RGB), or 4 (RGBA)), return the index of that axis.

    With the default parameters, the smallest array shapes that will be
    recognized as having a channel axis are (2, 32, 32) and (32, 32, 2).
    Note that shapes (32, 2) and (32, 16, 2) will NOT be recognized due
    to the minimum_image_size parameter. If you have arrays that look
    like this and you want to recognize them as having a channel axis,
    you can adjust the minimum_image_size parameter accordingly.

    Parameters
    ----------
    data : numpy.ndarray
        The numpy array to check for a channel axis.

    possible_channel_axes : int or list of int, default [-1, 0]
        If None, any axis having length in possible_channel_lengths will
        be considered a channel axis.
        If an int, only that axis index will be checked.
        If a list of ints, all axes with those indices will be checked,
        and the first one that satisfies the other criteria will be returned.
        The default value of [-1, 0] checks the last and first axes (in that
        order), which is almost always where a channel axis will be found.

    possible_channel_lengths : int or list of int, default [2, 3, 4]
        If an int, only that length will be considered a channel axis.
        If a list of ints, an axis with any of those lengths will be considered
        a channel axis.

    minimum_image_size : tuple of int, default (32, 32)
        In addition to having an axis with a length in possible_channel_lengths,
        the data must also have other axes with at least these lengths in order
        to be considered to have a channel axis. This prevents small arrays
        with shapes like (3, 3, 3) or (128, 3) from being misinterpreted
        as having a channel axis when they are probably not intended as such.

    Returns
    -------
    int or None
        The index of the channel axis, or None if no channel axis was found.

        If possible_channel_axes is given, the returned value will be one of
        the possible_channel_axes values, or None if no channel axis was found.

        If possible_channel_axes is None, the returned value will be between
        0 and data.ndim - 1, inclusive, or None if no channel axis was found.

        Note that returning 0 means the channel axis was found and is the first
        axis, so be careful not to do a test like `if find_channel_axis(data):`
        because 0 will evaluate to False even though the data has a channel axis.
        Instead write `if find_channel_axis(data) is not None:`

    Examples
    --------
    >>> npimage.find_channel_axis(np.empty((3, 1024, 1024)))
    0
    >>> npimage.find_channel_axis(np.empty((1024, 1024, 3)))
    2
    >>> npimage.find_channel_axis(np.empty((1024, 3, 1024)))
    None  # Due to possible_channel_axes=[-1, 0] not being met
    >>> npimage.find_channel_axis(np.empty((256, 128, 128)))
    None  # Due to possible_channel_lengths=[2, 3, 4] not being met
    >>> npimage.find_channel_axis(np.empty((1024, 4)))
    None  # Due to minimum_image_size=(32, 32) not being met
    >>> npimage.find_channel_axis(np.empty((30, 30, 4)))
    None  # Due to minimum_image_size=(32, 32) not being met
    >>> npimage.find_channel_axis(np.empty((30, 4)), minimum_image_size=0)
    1
    """
    if np.issubdtype(type(possible_channel_axes), np.integer):
        possible_channel_axes = [possible_channel_axes]
    if possible_channel_axes is None:
        possible_channel_axes = range(data.ndim)

    if np.issubdtype(type(possible_channel_lengths), np.integer):
        possible_channel_lengths = [possible_channel_lengths]

    if np.issubdtype(type(minimum_image_size), np.integer):
        minimum_image_size = (minimum_image_size,)

    for axis in possible_channel_axes:
        if data.shape[axis] not in possible_channel_lengths:
            continue
        other_axis_lengths = [data.shape[i] for i in range(data.ndim) if i != axis]
        if len(other_axis_lengths) < len(minimum_image_size):
            continue
        other_axis_lengths = sorted(other_axis_lengths, reverse=True)
        other_axis_lengths = other_axis_lengths[:len(minimum_image_size)]
        if any(i < j for i, j in zip(other_axis_lengths, minimum_image_size)):
            continue
        return axis % data.ndim
    return None


def transpose_metadata(metadata: Union[dict, OrderedDict],
                       inplace: bool = True) -> Optional[Union[dict, OrderedDict]]:
    """
    Reverse the order of all metadata values that have a separate entry
    for each axis of the data.

    For example, flip voxel size metadata of (1.5, 1.5, 2) to (2, 1.5, 1.5).
    Useful when transposing an array from zyx to xyz order and wanting
    to update the metadata order to reflect the change, for example.
    Designed to work with OrderedDicts that contain nrrd files'
    metadata, but may be useful for metadata from other formats.
    """
    if metadata is None:
        return None
    if not inplace:
        metadata = metadata.copy()
    for key, value in metadata.items():
        if isinstance(value, str) or not hasattr(value, '__iter__'):
            continue
        if key == 'scales':
            value = [transpose_metadata(scale, inplace=False)
                     for scale in value]
        if isinstance(value, np.ndarray):
            value = np.flip(value)
        else:
            value = value[::-1]
        metadata[key] = value
    if not inplace:
        return metadata


def is_out_of_bounds(coords, shape, allow_negative_wrapping=False, convention='corner'):
    """
    Check if coordinates are out of the bounds of a given shape.

    Parameters
    ----------
    coords: Coordinates to check (can be a list or numpy array).

    shape: Shape of the volume (tuple or list).

    allow_negative_wrapping
      If False, all negative coordinates are considered out of bounds
      If True, negative coordinates with absolute value less than the shape
      are considered in bounds (in which case they can be used as indices
      to a list or np.ndarray and they will wrap around).

    convention: 'corner' or 'center' to specify whether the coordinate
      0 refers to the top-left corner of the first pixel (in which case
      -0.1 is out of bounds) or the center of the first pixel (in which
      case -0.1 is in bounds, down to -0.5 being the last in-bounds value).

    Returns
    -------
    Boolean array indicating whether each coordinate is within bounds.
    """
    if convention not in ['corner', 'center']:
        raise ValueError("Convention must be 'corner' or 'center'.")
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    if coords.ndim == 1 and coords.shape[0] == len(shape):
        return is_out_of_bounds(coords[np.newaxis, :], shape,
                                allow_negative_wrapping, convention)[0]
    if coords.ndim != 2 or coords.shape[1] != len(shape):
        raise ValueError(f'Coordinates must be a Nx{len(shape)} array, '
                         f'but got shape {coords.shape}.')

    upper_limit = shape
    if allow_negative_wrapping:
        lower_limit = tuple(-i for i in shape)
    else:
        lower_limit = 0

    if convention == 'center':
        lower_limit = np.array(lower_limit) - 0.5
        upper_limit = np.array(upper_limit) - 0.5

    underflows = coords < lower_limit
    overflows = coords >= upper_limit

    return np.logical_or(underflows, overflows).any(axis=-1)


def is_in_bounds(*args, **kwargs):
    return np.logical_not(is_out_of_bounds(*args, **kwargs))


def remove_out_of_bounds(coords, shape, allow_negative_wrapping=False,
                         convention='corner'):
    in_bounds = is_in_bounds(coords, shape, allow_negative_wrapping, convention)
    return coords[in_bounds]


def limit_fraction(fraction: Union[Fraction, float], to: int = 2**31) -> Fraction:
    """
    Limit a Fraction to ensure both numerator and denominator are smaller than a given value.

    This is useful for ensuring compatibility with FFmpeg, which expresses framerates
    as ratios of two 32-bit signed integers.
    """
    fraction = Fraction(fraction)
    new_fraction = fraction
    limit = to
    while abs(new_fraction.numerator) >= to or abs(new_fraction.denominator) >= to:
        new_fraction = fraction.limit_denominator(limit)
        limit = limit // 2
    return new_fraction
