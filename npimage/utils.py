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


#TODO duck type this to accept a more general class of objects that extend ints
def isint(n):
    try:
        return [isinstance(a, int) or np.issubdtype(type(a), np.integer)
                for a in n]
    except TypeError:
        return isinstance(n, int) or np.issubdtype(type(n), np.integer)


def find_channel_axis(data,
                      possible_channel_axes=[-1, 0],
                      possible_channel_lengths=[2, 3, 4]) -> Union[int, None]:
    """
    If the given numpy array has a shape suggesting that it has a
    channel (color) axis (that is, any axis with length 2 (2-color),
    3 (RGB), or 4 (RGBA)), return the index of that axis.

    Parameters
    ----------
    data : numpy.ndarray
        The numpy array to check for a channel axis.

    possible_channel_axes : int or list of int, default [0, -1]
        If None, any axis having length 2, 3, or 4 will be considered
        a channel axis.
        If an int, only that axis index will be checked.
        If a list of ints, all axes with those indices will be checked.
        The default value of [-1, 0] checks the last and first axes, which is
        almost always where a channel axis will be found.

    possible_channel_lengths : int or list of int, default [2, 3, 4]
        If an int, only that length will be considered a channel axis.
        If a list of ints, an axis with any of those lengths will be considered
        a channel axis.

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
    """
    if isinstance(possible_channel_axes, int):
        possible_channel_axes = [possible_channel_axes]
    if possible_channel_axes is None:
        possible_channel_axes = range(data.ndim)
    if isinstance(possible_channel_lengths, int):
        possible_channel_lengths = [possible_channel_lengths]
    for axis in possible_channel_axes:
        if data.shape[axis] in possible_channel_lengths:
            return axis
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
