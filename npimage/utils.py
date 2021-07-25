#!/usr/bin/env python3

import numpy as np


def iround(n):
    # This rounds to the nearest integer, rounding values ending in .5 up
    # (toward positive infinity)
    return np.floor(n + 0.5).astype(int)

    # "For values exactly halfway between rounded decimal values, NumPy.round
    # rounds to the nearest even value. Thus 1.5 and 2.5 round to 2.0, -0.5 and
    # 0.5 round to 0.0, etc."
    # I think this convention is potentially problematic for graphics
    # applications, that is, it could break watertightness of shapes that are
    # supposed to be watertight, so I'm not using it here.
    #return np.round(n).astype(int)


def ifloor(n):
    return np.floor(n).astype(int)


def iceil(n):
    return np.ceil(n).astype(int)


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


