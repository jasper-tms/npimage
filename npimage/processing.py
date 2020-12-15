#!/usr/bin/env python3

import numpy as np

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

    """
    #TODO implement
    raise NotImplementedError
