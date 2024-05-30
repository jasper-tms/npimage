#!/usr/bin/env python3
"""
Functions for aligning images.
"""

import numpy as np
import cv2 as cv


def find_landmark(image: np.ndarray,
                  landmark: np.ndarray,
                  search_bbox=None,
                  metric=cv.TM_CCOEFF_NORMED):
    """
    Find the region in an image that most resembles a particular landmark.

    Implementation leverages OpenCV functions, following tutorial at
    https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html

    Parameters
    ----------
    image : np.ndarray
        The image in which to search for the landmark.

    landmark : np.ndarray
        The landmark to search for in the image.

    search_bbox : list/tuple, optional
        The bounding box in which to search for the landmark. If you know
        the landmark you're searching for is in a subset of the image then
        specifying a search region will save time.
        If None, the entire image will be searched. If not None, search_bbox
        should be a list/tuple with two elements. Each element should be either
        a slice object or a list/tuple of two integers that specify the min and
        max values of the range.
        For example, search_bbox=(slice(0, 100), slice(0, 200)) or
        search_box=([0, 100], [0, 200]) would search the top-left
        100x200 region of the image.

    metric : cv.TemplateMatchModes, optional
        The metric to use to compare the landmark to the image. Options are
        cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED, cv.TM_CCORR, cv.TM_CCORR_NORMED,
        cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED. Default is cv.TM_CCOEFF_NORMED.

    Returns
    -------
    (top_left, match_score) : tuple
        top_left : tuple
            The top-left corner of the bounding box in the image that
            is most similar to the landmark.
        score : float
            The score of the match, from 0 to 1. A score of 1 indicates
            a perfect match.
    """
    if search_bbox is not None:
        if all(isinstance(el, slice) for el in search_bbox):
            img_to_search = image[search_bbox[0], search_bbox[1]]
            offset = (search_bbox[0].start, search_bbox[1].start)
        elif all(isinstance(el, (list, tuple)) and len(el) == 2
                 for el in search_bbox):
            img_to_search = image[search_bbox[0][0]:search_bbox[0][1],
                                  search_bbox[1][0]:search_bbox[1][1]]
            offset = (search_bbox[0][0], search_bbox[1][0])
        else:
            raise ValueError(f"Invalid search_bbox: {search_bbox}. Must "
                             "be two slices or two (min, max) pairs.")
        print('img_to_search.shape:', img_to_search.shape)
        print('offset:', offset)
    else:
        img_to_search = image
        offset = (0, 0)

    # Perform template matching
    scores = cv.matchTemplate(img_to_search, landmark, metric)

    # TODO implement subpixel max_loc finding.
    # Perhaps fitting a gaussian peak?
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(scores)
    if metric in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc[::-1]  # Includes a flip from (x, y) to (y, x)
        match_score = 1 - min_val
    else:
        top_left = max_loc[::-1]  # Includes a flip from (x, y) to (y, x)
        match_score = max_val
    top_left = (top_left[0] + offset[0], top_left[1] + offset[1])
    return top_left, match_score
