#!/usr/bin/env python3
"""
Functions for aligning images.
"""
from typing import Optional, Tuple, Literal

import numpy as np
import cv2 as cv


def find_landmark(image: np.ndarray,
                  landmark: np.ndarray,
                  search_bbox: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
                  metric: Literal[cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED,
                                  cv.TM_CCORR, cv.TM_CCORR_NORMED,
                                  cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED] = cv.TM_CCOEFF_NORMED,
                  subpixel_accuracy: bool = True
                  ) -> Tuple[Tuple[int, int], float]:
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

    search_bbox : ([axis1min, axis1max], [axis2min, axis2max])
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
            search_offset = (search_bbox[0].start, search_bbox[1].start)
        elif all(isinstance(el, (list, tuple)) and len(el) == 2
                 for el in search_bbox):
            img_to_search = image[search_bbox[0][0]:search_bbox[0][1],
                                  search_bbox[1][0]:search_bbox[1][1]]
            search_offset = (search_bbox[0][0], search_bbox[1][0])
        else:
            raise ValueError(f"Invalid search_bbox: {search_bbox}. Must "
                             "be two slices or two (min, max) pairs.")
    else:
        img_to_search = image
        search_offset = (0, 0)

    # Perform template matching
    scores = cv.matchTemplate(img_to_search, landmark, metric)

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(scores)
    if metric in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc[::-1]  # Includes a flip from (x, y) to (y, x)
        match_score = 1 - min_val
    else:
        top_left = max_loc[::-1]  # Includes a flip from (x, y) to (y, x)
        match_score = max_val
    if not subpixel_accuracy:
        return top_left, match_score

    # Subpixel accuracy: Fit a quadratic to a patch around the best score
    patch_size = 5
    patch_top_left = (max(0, top_left[0] - patch_size//2),
                      max(0, top_left[1] - patch_size//2))
    patch_bottom_right = (min(scores.shape[0], top_left[0] + patch_size//2),
                          min(scores.shape[1], top_left[1] + patch_size//2))
    patch = scores[patch_top_left[0]:patch_bottom_right[0],
                   patch_top_left[1]:patch_bottom_right[1]]

    def eval_quadratic(coeffs, x, y):
        """
        Evaluate a quadratic surface at a given point.

        The quadratic surface is defined as:
            quadratic(x, y) = a*x**2 + b*y**2 + c*x*y + d*x + e*y + f
        """
        a, b, c, d, e, f = coeffs
        return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

    def quadratic_interpolate_peak(patch):
        """
        Fit a 2D quadratic surface to a matrix of values, and find
        the peak position and value of the surface.

        Returns
        -------
        tuple containing
        - The coordinate of the peak value of the quadratic fit to the patch
        - The peak value
        """
        A = [[i**2, j**2, i*j, i, j, 1] for i, j in np.ndindex(patch.shape)]
        z = [patch[i, j] for i, j in np.ndindex(patch.shape)]
        A = np.array(A)
        z = np.array(z)

        coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
        a, b, c, d, e, f = coeffs

        # Find peak by solving gradient = 0:
        # df/di = 2a i + c j + d = 0
        # df/dj = 2b j + c i + e = 0
        A_grad = np.array([[2*a, c],
                           [c, 2*b]])
        b_grad = -np.array([d, e])
        try:
            peak_loc = np.linalg.solve(A_grad, b_grad)
        except np.linalg.LinAlgError:
            peak_loc = np.array([np.nan, np.nan])  # singular matrix, fallback

        peak_value = eval_quadratic(coeffs, peak_loc[0], peak_loc[1])
        return peak_loc, peak_value

    patch_peak, peak_value = quadratic_interpolate_peak(patch)
    image_peak = patch_peak + patch_top_left + search_offset
    image_peak = (round(image_peak[0], ndigits=2),
                  round(image_peak[1], ndigits=2))

    return image_peak, peak_value
