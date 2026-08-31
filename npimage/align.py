#!/usr/bin/env python3
"""
Functions for aligning images.
"""
from typing import Optional, Tuple, Union

import numpy as np

from .utils import iround


def find_landmark(image: np.ndarray,
                  landmark: np.ndarray,
                  search_bbox: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
                  metric: Optional[int] = None,
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
        cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED.
        If left as None, defaults to cv.TM_CCOEFF_NORMED.

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
    try:
        import cv2 as cv
    except ImportError as e:
        raise ImportError(
            'find_landmark requires opencv. Install it with '
            '`pip install numpyimage[align]` or `pip install opencv-python-headless`.'
        ) from e

    if metric is None:
        metric = cv.TM_CCOEFF_NORMED

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
        top_left = (top_left[0] + search_offset[0],
                    top_left[1] + search_offset[1])
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


def series_to_target(array,
                     series_axis: int = 0,
                     target_index: Union[int, range, slice] = 0,
                     *,
                     roi: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
                     max_step: Optional[Union[int, Tuple[int, int]]] = None,
                     channel_axis: Optional[int] = None,
                     channel_reduction: str = 'mean',
                     target_reduction: str = 'median',
                     metric: Optional[int] = None,
                     padding_value=0,
                     expand_to_fit: bool = False,
                     return_offsets: bool = False,
                     progress: bool = False):
    """
    Translationally align every image in a series to a common target.

    Each image is matched to a fixed target by whole-pixel template matching
    (`find_landmark`) and shifted so its content lines up with the target's.
    This removes whole-frame motion such as camera jitter or a slow pan from a
    video, or stage drift from a microscopy z- or t-stack. Alignment is
    whole-pixel only (no subpixel interpolation).

    Parameters
    ----------
    array : np.ndarray, str, pathlib.Path, or npimage.VideoStreamer
        The image series to align.
        - As an ndarray, one axis (`series_axis`) indexes the images and two
          axes are the spatial (row, column) axes that get aligned; an optional
          `channel_axis` holds color channels. So a color video is typically
          (frames, height, width, 3) with series_axis=0, channel_axis=-1, and a
          grayscale stack is (frames, height, width). After the series and
          channel axes are set aside exactly two spatial axes must remain. The
          whole aligned array is returned.
        - As a video filename or a VideoStreamer, the frames are read and
          aligned one at a time so only a single frame is ever held in memory,
          and a generator that yields the aligned frames in order is returned
          instead (see Returns). `series_axis` and `channel_axis` do not apply;
          frames are (height, width, channels).

    series_axis : int, default 0
        The axis of `array` that indexes the images in the series.

    target_index : int, range, slice, or sequence of int, default 0
        Which image(s) form the alignment target. An int names a single image
        to align to. A range/slice/sequence names several images that are
        combined (see `target_reduction`) into one target, which helps when a
        single frame has moving foreground: combining a stretch over which the
        camera is still leaves the static background and averages the moving
        objects away.

    roi : ((row_min, row_max), (col_min, col_max)) or None, default None
        The region of the target used as the match template.
        - A bounding box crops a fixed template from the target. Use this to
          lock onto a specific static landmark (e.g. the net of a tennis court);
          being smaller than the frame, it imposes no cap on total drift.
        - None uses an ADAPTIVE template: each image is matched against the
          largest region of the target that still lies fully inside the image at
          the previous image's offset, so the whole static background dominates
          the match with no cap on drift. Requires `max_step`.

    max_step : int, (int, int), or None, default None
        The most the offset may change between consecutive images, in pixels, as
        a single value or a (row, col) pair. Each image is searched only within
        this radius of the previous image's offset, which stops a moving object
        from yanking the match far away in a single step. None searches each
        image independently over the whole frame (only allowed with a fixed
        `roi`).

    channel_axis : int or None, default None
        The axis holding color channels of an ndarray input, if any. Channels
        are collapsed to one (see `channel_reduction`) for the matching only;
        the returned images keep all of their channels.

    channel_reduction : 'mean' or 'luminance', default 'mean'
        How to collapse channels to the single channel the matcher needs.

    target_reduction : 'median' or 'mean', default 'median'
        How to combine multiple target images into one when `target_index` names
        more than one. 'median' rejects moving foreground best. Uses
        `fastpercentile` for the median if installed, else `numpy.median`.

    metric : int or None, default None
        OpenCV template-match metric, passed through to `find_landmark`.

    padding_value : scalar, default 0
        Value used to fill any area of the output not covered by a shifted
        image: the exposed border of each frame when `expand_to_fit` is False,
        and the corners of the enlarged canvas that no frame reaches when it is
        True.

    expand_to_fit : bool, default False
        Where the aligned images are placed.
        - False keeps every output image the same size as its input, so a shift
          slides content off one edge (lost) and exposes the opposite edge
          (filled with `padding_value`). This is a single pass over the images.
        - True places every image on a common canvas made large enough that no
          pixel of any image is ever pushed off it -- the canvas grows by the
          full spread of the offsets (its height by `max_row_offset -
          min_row_offset`, its width likewise). Every output image has this one
          enlarged size. Because the canvas cannot be sized until every offset
          is known, this makes two passes over the images (one frame is still
          held at a time, so a streaming source is simply read twice).

    return_offsets : bool, default False
        If True, also return the per-image offsets.

    progress : bool, default False
        If True, show a tqdm progress bar.

    Returns
    -------
    aligned : np.ndarray or generator
        For an ndarray input: `array` with every image shifted into alignment
        with the target, dtype and axis order preserved. The shape matches the
        input when `expand_to_fit` is False, or has enlarged spatial axes (equal
        for every image) when it is True.
        For a video filename or VideoStreamer input: a generator that yields the
        aligned frames one at a time, in order (each an (H, W, C) array, all the
        same enlarged size when `expand_to_fit` is True), so they can be fed
        straight to a writer without holding them all in memory (e.g.
        `for f in series_to_target(path, ...): writer.write(f)`).
    offsets : np.ndarray, only if `return_offsets` is True (ndarray input)
        An (n_images, 2) int array of each image's found (row, col) offset
        relative to the target, in series order. `aligned` image i is `array`
        image i with the scene shifted by `-offsets[i]`. For a streaming input,
        `return_offsets` instead makes the generator yield (aligned_frame,
        (row, col) offset) pairs.
    """
    from pathlib import Path

    from . import operations
    from .vidio import VideoStreamer, lazy_load_video

    if max_step is None:
        step = None
    elif isinstance(max_step, (int, np.integer)):
        step = (int(max_step), int(max_step))
    else:
        step = (int(max_step[0]), int(max_step[1]))
    if roi is None and step is None:
        raise ValueError('max_step is required when roi is None, since the '
                         'adaptive template tracks the running offset.')

    streaming = isinstance(array, (str, Path, VideoStreamer))
    # Video frames are always (height, width, channels); an ndarray declares its
    # own layout via series_axis / channel_axis.
    has_channels = True if streaming else channel_axis is not None

    def to_match(image):
        """The single-channel, OpenCV-friendly (uint8 or float32) version of an
        image that find_landmark matches on."""
        if has_channels:
            if channel_reduction == 'mean':
                gray = image.mean(axis=-1)
            elif channel_reduction == 'luminance':
                gray = image[..., :3] @ np.array([0.299, 0.587, 0.114])
            else:
                raise ValueError(f"Unknown channel_reduction '{channel_reduction}'."
                                 " Use 'mean' or 'luminance'.")
            if image.dtype == np.uint8:
                gray = iround(gray, np.uint8)
        else:
            gray = image
        if gray.dtype == np.uint8:
            return np.ascontiguousarray(gray)
        return np.ascontiguousarray(gray.astype(np.float32))

    def build_target(get_frame, n):
        """Read the target via a frame-getter: one frame for an int
        target_index, else the median/mean of the frames it names."""
        if isinstance(target_index, (int, np.integer)):
            return get_frame(int(target_index))
        if isinstance(target_index, slice):
            indices = list(range(*target_index.indices(n)))
        else:
            indices = list(target_index)
        stack = np.stack([get_frame(i) for i in indices])
        if target_reduction == 'mean':
            combined = stack.mean(axis=0)
        elif target_reduction == 'median':
            try:
                import fastpercentile
                combined = fastpercentile.median(stack, axis=0)
            except ImportError:
                combined = np.median(stack, axis=0)
        else:
            raise ValueError(f"Unknown target_reduction '{target_reduction}'. "
                             "Use 'median' or 'mean'.")
        # A median promotes uint8 to float; bring it back so the target's match
        # image shares the frames' dtype (OpenCV requires that).
        if np.issubdtype(stack.dtype, np.integer):
            return iround(combined, stack.dtype)
        return combined.astype(stack.dtype)

    # Assemble the target image and a re-callable `frame_iter` that returns a
    # fresh generator of the frames in order (called once normally, twice when
    # expand_to_fit sizes the canvas from every offset before placing a frame).
    if streaming:
        if isinstance(array, VideoStreamer):
            n_images = array.n_frames
            target_image = build_target(lambda i: array[i], n_images)
            frame_iter = lambda: (array[i] for i in range(n_images))
        else:
            with VideoStreamer(array) as streamer:
                n_images = streamer.n_frames
                target_image = build_target(lambda i: streamer[i], n_images)
            frame_iter = lambda: lazy_load_video(array)
    else:
        array = np.asarray(array)
        ndim = array.ndim
        series_axis = series_axis % ndim
        if has_channels:
            if ndim != 4:
                raise ValueError('With channel_axis set, array must be 4-D '
                                 '(series, height, width, channels) in some '
                                 f'order; got {ndim}-D.')
            work = np.moveaxis(array, [series_axis, channel_axis % ndim], [0, -1])
        else:
            if ndim != 3:
                raise ValueError('Without channel_axis, array must be 3-D '
                                 '(series, height, width) in some order; '
                                 f'got {ndim}-D.')
            work = np.moveaxis(array, series_axis, 0)
        n_images = work.shape[0]
        target_image = build_target(lambda i: work[i], n_images)
        frame_iter = lambda: (work[i] for i in range(n_images))

    target_match = to_match(target_image)
    target_height, target_width = target_match.shape
    if roi is None:
        template = None
        reference = None
    else:
        (row_min, row_max), (col_min, col_max) = roi
        template = np.ascontiguousarray(target_match[row_min:row_max,
                                                     col_min:col_max])
        reference = (row_min, col_min)

    def find_offset(frame_match, previous):
        """The (row, col) offset of one frame relative to the target, searched
        within `step` of the previous frame's offset."""
        if template is None:
            # Adaptive: match against the largest target region still fully
            # in-frame at `previous` (widened by `step`); if the overlap has
            # collapsed, keep the previous offset.
            previous_row, previous_col = previous
            r0 = max(0, step[0] - previous_row)
            r1 = min(target_height, target_height - step[0] - previous_row)
            c0 = max(0, step[1] - previous_col)
            c1 = min(target_width, target_width - step[1] - previous_col)
            if r1 - r0 <= 0 or c1 - c0 <= 0:
                return previous
            crop = np.ascontiguousarray(target_match[r0:r1, c0:c1])
            search_bbox = [[r0 + previous_row - step[0], r1 + previous_row + step[0]],
                           [c0 + previous_col - step[1], c1 + previous_col + step[1]]]
            top_left, _ = find_landmark(frame_match, crop, search_bbox=search_bbox,
                                        metric=metric, subpixel_accuracy=False)
            return (top_left[0] - r0, top_left[1] - c0)
        # Fixed template: search a window around `previous`, or the whole frame.
        reference_row, reference_col = reference
        if step is None:
            search_bbox = None
        else:
            expected_row = reference_row + previous[0]
            expected_col = reference_col + previous[1]
            search_bbox = [
                [max(0, expected_row - step[0]),
                 min(frame_match.shape[0], expected_row + step[0] + template.shape[0])],
                [max(0, expected_col - step[1]),
                 min(frame_match.shape[1], expected_col + step[1] + template.shape[1])]]
        top_left, _ = find_landmark(frame_match, template, search_bbox=search_bbox,
                                    metric=metric, subpixel_accuracy=False)
        return (top_left[0] - reference_row, top_left[1] - reference_col)

    def with_progress(source, desc):
        if not progress:
            return source
        from tqdm import tqdm
        return tqdm(source, total=n_images, desc=desc)

    def all_offsets():
        """Every frame's (row, col) offset, in order (each search seeded at the
        previous frame's answer)."""
        previous = (0, 0)
        offsets = []
        for frame in with_progress(frame_iter(),
                                   'Aligning series (finding offsets)'
                                   if expand_to_fit else 'Aligning series'):
            row, col = find_offset(to_match(frame), previous)
            previous = (int(row), int(col))
            offsets.append(previous)
        return offsets

    def aligned_pairs():
        """Yield (aligned_frame, (row, col) offset) for each frame, in order."""
        if not expand_to_fit:
            # Single pass: find each offset and immediately shift-and-crop the
            # frame back to its own size, filling the exposed border.
            previous = (0, 0)
            for frame in with_progress(frame_iter(), 'Aligning series'):
                row, col = find_offset(to_match(frame), previous)
                previous = (int(row), int(col))
                yield operations.offset(frame, (-row, -col),
                                        fill_empty_with=padding_value), previous
            return
        # expand_to_fit: size a common canvas from the full offset spread (first
        # pass), then place each frame onto it so no pixel is lost (second pass).
        offsets = all_offsets()
        if not offsets:
            return
        row_high = max(row for row, _ in offsets)
        row_low = min(row for row, _ in offsets)
        col_high = max(col for _, col in offsets)
        col_low = min(col for _, col in offsets)
        pairs = zip(with_progress(frame_iter(), 'Aligning series (placing frames)'),
                    offsets)
        for frame, (row, col) in pairs:
            height, width = frame.shape[:2]
            canvas = np.full((height + row_high - row_low,
                              width + col_high - col_low) + frame.shape[2:],
                             padding_value, dtype=frame.dtype)
            top, left = row_high - row, col_high - col
            canvas[top:top + height, left:left + width] = frame
            yield canvas, (row, col)

    if streaming:
        if return_offsets:
            return aligned_pairs()
        return (frame for frame, _offset in aligned_pairs())

    # In-memory input: gather the aligned frames back into an array laid out
    # like the input. Without expansion every frame keeps its size, so prefill
    # one array; with expansion the frames share a larger size found mid-stream,
    # so collect and stack them.
    if expand_to_fit:
        gathered = list(aligned_pairs())
        aligned = (np.stack([frame for frame, _ in gathered], axis=0)
                   if gathered else work[:0].copy())
        offsets = np.array([offset for _, offset in gathered],
                           dtype=int).reshape(-1, 2)
    else:
        aligned = np.empty_like(work)
        offsets = np.zeros((n_images, 2), dtype=int)
        for i, (frame, offset) in enumerate(aligned_pairs()):
            aligned[i] = frame
            offsets[i] = offset
    if has_channels:
        aligned = np.moveaxis(aligned, [0, -1], [series_axis, channel_axis % ndim])
    else:
        aligned = np.moveaxis(aligned, 0, series_axis)

    if return_offsets:
        return aligned, offsets
    return aligned
