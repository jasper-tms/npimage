#!/usr/bin/env python3
"""
Tests for npimage.align.series_to_target.

The test scenes are built by cropping a fixed, fully-textured "world" image at
shifting top-left corners, so each frame is the same scene translated by a known
amount and the true offset of every frame is known exactly. A world feature at
target-crop position (r, c) appears in a crop taken `delta` away from the
target's crop at (r - delta_row, c - delta_col), so that crop's offset relative
to the target is `-delta`.
"""
import numpy as np
import pytest

# series_to_target matches via OpenCV; skip the whole module if it is absent.
pytest.importorskip('cv2')

import npimage
from npimage.align import series_to_target

CROP_HEIGHT, CROP_WIDTH = 100, 120
BASE_ORIGIN = (8, 8)  # top-left of the target crop within the world


def _world(seed=0):
    """A fully-textured world large enough to crop BASE_ORIGIN +/- 8 px."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256,
                        size=(CROP_HEIGHT + 2 * BASE_ORIGIN[0],
                              CROP_WIDTH + 2 * BASE_ORIGIN[1], 3),
                        dtype=np.uint8)


def _series(world, deltas):
    """
    Build a (n, CROP_HEIGHT, CROP_WIDTH, 3) series of crops of `world`, one per
    (delta_row, delta_col) in `deltas`, plus the array of true offsets (-delta).
    `deltas[0]` should be (0, 0) so frame 0 is the target crop.
    """
    base_row, base_col = BASE_ORIGIN
    frames, truth = [], []
    for delta_row, delta_col in deltas:
        row = base_row + delta_row
        col = base_col + delta_col
        frames.append(world[row:row + CROP_HEIGHT, col:col + CROP_WIDTH])
        truth.append((-delta_row, -delta_col))
    return np.stack(frames), np.array(truth)


def _assert_interior_matches(aligned, target, margin=8):
    """The interior (away from the shift-exposed border) of every aligned frame
    should exactly reproduce the target crop."""
    core = (slice(margin, -margin), slice(margin, -margin))
    for i in range(aligned.shape[0]):
        assert np.array_equal(aligned[i][core], target[core]), f'frame {i}'


def test_fixed_roi_recovers_known_offsets_global_search():
    """A fixed central-landmark ROI with a full (max_step=None) search recovers
    each frame's known offset and lines the scene back up."""
    world = _world()
    rng = np.random.default_rng(1)
    deltas = [(0, 0)] + [tuple(rng.integers(-8, 9, size=2)) for _ in range(15)]
    frames, truth = _series(world, deltas)

    aligned, offsets = series_to_target(
        frames, channel_axis=-1,
        roi=((30, 70), (40, 90)), max_step=None,
        return_offsets=True)

    assert np.array_equal(offsets, truth)
    assert aligned.shape == frames.shape
    assert aligned.dtype == frames.dtype
    _assert_interior_matches(aligned, frames[0])


def test_fixed_roi_chained_search_follows_a_ramp():
    """With max_step set, a fixed ROI tracks a smooth pan whose per-frame step
    stays within the cap."""
    world = _world()
    deltas = [(0, d) for d in [0, 2, 4, 6, 8, 6, 4, 2, 0, -2, -4, -6, -8]]
    frames, truth = _series(world, deltas)

    aligned, offsets = series_to_target(
        frames, channel_axis=-1,
        roi=((30, 70), (40, 90)), max_step=(2, 3),
        return_offsets=True)

    assert np.array_equal(offsets, truth)
    _assert_interior_matches(aligned, frames[0])


def test_adaptive_template_follows_a_ramp():
    """The adaptive (roi=None) whole-background template recovers a smooth pan
    with no fixed drift cap."""
    world = _world(seed=2)
    deltas = [(0, d) for d in [0, 3, 6, 8, 6, 3, 0, -3, -6, -8, -6, -3, 0]]
    frames, truth = _series(world, deltas)

    aligned, offsets = series_to_target(
        frames, channel_axis=-1, roi=None, max_step=(3, 4),
        return_offsets=True)

    assert np.array_equal(offsets, truth)
    _assert_interior_matches(aligned, frames[0])


def test_grayscale_series_without_channel_axis():
    """A 3-D grayscale series (no channel axis) aligns just as well."""
    world = _world()[..., 0]  # drop to a single channel
    base_row, base_col = BASE_ORIGIN
    deltas = [(0, 0), (1, -2), (-3, 4), (5, 5), (-6, 2)]
    frames = np.stack([world[base_row + dr:base_row + dr + CROP_HEIGHT,
                             base_col + dc:base_col + dc + CROP_WIDTH]
                       for dr, dc in deltas])
    truth = np.array([(-dr, -dc) for dr, dc in deltas])

    aligned, offsets = series_to_target(
        frames, roi=((30, 70), (40, 90)), max_step=None, return_offsets=True)

    assert np.array_equal(offsets, truth)
    _assert_interior_matches(aligned, frames[0])


def test_series_axis_other_than_zero_is_preserved():
    """The series axis may be anywhere; the output keeps the input's layout."""
    world = _world()
    deltas = [(0, 0), (2, -1), (-3, 3), (4, 4)]
    frames, truth = _series(world, deltas)  # (n, H, W, 3)
    moved = np.moveaxis(frames, 0, 2)       # (H, W, n, 3)

    aligned, offsets = series_to_target(
        moved, series_axis=2, channel_axis=-1,
        roi=((30, 70), (40, 90)), max_step=None, return_offsets=True)

    assert aligned.shape == moved.shape
    assert np.array_equal(offsets, truth)
    # Undo the move and check interiors against the target frame.
    _assert_interior_matches(np.moveaxis(aligned, 2, 0), frames[0])


def test_target_from_range_median_ignores_a_moving_object():
    """A static camera with a bright object moving across it: a single-frame
    target could be pulled off by the object, but a median over the range
    removes it, so every frame aligns at zero offset."""
    world = _world(seed=3)
    base_row, base_col = BASE_ORIGIN
    background = world[base_row:base_row + CROP_HEIGHT,
                       base_col:base_col + CROP_WIDTH]
    frames = np.stack([background.copy() for _ in range(12)])
    # A 15x15 white block that marches across the frames (different place each).
    for i in range(frames.shape[0]):
        top = 10 + i
        left = 5 * i
        frames[i, top:top + 15, left:left + 15] = 255

    aligned, offsets = series_to_target(
        frames, channel_axis=-1, target_index=range(frames.shape[0]),
        target_reduction='median', roi=None, max_step=(2, 2),
        return_offsets=True)

    assert np.array_equal(offsets, np.zeros_like(offsets))
    # Zero offset means no shift: frames come back untouched.
    assert np.array_equal(aligned, frames)


def test_constant_padding_fills_exposed_border_with_value():
    """A shift exposes a border; constant padding fills it with padding_value."""
    world = _world()
    frames, _ = _series(world, [(0, 0), (0, 5)])  # frame 1 shifts by col -5

    aligned = series_to_target(
        frames, channel_axis=-1, roi=((30, 70), (40, 90)), max_step=None,
        padding_value=0)

    # Frame 1 has offset (0, -5): aligned[y, x] = frame[y, x - 5], so the left
    # 5 columns are exposed and filled with 0.
    assert np.all(aligned[1][:, :5] == 0)
    assert not np.all(aligned[1][:, 5:] == 0)


def test_expand_to_fit_keeps_every_pixel_on_a_common_canvas():
    """With expand_to_fit, every frame is placed on one enlarged canvas big
    enough that no pixel is ever shifted off it; the frames end up aligned
    there and the canvas grows by the full spread of the offsets."""
    world = _world(seed=6)
    deltas = [(0, 0), (2, -3), (-4, 5), (3, 8), (-2, -6)]
    frames, truth = _series(world, deltas)  # each frame's offset is -delta

    aligned, offsets = series_to_target(
        frames, channel_axis=-1, roi=((30, 70), (40, 90)), max_step=None,
        expand_to_fit=True, padding_value=0, return_offsets=True)

    # The offsets are exactly what the non-expanding path would find.
    assert np.array_equal(offsets, truth)

    rows, cols = offsets[:, 0], offsets[:, 1]
    row_span = int(rows.max() - rows.min())
    col_span = int(cols.max() - cols.min())
    # One canvas for all frames, grown by the full offset spread.
    assert aligned.shape == (len(frames),
                             CROP_HEIGHT + row_span, CROP_WIDTH + col_span, 3)

    # The region every frame covers holds the aligned scene, so all frames must
    # agree there (an alignment check that doesn't assume the placement math).
    core = aligned[:, row_span:CROP_HEIGHT, col_span:CROP_WIDTH]
    assert np.all(core == core[0])

    # No pixel of any input frame is lost: each frame appears intact on the
    # canvas, at the slot that lines its offset up with the others.
    for i, (row, col) in enumerate(offsets):
        top, left = int(rows.max() - row), int(cols.max() - col)
        assert np.array_equal(
            aligned[i, top:top + CROP_HEIGHT, left:left + CROP_WIDTH], frames[i])

    # The canvas corners no frame reaches are filled with padding_value.
    assert aligned[0, 0, 0, 0] == 0


@pytest.mark.slow
def test_streaming_from_video_matches_eager(tmp_path):
    """
    Given a video filename or a VideoStreamer, series_to_target streams the
    frames and yields the aligned ones one at a time, producing exactly what the
    in-memory path produces on the same decoded frames (holding one frame at a
    time rather than the whole clip).
    """
    pytest.importorskip('av')
    world = _world(seed=5)
    frames, _ = _series(world, [(0, 0), (0, 2), (0, 4), (0, 3), (0, 1)])
    video = tmp_path / 'series.mp4'
    npimage.save_video(frames, str(video))

    roi = ((30, 70), (40, 90))
    kwargs = dict(roi=roi, max_step=(2, 3), target_index=range(3),
                  return_offsets=True)

    # Reference: the eager path on the frames exactly as the streamer decodes
    # them (so the comparison isolates the streaming logic from lossy encoding).
    with npimage.VideoStreamer(str(video)) as stream:
        decoded = np.stack([stream[i] for i in range(stream.n_frames)])
    eager, eager_offsets = series_to_target(decoded, channel_axis=-1, **kwargs)

    # VideoStreamer source: reads target and iterates frames all via stream[i],
    # so it must match the eager result bit-for-bit.
    with npimage.VideoStreamer(str(video)) as stream:
        pairs = list(series_to_target(stream, **kwargs))
    stream_frames = np.stack([frame for frame, _ in pairs])
    stream_offsets = np.array([offset for _, offset in pairs])
    assert stream_offsets.tolist() == eager_offsets.tolist()
    assert np.array_equal(stream_frames, eager)

    # Filename source: same offsets, one (H, W, 3) frame yielded per input frame.
    file_pairs = list(series_to_target(str(video), **kwargs))
    assert len(file_pairs) == len(frames)
    assert np.array([offset for _, offset in file_pairs]).tolist() == \
        eager_offsets.tolist()
    assert all(frame.shape == frames[0].shape for frame, _ in file_pairs)

    # expand_to_fit streams too (reading the source twice, one frame at a time)
    # and yields the same enlarged canvases the eager path builds.
    expand_kwargs = dict(kwargs, expand_to_fit=True)
    eager_expanded, _ = series_to_target(decoded, channel_axis=-1, **expand_kwargs)
    with npimage.VideoStreamer(str(video)) as stream:
        expanded_pairs = list(series_to_target(stream, **expand_kwargs))
    assert np.array_equal(np.stack([frame for frame, _ in expanded_pairs]),
                          eager_expanded)
    assert eager_expanded.shape[1] >= frames[0].shape[0]
    assert eager_expanded.shape[2] > frames[0].shape[1]  # the pan widens it
