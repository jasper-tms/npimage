"""Tests for FFmpegVideoWriter and AVVideoWriter."""

import numpy as np
import pytest

import npimage
from npimage.vidio import FFmpegVideoWriter, AVVideoWriter


def _animated_frames(base_image, num_frames=30, movement_range=30):
    """Generate frames that move `base_image` in a small circle."""
    frames = []
    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        dx = int(movement_range * np.cos(angle))
        dy = int(movement_range * np.sin(angle))
        frames.append(npimage.operations.offset(base_image, (dy, dx)))
    return frames


@pytest.mark.slow
@pytest.mark.parametrize('writer_class', [FFmpegVideoWriter, AVVideoWriter])
def test_video_writer_roundtrip(writer_class, table_tennis_emoji, tmp_path):
    """Writing frames with each writer produces a readable, non-empty file."""
    frames = _animated_frames(table_tennis_emoji, num_frames=30)
    output = tmp_path / f'{writer_class.__name__}.mp4'

    with writer_class(str(output), framerate=30, overwrite=True) as writer:
        for frame in frames:
            writer.write(frame)

    assert output.exists()
    assert output.stat().st_size > 0

    vid = npimage.VideoStreamer(str(output))
    assert vid.n_frames == len(frames)
    assert vid[0].shape[:2] == table_tennis_emoji.shape[:2]


@pytest.mark.slow
@pytest.mark.parametrize('writer_class', [FFmpegVideoWriter, AVVideoWriter])
@pytest.mark.parametrize('height, width', [(101, 100), (100, 101), (101, 101)])
def test_writer_pads_odd_dimensions_to_even(writer_class, height, width, tmp_path):
    """The yuv420p pixel format needs even dimensions, so a frame with an odd
    height and/or width is padded up by one (duplicating the edge) rather than
    failing to encode. Both writers do this."""
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(8, height, width, 3), dtype=np.uint8)
    output = tmp_path / f'{writer_class.__name__}_{height}x{width}.mp4'

    with writer_class(str(output), framerate=30, overwrite=True) as writer:
        for frame in frames:
            writer.write(frame)

    assert output.exists() and output.stat().st_size > 0
    vid = npimage.VideoStreamer(str(output))
    assert vid.n_frames == len(frames)
    out_height, out_width = vid[0].shape[:2]
    assert out_height == height + (height % 2)
    assert out_width == width + (width % 2)


@pytest.mark.slow
@pytest.mark.parametrize('extension', ['mp4', 'webm'])
def test_av_writer_preserves_variable_frame_times(extension, tmp_path):
    """Frames written with `time` keep those timestamps."""
    rng = np.random.default_rng(0)
    times = [0.0, 0.024, 0.055, 0.09, 0.124, 0.131, 0.189, 0.2, 0.25, 0.3]
    frames = rng.integers(0, 256, size=(len(times), 64, 64, 3), dtype=np.uint8)
    output = tmp_path / f'variable.{extension}'

    with AVVideoWriter(str(output), overwrite=True) as writer:
        for frame, time in zip(frames, times):
            writer.write(frame, time=time)

    vid = npimage.VideoStreamer(str(output))
    assert vid.n_frames == len(times)
    read_times = [vid.frame_number_to_time(i) for i in range(vid.n_frames)]
    np.testing.assert_allclose(read_times, times, atol=0.0005)


def test_av_writer_refuses_mixed_or_unordered_times(tmp_path):
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    with AVVideoWriter(str(tmp_path / 'mixed.mp4'), overwrite=True) as writer:
        writer.write(frame, time=0.0)
        with pytest.raises(ValueError):
            writer.write(frame)
        with pytest.raises(ValueError):
            writer.write(frame, time=0.0)
        writer.write(frame, time=0.04)


def test_ffmpeg_writer_refuses_time(tmp_path):
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    writer = FFmpegVideoWriter(str(tmp_path / 'constant.mp4'), overwrite=True)
    with pytest.raises(NotImplementedError):
        writer.write(frame, time=0.0)
