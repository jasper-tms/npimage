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
