#! /usr/bin/env python3

import numpy as np
import npimage
from pathlib import Path


def test_save_and_load_video(tmp_path=Path.cwd()):
    # Create synthetic video data
    data = np.full((120, 128, 256, 3), 255, dtype=np.uint8)
    for i in range(120):
        # Zero out the green (channel 1) channel along a moving line
        npimage.drawline(
            data[i],
            pt1=(10 + i / 4, 10 + i, 1),
            pt2=(90 + i / 4, 128 + i, 1),
            value=0,
            thickness=(3, 3, 1)  # Thickness 1 along the color axis
        )
    video_path = tmp_path / 'test_video.mp4'
    npimage.save_video(data, str(video_path), crf=15, overwrite=True)
    loaded = npimage.load_video(str(video_path), progress_bar=False)
    # Check shape
    assert loaded.shape == data.shape
    # Allow for some difference due to lossy encoding/decoding
    assert np.allclose(loaded.mean(axis=-1), data.mean(axis=-1), atol=64)
    print('test_save_and_load_video PASSED')


if __name__ == '__main__':
    test_save_and_load_video()
