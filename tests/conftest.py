"""Shared pytest fixtures for npimage tests."""

from pathlib import Path

import pytest

import npimage

# Load libheif before anything pulls in ffmpeg's native libraries.
#
# On macOS, initializing libheif (pillow_heif) AFTER a library that has already
# loaded ffmpeg's libav* native libs -- PyAV (test_video_streamer, test_video_
# writers) or OpenCV (cv2, used by test_align) -- segfaults inside pillow_heif;
# see npimage.imageio._check_ffmpeg_libs_not_loaded_before_heif. pytest imports
# conftest before it collects the test modules, so registering the HEIF opener
# here makes libheif load first, after which HEIF, PyAV, and OpenCV all coexist
# in the one test process. Without this the crash is order-dependent (it
# surfaces once any cv2- or av-importing test is collected before test_heic).
npimage.imageio._ensure_heif_opener_registered()

TESTS_DIR = Path(__file__).parent
TABLE_TENNIS_EMOJI = TESTS_DIR / 'table-tennis-emoji.png'


@pytest.fixture
def table_tennis_emoji():
    """The table tennis emoji image loaded as a numpy array."""
    return npimage.load(TABLE_TENNIS_EMOJI)
