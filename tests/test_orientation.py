"""Tests that load() applies the EXIF orientation tag for photo formats."""

import numpy as np
import pytest
from PIL import Image, ImageOps

import npimage


def _save_with_orientation(array, path, orientation):
    """
    Save `array` as an image at `path`, stamping the given EXIF Orientation
    value (e.g. 6 = Rotate 90 CW) without rotating the stored pixels, so that
    only a reader honoring the tag will display it upright.
    """
    image = Image.fromarray(array)
    exif = image.getexif()
    exif[0x0112] = orientation  # 0x0112 is the EXIF Orientation tag
    image.save(str(path), exif=exif)


@pytest.mark.parametrize('extension', ['jpg', 'png'])
def test_load_applies_rotate_90(tmp_path, extension):
    """
    A photo whose pixels are stored landscape but tagged Rotate 90 CW loads
    with its width and height swapped (i.e. the rotation is applied).
    """
    height, width = 20, 40
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, :width // 2] = [255, 0, 0]  # left half red, so rotation is visible
    path = tmp_path / f'rotated.{extension}'
    _save_with_orientation(array, path, orientation=6)  # Rotate 90 CW

    loaded = npimage.load(str(path))
    # Applying Rotate 90 CW swaps the axes, so a (height, width) image
    # becomes (width, height).
    assert loaded.shape[:2] == (width, height)
    # And it matches PIL's orientation-applied result exactly.
    truth = np.array(ImageOps.exif_transpose(Image.open(str(path))))
    assert np.array_equal(loaded, truth)


def test_load_without_orientation_is_unchanged(tmp_path):
    """An image with no Orientation tag loads with its pixels untouched."""
    array = np.random.randint(0, 256, (20, 40, 3), dtype=np.uint8)
    path = tmp_path / 'plain.png'
    Image.fromarray(array).save(str(path))

    loaded = npimage.load(str(path))
    assert loaded.shape == array.shape
    assert np.array_equal(loaded, array)
