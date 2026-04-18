"""Tests for PBM read/write roundtripping."""

import numpy as np

import npimage


def _roundtrip(data, tmp_path, filename):
    path = tmp_path / filename
    npimage.save(data, str(path))

    expected_size = npimage.filetypes.pbm.predict_file_size(data)
    assert path.stat().st_size == expected_size

    loaded = npimage.load(str(path))
    assert np.array_equal(data, loaded)


def test_pbm_arbitrary_width(tmp_path):
    """PBM roundtrip with a width that is not a multiple of 8."""
    data = np.array([
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 1, 1, 0, 0],
    ], dtype=bool)
    _roundtrip(data, tmp_path, 'arbitrary_width.pbm')


def test_pbm_width_multiple_of_8(tmp_path):
    """PBM roundtrip with a width that is a multiple of 8."""
    data = np.array([
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
    ], dtype=bool)
    _roundtrip(data, tmp_path, 'multiple_of_8.pbm')
