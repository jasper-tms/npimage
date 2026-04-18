"""Tests for HEIC read/write support."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

import npimage


def test_heic_extension_registered():
    """HEIC is in the set of supported file extensions."""
    assert 'heic' in npimage.imageio.supported_extensions


def test_heic_load_save_roundtrip(tmp_path):
    """A uint8 RGB image roundtripped through HEIC keeps its shape and dtype."""
    data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    path = tmp_path / 'test_image.heic'

    npimage.save(data, str(path))
    assert path.exists()

    loaded = npimage.load(str(path))
    assert loaded.shape == data.shape
    assert loaded.dtype == np.uint8


@pytest.mark.skipif(sys.platform != 'darwin', reason='guard only fires on macOS')
def test_heic_guards_against_pyav_already_imported(tmp_path):
    """
    On macOS, importing PyAV before libheif causes a segfault inside
    pillow_heif. npimage guards against this by raising a RuntimeError with a
    clear message instead of letting the segfault happen.
    """
    script = textwrap.dedent(f"""
        import av  # noqa: F401
        import numpy as np
        import npimage
        data = np.zeros((10, 10, 3), dtype=np.uint8)
        try:
            npimage.save(data, r'{tmp_path / "x.heic"}')
        except RuntimeError as e:
            print('GUARD_FIRED:', e)
            raise SystemExit(0)
        raise SystemExit('guard did not fire')
    """)
    result = subprocess.run(
        [sys.executable, '-c', script], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f'subprocess exited {result.returncode}. stdout={result.stdout!r} '
        f'stderr={result.stderr!r}'
    )
    assert 'GUARD_FIRED:' in result.stdout
    assert 'PyAV' in result.stdout
