"""Shared pytest fixtures for npimage tests."""

from pathlib import Path

import pytest

import npimage

TESTS_DIR = Path(__file__).parent
TABLE_TENNIS_EMOJI = TESTS_DIR / 'table-tennis-emoji.png'


@pytest.fixture
def table_tennis_emoji():
    """The table tennis emoji image loaded as a numpy array."""
    return npimage.load(TABLE_TENNIS_EMOJI)
