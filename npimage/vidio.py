#!/usr/bin/env python3
"""
Functions for reading and writing video files.

Function list:
- load_video(filename) -> np.ndarray
- lazy_load_video(filename) -> Iterator[np.ndarray]
- save_video(data, filename) -> None
    Saves a numpy array of pixel values as a video file.
    Arguments for setting framerate, video bitrate, etc are provided.
    With filename=None, returns the encoded video as bytes instead.

Class list:
- VideoStreamer: Provides fast random access to frames in a video file
    via VideoStreamer[frame_number], or by time in seconds via
    VideoStreamer.t[time_in_seconds].
- ConstantFrameratePTS: The frame timestamps of a constant-framerate video,
    stored as a formula instead of as one timestamp per frame. This is what
    VideoStreamer.frames_pts holds for a constant-framerate video.
- VideoWriter: Allows writing frames one-by-one to a video file via
    VideoWriter.write(image). This can be advantageous compared to save_video
    because you don't ever have to have all the frames in memory at once.
    With filename=None, close() returns the encoded video as bytes instead.
"""

from typing import Union, Tuple, Iterator, Literal, Optional
from pathlib import Path
from collections import OrderedDict
import subprocess
import shutil
import tempfile
import io
import os
import sys
import threading
import json
import re
import bisect
from fractions import Fraction

import numpy as np
from tqdm import tqdm

from . import utils

codec_aliases = {
    'libx264': 'libx264',
    'avc1': 'libx264',
    'h264': 'libx264',
    'H.264': 'libx264',
    'libx265': 'libx265',
    'hevc': 'libx265',
    'hvc1': 'libx265',
    'hev1': 'libx265',
    'h265': 'libx265',
    'H.265': 'libx265',
    'libvpx': 'libvpx',
    'vp8': 'libvpx',
    'libvpx-vp9': 'libvpx-vp9',
    'vp9': 'libvpx-vp9',
}

# Default codec for each container format
extension_default_codecs = {
    'mp4': 'libx264',
    'mkv': 'libx264',
    'avi': 'libx264',
    'mov': 'libx264',
    'webm': 'libvpx-vp9',
}

supported_extensions = ['mp4', 'mkv', 'avi', 'mov', 'webm', 'gif']

# Container formats the video writers can encode to bytes (filename=None).
# Values are the matching FFmpeg/PyAV muxer names.
bytes_formats = {
    'mp4': 'mp4',
    'mkv': 'matroska',
    'avi': 'avi',
    'mov': 'mov',
    'webm': 'webm',
}

# Time base for AVVideoWriter frames written with explicit timestamps.
# 90000 is evenly divisible by common frame rates (24, 25, 30, 30000/1001, 60).
timestamped_time_base = Fraction(1, 90000)


def _import_av():
    try:
        import av
        return av
    except ImportError:
        raise ImportError('Missing optional dependency for video processing,'
                          ' run `pip install av` and try again')


def _check_ffmpeg_available(program: Literal['ffmpeg', 'ffprobe'] = 'ffmpeg'):
    """
    Check that ffmpeg or ffprobe is on PATH, and raise a
    FileNotFoundError with installation instructions if not.
    """
    if shutil.which(program) is not None:
        return
    if sys.platform.startswith('linux'):
        install_hint = (
            'On Debian/Ubuntu:  sudo apt install ffmpeg\n'
            '  On Fedora/RHEL:    sudo dnf install ffmpeg\n'
            '  On Arch:           sudo pacman -S ffmpeg'
        )
    elif sys.platform == 'darwin':
        install_hint = (
            'With Homebrew:     brew install ffmpeg\n'
            '  With MacPorts:     sudo port install ffmpeg'
        )
    elif sys.platform.startswith('win'):
        install_hint = (
            'With winget:       winget install ffmpeg\n'
            '  With Chocolatey:   choco install ffmpeg\n'
            '  Or download a static build from https://ffmpeg.org/download.html'
            ' and add it to your PATH.'
        )
    else:
        install_hint = (
            'See https://ffmpeg.org/download.html for installation'
            ' instructions for your platform.'
        )
    raise FileNotFoundError(
        f"`{program}` was not found on PATH. npimage.vidio needs the ffmpeg"
        f" suite installed to read and write video files.\n\n"
        f"Install ffmpeg:\n  {install_hint}"
    )


def load_video(filename,
               return_framerate=False,
               progress_bar=True) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """
    Load all images in a video file as a numpy array.

    Parameters
    ----------
    filename : str
        Path to the video file
    return_framerate : bool, default False
        If True, return the frame rate of the video
    progress_bar : bool, default True
        If True, display a progress bar

    Returns
    -------
    If return_framerate is False:
        data : numpy.ndarray
            The video frames as a numpy array, shape (num_frames, height, width, colors)
    If return_framerate is True:
        (data, framerate) : tuple, where data is as above and:
        framerate : float
            The frame rate of the video in frames per second
    """
    extension = Path(filename).suffix.lower().lstrip('.')
    if extension == 'gif':
        from PIL import Image
        frames = []
        durations = []
        with Image.open(filename) as img:
            while True:
                frames.append(np.array(img.convert('RGB')))
                durations.append(img.info.get('duration', None))
                try:
                    img.seek(img.tell() + 1)
                except EOFError:
                    break
        data = np.array(frames)
        if return_framerate:
            if None in durations:
                raise ValueError('Cannot return framerate for GIF because one or'
                                 ' more frames are missing duration metadata')
            if len(durations) == 0:
                return data, None
            framerate = 1000 / np.mean(durations)
            return data, float(framerate)
        return data

    av = _import_av()
    with av.open(filename) as container:
        stream = container.streams.video[0]
        num_frames = stream.frames
        if not num_frames or num_frames == 0:
            # If we don't know the number of frames, we can't preallocate, so
            # it's hard to do better than the following approach which temporarily
            # uses double the amount of RAM compared to the preallocated approach.
            data = np.array(list(lazy_load_video(filename)))
            if return_framerate:
                return data, float(stream.average_rate)
            else:
                return data
        else:
            # Load first image to get shape and dtype
            frame_iter = container.decode(stream)
            first_frame = next(frame_iter)
            first_img = first_frame.to_ndarray(format='rgb24')
            # Preallocate memory for the entire array
            data = np.empty((num_frames, *first_img.shape), dtype=first_img.dtype)
            # Then fill it up frame by frame
            data[0] = first_img
            container.seek(0, stream=stream)
            for i, frame in tqdm(enumerate(frame_iter), total=num_frames,
                                 desc='Loading video', disable=not progress_bar):
                img = frame.to_ndarray(format='rgb24')
                if i == 0 and not np.array_equal(img, first_img):
                    raise RuntimeError('PyAV seek failed. Please report how this happened'
                                       ' at github.com/jasper-tms/npimage/issues')
                data[i] = img
            if (data[-1] == 0).all():
                print('WARNING: Last frame of video is all zeros, this may'
                      ' indicate an error in video loading unless you expected this.')
            if return_framerate:
                return data, float(stream.average_rate)
            else:
                return data


def lazy_load_video(filename) -> Iterator[np.ndarray]:
    """
    Lazily load video frames as numpy arrays using PyAV (or using PIL for gifs)

    This iterator yields images in the order they appear in the video.
    If you want reasonably fast random access to arbitrary frames in
    a video, use the VideoStreamer class instead.

    Parameters
    ----------
    filename : str
        Path to the video file.

    Yields
    ------
    frame : np.ndarray
        Video frame as a numpy array, shape (height, width, colors).
    """
    extension = Path(filename).suffix.lower().lstrip('.')
    if extension == 'gif':
        from PIL import Image
        with Image.open(filename) as img:
            while True:
                img_array = np.array(img.convert('RGB'))
                yield img_array
                try:
                    img.seek(img.tell() + 1)
                except EOFError:
                    break
        return

    av = _import_av()
    rotation = _get_rotation_from_metadata(filename)
    with av.open(Path(filename).expanduser()) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            img = frame.to_ndarray(format='rgb24')
            if rotation not in [None, '0', 0]:
                img = np.rot90(img, k=-int(rotation) // 90)
            yield img


class VideoSeekError(RuntimeError):
    pass


_cache_size_units = {
    'b': 1,
    'kb': 1024,
    'mb': 1024 ** 2,
    'gb': 1024 ** 3,
}

# Bumped whenever the on-disk .index format changes in a way that makes
# previously written index files unusable or obsolete. Index files without
# this key, or with an older value, are discarded and rebuilt.
_index_format_version = 2

# Framerates that real videos are actually shot at, including the awkward
# 1000/1001 ("NTSC") rates. A short clip's timestamps often can't distinguish
# the true framerate from some nearby fraction that happens to reproduce them
# just as exactly, so when we have to work the framerate out from the
# timestamps we check these first and report one of them if it fits. That way
# a 29.97 fps video is reported as 30000/1001 rather than as whatever odd
# fraction happens to land on the same timestamps.
_common_framerates = (
    Fraction(24000, 1001), Fraction(24), Fraction(25),
    Fraction(30000, 1001), Fraction(30), Fraction(48), Fraction(50),
    Fraction(60000, 1001), Fraction(60), Fraction(100),
    Fraction(120000, 1001), Fraction(120), Fraction(240),
)


def _round_half_away_from_zero(value: Fraction) -> int:
    """
    Round a `Fraction` to the nearest integer, breaking ties away from zero.

    ffmpeg rounds this way (its `AV_ROUND_NEAR_INF` mode) when it converts a
    frame's ideal presentation time into an integer timestamp in the
    container's time base, so this is the convention that must be used to
    reproduce the timestamps ffmpeg wrote. Python's built-in `round()` breaks
    ties toward the nearest even integer instead, which disagrees whenever a
    frame's ideal time lands exactly halfway between two ticks: a 16 fps webm
    stores its second frame at tick 63, whereas `round(Fraction(125, 2))`
    gives 62.

    Parameters
    ----------
    value : Fraction
        The value to round.

    Returns
    -------
    int
        `value` rounded to the nearest integer, with exact halves rounded
        away from zero.
    """
    numerator, denominator = value.numerator, value.denominator
    if numerator >= 0:
        return (2 * numerator + denominator) // (2 * denominator)
    return -((-2 * numerator + denominator) // (2 * denominator))


class ConstantFrameratePTS:
    """
    The presentation timestamps (PTS) of a constant-framerate video, stored
    as a formula rather than as an explicit list of values.

    Frame `i` of a constant-framerate video is shown `i / framerate` seconds
    after the first frame, but a container can only store a timestamp as a
    whole number of ticks of its `time_base`, so the PTS actually written to
    the file is that ideal time rounded to the nearest tick:

        pts[i] = pts0 + round(i / framerate / time_base)

    When one frame spans a whole number of ticks the rounding does nothing and
    consecutive PTS values come out evenly spaced. This is the case for a
    typical mp4, whose `time_base` is chosen to divide evenly by the framerate
    (at 30 fps with a time_base of 1/15360, every frame is exactly 512 ticks).

    When one frame does not span a whole number of ticks, the rounding makes
    consecutive PTS values differ by a tick here and there even though the
    video is perfectly constant-framerate. This is unavoidable for webm and
    other Matroska files, whose `time_base` is 1/1000: at 30 fps a frame lasts
    100/3 ticks, so the stored timestamps go 0, 33, 67, 100, 133, ... and the
    gaps between them alternate between 33 and 34. Such a video is still
    constant-framerate in every sense that matters here, because the timestamps
    are still fully described by the formula above.

    This class represents both cases exactly. It behaves like an immutable
    sequence of the video's PTS values, so it can be indexed, sliced,
    iterated, and searched just like the plain list of PTS values that
    `VideoStreamer` uses for a variable-framerate video. Unlike that list it
    stores only four numbers no matter how long the video is, and it can
    convert a PTS back to a frame number by arithmetic instead of by scanning.
    """
    def __init__(self,
                 pts0: int,
                 framerate: Union[int, Fraction],
                 time_base: Fraction,
                 n_frames: int):
        """
        Parameters
        ----------
        pts0 : int
            The PTS of the first frame.

        framerate : int or Fraction
            Frames per second.

        time_base : Fraction
            The duration of one PTS tick, in seconds.

        n_frames : int
            The number of frames in the video.
        """
        self.pts0 = int(pts0)
        self.framerate = Fraction(framerate)
        self.time_base = Fraction(time_base)
        self.n_frames = int(n_frames)
        if self.framerate <= 0:
            raise ValueError(f'framerate must be positive, got {framerate}')
        if self.n_frames < 0:
            raise ValueError(f'n_frames must not be negative, got {n_frames}')
        # The number of ticks one frame lasts. Not necessarily a whole number,
        # which is the entire reason this class exists.
        self.ticks_per_frame = 1 / (self.framerate * self.time_base)

    def _pts_at(self, frame_number: int) -> int:
        """
        Evaluate the PTS formula at a single frame number, which must already
        be normalized to lie in `range(self.n_frames)`.
        """
        return self.pts0 + _round_half_away_from_zero(frame_number
                                                      * self.ticks_per_frame)

    def as_array(self) -> np.ndarray:
        """
        Evaluate the PTS formula at every frame number at once.

        Returns
        -------
        np.ndarray
            The video's PTS values, as an array of `self.n_frames` integers.
        """
        numerator = self.ticks_per_frame.numerator
        denominator = self.ticks_per_frame.denominator
        # The vectorized form of _round_half_away_from_zero(i * ticks_per_frame),
        # relying on i, numerator, and denominator all being positive. numpy
        # integers wrap around silently on overflow instead of raising, so fall
        # back to (slower) arbitrary-precision python ints if int64 can't hold
        # the largest intermediate value this would compute.
        largest_intermediate = 2 * max(self.n_frames - 1, 0) * numerator + denominator
        if largest_intermediate > np.iinfo(np.int64).max:
            return np.array([self._pts_at(i) for i in range(self.n_frames)],
                            dtype=object)
        frame_numbers = np.arange(self.n_frames, dtype=np.int64)
        return self.pts0 + ((2 * frame_numbers * numerator + denominator)
                            // (2 * denominator))

    def reproduces(self, frames_pts: Union[list, np.ndarray]) -> bool:
        """
        Check whether this formula reproduces a list of PTS values exactly.

        This is the test that decides whether a video counts as constant
        framerate. It is an exact test, not an approximate one: every single
        timestamp must come out bit-for-bit identical, and the timestamps must
        be strictly increasing (so that each PTS maps back to exactly one
        frame number).

        Parameters
        ----------
        frames_pts : list or np.ndarray
            The PTS values read out of the video file, in presentation order.

        Returns
        -------
        bool
            True if `pts[i] == pts0 + round(i / framerate / time_base)` holds
            for every frame, and the values strictly increase.
        """
        if len(frames_pts) != self.n_frames:
            return False
        reconstructed = self.as_array()
        if not np.array_equal(reconstructed, np.asarray(frames_pts)):
            return False
        return bool(self.n_frames < 2 or (np.diff(reconstructed) > 0).all())

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, key) -> Union[int, list]:
        if isinstance(key, slice):
            return [self._pts_at(i) for i in range(*key.indices(self.n_frames))]
        if not np.issubdtype(type(key), np.integer):
            raise TypeError(f'Frame number must be an int or slice, got '
                            f'{type(key).__name__}')
        frame_number = int(key)
        if frame_number < 0:
            frame_number += self.n_frames
        if not 0 <= frame_number < self.n_frames:
            raise IndexError(f'Frame number {key} is out of range for a video '
                             f'with {self.n_frames} frames.')
        return self._pts_at(frame_number)

    def __iter__(self) -> Iterator[int]:
        return (self._pts_at(i) for i in range(self.n_frames))

    def index(self, pts: int) -> int:
        """
        Find the frame number whose PTS is `pts`, in constant time.

        Inverting the PTS formula gives a frame number that is off by at most
        one from the true one (the rounding in the formula moves a timestamp by
        less than a tick, and a frame lasts at least a tick in any video whose
        timestamps strictly increase), so it is enough to check the estimate
        and its two neighbors.

        Parameters
        ----------
        pts : int
            The PTS to look up.

        Returns
        -------
        int
            The frame number with this PTS.

        Raises
        ------
        ValueError
            If no frame has this PTS.
        """
        pts = int(pts)
        estimate = int((pts - self.pts0) / self.ticks_per_frame)
        for frame_number in (estimate - 1, estimate, estimate + 1):
            if 0 <= frame_number < self.n_frames and self._pts_at(frame_number) == pts:
                return frame_number
        raise ValueError(f'PTS {pts} is not in this video.')

    def __contains__(self, pts) -> bool:
        try:
            self.index(pts)
        except (ValueError, TypeError):
            return False
        return True

    def __eq__(self, other) -> bool:
        if isinstance(other, ConstantFrameratePTS):
            return (self.pts0 == other.pts0
                    and self.framerate == other.framerate
                    and self.time_base == other.time_base
                    and self.n_frames == other.n_frames)
        if isinstance(other, (list, tuple)):
            return list(self) == list(other)
        return NotImplemented

    def __repr__(self) -> str:
        return (f'{type(self).__name__}(pts0={self.pts0}, '
                f'framerate={self.framerate}, time_base={self.time_base}, '
                f'n_frames={self.n_frames})')


def _detect_constant_framerate(frames_pts: list,
                               time_base: Fraction,
                               declared_framerate=None
                               ) -> Optional[ConstantFrameratePTS]:
    """
    Determine whether a video's PTS values are exactly described by a constant
    framerate, and if so, by which one.

    A video counts as constant framerate when some framerate `r` satisfies
    `pts[i] == pts0 + round(i / r / time_base)` for every frame (see
    `ConstantFrameratePTS`). Rather than solving for `r`, this tries a short
    list of plausible candidates and verifies each one against every timestamp
    in the video, so a candidate is only ever accepted on the strength of
    exactly reproducing the data. A framerate the container declares but does
    not honor is therefore rejected, as is a framerate derived here that turns
    out not to fit.

    Parameters
    ----------
    frames_pts : list of int
        The video's PTS values, in presentation order.

    time_base : Fraction
        The duration of one PTS tick, in seconds.

    declared_framerate : int, Fraction, or None, default None
        The framerate the container claims, if it claims one. Used only as a
        candidate to be verified, never trusted on its own.

    Returns
    -------
    ConstantFrameratePTS or None
        A formula that exactly reproduces `frames_pts`, or None if the video
        is genuinely variable framerate.
    """
    n_frames = len(frames_pts)
    if n_frames == 0:
        return None
    if n_frames == 1:
        # A single frame is trivially consistent with any framerate. Pick the
        # declared one if there is one so that `framerate` is still reported.
        framerate = Fraction(declared_framerate) if declared_framerate else Fraction(1)
        return ConstantFrameratePTS(frames_pts[0], framerate, time_base, 1)

    candidates = []
    if declared_framerate:
        candidates.append(Fraction(declared_framerate))

    # The framerate implied by the gap between the first two frames. This is
    # the exact answer whenever a frame spans a whole number of ticks, which
    # covers the evenly-spaced case that mp4 files fall into.
    first_gap = frames_pts[1] - frames_pts[0]
    if first_gap > 0:
        candidates.append(1 / (first_gap * time_base))

    # The average framerate across the whole video, which for a constant
    # framerate video lands within a rounding error of the true rate. Snapping
    # it to a nearby standard framerate, and failing that to a nearby simple
    # fraction, recovers the true rate. Standard framerates are tried first
    # because a short video's timestamps can be reproduced exactly by more than
    # one framerate, and in that case the standard one is the right answer to
    # report.
    span = frames_pts[-1] - frames_pts[0]
    if span > 0:
        average = Fraction(n_frames - 1) / (span * time_base)
        nearby = sorted((framerate for framerate in _common_framerates
                         if abs(framerate - average) < average / 1000),
                        key=lambda framerate: abs(framerate - average))
        candidates.extend(nearby)
        for largest_denominator in (1, 1001, 100000):
            candidates.append(average.limit_denominator(largest_denominator))

    checked = set()
    for framerate in candidates:
        framerate = Fraction(framerate)
        if framerate <= 0 or framerate in checked:
            continue
        checked.add(framerate)
        candidate = ConstantFrameratePTS(frames_pts[0], framerate,
                                         time_base, n_frames)
        if candidate.reproduces(frames_pts):
            return candidate
    return None


def _parse_cache_size(cache_size: Optional[Union[int, str]]
                      ) -> Tuple[Optional[int], Optional[int]]:
    """
    Interpret the `cache_size` argument of `VideoStreamer` and return a
    `(max_frames, max_bytes)` tuple in which exactly one element is set when
    caching is enabled, or `(None, None)` when caching is disabled.

    See the `VideoStreamer` initializer docstring for the accepted formats.
    """
    if cache_size is None:
        return None, None
    if isinstance(cache_size, bool):
        raise TypeError(f'cache_size must be an int, str, or None, not a bool '
                        f'(got {cache_size})')
    if isinstance(cache_size, (int, np.integer)):
        if cache_size < 0:
            raise ValueError(f'cache_size must be non-negative, got {cache_size}')
        if cache_size == 0:
            return None, None
        return int(cache_size), None
    if isinstance(cache_size, str):
        match = re.fullmatch(r'\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)\s*', cache_size)
        if match is None:
            raise ValueError(f'Could not interpret cache_size string "{cache_size}". '
                             "Expected something like '64', '256MB', or '1.5GB'.")
        number, unit = match.group(1), match.group(2).lower()
        if unit == '':
            # A bare number means a frame count, which must be a whole number.
            value = float(number)
            if not value.is_integer():
                raise ValueError(f'A cache_size given as a number of frames must be '
                                 f'a whole number, got "{cache_size}"')
            return (int(value), None) if value > 0 else (None, None)
        # Accept binary-prefix spellings like 'MiB' as aliases for 'MB'.
        unit = unit.replace('ib', 'b')
        if unit not in _cache_size_units:
            raise ValueError(f'Unrecognized cache_size unit "{match.group(2)}". '
                             f'Recognized units are B, KB, MB, and GB.')
        num_bytes = int(float(number) * _cache_size_units[unit])
        return (None, num_bytes) if num_bytes > 0 else (None, None)
    raise TypeError(f'cache_size must be an int, str, or None, got '
                    f'{type(cache_size).__name__}')


class VideoStreamer:
    def __init__(self,
                 filename: str,
                 verbose: bool = False,
                 cache_index: Literal['auto', True, False] = 'auto',
                 cache_size: Optional[Union[int, str]] = '256MB'):
        """
        Parameters
        ----------
        filename : str
            Path to the video file.

        verbose : bool, default False
            If True, print progress messages.

        cache_index : 'auto' (default), True, or False
            **Definitely set this to True in code where speed matters!**
            Whether to cache the frame timestamp index to a .index file
            next to the video file for faster loading next time.
            If 'auto', the index is cached only if building it takes
            more than 0.5 seconds, which only happens for ~1+ GB videos.
            An index file written by an older version of npimage is
            discarded and rebuilt.

        cache_size : int, str, or None, default '256MB'
            Sets the size of an in-memory cache of decoded frames. The cache
            makes a three-way tradeoff: in exchange for holding up to
            `cache_size` worth of decoded frames in memory, reading a frame
            that is already cached is dramatically faster (~20x in benchmarks
            on a 1080p video) than decoding it, while reading a not-yet-cached
            frame is only marginally slower (~2%, the cost of copying the
            decoded frame before returning it). This is a large net win
            whenever you revisit frames, for example when scrubbing back and
            forth through a video, which is why caching is on by default. The
            least recently used frames are evicted once the cache is full.
            (The ~20x and ~2% figures are approximate and vary with the video.)

            - None or 0 disables caching.
            - An int sets the maximum number of frames to cache, e.g.
              `cache_size=64`.
            - A string sets a maximum memory budget, e.g. `cache_size='256MB'`
              or `cache_size='1.5GB'`. Recognized units are B, KB, MB, and GB
              (interpreted as powers of 1024, so '1MB' means 1024*1024 bytes).
              A bare numeric string like '64' is treated as a frame count.

            Frames returned by indexing are always independent, writeable
            arrays, exactly as when caching is disabled: the cache holds its own
            private copy of each frame, so modifying a returned frame never
            affects the cache or any other returned frame.
        """
        extension = Path(filename).suffix.lower().lstrip('.')
        if extension == 'gif':
            raise NotImplementedError('Streaming from gifs not yet implemented. '
                                      'Use load_video() or lazy_load_video() instead.')
        self.av = _import_av()
        self.verbose = verbose
        self.filename = Path(filename).expanduser()
        if not self.filename.exists():
            raise FileNotFoundError(f'File {filename} not found')

        self.container = self.av.open(str(self.filename))
        self.stream = self.container.streams.video[0]
        self.time_base = self.stream.time_base
        self._frame_iterator = self.container.decode(self.stream)
        self._shape = None
        self._first_frame = None
        self._width = None
        self._height = None
        self._ndim = None
        self._dtype = None
        self._current_frame_number = None
        self._lock = threading.Lock()

        # In-memory frame cache (see the cache_size docstring above). Exactly
        # one of _cache_max_frames / _cache_max_bytes is set when caching is on.
        self._cache_max_frames, self._cache_max_bytes = _parse_cache_size(cache_size)
        if self._cache_max_frames is None and self._cache_max_bytes is None:
            self._cache = None
        else:
            self._cache = OrderedDict()
        self._cache_bytes = 0

        self.index_filename = self.filename.with_suffix(self.filename.suffix + '.index')
        self._build_index(cache_index=cache_index)
        self.t = _VideoStreamerTimeIndexer(self)

    def _build_index(self, cache_index='auto'):
        if cache_index and self.index_filename.exists() and self._load_index():
            return
        if self.verbose:
            print('Building frame timestamp index for fast random frame access...')

        import time
        start_time = time.time()

        frames_pts = []
        # Try getting frame PTS values fast using PyAV
        try:
            from tqdm import tqdm

            with self.av.open(str(self.filename)) as container:
                stream = container.streams.video[0]
                start_pts = stream.start_time if stream.start_time is not None else 0
                # Access as packets to only read metadata (fast), not pixel data (slow)
                for packet in tqdm(container.demux(stream), total=stream.frames,
                                   desc='Indexing frames', disable=not self.verbose):
                    if packet.pts is not None and packet.pts >= start_pts:
                        frames_pts.append(packet.pts)

            # demux gave us packets in the order they appeared in the file
            # (decoding order), but we want the frames in presentation order,
            # so we sort them.
            frames_pts.sort()
        except Exception as e:
            if self.verbose:
                print(f'Error getting frame PTS values using PyAV: {e}')
                print('Falling back to ffprobe...')
            frames_pts = []
            # Fallback to ffprobe if PyAV didn't work
            _check_ffmpeg_available('ffprobe')
            cmd = ['ffprobe',
                   '-select_streams', 'v:0',
                   '-show_frames',
                   '-show_entries', 'frame=pts',
                   '-of', 'default=noprint_wrappers=1:nokey=1',
                   '-v', 'quiet',
                   self.filename]
            ffprobe_result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, text=True)
            if ffprobe_result.returncode != 0:
                raise RuntimeError(f'ffprobe failed: {ffprobe_result.stderr}')

            for line in ffprobe_result.stdout.strip().split('\n'):
                if line.strip().startswith('pts='):
                    pts_str = line.split('=')[1]
                    frames_pts.append(int(pts_str))

        if len(frames_pts) == 0:
            raise RuntimeError('No timestamps found in frame metadata')

        self.n_frames = len(frames_pts)
        self.pts0 = frames_pts[0]
        self.rotation = _get_rotation_from_metadata(self.filename)
        index = {'index_format_version': _index_format_version}

        # Determine whether the video is constant or variable framerate. The
        # video counts as constant framerate if some single framerate exactly
        # reproduces every one of its timestamps, which is a weaker condition
        # than the timestamps being evenly spaced (see ConstantFrameratePTS)
        # but still an exact one.
        declared_framerate = getattr(self.stream, 'average_rate', None)
        constant_framerate_pts = _detect_constant_framerate(
            frames_pts, self.time_base, declared_framerate)
        if constant_framerate_pts is not None:
            # The video is constant framerate
            self.frames_pts = constant_framerate_pts
            self._framerate = constant_framerate_pts.framerate
            if self._framerate.denominator == 1:
                self._framerate = self._framerate.numerator
                index['framerate'] = self._framerate
            else:
                index['framerate'] = {'numerator': self._framerate.numerator,
                                      'denominator': self._framerate.denominator}
            index['pts0'] = self.pts0
        else:
            # The video is variable framerate
            self._framerate = 'variable'
            index['framerate'] = 'variable'
            self.frames_pts = frames_pts
            index['frames_pts'] = frames_pts

        index['n_frames'] = self.n_frames
        index['rotation'] = self.rotation
        index['time_base'] = {'numerator': self.time_base.numerator,
                              'denominator': self.time_base.denominator}

        if cache_index is True or (cache_index == 'auto'
                                   and time.time() - start_time > 0.5):
            with open(self.index_filename, 'w') as f:
                json.dump(index, f)
            if self.verbose:
                print(f'Cached index at "{self.index_filename}"')

    def _load_index(self) -> bool:
        """
        Load a previously cached frame timestamp index.

        Returns
        -------
        bool
            True if the index was loaded. False if it was written by an older
            version of npimage or is unreadable, in which case the caller
            should rebuild it from the video file.
        """
        if self.verbose:
            print(f'Loading frame timestamp index from "{self.index_filename}"')

        try:
            with open(self.index_filename, 'r') as f:
                index = json.load(f)
            if index.get('index_format_version') != _index_format_version:
                if self.verbose:
                    print(f'Index at "{self.index_filename}" was written in an'
                          ' outdated format. Rebuilding it.')
                return False
            self.n_frames = index['n_frames']
            self.rotation = index.get('rotation', None)
            time_base = index['time_base']
            self.time_base = Fraction(time_base['numerator'],
                                      time_base['denominator'])

            if index['framerate'] == 'variable':
                self._framerate = 'variable'
                self.frames_pts = index['frames_pts']
                self.pts0 = self.frames_pts[0]
            else:
                self.pts0 = index['pts0']
                if isinstance(index['framerate'], dict):
                    self._framerate = Fraction(index['framerate']['numerator'],
                                               index['framerate']['denominator'])
                else:
                    self._framerate = index['framerate']
                self.frames_pts = ConstantFrameratePTS(self.pts0, self._framerate,
                                                       self.time_base, self.n_frames)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as e:
            if self.verbose:
                print(f'Could not read index at "{self.index_filename}" ({e}).'
                      ' Rebuilding it.')
            return False
        return True

    @property
    def ticks_per_frame(self) -> Fraction:
        """
        How long one frame lasts, in PTS ticks, as an exact `Fraction`.

        This is not necessarily a whole number: a 30 fps webm has a time_base
        of 1/1000, so each of its frames lasts 100/3 ticks. The PTS actually
        stored for a frame is this ideal spacing rounded to a whole number of
        ticks, so use `frame_number_to_pts()` to get a frame's real PTS rather
        than multiplying by this.

        Raises
        ------
        AttributeError
            If the video is variable framerate, in which case its frames have
            no single duration.
        """
        if self._framerate == 'variable':
            raise AttributeError('A variable-framerate video has no single '
                                 'ticks_per_frame. Use frame_number_to_pts() '
                                 'instead.')
        return self.frames_pts.ticks_per_frame

    @property
    def pts_delta(self) -> int:
        """
        The whole number of PTS ticks between consecutive frames.

        This only exists for a video whose frames each last a whole number of
        ticks, which is the case when the container's time_base divides evenly
        by the framerate (as a typical mp4's does). Such a video's PTS values
        are evenly spaced, so `pts0 + frame_number * pts_delta` is a valid way
        to compute them.

        It deliberately does not exist for a video whose frames do not last a
        whole number of ticks, such as any 30 fps webm (whose time_base is
        1/1000, giving 100/3 ticks per frame). Such a video's PTS values are
        not evenly spaced, so there is no correct integer to return here, and
        computing timestamps by repeatedly adding one would drift away from the
        real timestamps. Use `frame_number_to_pts()`, which is exact for every
        video, or `ticks_per_frame` for the exact fractional frame duration.

        Raises
        ------
        AttributeError
            If the video is variable framerate, or if its frames do not last a
            whole number of ticks.
        """
        ticks_per_frame = self.ticks_per_frame
        if ticks_per_frame.denominator != 1:
            raise AttributeError(
                f'The frames of this video each last {ticks_per_frame} PTS '
                f'ticks, which is not a whole number, so its PTS values are '
                f'not evenly spaced and it has no integer pts_delta. Use '
                f'frame_number_to_pts() to get a frame\'s exact PTS, or '
                f'ticks_per_frame for the exact frame duration.')
        return ticks_per_frame.numerator

    @property
    def framerate(self) -> Union[float, Literal['variable']]:
        if self._framerate == 'variable':
            return 'variable'
        return float(self._framerate)

    @property
    def fps(self) -> float:
        """
        Frames per second, returned as a float even when the framerate is
        'variable'. For variable-framerate videos this is the average rate,
        equal to `n_frames / duration`, equivalently the number of frame
        intervals divided by the time span between the first and last frames'
        timestamps.
        """
        if self.framerate == 'variable':
            return float((self.n_frames - 1) / self.time_base
                         / (self.frames_pts[-1] - self.frames_pts[0]))
        else:
            return self.framerate

    @property
    def timestep(self) -> Union[float, Literal['variable']]:
        if self._framerate == 'variable':
            return 'variable'
        return float(1.0 / self._framerate)

    @property
    def duration(self) -> float:
        """
        Total playback duration of the video in seconds, equal to
        `n_frames / fps`.

        Each frame is treated as occupying one inter-frame interval on
        screen, so the duration extends from the first frame's timestamp to
        one mean inter-frame interval past the last frame's timestamp. For
        constant-framerate videos this is exactly `n_frames * timestep`.
        For variable-framerate videos the last frame's display interval is
        assumed to equal the mean inter-frame interval, which may differ
        slightly from ffprobe's reported duration (ffprobe trusts the
        encoder-written last-frame duration metadata, which is encoder-
        dependent).
        """
        return float(self.n_frames / self.fps)

    def frame_number_to_pts(self, frame_number: int) -> int:
        if hasattr(frame_number, '__iter__'):
            return [self.frame_number_to_pts(n) for n in frame_number]
        frame_number = self._normalize_frame_number(frame_number)
        return int(self.frames_pts[frame_number])

    def frame_number_to_time(self, frame_number: int) -> float:
        if hasattr(frame_number, '__iter__'):
            return [self.frame_number_to_time(n) for n in frame_number]
        return float(self.frame_number_to_pts(frame_number) * self.time_base)

    def pts_to_frame_number(self, pts: int) -> int:
        if hasattr(pts, '__iter__'):
            return [self.pts_to_frame_number(p) for p in pts]
        first_pts, last_pts = self.frames_pts[0], self.frames_pts[-1]
        if pts < first_pts:
            raise ValueError(f'PTS {pts} is before the start of the'
                             f' video (PTS {first_pts}).')
        if pts > last_pts:
            raise ValueError(f'PTS {pts} is after the end of the'
                             f' video (PTS {last_pts}).')
        try:
            # O(1) for a constant-framerate video, O(n_frames) for a
            # variable-framerate one, whose PTS values are a plain list.
            return self.frames_pts.index(pts)
        except ValueError:
            raise ValueError(f'PTS {pts} is between frames for this'
                             ' video.') from None

    def __getitem__(self, key) -> np.ndarray:
        if np.issubdtype(type(key), np.integer):
            return self._get_frame(key)
        if isinstance(key, slice):  # Support slicing
            key = (key,)  # Logic is handled in the tuple case below
        if not isinstance(key, tuple):
            raise TypeError('Key must be an int, slice, or a tuple of ints/slices')

        frame_idx = key[0]
        if np.issubdtype(type(frame_idx), np.integer):
            frames = self._get_frame(frame_idx)
            key = key[1:]
        elif isinstance(frame_idx, slice):  # Support slicing
            start, stop, step = frame_idx.indices(self.n_frames)
            frames = np.array([self._get_frame(i) for i in range(start, stop, step)])
            key = (slice(None),) + key[1:]
        elif isinstance(frame_idx, (list, tuple, np.ndarray)):  # Support sequences of ints
            if not all(utils.isint(frame_idx)):
                raise TypeError('Sequences of frame indices must contain only integers')
            frames = np.array([self._get_frame(i) for i in frame_idx])
            key = (slice(None),) + key[1:]
        else:
            raise TypeError("Key's first element must be an int, slice, or sequence of ints")
        return frames[key]

    def _get_frame(self, frame_number) -> np.ndarray:
        """
        Provides access to random frames as fast as is reasonable when getting
        frames from a compressed video in python.

        Returns
        -------
        frame : np.ndarray
            The pixel values of the frame as a numpy array.
        """

        def decode_until(frame_number) -> np.ndarray:
            """
            Decode forward from the current frame in the stream until we get
            to the requested frame number.
            """
            target_pts = self.frame_number_to_pts(frame_number)
            for frame in self._frame_iterator:
                if frame.pts is None:
                    if self.verbose:
                        print('WARNING: Skipping a frame with no PTS.')
                    continue
                if frame.pts == target_pts:
                    frame = frame.to_ndarray(format='rgb24')
                    self._current_frame_number = frame_number
                    return frame
                if frame.pts > target_pts:
                    self._current_frame_number = self.pts_to_frame_number(frame.pts)
                    raise VideoSeekError(f'Frame with PTS {target_pts} not found after'
                                         f' seeking – current frame PTS: {frame.pts}')
                if type(self.verbose) is int and self.verbose:  # Set verbose=1 to use
                    print(f'Passing frame {self.pts_to_frame_number(frame.pts)} (PTS {frame.pts})'
                          f' while decoding to frame {frame_number} (PTS {target_pts})')
            raise VideoSeekError('Hit end of video before finding frame {frame_number} (PTS '
                                 f'{target_pts}). Last seen frame was {self._current_frame_number}'
                                 f' (PTS {self.frame_number_to_pts(self._current_frame_number)}')

        with self._lock:
            frame_number = self._normalize_frame_number(frame_number)
            if self._cache is not None and frame_number in self._cache:
                # Cache hit: mark as most recently used and hand back an
                # independent, writeable copy so callers can't corrupt the
                # cached frame by modifying what they receive.
                self._cache.move_to_end(frame_number)
                return self._cache[frame_number].copy()
            if (self._current_frame_number is None
                    or frame_number <= self._current_frame_number
                    or frame_number > self._current_frame_number + 100):
                # We seek to a few frames before the requested frame because
                # seeking has the undesirable behavior of sometimes landing
                # at a keyframe just after the requested frame, if a keyframe
                # exists one or two frames after the requested frame.
                seek_to_frame = max(0, frame_number - 3)
                seek_to_pts = self.frame_number_to_pts(seek_to_frame)
                if self.verbose:
                    print(f'Frame {frame_number} requested: Seeking to'
                          f' frame {seek_to_frame} (PTS {seek_to_pts})')
                # The seek call actually seeks to the closest keyframe before
                # target_pts, because it's not possible to seek directly to
                # non-keyframes due to video files being compressed.
                self.container.seek(seek_to_pts, any_frame=False,
                                    backward=True, stream=self.stream)
                self._frame_iterator = self.container.decode(self.stream)
            try:
                # Now we decode frames forward until we get to the requested frame
                image = decode_until(frame_number)
            except VideoSeekError as e:
                # If we fail on the first attempt, try seeking back
                # 30 frames (instead of 3) then decoding forward again.
                seek_to_frame = max(0, frame_number - 30)
                seek_to_pts = self.frame_number_to_pts(seek_to_frame)
                if self.verbose:
                    print(f'[WARNING] {e}')
                    print(f'[RETRY] Frame {frame_number} requested: Seeking'
                          f' to frame {seek_to_frame} (PTS {seek_to_pts})')
                self.container.seek(seek_to_pts, stream=self.stream)
                self._frame_iterator = self.container.decode(self.stream)
                # If this one fails too, we let its exception raise.
                # I haven't seen this ever fail, but who knows.
                image = decode_until(frame_number)

            if self.rotation not in [None, '0', 0]:
                image = np.rot90(image, k=-int(self.rotation) // 90)
            if self._cache is not None:
                self._cache_store(frame_number, image)
                # Return a copy, keeping the cached frame as a private canonical
                # copy that callers can never corrupt (see the hit path above).
                return image.copy()
            return image

    def _cache_store(self, frame_number: int, image: np.ndarray) -> None:
        """
        Insert a decoded frame into the cache as the most recently used entry,
        then evict least recently used entries until the cache is within its
        configured size limit.
        """
        self._cache[frame_number] = image
        self._cache.move_to_end(frame_number)
        if self._cache_max_bytes is not None:
            self._cache_bytes += image.nbytes
            while self._cache_bytes > self._cache_max_bytes and len(self._cache) > 1:
                _, evicted = self._cache.popitem(last=False)
                self._cache_bytes -= evicted.nbytes
        else:
            while len(self._cache) > self._cache_max_frames:
                self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        """
        Empty the in-memory frame cache. Has no effect if caching is disabled.
        """
        with self._lock:
            if self._cache is not None:
                self._cache.clear()
                self._cache_bytes = 0

    def _normalize_frame_number(self, frame_number: int) -> int:
        """
        Support negative indexing by converting negative frame numbers to
        positive ones, e.g. -1 becomes n_frames - 1, -2 becomes n_frames - 2, etc.
        """
        try:
            frame_number = int(frame_number)
        except ValueError:
            raise TypeError(f'Frame number must be castable to int but got "{frame_number}"')

        if frame_number < -self.n_frames:
            raise IndexError(f'Negative frame {frame_number} not in'
                             f' valid range [-{self.n_frames}, -1]')
        elif -self.n_frames <= frame_number and frame_number < 0:
            return frame_number + self.n_frames
        elif 0 <= frame_number and frame_number < self.n_frames:
            return frame_number
        elif self.n_frames <= frame_number:
            raise IndexError(f'Frame {frame_number} not in'
                             f' valid range [0, {self.n_frames-1}]')
        raise IndexError(f'Frame {frame_number} not understood')

    @property
    def first_frame(self):
        if self._first_frame is None:
            self._first_frame = self[0]
        return self._first_frame

    @property
    def shape(self):
        # (num_frames, height, width, channels)
        if self._shape is None:
            self._shape = (self.n_frames,) + self.first_frame.shape
        return self._shape

    @property
    def width(self):
        if self._width is None:
            self._width = self.first_frame.shape[1]
        return self._width

    @property
    def height(self):
        if self._height is None:
            self._height = self.first_frame.shape[0]
        return self._height

    @property
    def ndim(self):
        if self._ndim is None:
            self._ndim = len(self.shape)
        return self._ndim

    @property
    def dtype(self):
        if self._dtype is None:
            self._dtype = self.first_frame.dtype
        return self._dtype

    def __len__(self):
        return self.n_frames

    def close(self):
        self.container.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class _VideoStreamerTimeIndexer:
    """
    Helper exposing time-based indexing on a VideoStreamer via
    ``streamer.t[time_in_seconds]``.

    Returns a 3-tuple ``(image, timestamp, frame_index)`` for the frame
    that is "on screen" at the requested time:
      - ``image`` (np.ndarray): the frame's pixel data
      - ``timestamp`` (float): the frame's actual timestamp in seconds
      - ``frame_index`` (int): the frame's integer index in the video

    Each frame N is treated as occupying the half-open interval
    ``[frame_N_timestamp, frame_{N+1}_timestamp)``, and the last frame
    occupies one mean inter-frame interval past its timestamp. The lookup
    is therefore "largest frame timestamp less than or equal to the
    requested time", with a half-tick eps tolerance so that a request for
    the exact timestamp of frame N reliably returns frame N even when
    floating-point arithmetic has nudged the value slightly below.

    Valid times span ``[first_frame_timestamp, end_of_playback)``, where
    ``end_of_playback = first_frame_timestamp + duration``.

    Negative times wrap around relative to ``end_of_playback``, mirroring
    how negative integer indices wrap on ``VideoStreamer``: a time of
    ``-x`` resolves to ``end_of_playback - x``. Times more negative than
    ``-duration`` raise ``IndexError``.
    """
    def __init__(self, streamer: 'VideoStreamer'):
        self._streamer = streamer

    def __getitem__(self, time) -> Tuple[np.ndarray, float, int]:
        if not np.issubdtype(type(time), np.number):
            raise TypeError(f'Time index must be a number, got {type(time).__name__}')
        s = self._streamer
        time = float(time)
        first_frame_time = s.frame_number_to_time(0)
        end_of_playback = first_frame_time + s.duration
        if time < 0:
            if time < -s.duration:
                raise IndexError(f'Negative time {time:g} s is more than one'
                                 f' video duration ({s.duration:g} s) before'
                                 f' the end of the video.')
            time = time + end_of_playback
        if time >= end_of_playback:
            raise IndexError(f'No frame at time {time:g} s (playback'
                             f' ends at {end_of_playback:g} s).')
        if time < first_frame_time:
            raise IndexError(f'Time {time:g} s is before the first frame'
                             f' (timestamp {first_frame_time:g} s).')
        # Convert the requested time to a fractional PTS, then add half a
        # time_base tick. The eps shift means a request for the exact
        # timestamp of frame N returns frame N (rather than frame N-1) even
        # if float arithmetic has nudged the converted value below the
        # stored integer PTS.
        target_pts = time / float(s.time_base) + 0.5
        # Largest index i with s.frames_pts[i] <= target_pts. PTS values
        # increase with frame number whether s.frames_pts is the plain list of
        # a variable-framerate video or the ConstantFrameratePTS formula of a
        # constant-framerate one, and bisect only needs indexing and a length,
        # so the same search works for both.
        frame_number = bisect.bisect_right(s.frames_pts, target_pts) - 1
        # The bound checks above guarantee frame_number lands in
        # [0, n_frames - 1] under exact arithmetic. Clamp defensively for
        # the float-roundoff edge at exactly time == end_of_playback - eps.
        frame_number = max(0, min(frame_number, s.n_frames - 1))
        image = s._get_frame(frame_number)
        timestamp = s.frame_number_to_time(frame_number)
        return (image, timestamp, frame_number)


class AVVideoWriter:
    """
    Create a video writer object for saving frames to a video file.

    Example usage:
    >>> with VideoWriter('output.mp4', framerate=30) as writer:
    >>>     for i in range(n_frames):
    >>>         frame = do_something_to_build_an_image(i)
    >>>         writer.write(frame)

    This allows you to write a bunch of frames to a video file without
    ever needing to store all the frames in memory at once. If you have all
    your frames in memory already, you could use save_video(data, filename)

    To get the encoded video as bytes instead of writing a file, pass
    filename=None and take the return value of close():
    >>> writer = VideoWriter(None, format='mp4')
    >>> writer.write(frames)
    >>> video_bytes = writer.close()

    Parameters
    ----------
    filename : str or None
        The filename to save the video to. If None, the video is encoded to
        bytes, which close() returns and which are also kept on `self.bytes`.
    framerate : int or float, default 30
        The frame rate of the video. Ignored if frames are written with
        `time` (see write()).
    crf : int, default 23
        Constant Rate Factor for encoding quality (lower is better quality).
    compression_speed : str, default 'medium'
        Compression speed preset: 'ultrafast', 'superfast', 'veryfast',
        'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'.
    codec : str or None, default None
        The video codec to use for encoding. If None, automatically chosen based
        on the file extension (e.g. libx264 for .mp4, libvpx-vp9 for .webm).
        Accepts aliases like h264, h265, vp8, vp9, etc.
    overwrite : bool, default False
        Whether to overwrite the file if it already exists.
    format : 'mp4', 'mkv', 'avi', 'mov', 'webm', or None, default None
        The container format to encode to when filename=None. Defaults to
        'mp4' in that case. Must be None when a filename is given, since the
        file extension sets the format.
    """
    def __init__(self, filename, framerate=30, crf=23, compression_speed='medium',
                 codec: Literal['libx264', 'libx265', 'libvpx', 'libvpx-vp9', None] = None,
                 overwrite=False,
                 format: Literal['mp4', 'mkv', 'avi', 'mov', 'webm', None] = None):
        self.av = _import_av()
        if filename is None:
            if format is None:
                format = 'mp4'
            if format not in bytes_formats:
                raise ValueError(f'format must be one of {list(bytes_formats)}'
                                 f' but was {format!r}')
            extension = format
        else:
            if format is not None:
                raise ValueError('format is only used when filename=None. When'
                                 ' saving to a file, its extension sets the format.')
            filename = Path(filename).expanduser()
            extension = filename.suffix.lower().lstrip('.')
            if extension == 'gif':
                raise NotImplementedError('Saving to GIF format not yet implemented. '
                                          'Use save_video() instead.')
            if filename.exists() and not overwrite:
                raise FileExistsError(f'File {filename} already exists. '
                                      'Set overwrite=True to overwrite.')
        self.filename = filename
        # The encoded video, set by close() when filename is None
        self.bytes = None
        if not np.issubdtype(type(framerate), np.number):
            raise TypeError('framerate must be a number but got'
                            f' type {type(framerate)} instead')
        self._framerate = utils.limit_fraction(framerate)
        self.crf = crf
        self.compression_speed = compression_speed

        if codec is None:
            self.codec = extension_default_codecs.get(extension, 'libx264')
        else:
            self.codec = codec_aliases[codec.lower()]

        if filename is None:
            self._buffer = io.BytesIO()
            self.container = self.av.open(self._buffer, mode='w',
                                          format=bytes_formats[extension])
        else:
            self._buffer = None
            self.container = self.av.open(filename, mode='w')
        self.stream = self.container.add_stream(self.codec, rate=self._framerate)
        self.stream.pix_fmt = 'yuv420p'
        if self.codec in ('libvpx', 'libvpx-vp9'):
            self.stream.options = {'crf': str(crf), 'b:v': '0'}
        else:
            self.stream.options = {'crf': str(crf), 'preset': compression_speed}
        self._closed = False
        self.stream.width = 0
        self.stream.height = 0
        # np.pad spec that evens out odd frame dimensions for yuv420p; computed
        # from the first frame (see _pad_to_even) and reused for the rest.
        self._pad = None
        # Set by the first write() call.
        self._timestamped = None
        self._last_pts = None

    @property
    def framerate(self):
        return float(self._framerate)

    def _pad_to_even(self, frame):
        """
        Duplicate the bottom row and/or right column of an ndarray frame so its
        height and width are both even, as the yuv420p pixel format requires
        (an odd dimension otherwise fails to encode). The pad is worked out from
        the first frame and cached, so the info message prints only once.
        """
        if self.stream.pix_fmt != 'yuv420p':
            return frame
        if self._pad is None:
            height, width = frame.shape[:2]
            pad = [[0, 0] for _ in range(frame.ndim)]
            if height % 2 != 0:
                print('INFO: Height must be even for yuv420p pixel format but image'
                      f' has height {height}, so the bottom row will be duplicated.')
                pad[0][1] = 1
            if width % 2 != 0:
                print('INFO: Width must be even for yuv420p pixel format but image'
                      f' has width {width}, so the right column will be duplicated.')
                pad[1][1] = 1
            self._pad = pad
        if any(after for _, after in self._pad):
            return np.pad(frame, self._pad, mode='edge')
        return frame

    def write(self, frame, time: Optional[float] = None):
        """
        Write a frame to the video file.

        Parameters
        ----------
        frame : np.ndarray or av.VideoFrame
            An image of shape (H, W), (H, W, 3) or (H, W, 4), or a stack of
            images of shape (t, H, W, 3) or (t, H, W, 4).
        time : float, optional
            The frame's timestamp in seconds, for variable-framerate video.
            Must strictly increase, and be passed on every call or none.
        """
        if self._closed:
            raise RuntimeError('AVVideoWriter is closed, cannot write more frames.')
        if self._timestamped is None:
            self._timestamped = time is not None
            if self._timestamped:
                # Must be set before the first frame opens the encoder.
                self.stream.codec_context.time_base = timestamped_time_base
                self.stream.time_base = timestamped_time_base
        elif self._timestamped != (time is not None):
            raise ValueError('Pass `time` to every write() call or to none of them'
                             ' (the first write() call '
                             + ('did' if self._timestamped else 'did not') + ').')
        if not isinstance(frame, self.av.VideoFrame):
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            if frame.ndim == 4:
                if time is not None:
                    raise ValueError('`time` applies to a single frame, so write'
                                     ' a stack of images one frame at a time.')
                for i in range(frame.shape[0]):
                    self.write(frame[i])
                return
            frame = self._pad_to_even(frame)
            if frame.ndim == 3 and frame.shape[-1] == 3:
                frame = self.av.VideoFrame.from_ndarray(frame, format='rgb24')
            elif frame.ndim == 3 and frame.shape[-1] == 4:
                # While some video codecs support an alpha channel, most don't,
                # so for now we're just going to ignore the alpha channel
                frame = self.av.VideoFrame.from_ndarray(frame[..., :3], format='rgb24')
            elif frame.ndim == 2:
                frame = self.av.VideoFrame.from_ndarray(frame, format='gray')
            else:
                raise ValueError('Frame must have shape (H, W) (H, W, 3) (H, W, 4)'
                                 f' (t, H, W, 3) or (t, H, W, 4) but was {frame.shape}')
        if self.stream.width == 0:
            self.stream.width = frame.width
        if self.stream.height == 0:
            self.stream.height = frame.height
        if time is not None:
            pts = _round_half_away_from_zero(Fraction(time) / timestamped_time_base)
            if self._last_pts is not None and pts <= self._last_pts:
                raise ValueError(f'Frame time {time} s is not after the previous'
                                 " frame's time. Timestamps must strictly increase.")
            self._last_pts = pts
            frame.pts = pts
            frame.time_base = timestamped_time_base
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
            del packet
        del frame

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> Optional[bytes]:
        """
        Finish encoding and close the video. Returns the encoded video as
        bytes if filename was None, otherwise None.
        """
        if self._closed:
            return self.bytes
        # Flush stream
        for packet in self.stream.encode():
            self.container.mux(packet)
        del packet
        # Close and delete everything
        self.stream = None
        self.container.close()
        self.container = None
        if self._buffer is not None:
            self.bytes = self._buffer.getvalue()
            self._buffer = None
        self._closed = True
        import gc
        gc.collect()
        return self.bytes


class FFmpegVideoWriter:
    """
    Create a video writer object for saving frames to a video file using FFmpeg subprocess.

    Example usage:
    >>> with FFmpegVideoWriter('output.mp4', framerate=30) as writer:
    >>>     for i in range(n_frames):
    >>>         frame = do_something_to_build_an_image(i)
    >>>         writer.write(frame)

    This allows you to write a bunch of frames to a video file without
    ever needing to store all the frames in memory at once. If you have all
    your frames in memory already, you could use save_video(data, filename)

    To get the encoded video as bytes instead of writing a file, pass
    filename=None and take the return value of close():
    >>> writer = VideoWriter(None, format='mp4')
    >>> writer.write(frames)
    >>> video_bytes = writer.close()

    Parameters
    ----------
    filename : str or None
        The filename to save the video to. If None, the video is encoded to
        bytes, which close() returns and which are also kept on `self.bytes`.
    framerate : int or float, default 30
        The frame rate of the video.
    crf : int, default 23
        Constant Rate Factor for encoding quality (lower is better quality).
    compression_speed : str, default 'medium'
        Compression speed preset: 'ultrafast', 'superfast', 'veryfast',
        'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'.
    codec : str or None, default None
        The video codec to use for encoding. If None, automatically chosen based
        on the file extension (e.g. libx264 for .mp4, libvpx-vp9 for .webm).
        Accepts aliases like h264, h265, vp8, vp9, etc.
    overwrite : bool, default False
        Whether to overwrite the file if it already exists.
    format : 'mp4', 'mkv', 'avi', 'mov', 'webm', or None, default None
        The container format to encode to when filename=None. Defaults to
        'mp4' in that case. Must be None when a filename is given, since the
        file extension sets the format.
    """
    def __init__(self, filename, framerate=30, crf=23, compression_speed='medium',
                 codec: Literal['libx264', 'libx265', 'libvpx', 'libvpx-vp9', None] = None,
                 overwrite=False,
                 format: Literal['mp4', 'mkv', 'avi', 'mov', 'webm', None] = None):
        if filename is None:
            if format is None:
                format = 'mp4'
            if format not in bytes_formats:
                raise ValueError(f'format must be one of {list(bytes_formats)}'
                                 f' but was {format!r}')
            extension = format
        else:
            if format is not None:
                raise ValueError('format is only used when filename=None. When'
                                 ' saving to a file, its extension sets the format.')
            filename = Path(filename).expanduser()
            extension = filename.suffix.lower().lstrip('.')
            if filename.exists() and not overwrite:
                raise FileExistsError(f'File {filename} already exists. '
                                      'Set overwrite=True to overwrite.')
        self.filename = filename
        # The encoded video, set by close() when filename is None
        self.bytes = None
        if not np.issubdtype(type(framerate), np.number):
            raise TypeError('framerate must be a number but got'
                            f' type {type(framerate)} instead')
        self._framerate = utils.limit_fraction(framerate)
        self.crf = crf
        self.compression_speed = compression_speed

        if codec is None:
            self.codec = extension_default_codecs.get(extension, 'libx264')
        else:
            self.codec = codec_aliases[codec.lower()]
        self._extension = extension
        # When encoding to bytes, ffmpeg writes to this temporary file, which
        # close() reads back and deletes. A pipe to stdout would avoid touching
        # disk, but mp4 can't be written to a pipe without fragmenting it.
        self._temporary_path = None

        # Initialize process state
        self._process = None
        self._stdin = None
        self._stderr_thread = None
        self._stderr_buffer = b''
        self._closed = False
        self._width = None
        self._height = None
        self._pixel_format_in = None
        self._pixel_format_out = 'yuv420p'

    @property
    def framerate(self):
        return float(self._framerate)

    def _initialize_process(self, width, height, pixel_format_in):
        """Initialize the FFmpeg subprocess for video encoding"""
        self._width = width
        self._height = height
        self._pixel_format_in = pixel_format_in

        # Check if width or height needs to be padded to an even value
        if (self._pixel_format_out == 'yuv420p'
                and (width % 2 != 0 or height % 2 != 0)):
            pad = [[0, 0], [0, 0]]  # [h, w]
            pad += [[0, 0]] if pixel_format_in == 'rgb24' else []
            if height % 2 != 0:
                print('INFO: Height must be even for yuv420p pixel format but image'
                      f' has height {height}, so the bottom row will be duplicated.')
                pad[0][1] = 1
                height = height + 1
            if width % 2 != 0:
                print('INFO: Width must be even for yuv420p pixel format but image'
                      f' has width {width}, so the right column will be duplicated.')
                pad[1][1] = 1
                width = width + 1
            self._pad = pad
        else:
            self._pad = 0

        command = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-nostats',
            '-y',  # Overwrite output
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', pixel_format_in,  # Input pixel format comes before -i
            '-r', str(self._framerate),
            '-i', '-',  # Read from stdin
            '-an',  # No audio
            '-c:v', self.codec,
            '-pix_fmt', self._pixel_format_out,  # Output pixel format after -i
            '-crf', str(self.crf),
        ]
        if self.codec in ('libvpx', 'libvpx-vp9'):
            # VP8/VP9 need -b:v 0 for constant quality (CRF) mode
            command += ['-b:v', '0']
        else:
            command += ['-preset', self.compression_speed]
        if self.filename is None:
            file_descriptor, temporary_path = tempfile.mkstemp(
                suffix='.' + self._extension)
            os.close(file_descriptor)
            self._temporary_path = Path(temporary_path)
            command.append(str(self._temporary_path))
        else:
            command.append(str(self.filename))

        # Start FFmpeg process
        _check_ffmpeg_available('ffmpeg')
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE  # Capture errors
        )
        self._stdin = self._process.stdin

        # Drain ffmpeg's stderr in a background thread so a verbose or
        # erroring ffmpeg can't fill the pipe buffer (~64 KB on Linux, smaller
        # on Windows) and deadlock our writes to stdin.
        def _drain_stderr():
            self._stderr_buffer = self._process.stderr.read()
        self._stderr_thread = threading.Thread(target=_drain_stderr,
                                               daemon=True)
        self._stderr_thread.start()

    def write(self, frame, time: Optional[float] = None):
        """Write a frame to the video file. `time` is only supported by AVVideoWriter."""
        if time is not None:
            raise NotImplementedError('FFmpegVideoWriter writes constant-framerate'
                                      ' video only. Use AVVideoWriter to write'
                                      ' frames with explicit timestamps.')
        if self._closed:
            raise RuntimeError('FFmpegVideoWriter is closed, cannot write more frames.')

        # Convert frame to numpy array if needed
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        # Handle batch frames (4D arrays)
        if frame.ndim == 4:
            for i in range(frame.shape[0]):
                self.write(frame[i])
            return

        # Determine frame format and dimensions
        if frame.ndim == 2:  # Grayscale
            height, width = frame.shape
            pixel_format = 'gray'
        elif frame.ndim == 3:
            height, width, channels = frame.shape
            if channels == 3:  # RGB
                pixel_format = 'rgb24'
            elif channels == 4:  # RGBA - ignore alpha
                frame = frame[..., :3]
                pixel_format = 'rgb24'
            else:
                raise ValueError(f'Unsupported channel count: {channels}')
        else:
            raise ValueError('Frame must have shape (H, W) (H, W, 3) (H, W, 4)'
                             f' (t, H, W, 3) or (t, H, W, 4) but was {frame.shape}')

        # Initialize FFmpeg process on first frame
        if self._process is None:
            self._initialize_process(width, height, pixel_format)

        # Validate frame dimensions
        if (width, height) != (self._width, self._height):
            raise ValueError(f'Cannot write image of size (w={width}, h={height}) to video'
                             f' already containing images of size {(self._width, self._height)}')

        if self._pad:
            frame = np.pad(frame, self._pad, mode='edge')

        # Convert frame to bytes and write to FFmpeg
        frame_bytes = frame.tobytes()
        self._stdin.write(frame_bytes)

    def close(self) -> Optional[bytes]:
        """
        Finish encoding and close the video. Returns the encoded video as
        bytes if filename was None, otherwise None.
        """
        if self._closed:
            return self.bytes

        try:
            # Close stdin to signal end of input
            if self._stdin:
                self._stdin.close()

            # Wait for FFmpeg to finish
            if self._process:
                return_code = self._process.wait()
                if self._stderr_thread is not None:
                    self._stderr_thread.join()
                stderr_data = self._stderr_buffer

                # Check for errors
                if return_code != 0:
                    raise RuntimeError(f'FFmpeg failed with return code {return_code}:'
                                       f' {stderr_data.decode()}')
            if self.filename is None:
                if self._temporary_path is None:
                    # No frames were written, so ffmpeg never ran
                    self.bytes = b''
                else:
                    self.bytes = self._temporary_path.read_bytes()
        finally:
            # Clean up process references
            self._process = None
            self._stdin = None
            self._stderr_thread = None
            if self._temporary_path is not None:
                self._temporary_path.unlink(missing_ok=True)
                self._temporary_path = None
            self._closed = True
        return self.bytes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Alias the preferred video writer class
VideoWriter = FFmpegVideoWriter


def save_video(data, filename=None, time_axis=0, color_axis=None, overwrite=False,
               dim_order='yx', framerate=30, crf=23, compression_speed='medium',
               progress_bar=True, codec: Literal['libx264', 'libx265', 'libvpx',
                                                 'libvpx-vp9', None] = None,
               format: Literal['mp4', 'mkv', 'avi', 'mov', 'webm', 'gif', None] = None
               ) -> Optional[bytes]:
    """
    Save a 3D numpy array of greyscale values OR a 4D numpy array of RGB values as a video

    With filename=None, the video is not saved to a file and is instead
    returned as bytes, e.g. `video_bytes = save_video(data, format='webm')`.

    Follows the PyAV cookbook section on generating video from numpy arrays:
    https://pyav.basswood-io.com/docs/develop/cookbook/numpy.html#generating-video

    Parameters
    ----------
    data : numpy.ndarray or list of images
        A 3D (grayscale) or 4D (RGB) numpy array of pixel values.

    filename : str or None, default None
        The filename to save the video to. If None, the encoded video is
        returned as bytes instead.

    time_axis : int, default 0
        The axis of the data array that will be played as time in the video.

    color_axis : int or None, default None
        If not None, specifies the axis of the color channels (e.g., -1 for last axis,
        1 for second axis).
        If None, data must be 3D (greyscale) or 4D with one length 3 axis (RGB)
        or length 4 axis (RGBA, whose alpha channel will be dropped).

    overwrite : bool, default False
        Whether to overwrite the file if it already exists.

    dim_order : 'yx' (default) or 'xy'
        The order of the spatial dimensions in the input data.

    framerate : int, default 30
        The frame rate of the video.

    crf : int, default 23
        Constant Rate Factor that specifies amount of lossiness allowed in compression.
        Lower values produce better quality, larger videos. crf=17 has no human-visible
        compression artifacts. File size approximately doubles/halves each time you
        add/subtract 6 from crf, so crf=17 produces files about twice as large as
        the default crf=23.

    compression_speed : str, default 'medium'
        Compression speed preset: 'ultrafast', 'superfast', 'veryfast',
        'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'.

    progress_bar : bool, default True
        If True, display a progress bar.

    codec : str or None, default None
        The video codec to use for encoding. If None, automatically chosen based
        on the file extension (e.g. libx264 for .mp4, libvpx-vp9 for .webm).
        Accepts aliases like h264, h265, vp8, vp9, etc.

    format : 'mp4', 'mkv', 'avi', 'mov', 'webm', 'gif', or None, default None
        The container format to encode to when filename=None. Defaults to
        'mp4' in that case. Must be None when a filename is given, since the
        file extension sets the format.

    Returns
    -------
    bytes or None
        The encoded video if filename is None, otherwise None.
    """

    if filename is None:
        if format is None:
            format = 'mp4'
        if format not in supported_extensions:
            raise ValueError(f'format must be one of {supported_extensions}'
                             f' but was {format!r}')
        extension = format
    else:
        if format is not None:
            raise ValueError('format is only used when filename=None. When'
                             ' saving to a file, its extension sets the format.')
        filename = str(filename)
        if filename.split('.')[-1].lower() not in supported_extensions:
            filename += '.mp4'
        filename = Path(filename).expanduser()
        if filename.exists() and not overwrite:
            raise FileExistsError(f'File {filename} already exists. '
                                  'Set overwrite=True to overwrite.')
        extension = filename.suffix.lower().lstrip('.')

    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if color_axis is None and data.ndim == 4:
        color_axis = utils.find_channel_axis(data, possible_channel_lengths=[3, 4])
        if color_axis is None:
            raise ValueError('4D input data must have an RGB (length 3) or '
                             'RGBA (length 4) axis.')
    if color_axis is not None:
        if data.ndim != 4:
            raise ValueError('Input data must be 4D when color_axis is specified.')
        # Move time axis to 0, color axis to -1
        data = np.moveaxis(data, time_axis, 0)
        if color_axis != -1:
            data = np.moveaxis(data, color_axis, -1)
        if 'xy' in dim_order:
            data = data.swapaxes(1, 2)
        n_frames = data.shape[0]
        height, width, channels = data.shape[1:]
        if channels == 4:
            # While some video codecs support an alpha channel, most don't,
            # so we drop it and keep just the RGB channels
            data = data[..., :3]
            channels = 3
        if channels != 3:
            raise ValueError(f'Color video must have 3 channels (RGB) but had {channels}.')
    else:
        if data.ndim != 3:
            raise ValueError('Input data must be 3D when color_axis is not specified.')
        data = np.moveaxis(data, time_axis, 0)
        if 'xy' in dim_order:
            data = data.swapaxes(1, 2)
        n_frames = data.shape[0]
        height, width = data.shape[1:]

    if extension == 'gif':
        # We make gifs with PIL instead of using FFmpeg
        from PIL import Image
        pil_images = [Image.fromarray(frame) for frame in data]

        # GIF frame delays must be multiples of 10ms, so GIF doesn't
        # natively suppport framerates like 30 fps which require an
        # inter-frame delay of 33.33 ms. To support arbitrary
        # framerates, we copy the solution used in ffmpeg which is to
        # use variable inter-frame intervals: 30 ms some frames and 40 ms
        # others to achieve an average interval of 33.33 ms (for the
        # example of 30 fps). Implementation of alternating delays:
        ideal_delay_cs = 100 / framerate
        lo = int(ideal_delay_cs)
        hi = lo + 1
        durations_ms = []
        accumulated = 0.0
        for _ in range(len(pil_images)):
            accumulated += ideal_delay_cs
            if accumulated >= hi:
                durations_ms.append(hi * 10)
                accumulated -= hi
            else:
                durations_ms.append(lo * 10)
                accumulated -= lo

        output = io.BytesIO() if filename is None else filename
        pil_images[0].save(output, format='GIF', save_all=True,
                           append_images=pil_images[1:],
                           duration=durations_ms, loop=0)
        return output.getvalue() if filename is None else None

    with VideoWriter(filename, framerate=framerate, crf=crf,
                     compression_speed=compression_speed, codec=codec,
                     overwrite=overwrite,
                     format=extension if filename is None else None) as writer:
        for frame_i in tqdm(range(n_frames), total=n_frames,
                            desc='Saving video', disable=not progress_bar):
            writer.write(data[frame_i])
    return writer.bytes


def _get_rotation_from_metadata(filename):
    """
    Get rotation metadata from a video file.

    We could use PyAV to do this perhaps faster, but for an already fast operation
    like this, we stick with ffprobe to avoid the memory leaks in PyAV.
    """
    if shutil.which('ffprobe') is None:
        return None
    cmd = ['ffprobe',
           '-v', 'quiet',
           '-select_streams', 'v:0',
           '-show_entries', 'stream',
           '-of', 'json',
           str(filename)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return None

    try:
        stream = json.loads(result.stdout)['streams'][0]
        if 'tags' in stream and 'rotate' in stream['tags']:
            # Older versions of ffprobe return rotation metadata in this location,
            # as a string, indicating **clockwise** rotation by this many degrees.
            return int(stream['tags']['rotate'])
        for side_data in stream.get('side_data_list', []):
            # Newer versions of ffprobe return rotation metadata in this location,
            # as an integer, indicating **counter-clockwise** rotation by this many
            # degrees. We negate it to convert to clockwise rotation so that this
            # function returns the same value for all versions of ffprobe.
            if 'rotation' in side_data:
                return -side_data['rotation']
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return None
