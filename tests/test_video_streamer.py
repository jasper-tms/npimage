"""
Tests for VideoStreamer, particularly handling of HEVC videos with
negative-PTS priming packets (common in iPhone recordings).
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

import npimage
from npimage.vidio import _parse_cache_size

TESTS_DIR = Path(__file__).parent
SPINNING_MP4 = TESTS_DIR / 'table-tennis-emoji-spinning.mp4'


@pytest.fixture
def spinning_streamer():
    """A VideoStreamer over the bundled 8-frame, 4 fps test video."""
    vid = npimage.VideoStreamer(str(SPINNING_MP4))
    yield vid
    vid.close()


@pytest.mark.slow
def test_negative_pts_priming_packet(tmp_path):
    """
    VideoStreamer correctly handles HEVC videos where the first packet has a
    negative PTS (a "priming" packet used for B-frame prediction that doesn't
    produce a decoded frame).

    This is common in iPhone HEVC recordings. The bug was that the index was
    built from demuxed packets, which included the priming packet's negative
    PTS. When trying to access frame 0, it would seek to that negative PTS,
    which no decoded frame ever has, causing a VideoSeekError.
    """
    import av

    video_path = tmp_path / 'test_negative_pts.mp4'

    # Create a small HEVC video with a negative-PTS priming packet.
    # -output_ts_offset shifts timestamps so the first keyframe packet gets a
    # negative PTS, mimicking iPhone HEVC behavior.
    result = subprocess.run([
        'ffmpeg', '-y',
        '-f', 'lavfi', '-i', 'color=c=red:size=320x240:rate=30:d=1',
        '-c:v', 'libx265', '-preset', 'ultrafast',
        '-bf', '2', '-x265-params', 'bframes=2',
        '-tag:v', 'hvc1',
        '-output_ts_offset', '-0.033',
        str(video_path),
    ], capture_output=True, text=True)
    assert result.returncode == 0, f'ffmpeg failed: {result.stderr}'

    # Verify the test video actually has the problematic structure: more
    # packets than decoded frames due to the priming packet.
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        pkt_pts = sorted(
            p.pts for p in container.demux(stream) if p.pts is not None
        )
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        frame_pts = [f.pts for f in container.decode(stream)]

    assert len(pkt_pts) > len(frame_pts), (
        f'Test video should have more packets ({len(pkt_pts)}) than decoded '
        f'frames ({len(frame_pts)}). The test setup may need updating if '
        f'ffmpeg behavior has changed.'
    )
    assert pkt_pts[0] < 0, f'First packet PTS should be negative, got {pkt_pts[0]}'

    vid = npimage.VideoStreamer(str(video_path))

    assert vid.n_frames == len(frame_pts)

    frame = vid[0]
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (240, 320, 3)

    last_frame = vid[vid.n_frames - 1]
    assert isinstance(last_frame, np.ndarray)


def test_duration_matches_n_frames_over_fps():
    """
    `duration` follows the View-B convention: each frame occupies one
    inter-frame interval on screen, so total duration is
    ``n_frames / fps`` (equivalent to ``n_frames * timestep`` for CFR).

    The bundled test video has 8 frames at 4 fps. View A would give
    7 * 0.25 = 1.75 s; View B gives 8 * 0.25 = 2.00 s, matching the
    duration ffprobe reports for the same file.
    """
    vid = npimage.VideoStreamer(str(SPINNING_MP4))
    try:
        assert vid.n_frames == 8
        assert vid.fps == 4.0
        assert vid.duration == pytest.approx(2.0)
        # Self-consistency: duration, fps, and n_frames form a closed triple.
        assert vid.duration == pytest.approx(vid.n_frames / vid.fps)
        # For CFR specifically, duration should also equal n_frames * timestep.
        assert vid.duration == pytest.approx(vid.n_frames * vid.timestep)
    finally:
        vid.close()


def test_duration_matches_ffprobe_on_cfr():
    """
    For constant-framerate videos, our `duration` should agree with
    ffprobe to within a small tolerance.
    """
    result = subprocess.run([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(SPINNING_MP4),
    ], capture_output=True, text=True)
    assert result.returncode == 0, f'ffprobe failed: {result.stderr}'
    ffprobe_duration = float(result.stdout.strip())

    vid = npimage.VideoStreamer(str(SPINNING_MP4))
    try:
        assert vid.duration == pytest.approx(ffprobe_duration, abs=1e-6)
    finally:
        vid.close()


def test_t_indexer_exact_timestamp_returns_that_frame(spinning_streamer):
    """vid.t[exact_frame_time] returns that frame, not the next one."""
    vid = spinning_streamer
    for i in range(vid.n_frames):
        _, _, idx = vid.t[vid.frame_number_to_time(i)]
        assert idx == i


def test_t_indexer_returns_3tuple(spinning_streamer):
    """The return value is a plain (image, timestamp, frame_index) tuple."""
    vid = spinning_streamer
    result = vid.t[0.5]
    assert type(result) is tuple
    assert len(result) == 3
    image, timestamp, idx = result
    assert isinstance(image, np.ndarray)
    assert image.shape == vid.shape[1:]
    assert isinstance(timestamp, float)
    assert isinstance(idx, int)


def test_t_indexer_supports_tuple_unpacking(spinning_streamer):
    """Callers can unpack the result positionally as (image, time, index)."""
    vid = spinning_streamer
    image, timestamp, idx = vid.t[0.75]
    assert isinstance(image, np.ndarray)
    assert timestamp == pytest.approx(0.75)
    assert idx == 3


def test_t_indexer_eps_below_timestamp_still_returns_that_frame(spinning_streamer):
    """A request slightly under a frame's timestamp still returns that frame.

    This is the float-eps-tolerance behavior: small negative drift from the
    exact timestamp must not cause the indexer to fall back to the previous
    frame. (We only test that direction; small drift above the timestamp is
    correctly resolved to the next frame, see the next test.)
    """
    vid = spinning_streamer
    t5 = vid.frame_number_to_time(5)
    assert vid.t[t5 - 1e-12][2] == 5
    assert vid.t[t5 - 1e-9][2] == 5


def test_t_indexer_just_above_timestamp_stays_on_that_frame(spinning_streamer):
    """A request just inside a frame's display interval returns that frame.

    Under View B each frame is on screen during ``[pts_N, pts_{N+1})``, so
    a time 1 ms after frame 5's timestamp is still inside frame 5's
    display interval (which spans 250 ms on this 4 fps video).
    """
    vid = spinning_streamer
    t5 = vid.frame_number_to_time(5)
    assert vid.t[t5 + 1e-3][2] == 5


def test_t_indexer_between_frames_returns_earlier_frame(spinning_streamer):
    """A time strictly between two frames returns the earlier (still-displayed) one."""
    vid = spinning_streamer
    midpoint = (vid.frame_number_to_time(2) + vid.frame_number_to_time(3)) / 2
    assert vid.t[midpoint][2] == 2


def test_t_indexer_just_before_next_frame_still_returns_current(spinning_streamer):
    """A time meaningfully shy of frame N+1's timestamp still returns frame N.

    This is the "scrub to the deep tail of frame N's display interval"
    case — under View B the next frame hasn't appeared yet. The delta
    must be larger than half a time_base tick or the eps tolerance will
    snap the request up to frame N+1.
    """
    vid = spinning_streamer
    t6 = vid.frame_number_to_time(6)
    # 1 ms before frame 6 starts (well above half-tick eps) -> still frame 5
    assert vid.t[t6 - 1e-3][2] == 5


def test_t_indexer_zero_returns_first_frame(spinning_streamer):
    """Time 0.0 resolves to frame 0."""
    assert spinning_streamer.t[0.0][2] == 0


def test_t_indexer_negative_time_wraps_relative_to_end(spinning_streamer):
    """Negative times wrap, mirroring the integer indexer's negative semantics.

    ``vid.t[-x]`` resolves to ``end_of_playback - x``. On the 8-frame,
    4 fps test video (frames at 0.00, 0.25, ..., 1.75; duration 2.00 s)
    a request for -0.25 s lands at absolute time 1.75 — frame 7's
    timestamp, the start of its display interval, so frame 7. -1.0 s
    lands at 1.0 = frame 4. -duration lands at 0.0 = frame 0.
    """
    vid = spinning_streamer
    assert vid.t[-0.25][2] == 7
    assert vid.t[-1.0][2] == 4
    assert vid.t[-vid.duration][2] == 0


def test_t_indexer_negative_time_eps_tolerance(spinning_streamer):
    """The eps shift also applies after negative-time wrap.

    A wrapped time that equals an exact frame timestamp should resolve to
    that frame, not the previous one, due to float drift. -0.25 s wraps
    to exactly 1.75 s = frame 7's timestamp.
    """
    vid = spinning_streamer
    assert vid.t[-0.25][2] == 7
    assert vid.t[-0.25 - 1e-12][2] == 7


def test_t_indexer_too_negative_raises(spinning_streamer):
    """Negative times more negative than -duration raise IndexError."""
    vid = spinning_streamer
    with pytest.raises(IndexError):
        vid.t[-vid.duration - 1e-3]
    with pytest.raises(IndexError):
        vid.t[-100.0]


def test_t_indexer_past_last_frame_timestamp_still_returns_last_frame(spinning_streamer):
    """Times within the last frame's display interval still return it.

    Under View B the last frame is on screen during
    ``[last_frame_timestamp, end_of_playback)``, so a request 1 ms past
    the last frame's timestamp (well inside the 250 ms last interval) is
    still the last frame, not an error.
    """
    vid = spinning_streamer
    last = vid.frame_number_to_time(vid.n_frames - 1)
    assert vid.t[last + 1e-3][2] == vid.n_frames - 1
    # Just shy of end_of_playback should also still be the last frame.
    assert vid.t[vid.duration - 1e-6][2] == vid.n_frames - 1


def test_t_indexer_past_end_of_playback_raises(spinning_streamer):
    """Times at or past end_of_playback raise IndexError."""
    vid = spinning_streamer
    with pytest.raises(IndexError):
        vid.t[vid.duration]
    with pytest.raises(IndexError):
        vid.t[vid.duration + 1e-3]
    with pytest.raises(IndexError):
        vid.t[vid.duration + 100.0]


def test_t_indexer_image_matches_frame_lookup(spinning_streamer):
    """The image returned by t[time] is identical to streamer[returned_index]."""
    vid = spinning_streamer
    image, _, idx = vid.t[0.6]
    assert np.array_equal(image, vid[idx])


def test_t_indexer_rejects_non_numeric(spinning_streamer):
    """Non-numeric keys raise TypeError rather than failing deeper in."""
    vid = spinning_streamer
    with pytest.raises(TypeError):
        vid.t['0.5']
    with pytest.raises(TypeError):
        vid.t[None]


def test_t_indexer_variable_framerate_uses_bisect(spinning_streamer):
    """The variable-framerate code path correctly resolves between frames.

    The bundled test video is constant-framerate, so we coerce the streamer
    into the variable-framerate branch by swapping in a list of PTS values
    and flipping `_framerate`. This exercises the ``bisect_right`` path in
    ``_VideoStreamerTimeIndexer.__getitem__`` against the same View B
    semantics.
    """
    vid = spinning_streamer
    original_framerate = vid._framerate
    original_frames_pts = vid.frames_pts
    try:
        vid._framerate = 'variable'
        # Materialize the range as a list so bisect operates on real ints.
        vid.frames_pts = list(original_frames_pts)

        # Exact timestamp -> that frame.
        t4 = vid.frame_number_to_time(4)
        assert vid.t[t4][2] == 4
        # Just below -> still that frame (eps tolerance).
        assert vid.t[t4 - 1e-9][2] == 4
        # Between two frames -> the earlier one (still on screen).
        midpoint = (vid.frame_number_to_time(1) + vid.frame_number_to_time(2)) / 2
        assert vid.t[midpoint][2] == 1
        # Just inside the last frame's display interval -> last frame.
        assert vid.t[vid.duration - 1e-6][2] == vid.n_frames - 1
        # At or past end_of_playback -> IndexError.
        with pytest.raises(IndexError):
            vid.t[vid.duration]
    finally:
        vid._framerate = original_framerate
        vid.frames_pts = original_frames_pts


# ----------------------------------------------------------------------------
# Frame caching
# ----------------------------------------------------------------------------

def test_parse_cache_size_disabled():
    """None and 0 (in either int or string form) disable caching."""
    assert _parse_cache_size(None) == (None, None)
    assert _parse_cache_size(0) == (None, None)
    assert _parse_cache_size('0') == (None, None)
    assert _parse_cache_size('0MB') == (None, None)


def test_parse_cache_size_frame_counts():
    """Bare ints and bare numeric strings are interpreted as frame counts."""
    assert _parse_cache_size(64) == (64, None)
    assert _parse_cache_size('64') == (64, None)


def test_parse_cache_size_byte_budgets():
    """Strings with units are interpreted as memory budgets (powers of 1024)."""
    assert _parse_cache_size('1B') == (None, 1)
    assert _parse_cache_size('1KB') == (None, 1024)
    assert _parse_cache_size('256MB') == (None, 256 * 1024 ** 2)
    assert _parse_cache_size('1.5GB') == (None, int(1.5 * 1024 ** 3))
    # Binary-prefix spellings are accepted as aliases.
    assert _parse_cache_size('1MiB') == (None, 1024 ** 2)
    # Whitespace and case are tolerated.
    assert _parse_cache_size('  2 gb ') == (None, 2 * 1024 ** 3)


def test_parse_cache_size_invalid():
    """Bad units, non-whole frame counts, bools, and negatives all raise."""
    with pytest.raises(ValueError):
        _parse_cache_size('abc')
    with pytest.raises(ValueError):
        _parse_cache_size('10TB')
    with pytest.raises(ValueError):
        _parse_cache_size('1.5')  # fractional frame count
    with pytest.raises(ValueError):
        _parse_cache_size(-5)
    with pytest.raises(TypeError):
        _parse_cache_size(True)


def test_cache_returns_correct_frames():
    """Cached frames are pixel-identical to uncached ones, even out of order."""
    reference = npimage.VideoStreamer(str(SPINNING_MP4), cache_size=None)
    cached = npimage.VideoStreamer(str(SPINNING_MP4), cache_size=4)
    try:
        for i in [0, 5, 2, 5, 0, 7, 3, 7, 1]:
            assert np.array_equal(reference[i], cached[i]), f'mismatch at frame {i}'
    finally:
        reference.close()
        cached.close()


def test_cache_disabled_returns_writeable_frames():
    """With caching off, returned frames are writeable (unchanged behavior)."""
    vid = npimage.VideoStreamer(str(SPINNING_MP4), cache_size=None)
    try:
        assert vid._cache is None
        assert vid[0].flags.writeable
    finally:
        vid.close()


def test_cache_enabled_returns_writeable_independent_frames():
    """
    With caching on, returned frames are still writeable, and mutating a
    returned frame must not corrupt the cached copy handed to later callers.

    This exercises both the cache-miss return path (the first read of a frame)
    and the cache-hit return path (subsequent reads): mutating the frame
    returned from either must leave the cache untouched.
    """
    vid = npimage.VideoStreamer(str(SPINNING_MP4), cache_size=4)
    try:
        first = vid[0]                  # cache miss: this return is freshly decoded
        assert first.flags.writeable
        original = first.copy()
        # Guard against a vacuous test: the frame must have content to clobber.
        assert original.any()

        first[:] = 0                    # mutate the miss-path frame in place
        second = vid[0]                 # cache hit: must be the pristine frame
        assert np.array_equal(second, original), (
            'Mutating the frame returned on a cache miss corrupted the cache.'
        )

        second[:] = 0                   # now mutate the hit-path frame in place
        third = vid[0]                  # another cache hit: still pristine
        assert np.array_equal(third, original), (
            'Mutating the frame returned on a cache hit corrupted the cache.'
        )

        # Every read handed back an independent array object.
        assert first is not second
        assert second is not third
        assert first is not third
    finally:
        vid.close()


def test_cache_lru_eviction_by_frame_count():
    """The least recently used frames are evicted once the cache is full."""
    vid = npimage.VideoStreamer(str(SPINNING_MP4), cache_size=4)
    try:
        for i in range(5):  # fill past capacity: 0 should be evicted
            _ = vid[i]
        assert list(vid._cache.keys()) == [1, 2, 3, 4]
        # Touch frame 1 (most recently used), then add two more.
        _ = vid[1]
        _ = vid[5]
        _ = vid[6]
        # 2 and 3 (oldest untouched) evicted; 1 retained because it was reused.
        assert 1 in vid._cache
        assert 2 not in vid._cache
        assert 3 not in vid._cache
    finally:
        vid.close()


def test_cache_respects_byte_budget():
    """A memory-budgeted cache never holds more bytes than allowed."""
    reference = npimage.VideoStreamer(str(SPINNING_MP4), cache_size=None)
    frame_bytes = reference[0].nbytes
    # Budget for exactly two frames.
    vid = npimage.VideoStreamer(str(SPINNING_MP4), cache_size=f'{2 * frame_bytes}B')
    try:
        for i in range(5):
            _ = vid[i]
        assert len(vid._cache) == 2
        assert vid._cache_bytes <= 2 * frame_bytes
    finally:
        reference.close()
        vid.close()


def test_clear_cache():
    """clear_cache empties the cache and resets the byte tally."""
    vid = npimage.VideoStreamer(str(SPINNING_MP4), cache_size='64MB')
    try:
        for i in range(4):
            _ = vid[i]
        assert len(vid._cache) > 0
        vid.clear_cache()
        assert len(vid._cache) == 0
        assert vid._cache_bytes == 0
        # The streamer still works after clearing.
        assert np.array_equal(vid[0], vid[0])
    finally:
        vid.close()


def test_cache_hit_serves_without_redecoding():
    """
    The core promise of the cache: re-reading a cached frame returns it
    straight from memory without decoding it again.

    We verify this without mocking by watching `_current_frame_number`, which
    tracks the decoder's position. Decoding a frame advances it to that frame,
    whereas a cache hit returns before touching the stream and so leaves it
    unchanged. We also confirm the cached frame is still pixel-correct.
    """
    reference = npimage.VideoStreamer(str(SPINNING_MP4), cache_size=None)
    vid = npimage.VideoStreamer(str(SPINNING_MP4), cache_size=8)
    try:
        _ = vid[5]                              # cache miss: decodes frame 5
        _ = vid[2]                              # cache miss: decodes frame 2
        assert vid._current_frame_number == 2   # decoder is now parked at 2

        frame5 = vid[5]                          # cache hit: must NOT decode
        assert vid._current_frame_number == 2, (
            'Re-reading a cached frame moved the decoder, so it was decoded '
            'again instead of served from the cache.'
        )
        assert 5 in vid._cache
        # The frame served from the cache is identical to a fresh decode.
        assert np.array_equal(frame5, reference[5])
    finally:
        reference.close()
        vid.close()
