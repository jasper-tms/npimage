"""
Tests for VideoStreamer, particularly handling of HEVC videos with
negative-PTS priming packets (common in iPhone recordings).
"""

import subprocess

import numpy as np
import pytest

import npimage


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
