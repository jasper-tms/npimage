#!/usr/bin/env python3
"""
Tests for VideoStreamer, particularly handling of HEVC videos with
negative-PTS priming packets (common in iPhone recordings).
"""

import os
import subprocess
import tempfile

import numpy as np

import npimage


def test_negative_pts_priming_packet():
    """
    Test that VideoStreamer correctly handles HEVC videos where the first
    packet has a negative PTS (a "priming" packet used for B-frame prediction
    that doesn't produce a decoded frame).

    This is common in iPhone HEVC recordings. The bug was that the index was
    built from demuxed packets, which included the priming packet's negative
    PTS. When trying to access frame 0, it would seek to that negative PTS,
    which no decoded frame ever has, causing a VideoSeekError.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, 'test_negative_pts.mp4')

        # Create a small HEVC video with a negative-PTS priming packet.
        # The -output_ts_offset shifts timestamps so the first keyframe packet
        # gets a negative PTS, mimicking iPhone HEVC behavior.
        result = subprocess.run([
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', 'color=c=red:size=320x240:rate=30:d=1',
            '-c:v', 'libx265', '-preset', 'ultrafast',
            '-bf', '2', '-x265-params', 'bframes=2',
            '-tag:v', 'hvc1',
            '-output_ts_offset', '-0.033',
            video_path
        ], capture_output=True, text=True)
        assert result.returncode == 0, f'ffmpeg failed: {result.stderr}'

        # Verify the test video actually has the problematic structure:
        # more packets than decoded frames due to the priming packet
        import av
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            pkt_pts = sorted(
                p.pts for p in container.demux(stream) if p.pts is not None
            )
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            frame_pts = [f.pts for f in container.decode(stream)]

        assert len(pkt_pts) > len(frame_pts), (
            f'Test video should have more packets ({len(pkt_pts)}) than '
            f'decoded frames ({len(frame_pts)}). The test setup may need '
            f'updating if ffmpeg behavior has changed.'
        )
        assert pkt_pts[0] < 0, (
            f'First packet PTS should be negative, got {pkt_pts[0]}'
        )

        # Now test that VideoStreamer handles this correctly
        vid = npimage.VideoStreamer(video_path)

        # The number of frames should match decoded frames, not packets
        assert vid.n_frames == len(frame_pts), (
            f'Expected {len(frame_pts)} frames, got {vid.n_frames}'
        )

        # Accessing frame 0 should work without raising VideoSeekError
        frame = vid[0]
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (240, 320, 3)

        # Accessing the last frame should also work
        last_frame = vid[vid.n_frames - 1]
        assert isinstance(last_frame, np.ndarray)

        print(f'PASSED: Loaded {vid.n_frames} frames from video with '
              f'negative-PTS priming packet')


def main():
    test_negative_pts_priming_packet()
    print('\nAll VideoStreamer tests passed.')


if __name__ == '__main__':
    main()
