#!/usr/bin/env python3
"""
Functions for reading and writing videos.

Function list:
- load_video(filename) -> np.ndarray
- lazy_load_video(filename) -> Iterator[np.ndarray]
- save_video(data, filename) -> Saves a (time, height, width[, channels])
    numpy array of pixel values as a video file.
    Arguments for setting framerate, video bitrate, etc are provided.

Class list:
- VideoStreamer: Provides fast random access to frames in a video file
    via VideoStreamer[frame_number].
- VideoWriter: Allows writing frames one-by-one to a video file via
    VideoWriter.write(image). This can be advantageous compared to save_video
    because you don't ever have to have all the frames in memory at once.
"""

from typing import Union, Tuple, Iterator, Literal
from pathlib import Path
import subprocess
import threading
import json
from fractions import Fraction

import numpy as np

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
}

supported_extensions = ['mp4', 'mkv', 'avi', 'mov', 'webm']


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
    try:
        import av
        from tqdm import tqdm
    except ImportError:
        raise ImportError('Missing optional dependency for video processing,'
                          ' run `pip install av tqdm`')

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
    Lazily load video frames as numpy arrays using PyAV.

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
    try:
        import av
    except ImportError:
        raise ImportError('Missing optional dependency for video processing,'
                          ' run `pip install av tqdm`')
    with av.open(Path(filename).expanduser()) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            img = frame.to_ndarray(format='rgb24')
            yield img


class VideoSeekError(RuntimeError):
    pass


class VideoStreamer:
    def __init__(self, filename, verbose: bool = False):
        try:
            import av
            self.av = av
        except ImportError:
            raise ImportError('Missing optional dependency for video processing,'
                              ' run `pip install av tqdm`')
        self.verbose = verbose
        self.filename = Path(filename).expanduser()
        if not self.filename.exists():
            raise FileNotFoundError(f'File {filename} not found')

        self.container = av.open(str(self.filename))
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

        self.index_filename = self.filename.with_suffix(self.filename.suffix + '.index')
        self._load_index()

    def _build_index(self):
        if self.verbose:
            print('Building frame timestamp index for fast random frame access...')

        frames_pts = []
        # Try getting frame PTS values fast using PyAV
        try:
            from tqdm import tqdm

            with self.av.open(str(self.filename)) as container:
                stream = container.streams.video[0]
                # Access as packets to only read metadata (fast), not pixel data (slow)
                for packet in tqdm(container.demux(stream), total=stream.frames,
                                   desc='Indexing frames', disable=not self.verbose):
                    if packet.pts is not None:
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

        self.frames_pts = frames_pts
        self.n_frames = len(frames_pts)
        self.pts0 = self.frames_pts[0]
        self.rotation = _get_rotation_from_metadata(self.filename)
        index = {}

        # Determine whether the video is constant or variable framerate
        pts_deltas = np.diff(frames_pts) if len(frames_pts) > 1 else None
        if pts_deltas is not None and (pts_deltas == pts_deltas[0]).all():
            # The video is constant framerate
            self.pts_delta = pts_deltas[0]
            self._framerate = 1 / (self.pts_delta * self.time_base)
            if self._framerate.denominator == 1:
                self._framerate = self._framerate.numerator
                index['framerate'] = self._framerate
            else:
                index['framerate'] = {'numerator': self._framerate.numerator,
                                      'denominator': self._framerate.denominator}
        else:
            # The video is variable framerate
            self._framerate = 'variable'
            index['framerate'] = 'variable'

        index['n_frames'] = self.n_frames
        index['rotation'] = self.rotation
        index['time_base'] = {'numerator': self.time_base.numerator,
                              'denominator': self.time_base.denominator}
        if self._framerate == 'variable':
            index['frames_pts'] = frames_pts
        else:
            index['pts0'] = self.pts0
        with open(self.index_filename, 'w') as f:
            json.dump(index, f)
        if self.verbose:
            print(f'Cached index at "{self.index_filename}"')

    def _load_index(self):
        if not self.index_filename.exists():
            self._build_index()
        elif self.verbose:
            print(f'Loading frame timestamp index from "{self.index_filename}"')

        with open(self.index_filename, 'r') as f:
            index = json.load(f)
        self.n_frames = index['n_frames']
        self.rotation = index.get('rotation', None)

        # Load time_base and pts_values
        time_base_data = index['time_base']
        self.time_base = Fraction(time_base_data['numerator'], time_base_data['denominator'])

        if index['framerate'] == 'variable':
            self._framerate = 'variable'
            self.frames_pts = index['frames_pts']
        else:
            self.pts0 = index['pts0']
            if np.issubdtype(type(index['framerate']), np.integer):
                self._framerate = index['framerate']
            else:
                self._framerate = Fraction(index['framerate']['numerator'],
                                           index['framerate']['denominator'])
            self.pts_delta = 1 / self.time_base / self._framerate
            if self.pts_delta.denominator != 1:
                raise ValueError('pts_delta does not appear to be an integer. This is'
                                 ' unexpected and may indicate a malformed index.')
            self.pts_delta = self.pts_delta.numerator

    @property
    def framerate(self) -> Union[float, Literal['variable']]:
        if self._framerate == 'variable':
            return 'variable'
        return float(self._framerate)

    @property
    def fps(self) -> float:
        """
        Note that fps always returns a float even if the framerate is 'variable'.
        (If framerate is 'variable', fps returns exactly what its name, "frames per second",
        implies: the number of frame intervals divided by the video duration in seconds.)
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

    def frame_number_to_pts(self, frame_number: int) -> int:
        frame_number = self._normalize_frame_number(frame_number)
        if self._framerate == 'variable':
            return self.frames_pts[frame_number]
        else:
            return int(frame_number) * self.pts_delta + self.pts0

    def frame_number_to_time(self, frame_number: int) -> float:
        return float(self.frame_number_to_pts(frame_number) * self.time_base)

    def pts_to_frame_number(self, pts: int) -> int:
        if self._framerate == 'variable':
            if pts not in self.frames_pts:
                raise ValueError(f'PTS {pts} not in video index.')
            return self.frames_pts.index(pts)
        else:
            if pts < self.pts0:
                raise ValueError(f'PTS {pts} is before the start of the'
                                 f' video (PTS {self.pts0}).')
            if pts > self.pts_delta * (self.n_frames - 1) + self.pts0:
                raise ValueError(f'PTS {pts} is after the end of the video (PTS '
                                 f'{self.pts_delta * (self.n_frames - 1) + self.pts0}).')
            if (pts - self.pts0) % self.pts_delta != 0:
                raise ValueError(f'PTS {pts} is between frames for this video.')
            return (pts - self.pts0) // self.pts_delta

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
            return image

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

    Parameters
    ----------
    filename : str
        The filename to save the video to.
    framerate : int or float, default 30
        The frame rate of the video.
    crf : int, default 23
        Constant Rate Factor for encoding quality (lower is better quality).
    compression_speed : str, default 'medium'
        Compression speed preset: 'ultrafast', 'superfast', 'veryfast',
        'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'.
    codec : Literal['libx264', 'libx265'], default 'libx264'
        The video codec to use for encoding. Can be any of a number of aliases for
        these two codecs, including avc1/h264 vs hevc/hvc1/hev1/h265.
    """
    def __init__(self, filename, framerate=30, crf=23, compression_speed='medium',
                 codec: Literal['libx264', 'libx265'] = 'libx264',
                 overwrite=False):
        try:
            import av
        except ImportError:
            raise ImportError('Missing optional dependency for video processing,'
                              ' run `pip install av tqdm`')
        self.av = av
        filename = Path(filename).expanduser()
        if filename.exists() and not overwrite:
            raise FileExistsError(f'File {filename} already exists. '
                                  'Set overwrite=True to overwrite.')
        self.filename = filename
        self._framerate = utils.limit_fraction(framerate)
        self.crf = crf
        self.compression_speed = compression_speed
        self.codec = codec_aliases[codec.lower()]
        self.container = av.open(filename, mode='w')
        self.stream = self.container.add_stream(self.codec, rate=self._framerate)
        self.stream.pix_fmt = 'yuv420p'
        self.stream.options = {'crf': str(crf), 'preset': compression_speed}
        self._closed = False
        self.stream.width = 0
        self.stream.height = 0

    @property
    def framerate(self):
        return float(self._framerate)

    def write(self, frame):
        if self._closed:
            raise RuntimeError('AVVideoWriter is closed, cannot write more frames.')
        if not isinstance(frame, self.av.VideoFrame):
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            if frame.ndim == 4:
                for i in range(frame.shape[0]):
                    self.write(frame[i])
                return
            elif frame.ndim == 3 and frame.shape[-1] == 3:
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
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
            del packet
        del frame

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        # Flush stream
        for packet in self.stream.encode():
            self.container.mux(packet)
        del packet
        # Close and delete everything
        self.stream = None
        self.container.close()
        self.container = None
        self._closed = True
        import gc
        gc.collect()


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

    Parameters
    ----------
    filename : str
        The filename to save the video to.
    framerate : int or float, default 30
        The frame rate of the video.
    crf : int, default 23
        Constant Rate Factor for encoding quality (lower is better quality).
    compression_speed : str, default 'medium'
        Compression speed preset: 'ultrafast', 'superfast', 'veryfast',
        'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'.
    codec : Literal['libx264', 'libx265'], default 'libx264'
        The video codec to use for encoding. Can be any of a number of aliases for
        these two codecs, including avc1/h264 vs hevc/hvc1/hev1/h265.
    overwrite : bool, default False
        Whether to overwrite the file if it already exists.
    """
    def __init__(self, filename, framerate=30, crf=23, compression_speed='medium',
                 codec: Literal['libx264', 'libx265'] = 'libx264',
                 overwrite=False):
        filename = Path(filename).expanduser()
        if filename.exists() and not overwrite:
            raise FileExistsError(f'File {filename} already exists. '
                                  'Set overwrite=True to overwrite.')
        self.filename = filename
        self._framerate = utils.limit_fraction(framerate)
        self.crf = crf
        self.compression_speed = compression_speed
        self.codec = codec_aliases[codec.lower()]

        # Initialize process state
        self._process = None
        self._stdin = None
        self._closed = False
        self._width = None
        self._height = None
        self._pixel_format = None

    @property
    def framerate(self):
        return float(self._framerate)

    def _initialize_process(self, width, height, pixel_format):
        """Initialize the FFmpeg subprocess for video encoding"""
        command = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-nostats',
            '-y',  # Overwrite output
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', pixel_format,
            '-r', str(self._framerate),
            '-i', '-',  # Read from stdin
            '-an',  # No audio
            '-c:v', self.codec,
            '-pix_fmt', 'yuv420p',  # Output pixel format
            '-crf', str(self.crf),
            '-preset', self.compression_speed,
            self.filename
        ]

        # Start FFmpeg process
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE  # Capture errors
        )
        self._stdin = self._process.stdin
        self._width = width
        self._height = height
        self._pixel_format = pixel_format

    def write(self, frame):
        """Write a frame to the video file"""
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

        # Convert frame to bytes and write to FFmpeg
        frame_bytes = frame.tobytes()
        self._stdin.write(frame_bytes)

    def close(self):
        """Close the video writer and finalize the video file"""
        if self._closed:
            return

        try:
            # Close stdin to signal end of input
            if self._stdin:
                self._stdin.close()

            # Wait for FFmpeg to finish
            if self._process:
                stderr_data = self._process.stderr.read()
                return_code = self._process.wait()

                # Check for errors
                if return_code != 0:
                    raise RuntimeError(f'FFmpeg failed with return code {return_code}:'
                                       f' {stderr_data.decode()}')
        finally:
            # Clean up process references
            self._process = None
            self._stdin = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Alias the preferred video writer class
VideoWriter = FFmpegVideoWriter


def save_video(data, filename, time_axis=0, color_axis=None, overwrite=False,
               dim_order='yx', framerate=30, crf=23, compression_speed='medium',
               progress_bar=True, codec: Literal['libx264', 'libx265'] = 'libx264') -> None:
    """
    Save a 3D numpy array of greyscale values OR a 4D numpy array of RGB values as a video

    Follows the PyAV cookbook section on generating video from numpy arrays:
    https://pyav.basswood-io.com/docs/develop/cookbook/numpy.html#generating-video

    Parameters
    ----------
    data : numpy.ndarray or list of images
        A 3D (grayscale) or 4D (RGB) numpy array of pixel values.

    filename : str
        The filename to save the video to.

    time_axis : int, default 0
        The axis of the data array that will be played as time in the video.

    color_axis : int or None, default None
        If not None, specifies the axis of the color channels (e.g., -1 for last axis,
        1 for second axis).
        If None, data must be 3D (greyscale) or 4D with one length 3 axis (RGB).

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

    codec : Literal['libx264', 'libx265'], default 'libx264'
        The video codec to use for encoding. Can be any of a number of aliases for
        these two codecs, including avc1/h264 vs hevc/hvc1/hev1/h265.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        raise ImportError('Missing optional dependency for video processing,'
                          ' run `pip install av tqdm`')

    filename = str(filename)
    if filename.split('.')[-1].lower() not in supported_extensions:
        filename += '.mp4'
    filename = Path(filename).expanduser()
    if filename.exists() and not overwrite:
        raise FileExistsError(f'File {filename} already exists. '
                              'Set overwrite=True to overwrite.')

    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if color_axis is None and data.ndim == 4:
        color_axis = utils.find_channel_axis(data, possible_channel_lengths=3)
        if color_axis is None:
            raise ValueError('4D input data must have an RGB (length 3) axis.')
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

    extension = filename.suffix.lower().lstrip('.')
    if extension == 'mp4':
        pad = [[0, 0], [0, 0], [0, 0]]
        if height % 2 != 0:
            pad[1][1] = 1
        if width % 2 != 0:
            pad[2][1] = 1
        if pad != [[0, 0], [0, 0], [0, 0]]:
            data = np.pad(data, pad, mode='edge')

    with VideoWriter(filename, framerate=framerate, crf=crf,
                     compression_speed=compression_speed, codec=codec,
                     overwrite=overwrite) as writer:
        for frame_i in tqdm(range(n_frames), total=n_frames,
                            desc='Saving video', disable=not progress_bar):
            writer.write(data[frame_i])


def _get_rotation_from_metadata(filename):
    """
    Get rotation metadata from a video file.

    We could use PyAV to do this perhaps faster, but for an already fast operation
    like this, we stick with ffprobe to avoid the memory leaks in PyAV.
    """
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
