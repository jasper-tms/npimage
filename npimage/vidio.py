#!/usr/bin/env python3
"""
Functions for reading and writing videos.

Function list:
- load_video(filename) -> np.ndarray
- lazy_load_video(filename) -> Iterator[np.ndarray]
- save_video(data, filename) -> Saves a 3D numpy array as a video

Class list:
- VideoStreamer: Provides fast random access to frames in a video file
  via VideoStreamer[frame_number]
- VideoWriter: Allows writing frames one-by-one to a video file via
  VideoWriter.write(image). This can be advantageous compared to save_video
  when you don't want to ever have to have all the frames in memory at once.

"""

from typing import Union, Tuple, Iterator, Literal
from pathlib import Path
import os
import subprocess
import threading
import json

import numpy as np

from . import utils

codec_aliases = {
    "libx264": "libx264",
    "avc1": "libx264",
    "h264": "libx264",
    "H.264": "libx264",
    "libx265": "libx265",
    "hevc": "libx265",
    "hvc1": "libx265",
    "hev1": "libx265",
    "h265": "libx265",
    "H.265": "libx265",
}

supported_extensions = ['mp4', 'mkv', 'avi', 'mov']


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

    container = av.open(filename)
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
        for i, frame in tqdm(enumerate(frame_iter), total=num_frames,
                             desc='Loading video', disable=not progress_bar):
            if i == 0:
                continue
            img = frame.to_ndarray(format='rgb24')
            data[i] = img
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
    container = av.open(filename)
    stream = container.streams.video[0]
    for frame in container.decode(stream):
        img = frame.to_ndarray(format='rgb24')
        yield img


class VideoStreamer:
    def __init__(self, filename):
        try:
            import av
        except ImportError:
            raise ImportError('Missing optional dependency for video processing,'
                              ' run `pip install av tqdm`')
        self.filename = Path(filename)
        if not self.filename.exists():
            raise FileNotFoundError(f"File {filename} not found")
        self.index_filename = self.filename.parent / (self.filename.stem + '_index.json')
        self._load_index()

        self.container = av.open(str(self.filename))
        self.stream = self.container.streams.video[0]
        self.time_base = self.stream.time_base
        self._shape = None
        self._first_frame = None
        self._width = None
        self._height = None
        self._ndim = None
        self._dtype = None
        self._current_frame_number = None
        self._lock = threading.Lock()

    def _build_index(self):
        print("Building index for fast random frame access...")
        cmd = [
            'ffprobe',
            '-select_streams', 'v:0',
            '-show_frames',
            '-show_entries', 'frame=pkt_pos,pkt_pts_time,coded_picture_number',
            '-of', 'json',
            self.filename
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        frames = json.loads(result.stdout).get('frames', [])
        time_index = [float(frame['pkt_pts_time']) for frame in frames]
        if len(time_index) == 0:
            raise RuntimeError("No frames found in video")

        self.n_frames = len(time_index)
        self.t0 = time_index[0]
        self.rotation = _detect_rotation(self.filename)
        index = {
            'n_frames': self.n_frames,
            't0': self.t0,
            'rotation': self.rotation,
        }

        # Determine whether the video is constant or variable framerate
        time_deltas = np.diff(time_index) if len(time_index) > 1 else None
        if time_deltas is not None and np.allclose(time_deltas, time_deltas[0], atol=1e-6):
            self.timestep = float(time_deltas[0])  # The video is constant framerate
            index['timestep'] = self.timestep
        else:
            index.update({
                'timestep': 'variable',  # The video is variable framerate
                'time_index': time_index,
            })
        with open(self.index_filename, 'w') as f:
            json.dump(index, f)
        print(f"Saved index to {self.index_filename} so this slow step is not needed again.")

    def _load_index(self):
        if not self.index_filename.exists():
            self._build_index()

        with open(self.index_filename, 'r') as f:
            index = json.load(f)
        self.n_frames = index['n_frames']
        self.t0 = index['t0']
        self.rotation = index.get('rotation', None)

        if not isinstance(index['timestep'], (str, float, int)):
            raise ValueError('Malformed index: timestep is not a string or number')

        if index['timestep'] == 'variable':
            self.timestep = 'variable'
            self.time_index = index['time_index']
        else:
            self.timestep = float(index['timestep'])
            if self.timestep <= 0:
                raise ValueError('Malformed index: timestep is not positive')

    def frame_to_time(self, frame_number):
        if self.timestep == 'variable':
            return self.time_index[frame_number]
        else:
            return self.timestep * frame_number + self.t0

    def __getitem__(self, key):
        if isinstance(key, tuple):
            frame_idx = key[0]
            frame = self._get_frame(frame_idx)
            if len(key) == 1:
                return frame
            else:
                return frame[key[1:]]
        else:
            return self._get_frame(key)

    def _get_frame(self, frame_number):
        """
        Provides access to random frames as fast as is reasonable when getting
        frames from compressed video in python.
        """

        def decode_until(frame_number) -> np.ndarray:
            """
            Decode forward from the current frame in the stream until we get
            to the requested frame number.
            """
            target_time = self.frame_to_time(frame_number)
            for frame in self.container.decode(self.stream):
                frame_time = float(frame.pts * self.time_base)
                if abs(frame_time - target_time) < 1e-6:
                    frame = frame.to_ndarray(format='rgb24')
                    self._current_frame_number = frame_number
                    return frame
                if frame_time > target_time:
                    raise RuntimeError(f"Frame with time {target_time} not found after"
                                       f" seeking – current frame time: {frame_time}")
            raise RuntimeError(f"Frame with time {target_time} not found after"
                               f" seeking – current frame time: {frame_time}")

        if isinstance(frame_number, slice):  # Support slicing
            start, stop, step = frame_number.indices(self.n_frames)
            return np.array([self._get_frame(i) for i in range(start, stop, step)])
        with self._lock:
            if frame_number < 0 or frame_number >= self.n_frames:
                raise IndexError(f"Frame {frame_number} out of range: [0, {self.n_frames})")

            if (self._current_frame_number is None
                    or frame_number <= self._current_frame_number
                    or frame_number > self._current_frame_number + 100):
                target_time = self.frame_to_time(frame_number)
                seek_time = int(target_time / float(self.time_base))
                # The following actually seeks to the closest keyframe before
                # seek_time, because it's not possible to seek directly to
                # non-keyframes due to video files being compressed.
                self.container.seek(seek_time, any_frame=False,
                                    backward=True, stream=self.stream)
            # Now we decode frames forward until we get to the requested frame
            return _rotate(decode_until(frame_number), self.rotation)

    @property
    def average_timestep(self):
        """
        Returns the average time step between frames.
        """
        if self.timestep == 'variable':
            return (self.time_index[-1] - self.time_index[0]) / (self.n_frames - 1)
        else:
            return self.timestep

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


class VideoWriter:
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
        from fractions import Fraction
        filename = os.path.expanduser(str(filename))
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(f'File {filename} already exists. '
                                  'Set overwrite=True to overwrite.')
        self.filename = filename
        self.framerate = str(framerate)  # str instead of float to avoid precision issues
        while Fraction(self.framerate).denominator >= 2**32 or Fraction(self.framerate).numerator >= 2**32:
            # If framerate has too many decimals to be expressed as a
            # ratio of 32-bit ints, which is required by ffmpeg, crop
            # off one decimal point of precision until it is expressable
            self.framerate = self.framerate[:-1]
            if self.framerate[-1] == '.':
                self.framerate = self.framerate[:-1]
            if len(self.framerate) == 0:
                raise RuntimeError('Error occurred handling framerate argument')
        self.crf = crf
        self.compression_speed = compression_speed
        self.codec = codec_aliases[codec.lower()]
        self.container = av.open(filename, mode='w')
        self.stream = self.container.add_stream(self.codec, rate=Fraction(self.framerate))
        self.stream.pix_fmt = 'yuv420p'
        self.stream.options = {'crf': str(crf), 'preset': compression_speed}
        self._closed = False
        self.stream.width = 0
        self.stream.height = 0

    def write(self, frame):
        if not isinstance(frame, self.av.VideoFrame):
            if frame.ndim == 3 and frame.shape[-1] == 3:
                frame = self.av.VideoFrame.from_ndarray(frame, format='rgb24')
            elif frame.ndim == 3 and frame.shape[-1] == 4:
                # While some video codecs support an alpha channel, most don't,
                # so for now we're just going to ignore the alpha channel
                frame = self.av.VideoFrame.from_ndarray(frame[..., :3], format='rgb24')
            elif frame.ndim == 2:
                frame = self.av.VideoFrame.from_ndarray(frame, format='gray')
            else:
                raise ValueError(f'Frame must be (H, W) (H, W, 3) or (H, W, 4) but was {frame.shape}')
        if self.stream.width == 0:
            self.stream.width = frame.width
        if self.stream.height == 0:
            self.stream.height = frame.height
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Flush stream
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self._closed = True


def save_video(data, filename, time_axis=0, color_axis=None, overwrite=False,
               dim_order='yx', framerate=30, crf=23, compression_speed='medium',
               progress_bar=True, codec: Literal['libx264', 'libx265'] = 'libx264') -> None:
    """
    Save a 3D numpy array of greyscale values OR a 4D numpy array of RGB values as a video

    Follows the PyAV cookbook section on generating video from numpy arrays:
    https://pyav.basswood-io.com/docs/develop/cookbook/numpy.html#generating-video

    Parameters
    ----------
    data : numpy.ndarray or list of filenames
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

    filename = os.path.expanduser(str(filename))
    if filename.split('.')[-1].lower() not in supported_extensions:
        filename += '.mp4'
    if os.path.exists(filename) and not overwrite:
        raise FileExistsError(f'File {filename} already exists. '
                              'Set overwrite=True to overwrite.')

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

    extension = filename.split('.')[-1].lower()
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


def _detect_rotation(filename):
    """
    Detect rotation metadata from the video file using ffprobe.
    Returns the rotation angle as a string (e.g., '90', '180', '270'), or None if not present.
    """
    import subprocess
    import json
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-select_streams', 'v:0',
        '-show_entries', 'stream_tags=rotate',
        '-of', 'json',
        str(filename)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
        streams = data.get('streams', [])
        if streams and 'tags' in streams[0]:
            rotate_tag = streams[0]['tags'].get('rotate')
            if rotate_tag:
                return rotate_tag
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return None


def _rotate(data: np.ndarray, rotation: Union[str, int, None] = 0) -> np.ndarray:
    """
    Apply clockwise rotation to a numpy array

    Parameters
    ----------
    data : np.ndarray
        The numpy array to rotate.
    rotation : str, int, or None, default 0
        The rotation angle in degrees, as a string or integer.
        Valid values are 0, 90, 180, 270, or 360.
        If None, 0, or 360, no rotation is applied.

    TODO find an actual video file with rotation_tag of 90 or 270 and check that
    the rotation is applied in the right direction. If it's opposite the correct
    direction, remove axes=(1, 0) from the np.rot90 calls in this function.
    """
    if rotation is None:
        return data
    try:
        rotation = int(rotation)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid rotation value: {rotation}")

    if rotation in [0, 360]:
        return data
    if rotation == 90:
        return np.rot90(data, k=1, axes=(1, 0))
    elif rotation == 180:
        return np.rot90(data, k=2, axes=(1, 0))
    elif rotation == 270:
        return np.rot90(data, k=3, axes=(1, 0))
    else:
        raise ValueError(f"Invalid rotation value: {rotation}")
