#!/usr/bin/env python3
"""
Benchmark / memory-leak probe for FFmpegVideoWriter and AVVideoWriter.

This file lives outside the pytest-collected test suite on purpose — the
filename doesn't start with `test_`, so pytest will ignore it. Run it directly
when you want to measure:

    python tests/benchmarks/video_writer_memory.py

It writes a series of short videos back-to-back with each writer and prints
RSS memory before and after each iteration, so you can eyeball whether memory
is climbing over time.
"""

import gc
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil

import npimage
from npimage.vidio import FFmpegVideoWriter, AVVideoWriter

SCRIPT_DIR = Path(__file__).parent
TESTS_DIR = SCRIPT_DIR.parent
EMOJI_PATH = TESTS_DIR / 'table-tennis-emoji.png'

_process = psutil.Process(os.getpid())


def print_memory(label):
    gc.collect()
    mem_mb = _process.memory_info().rss / 1024 / 1024
    print(f'{label}: {mem_mb:.1f} MB')


def animated_frames(base_image, num_frames=60, movement_range=30):
    frames = []
    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames
        dx = int(movement_range * np.cos(angle))
        dy = int(movement_range * np.sin(angle))
        frames.append(npimage.operations.offset(base_image, (dy, dx)))
    return frames


def time_single_write(writer_class, filename, frames, name):
    print(f'\n=== {name} single-write benchmark ===')
    print_memory(f'Before {name}')
    start = time.time()
    with writer_class(filename, framerate=30, overwrite=True) as writer:
        for i, frame in enumerate(frames):
            if i % 20 == 0:
                print(f'  Writing frame {i}/{len(frames)}')
            writer.write(frame)
    elapsed = time.time() - start
    size_kb = os.path.getsize(filename) / 1024
    print_memory(f'After {name}')
    print(f'{name}: {elapsed:.2f}s, {size_kb:.1f} KB, {len(frames)/elapsed:.1f} fps')


def memory_leak_probe(writer_class, frames, name, output_dir, num_iterations=5):
    print(f'\n=== {name} memory-leak probe ===')
    print_memory('Initial')
    for i in range(num_iterations):
        filename = output_dir / f'memory_test_{name.lower()}_{i + 1}.mp4'
        print_memory(f'Before iteration {i + 1}')
        with writer_class(str(filename), framerate=30, overwrite=True) as writer:
            for frame in frames:
                writer.write(frame)
        print_memory(f'After iteration {i + 1}')
    print_memory(f'Final ({num_iterations} iterations)')


def main():
    print('Video Writer Benchmark')
    print('=' * 60)

    try:
        result = subprocess.run(
            ['ffmpeg', '-version'], capture_output=True, text=True
        )
        print(f'FFmpeg: {result.stdout.splitlines()[0]}')
    except FileNotFoundError:
        print('FFmpeg not found on PATH')

    base_image = npimage.load(str(EMOJI_PATH))
    print(f'Loaded emoji: {base_image.shape}, dtype={base_image.dtype}')

    frames = animated_frames(base_image, num_frames=60)
    print(f'Generated {len(frames)} frames')

    output_dir = SCRIPT_DIR / 'output'
    output_dir.mkdir(exist_ok=True)

    time_single_write(
        FFmpegVideoWriter, str(output_dir / 'table_tennis_ffmpeg.mp4'),
        frames, 'FFmpegVideoWriter',
    )
    time_single_write(
        AVVideoWriter, str(output_dir / 'table_tennis_av.mp4'),
        frames, 'AVVideoWriter',
    )

    probe_frames = frames[:30]
    memory_leak_probe(FFmpegVideoWriter, probe_frames, 'FFmpegVideoWriter', output_dir)
    memory_leak_probe(AVVideoWriter, probe_frames, 'AVVideoWriter', output_dir)


if __name__ == '__main__':
    main()
