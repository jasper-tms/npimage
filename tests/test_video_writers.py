#!/usr/bin/env python3
"""
Test script to compare FFmpegVideoWriter and AVVideoWriter implementations.
Uses the table tennis emoji image with animated movement to create test videos.
"""

import os
import time
import tempfile
import psutil
import gc
from pathlib import Path

import numpy as np

import npimage
from npimage.vidio import FFmpegVideoWriter, AVVideoWriter

script_dir = Path(__file__).parent

# Memory monitoring
process = psutil.Process(os.getpid())


def print_memory(label):
    """Print current memory usage"""
    gc.collect()
    mem = process.memory_info().rss / 1024 / 1024
    print(f"{label}: {mem:.1f} MB")


def create_animated_frames(base_image, num_frames=60, movement_range=128):
    """Create animated frames by moving the image around"""
    frames = []
    height, width = base_image.shape[:2]

    for i in range(num_frames):
        # Create circular motion
        angle = 2 * np.pi * i / num_frames
        dx = int(movement_range * np.cos(angle))
        dy = int(movement_range * np.sin(angle))

        # Apply offset to create movement
        frame = npimage.operations.offset(base_image, (dy, dx))
        frames.append(frame)

    return frames


def test_video_writer(writer_class, filename, frames, writer_name):
    """Test a specific video writer class"""
    print(f"\n=== Testing {writer_name} ===")
    print_memory(f"Before {writer_name}")

    start_time = time.time()

    try:
        with writer_class(filename, framerate=30, overwrite=True) as writer:
            for i, frame in enumerate(frames):
                if i % 20 == 0:  # Print progress every 20 frames
                    print(f"  Writing frame {i}/{len(frames)}")
                writer.write(frame)

        elapsed = time.time() - start_time
        file_size = os.path.getsize(filename) / 1024  # KB

        print_memory(f"After {writer_name}")
        print(f"✓ {writer_name} completed:")
        print(f"  Time: {elapsed:.2f} seconds")
        print(f"  File size: {file_size:.1f} KB")
        print(f"  Frames per second: {len(frames)/elapsed:.1f}")

        return True

    except Exception as e:
        print(f"✗ {writer_name} failed: {e}")
        return False


def test_memory_leak(writer_class, frames, writer_name, num_iterations=5):
    """Test for memory leaks by creating multiple videos"""
    print(f"\n=== Memory Leak Test for {writer_name} ===")
    print_memory("Initial memory")

    for i in range(num_iterations):
        filename = script_dir / "videos" / f"memory_test_{writer_name.lower()}_{i+1}.mp4"
        
        print_memory(f"Before iteration {i+1}")

        with writer_class(filename, framerate=30, overwrite=True) as writer:
            for frame in frames:
                writer.write(frame)

        print_memory(f"After iteration {i+1}")

    print_memory(f"Final memory after {num_iterations} iterations")


def main():
    print("Video Writer Comparison Test")
    print("=" * 60)
    
    # Check FFmpeg version
    print("\nChecking FFmpeg version...")
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        ffmpeg_version = result.stdout.split('\n')[0]
        print(f"FFmpeg version: {ffmpeg_version}")
    except Exception as e:
        print(f"Error checking FFmpeg version: {e}")

    # Load the table tennis emoji image
    print("\nLoading table tennis emoji image...")
    image_path = script_dir / "table-tennis-emoji.png"
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    base_image = npimage.load(image_path)
    print(f"Loaded image: {base_image.shape}, dtype: {base_image.dtype}")

    # Create animated frames
    print("\nCreating animated frames...")
    frames = create_animated_frames(base_image, num_frames=60, movement_range=30)
    print(f"Created {len(frames)} animated frames")

    # Test both video writers
    results = {}
    
    # Test FFmpegVideoWriter
    ffmpeg_filename = script_dir / "videos" / "table_tennis_ffmpeg.mp4"
    results['FFmpegVideoWriter'] = test_video_writer(
        FFmpegVideoWriter, ffmpeg_filename, frames, "FFmpegVideoWriter"
    )
    
    # Test AVVideoWriter
    av_filename = script_dir / "videos" / "table_tennis_av.mp4"
    results['AVVideoWriter'] = test_video_writer(
        AVVideoWriter, av_filename, frames, "AVVideoWriter"
    )

    # Memory leak tests
    print("\n" + "=" * 60)
    print("MEMORY LEAK TESTS")
    print("=" * 60)

    # Use a smaller set of frames for memory leak testing
    test_frames = frames[:30]  # 30 frames for faster testing

    if results.get('FFmpegVideoWriter', False):
        test_memory_leak(FFmpegVideoWriter, test_frames, "FFmpegVideoWriter")

    if results.get('AVVideoWriter', False):
        test_memory_leak(AVVideoWriter, test_frames, "AVVideoWriter")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for writer_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{writer_name}: {status}")

    print("\n🎉 Test completed!")


if __name__ == "__main__":
    main()
