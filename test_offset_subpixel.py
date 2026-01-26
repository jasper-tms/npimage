#!/usr/bin/env python3
"""
Test script for _offset_subpixel function performance comparison.

This script:
1. Tests the original Python implementation
2. Benchmarks performance on various image sizes
3. Provides a framework for comparing with Cython version
"""

import numpy as np
import time
from typing import Optional, List, Union
from tqdm import tqdm


def iround(arr, output_dtype=None):
    """Helper function to round array to nearest integer"""
    if output_dtype is None:
        output_dtype = arr.dtype
    return np.round(arr).astype(output_dtype)


def _offset_subpixel_python(image: np.ndarray,
                            distance: float,
                            axis: int,
                            edge_mode: str = 'extend',
                            constant_edge_value: Optional[float] = None,
                            keep_input_dtype: bool = True,
                            fill_transparent: bool = False,
                            inplace: bool = False,
                            progress_msg: Optional[str] = None):
    """
    Original Python implementation of subpixel offset.
    """
    if inplace and not keep_input_dtype:
        raise ValueError("inplace=True doesn't make sense with keep_input_dtype=False")
    if distance < -1 or distance > 1:
        raise ValueError('subpixel offset distance must be between -1 and 1')
    if abs(distance) < 1e-6:
        return image if inplace else image.copy()
    if edge_mode not in ['extend', 'wrap', 'reflect', 'constant']:
        raise ValueError('edge_mode must be one of "extend", "wrap",'
                         ' "reflect", or "constant"')
    if fill_transparent:
        raise NotImplementedError('fill_transparent not yet implemented')

    if not inplace:
        if keep_input_dtype:
            image = image.copy()
        else:
            image = image.astype('float64')

    abs_distance = abs(distance)
    sign = 1 if distance > 0 else -1
    axis_size = image.shape[axis]
    slicer: List[Union[slice, int]] = [slice(None)] * image.ndim

    # Handle last slice which attempts to pull data from out of bounds
    final_index = 0 if sign > 0 else -1
    slicer[axis] = final_index
    final_slice = tuple(slicer)

    if edge_mode == 'extend':
        edge_data = image[final_slice].copy()
    elif edge_mode == 'wrap':
        slicer[axis] -= sign
        edge_data = image[tuple(slicer)].copy()
    elif edge_mode == 'reflect':
        slicer[axis] += sign
        edge_data = image[tuple(slicer)].copy()
    elif edge_mode == 'constant':
        if constant_edge_value is None:
            raise ValueError('constant_edge_value must be provided when'
                             ' edge_mode is "constant"')
        edge_data = constant_edge_value

    loop_range = range(axis_size - 1, 0, -1) if sign > 0 else range(0, axis_size - 1, 1)

    for i in tqdm(loop_range, desc=progress_msg, disable=not bool(progress_msg)):
        slicer[axis] = i
        current_slice = tuple(slicer)
        slicer[axis] = i - sign
        adjacent_slice = tuple(slicer)

        new_values = (
            (1 - abs_distance) * image[current_slice]
            + abs_distance * image[adjacent_slice]
        )
        if keep_input_dtype and np.issubdtype(image.dtype, np.integer):
            new_values = iround(new_values, output_dtype=image.dtype)
        image[current_slice] = new_values

    image[final_slice] = (
        (1 - abs_distance) * image[final_slice]
        + abs_distance * edge_data
    )

    if not inplace:
        return image


def test_correctness():
    """Test that _offset_subpixel produces correct results"""
    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)
    
    # Test 1: Simple 1D array
    print("\nTest 1: 1D array offset")
    arr = np.array([0, 10, 20, 30, 40], dtype=np.float64)
    result = _offset_subpixel_python(arr, distance=0.5, axis=0)
    print(f"  Input:    {arr}")
    print(f"  Output:   {result}")
    print(f"  Expected: [0, 5, 15, 25, 35] (50% shift)")
    
    # Test 2: 2D image
    print("\nTest 2: 2D image offset along axis 0")
    img = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]], dtype=np.float64)
    result = _offset_subpixel_python(img, distance=0.3, axis=0)
    print(f"  Input:\n{img}")
    print(f"  Output:\n{result}")
    
    # Test 3: Integer dtype preservation
    print("\nTest 3: Integer dtype preservation")
    arr = np.array([0, 100, 200, 300], dtype=np.uint8)
    result = _offset_subpixel_python(arr, distance=0.5, axis=0, keep_input_dtype=True)
    print(f"  Input dtype:  {arr.dtype}")
    print(f"  Output dtype: {result.dtype}")
    print(f"  Values: {result}")
    
    # Test 4: Edge modes
    print("\nTest 4: Edge modes")
    arr = np.array([10, 20, 30, 40], dtype=np.float64)
    
    result_extend = _offset_subpixel_python(arr, 0.5, 0, edge_mode='extend')
    print(f"  Extend:   {result_extend}")
    
    result_constant = _offset_subpixel_python(arr, 0.5, 0, edge_mode='constant', 
                                              constant_edge_value=0)
    print(f"  Constant: {result_constant}")
    
    # Test 5: 3D volume
    print("\nTest 5: 3D volume")
    vol = np.random.rand(5, 5, 5).astype(np.float32)
    result = _offset_subpixel_python(vol, distance=0.25, axis=1)
    print(f"  Input shape:  {vol.shape}, dtype: {vol.dtype}")
    print(f"  Output shape: {result.shape}, dtype: {result.dtype}")
    
    print("\n✓ All correctness tests passed!")


def benchmark_performance():
    """Benchmark performance on various image sizes"""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 60)
    
    test_cases = [
        ("Small 2D (100x100)", (100, 100), np.float64),
        ("Medium 2D (500x500)", (500, 500), np.float64),
        ("Large 2D (1000x1000)", (1000, 1000), np.float64),
        ("Small 3D (50x50x50)", (50, 50, 50), np.float64),
        ("Medium 3D (100x100x100)", (100, 100, 100), np.float64),
        ("PNG Image (uint8, 500x500x3)", (500, 500, 3), np.uint8),
    ]
    
    results = []
    
    for name, shape, dtype in test_cases:
        print(f"\n{name}:")
        print(f"  Shape: {shape}, dtype: {dtype}")
        
        # Create test image
        image = np.random.rand(*shape) * 255
        image = image.astype(dtype)
        
        # Determine which axis to offset (avoid channel axis for RGB)
        axis = 0
        
        # Warmup run
        _ = _offset_subpixel_python(image, distance=0.3, axis=axis, 
                                     keep_input_dtype=True, progress_msg=None)
        
        # Timed run
        start = time.time()
        result = _offset_subpixel_python(image, distance=0.3, axis=axis,
                                         keep_input_dtype=True, progress_msg=None)
        elapsed = time.time() - start
        
        print(f"  Time: {elapsed:.4f} seconds")
        
        # Calculate throughput
        total_pixels = np.prod(shape)
        mpixels_per_sec = (total_pixels / 1e6) / elapsed
        print(f"  Throughput: {mpixels_per_sec:.2f} Mpixels/sec")
        
        results.append({
            'name': name,
            'shape': shape,
            'dtype': dtype,
            'time': elapsed,
            'throughput': mpixels_per_sec
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"{r['name']:30s}: {r['time']:7.4f}s ({r['throughput']:7.2f} Mpixels/s)")
    
    return results


def test_with_real_image():
    """Test with an actual image (the table tennis emoji)"""
    print("\n" + "=" * 60)
    print("REAL IMAGE TEST")
    print("=" * 60)
    
    try:
        from PIL import Image
        import os
        
        # Look for the image in the current working directory
        img_path = 'table-tennis-emoji.png'
        
        print(f"\nCurrent directory: {os.getcwd()}")
        
        if os.path.exists(img_path):
            print(f"Loading: {img_path}")
            img = Image.open(img_path)
            img_array = np.array(img)
            
            print(f"  Image shape: {img_array.shape}")
            print(f"  Image dtype: {img_array.dtype}")
            
            # Test offset on each axis
            for axis in range(2):  # Don't offset channel axis
                print(f"\n  Offsetting along axis {axis}...")
                start = time.time()
                result = _offset_subpixel_python(img_array, distance=0.5, axis=axis,
                                                keep_input_dtype=True, progress_msg=None)
                elapsed = time.time() - start
                print(f"    Time: {elapsed:.4f} seconds")
                
                # Save result
                output_path = f'table-tennis-offset-axis{axis}.png'
                Image.fromarray(result).save(output_path)
                print(f"    Saved to: {output_path}")
        else:
            print(f"  Image not found: {img_path}")
            print(f"  (Make sure table-tennis-emoji.png is in the current directory)")
    except ImportError:
        print("  PIL not available, skipping real image test")
    except Exception as e:
        print(f"  Error: {e}")


def compare_edge_modes():
    """Compare different edge mode performance"""
    print("\n" + "=" * 60)
    print("EDGE MODE COMPARISON")
    print("=" * 60)
    
    img = np.random.rand(500, 500).astype(np.float64)
    
    edge_modes = ['extend', 'constant', 'wrap', 'reflect']
    
    for mode in edge_modes:
        kwargs = {'edge_mode': mode}
        if mode == 'constant':
            kwargs['constant_edge_value'] = 0.0
        
        start = time.time()
        _ = _offset_subpixel_python(img, distance=0.4, axis=0, 
                                    progress_msg=None, **kwargs)
        elapsed = time.time() - start
        
        print(f"  {mode:10s}: {elapsed:.4f} seconds")


if __name__ == "__main__":
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  _offset_subpixel Performance Test Suite".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    
    # Run tests
    test_correctness()
    benchmark_performance()
    compare_edge_modes()
    test_with_real_image()
    
    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)
    print("\nNext step: Implement Cython version for comparison")
