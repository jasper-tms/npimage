#!/usr/bin/env python3
"""
Comprehensive comparison between Python and Cython implementations
of _offset_subpixel.

This script:
1. Tests both implementations for correctness
2. Benchmarks performance across various scenarios
3. Generates a comparison report
"""

import numpy as np
import time
from test_offset_subpixel import _offset_subpixel_python

# Try to import Cython version
try:
    from offset_subpixel_fast import offset_subpixel_cython, offset_subpixel_fast
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    print("=" * 60)
    print("WARNING: Cython module not built yet!")
    print("=" * 60)
    print("\nTo build the Cython extension, run:")
    print("  python setup_offset_subpixel.py build_ext --inplace")
    print("\nThen run this script again.")
    print("=" * 60)
    import sys
    sys.exit(1)


def verify_correctness():
    """Verify that Cython implementations match Python results"""
    print("\n" + "=" * 60)
    print("CORRECTNESS VERIFICATION")
    print("=" * 60)
    
    test_cases = [
        ("1D array", np.array([0, 10, 20, 30, 40], dtype=np.float64), 0),
        ("2D small", np.random.rand(10, 10) * 100, 0),
        ("2D medium", np.random.rand(50, 50) * 255, 1),
        ("3D small", np.random.rand(10, 10, 10) * 100, 0),
        ("RGB image", np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8), 0),
    ]
    
    distances = [0.1, 0.5, 0.9, -0.3]
    
    all_passed = True
    
    for name, img, axis in test_cases:
        for distance in distances:
            # Python version
            result_python = _offset_subpixel_python(
                img, distance, axis, edge_mode='constant',
                constant_edge_value=0, keep_input_dtype=True, progress_msg=None
            )
            
            # Cython standard version
            result_cython = offset_subpixel_cython(
                img, distance, axis, edge_mode='constant',
                constant_edge_value=0, keep_input_dtype=True, progress_msg=None
            )
            
            # Cython fast version
            result_fast = offset_subpixel_fast(
                img, distance, axis, edge_mode='constant',
                constant_edge_value=0, keep_input_dtype=True, verbose=False
            )
            
            # Compare with appropriate tolerance
            # For integer dtypes, allow up to 1 pixel difference due to rounding
            if np.issubdtype(img.dtype, np.integer):
                atol = 1.0  # Allow 1 pixel difference for integers
            else:
                atol = 1e-8  # Strict for floats
            
            match_cython = np.allclose(result_python, result_cython, rtol=1e-5, atol=atol)
            match_fast = np.allclose(result_python, result_fast, rtol=1e-5, atol=atol)
            
            status = "✓" if (match_cython and match_fast) else "✗"
            
            if not (match_cython and match_fast):
                all_passed = False
                print(f"  {status} {name:15s} dist={distance:4.1f} axis={axis} - MISMATCH!")
                if not match_cython:
                    print(f"      Cython standard differs by max: {np.max(np.abs(result_python - result_cython))}")
                if not match_fast:
                    print(f"      Cython fast differs by max: {np.max(np.abs(result_python - result_fast))}")
    
    if all_passed:
        print("\n  ✓ All correctness tests PASSED!")
        print("    Python, Cython standard, and Cython fast produce identical results")
    else:
        print("\n  ✗ Some tests FAILED - results don't match!")
    
    return all_passed


def benchmark_comparison():
    """Comprehensive performance comparison"""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    test_cases = [
        ("Small 2D (100x100)", (100, 100), np.float64, 0),
        ("Medium 2D (500x500)", (500, 500), np.float64, 0),
        ("Large 2D (1000x1000)", (1000, 1000), np.float64, 0),
        ("Small 3D (50x50x50)", (50, 50, 50), np.float64, 0),
        ("Medium 3D (100x100x100)", (100, 100, 100), np.float64, 1),
        ("PNG uint8 (500x500x3)", (500, 500, 3), np.uint8, 0),
        ("Large RGB (1000x1000x3)", (1000, 1000, 3), np.uint8, 1),
    ]
    
    results = []
    
    for name, shape, dtype, axis in test_cases:
        print(f"\n{name}:")
        print(f"  Shape: {shape}, dtype: {dtype}, axis: {axis}")
        
        # Create test image
        if dtype == np.uint8:
            image = np.random.randint(0, 256, shape, dtype=dtype)
        else:
            image = (np.random.rand(*shape) * 255).astype(dtype)
        
        distance = 0.3
        
        # Time Python version
        img_copy = image.copy()
        start = time.time()
        _ = _offset_subpixel_python(img_copy, distance, axis, edge_mode='constant',
                                    constant_edge_value=0, keep_input_dtype=True,
                                    progress_msg=None)
        time_python = time.time() - start
        
        # Time Cython standard version
        img_copy = image.copy()
        start = time.time()
        _ = offset_subpixel_cython(img_copy, distance, axis, edge_mode='constant',
                                   constant_edge_value=0, keep_input_dtype=True,
                                   progress_msg=None)
        time_cython = time.time() - start
        
        # Time Cython fast version
        img_copy = image.copy()
        start = time.time()
        _ = offset_subpixel_fast(img_copy, distance, axis, edge_mode='constant',
                                constant_edge_value=0, keep_input_dtype=True,
                                verbose=False)
        time_fast = time.time() - start
        
        # Calculate speedups
        speedup_cython = time_python / time_cython if time_cython > 0 else 0
        speedup_fast = time_python / time_fast if time_fast > 0 else 0
        
        print(f"  Python:        {time_python:7.4f}s")
        print(f"  Cython:        {time_cython:7.4f}s  ({speedup_cython:5.1f}x faster)")
        print(f"  Cython (fast): {time_fast:7.4f}s  ({speedup_fast:5.1f}x faster)")
        
        results.append({
            'name': name,
            'shape': shape,
            'dtype': dtype,
            'time_python': time_python,
            'time_cython': time_cython,
            'time_fast': time_fast,
            'speedup_cython': speedup_cython,
            'speedup_fast': speedup_fast,
        })
    
    # Summary table
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Test Case':<30s} {'Python':>10s} {'Cython':>10s} {'Fast':>10s} {'Speedup':>10s}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<30s} "
              f"{r['time_python']:>9.4f}s "
              f"{r['time_cython']:>9.4f}s "
              f"{r['time_fast']:>9.4f}s "
              f"{r['speedup_fast']:>9.1f}x")
    
    # Overall statistics
    avg_speedup_cython = np.mean([r['speedup_cython'] for r in results])
    avg_speedup_fast = np.mean([r['speedup_fast'] for r in results])
    max_speedup_fast = np.max([r['speedup_fast'] for r in results])
    
    print("\n" + "=" * 80)
    print(f"Average Cython speedup:       {avg_speedup_cython:5.1f}x")
    print(f"Average Fast path speedup:    {avg_speedup_fast:5.1f}x")
    print(f"Maximum Fast path speedup:    {max_speedup_fast:5.1f}x")
    print("=" * 80)
    
    return results


def test_with_table_tennis_emoji():
    """Test with the actual uploaded image"""
    print("\n" + "=" * 60)
    print("TABLE TENNIS EMOJI TEST")
    print("=" * 60)
    
    try:
        from PIL import Image
        import os
        
        # Look for the image in the current working directory
        img_path = 'table-tennis-emoji.png'
        
        # Show current directory for debugging
        print(f"\nCurrent directory: {os.getcwd()}")
        
        if not os.path.exists(img_path):
            print(f"  Image not found: {img_path}")
            print(f"  (Make sure table-tennis-emoji.png is in the current directory)")
            return
        
        print(f"\nLoading: {img_path}")
        img = Image.open(img_path)
        img_array = np.array(img)
        
        print(f"  Image shape: {img_array.shape}")
        print(f"  Image dtype: {img_array.dtype}")
        
        distance = 0.5
        axis = 0
        
        # Time all three versions
        print(f"\nOffsetting by {distance} along axis {axis}...")
        
        # Python
        img_copy = img_array.copy()
        start = time.time()
        result_python = _offset_subpixel_python(img_copy, distance, axis,
                                                edge_mode='constant',
                                                constant_edge_value=0,
                                                keep_input_dtype=True,
                                                progress_msg=None)
        time_python = time.time() - start
        print(f"  Python:        {time_python:.4f}s")
        
        # Cython standard
        img_copy = img_array.copy()
        start = time.time()
        result_cython = offset_subpixel_cython(img_copy, distance, axis,
                                               edge_mode='constant',
                                               constant_edge_value=0,
                                               keep_input_dtype=True)
        time_cython = time.time() - start
        print(f"  Cython:        {time_cython:.4f}s ({time_python/time_cython:.1f}x faster)")
        
        # Cython fast
        img_copy = img_array.copy()
        start = time.time()
        result_fast = offset_subpixel_fast(img_copy, distance, axis,
                                           edge_mode='constant',
                                           constant_edge_value=0,
                                           keep_input_dtype=True,
                                           verbose=False)
        time_fast = time.time() - start
        print(f"  Cython (fast): {time_fast:.4f}s ({time_python/time_fast:.1f}x faster)")
        
        # Save results
        Image.fromarray(result_python).save('table-tennis-python.png')
        Image.fromarray(result_cython).save('table-tennis-cython.png')
        Image.fromarray(result_fast).save('table-tennis-fast.png')
        
        print("\n  Saved results:")
        print("    table-tennis-python.png")
        print("    table-tennis-cython.png")
        print("    table-tennis-fast.png")
        
    except Exception as e:
        print(f"  Error: {e}")


def analyze_scalability():
    """Analyze how speedup scales with image size"""
    print("\n" + "=" * 60)
    print("SCALABILITY ANALYSIS")
    print("=" * 60)
    
    sizes = [50, 100, 200, 500, 1000, 2000]
    
    print(f"\n{'Size':>6s} {'Python':>10s} {'Cython':>10s} {'Fast':>10s} {'Speedup':>10s}")
    print("-" * 50)
    
    for size in sizes:
        img = np.random.rand(size, size).astype(np.float64)
        
        # Time each version (single run, no warmup for brevity)
        start = time.time()
        _offset_subpixel_python(img.copy(), 0.3, 0, edge_mode='constant',
                               constant_edge_value=0, progress_msg=None)
        t_python = time.time() - start
        
        start = time.time()
        offset_subpixel_cython(img.copy(), 0.3, 0, edge_mode='constant',
                              constant_edge_value=0)
        t_cython = time.time() - start
        
        start = time.time()
        offset_subpixel_fast(img.copy(), 0.3, 0, edge_mode='constant',
                            constant_edge_value=0, verbose=False)
        t_fast = time.time() - start
        
        speedup = t_python / t_fast if t_fast > 0 else 0
        
        print(f"{size:>6d} {t_python:>9.4f}s {t_cython:>9.4f}s {t_fast:>9.4f}s {speedup:>9.1f}x")


if __name__ == "__main__":
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Python vs Cython Performance Comparison".center(58) + "║")
    print("║" + "  _offset_subpixel Implementations".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    
    if not CYTHON_AVAILABLE:
        exit(1)
    
    # Run all comparisons
    correctness_passed = verify_correctness()
    
    if correctness_passed:
        benchmark_comparison()
        analyze_scalability()
        test_with_table_tennis_emoji()
    else:
        print("\n⚠ Skipping performance tests due to correctness failures")
    
    print("\n" + "=" * 60)
    print("Comparison complete!")
    print("=" * 60)
