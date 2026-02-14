#!/usr/bin/env python3
"""
Setup script for building the Cython-optimized offset_subpixel module.

Usage:
    python setup_offset_subpixel.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

import platform

# Determine appropriate compiler flags based on architecture
extra_compile_args = ["-O3"]
if platform.machine() in ['arm64', 'aarch64']:
    # Apple Silicon (M1/M2/M3) - don't use -march=native
    pass  # -O3 is sufficient
elif platform.system() == 'Darwin':
    # Intel Mac
    extra_compile_args.append("-march=native")
else:
    # Linux/Windows
    extra_compile_args.append("-march=native")

extensions = [
    Extension(
        "offset_subpixel_fast",
        ["offset_subpixel_fast.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=extra_compile_args,
    )
]

setup(
    name="offset_subpixel_fast",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'nonecheck': False,
            'cdivision': True,
        },
        annotate=True,  # Creates HTML file showing Python/C interactions
    ),
    zip_safe=False,
)
