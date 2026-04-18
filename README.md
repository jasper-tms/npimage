# npimage: Load, save & manipulate numpy arrays representing images & videos

[![Tests](https://github.com/jasper-tms/npimage/actions/workflows/tests.yml/badge.svg)](https://github.com/jasper-tms/npimage/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/jasper-tms/npimage/branch/main/graph/badge.svg)](https://codecov.io/gh/jasper-tms/npimage)
[![PyPI version](https://img.shields.io/pypi/v/numpyimage)](https://pypi.org/project/numpyimage/)
[![PyPI downloads](https://img.shields.io/pypi/dm/numpyimage)](https://pypi.org/project/numpyimage/)
[![License](https://img.shields.io/github/license/jasper-tms/npimage)](https://github.com/jasper-tms/npimage/blob/main/LICENSE)

Need to load pixel values from image files as numpy arrays, and hate having to remember whether you should use PIL, tifffile, matplotlib, or something else? Hate having to deal with the fact that those libraries all use different function names and syntaxes? Wish you could just provide a filename and get back a numpy array? This library's `imageio.py` does that, with `array = npimage.load(filename)`, `npimage.save(array, filename)`, and `npimage.show(array)` functions that let you easily handle a number of common image file formats without having to remember library-specific syntax. Additionally, `vidio.py` provides `array = npimage.load_video(filename)` and `npimage.save_video(array, filename)` for videos as well. (Another similar library to consider using is [imageio](https://pypi.org/project/imageio/).)

Want to draw simple shapes like lines, triangles, and circles into 3D numpy arrays? Frustrated that the python libraries you can find online like `opencv` and `skimage.draw` work on 2D arrays but not 3D/4D/+? I wrote some functions in `graphics.py` that do the trick.


### Documentation
- `imageio.py`: load, save, or show images.
- `vidio.py`: load or save videos.
- `graphics.py`: draw points, lines, triangles, circles, or spheres into 2D/3D/+D numpy arrays representing image data.
- `nrrd_utils.py`: compress or read metadata from `.nrrd` files.
- `operations.py`: perform operations on images.

Check each function's docstring for more details.


### Installation

As is always the case in python, consider making a virtual environment (using your preference of conda, virtualenv, or virtualenvwrapper) before installing.

**Option 1:** `pip install` from PyPI:

    pip install numpyimage
    pip install 'numpyimage[vid]'  # If you also want to install the video loading/saving dependencies
    pip install 'numpyimage[align]'  # If you also want to install the image alignment dependencies
    pip install 'numpyimage[all]'  # If you want to install both video and alignment dependencies

(Unfortunately the name `npimage` was already taken on PyPI, so `pip install npimage` will get you a different package.)

**Option 2:** `pip install` directly from GitHub:
    
    pip install git+https://github.com/jasper-tms/npimage.git

**Option 3:** First `git clone` this repo and then `pip install` it from your clone:

    cd ~/repos  # Or wherever on your computer you want to download this code to
    git clone https://github.com/jasper-tms/npimage.git
    cd npimage
    pip install '.[all]'

**After installing,** you can import this package in python using `import npimage` (not `import numpyimage`!)
