# npimage
Need to load pixel values from image files as numpy arrays, and hate having to remember whether you should use PIL, tifffile, matplotlib, or something else? Hate having to deal with the fact that those libraries all use different function names and syntaxes? Wish you could just provide a filename and get back a numpy array? This library's `core.py` does that, with `array = load(filename)`, `save(array, filename)`, and `show(array)` functions that let you easily handle a number of common image file formats without having to remember library-specific syntax.

Want to draw simple shapes like lines, triangles, and circles into 3D numpy arrays? Frustrated that the python libraries you can find online like `opencv` and `skimage.draw` work on 2D arrays but not 3D? I wrote some functions in `graphics.py` that do the trick in 3D. (If you know of another library that can do this, please let me know!)


### Documentation
- `core.py`: load, save, or show images.
- `graphics.py`: draw points, lines, triangles, circles, or spheres into 2D or 3D numpy arrays representing image volumes.
- `nrrd_utils.py`: compress or read metadata from `.nrrd` files.

For now, check each function's docstring for more. A jupyter notebook demonstrating this package's functions will come later.


### Installation

As is always the case in python, consider making a virtual environment (using your preference of conda, virtualenv, or virtualenvwrapper) before installing.

**Option 1:** `pip install` from PyPI:

    pip install numpyimage

(Unfortunately the name `npimage` was already taken on PyPI, so `pip install npimage` will get you a different package.)

**Option 2:** `pip install` directly from GitHub:
    
    pip install git+https://github.com/jasper-tms/npimage.git

**Option 3:** First `git clone` this repo and then `pip install` it from your clone:

    cd ~/repos  # Or wherever on your computer you want to download this code to
    git clone https://github.com/jasper-tms/npimage.git
    cd npimage
    pip install .

**After installing,** you can import this package in python using `import npimage` (not `import numpyimage`!)
