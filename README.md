# npimage
Need to load pixel values from image files as numpy arrays, and hate having to remember whether you should use PIL, tifffile, matplotlib, or something else? Hate having to deal with the fact that those libraries all use different function names and syntaxes? Wish you could just provide a filename and get back a numpy array? This library's `core.py` does that, with `array = load(filename)`, `save(array, filename)`, and `show(array)` functions that let you easily handle a number of common image file formats without having to remember library-specific syntax.

Want to draw simple shapes like lines, triangles, and circles into 3D numpy arrays? Frustrated that the python libraries you can find online like `opencv` and `skimage.draw` work on 2D arrays but not 3D? I wrote some functions in `graphics.py` that do the trick in 3D. (If you know of another library that can do this, please let me know!)

### Installation

**Either** `pip install` this package directly from GitHub:
    
    pip install git+https://github.com/jasper-tms/npimage.git

**or** first `git clone` it and then `pip install` it from your clone:

    cd ~/repos  # Or wherever on your computer you want to download this code to
    git clone https://github.com/jasper-tms/npimage.git
    cd npimage
    pip install .
