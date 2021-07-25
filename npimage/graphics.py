#!/usr/bin/env python3
# This package contains functions for drawing basic geometric shapes, like
# lines and triangles, into 3D numpy arrays (and the code almost works for nD
# arrays). opencv has drawline and drawtriangle functions, but those only
# operate on 2D image arrays. scipy.ndimage provides many other helpful,
# related functions for working with nD image arrays.

# Subpixel indexing conventions:
#
# 'corner' convention (default): An operation that tries to get or set a pixel
# located at (x, y, z) will actually get or set the pixel with index (floor(x),
# floor(y), floor(z)). Therefore the pixel at location (x, y, z) represents the
# cube of physical locations (a, b, c) such that a is in [x, x+1), b is in [y,
# y+1), and c is [z, z+1). This is called the 'corner' convention because
# integer locations like (5, 2, 10) point to the corner of a voxel.
#
# 'center' convention: An operation that tries to get or set a pixel located at
# (x, y, z) will actually get or set the pixel with index (round(x), round(y),
# round(z)). Therefore the pixel at location (x, y, z) represents the cube of
# physical locations (a, b, c) such that a is in [x-0.5, x+0.5), b is in
# [y-0.5, y+0.5), and c is in [z-0.5, z+0.5). This is called the 'center'
# convention because integer locations like (5, 2, 10) point to the middle of a
# voxel. This convention can be selected by passing convention='center' to any
# of the functions in this package (once I finish implementing it).
# 
# Functions in this package use non-integer data types to preserve accuracy,
# only flooring or rounding (depending on the convention) during get-pixel or
# set-pixel operations.


# LIST OF MAJOR TODOS

# Implement 'center' convention logic for all functions.

# Make the image argument optional for drawpoint, drawline, and drawtriangle,
# in which case they should create an image just large enough to hold the drawn
# object and then return that image

# Make get_voxels_within_distance be able to take a multidimensional distance
# argument, in which case it should return all the voxels within an ellipse
# having its radii along each axis specified by the elements of distance. This
# is an extension of the current functionality, since currently a sphere is
# returned with radius in all dimensions equal to the int/float distance.


import math
import itertools

import numpy as np
np.set_printoptions(suppress=True)

from .utils import *


# --- Primitive shapes - points, lines, triangles, circles, spheres --- #
#TODO out_of_bounds='raise' isn't actually raising a lot of the time. Fix it.
def drawpoint(image, coord, value, thickness=1,
              convention='corner', out_of_bounds='ignore'):
    """
    TODO docstring
    """
    coords = thicken(coord, thickness)
    imset(image, coords, value,
          convention=convention, out_of_bounds=out_of_bounds)


# TODO switch to kwargs
def drawline(image, pt1, pt2, value, thickness=1,
             convention='corner', out_of_bounds='ignore'):
    """
    DDA algorithm described in
    https://www.tutorialspoint.com/computer_graphics/line_generation_algorithm.htm
    with some additional logic that allows for non-integer point coordinates.
    """
    pt1, pt2, long_axis = preprocess_polygon_vertices(pt1, pt2)

    # Deal with the start and end points as special - always draw them
    drawpoint(image, pt1, value, thickness=thickness,
              convention=convention, out_of_bounds=out_of_bounds)
    drawpoint(image, pt2, value, thickness=thickness,
              convention=convention, out_of_bounds=out_of_bounds)

    vec_1to2 = pt2 - pt1
    if eq(vec_1to2[long_axis], 0):
        return  # The two points are the same, don't need to do anything else

    if convention == 'corner':
        # Will be marking a pixel every time the line connecting pt1 and
        # pt2 has a long_axis value ending in .5. Find the first and
        # last times that happens, ignoring the endpoints themselves:
        pt1_adjustment = ifloor(pt1[long_axis] + 0.5) + 0.5 - pt1[long_axis]
        pt2_adjustment = iceil(pt2[long_axis] - 0.5) - 0.5 - pt2[long_axis]
        pt1 += vec_1to2 * pt1_adjustment / vec_1to2[long_axis]
        pt2 += vec_1to2 * pt2_adjustment / vec_1to2[long_axis]
        #print(f'Adjusted endpoints:\npt1={pt1}\npt2={pt2}')
    elif convention == 'center':
        #TODO Test this block more rigorously for corner case performance
        # Will be marking a pixel every time the line connecting pt1 and
        # pt2 has a long_axis value that's an integer. Find the first and
        # last times that happens, ignoring the endpoints themselves:
        pt1_adjustment = ifloor(pt1[long_axis] + 1) - pt1[long_axis]
        pt2_adjustment = ifloor(pt2[long_axis] + 1) - pt2[long_axis]
        pt1 += vec_1to2 * pt1_adjustment / vec_1to2[long_axis]
        pt2 += vec_1to2 * pt2_adjustment / vec_1to2[long_axis]

    if pt1[long_axis] > pt2[long_axis]:
        return  # Line is so short that no more points need to be marked

    vec_1to2 = pt2 - pt1
    n_steps = vec_1to2[long_axis]
    assert eq(n_steps, iround(n_steps)), 'n_steps supposed to be integer'
    n_steps = iround(n_steps)

    if n_steps == 0:  # pt1 == pt2, so only one point to mark
        drawpoint(image, pt1, value, thickness=thickness,
                  convention=convention, out_of_bounds=out_of_bounds)
        return

    pts = np.outer(np.arange(n_steps+1) / n_steps, vec_1to2) + pt1
    # For some reason I don't understand, calling thicken and then imset runs
    # slightly (20%) faster than calling drawpoint - despite drawpoint
    # literally just calling thicken and then imset itself. ???
    pts = thicken(pts, thickness)
    imset(image, pts, value, 
          convention=convention, out_of_bounds=out_of_bounds)


# TODO switch to kwargs
def drawtriangle(image, pt1, pt2, pt3, value, thickness=1,
                 fill_value=None, convention='corner', out_of_bounds='ignore'):

    if fill_value is None:
        drawline(image, pt1, pt2, value, thickness=thickness,
                 convention=convention, out_of_bounds=out_of_bounds)
        drawline(image, pt2, pt3, value, thickness=thickness,
                 convention=convention, out_of_bounds=out_of_bounds)
        drawline(image, pt3, pt1, value, thickness=thickness,
                 convention=convention, out_of_bounds=out_of_bounds)
        return

    pt1, pt2, pt3, long_axis = preprocess_polygon_vertices(pt1, pt2, pt3)

    vec_1to3 = pt3 - pt1
    if eq(vec_1to3[long_axis], 0):
        return  # The three points are the same, don't need to do anything else
    vec_1to2 = pt2 - pt1
    vec_2to3 = pt3 - pt2

    pt1_adjustment = ifloor(pt1[long_axis] + 0.5) + 0.5 - pt1[long_axis]
    pt3_adjustment = iceil(pt3[long_axis] - 0.5) - 0.5 - pt3[long_axis]
    pt1_adjusted = pt1 + vec_1to3 * pt1_adjustment / vec_1to3[long_axis]
    pt3_adjusted = pt3 + vec_1to3 * pt3_adjustment / vec_1to3[long_axis]

    if pt1_adjusted[long_axis] > pt3_adjusted[long_axis]:
        return  # Triangle is so small that no more points need to be marked

    vec_1to3_adj = pt3_adjusted - pt1_adjusted
    n_steps = vec_1to3_adj[long_axis]
    assert eq(n_steps, iround(n_steps)), 'n_steps supposed to be integer'
    n_steps = iround(n_steps)

    try:
        iter(thickness)
    except:
        thickness = np.full(len(image.shape), thickness)

    def draw_perpendicular_from_triangle_base(basept):
        if basept[long_axis] < pt2[long_axis]:
            endpt_adjustment = basept[long_axis] - pt1[long_axis]
            endpt = pt1 + vec_1to2 * endpt_adjustment / vec_1to2[long_axis]
        else:
            endpt_adjustment = basept[long_axis] - pt2[long_axis]
            endpt = pt2 + vec_2to3 * endpt_adjustment / vec_2to3[long_axis]

        assert eq(basept[long_axis], endpt[long_axis]), f'{long_axis}, \
        {basept}, {basept[long_axis]}, {endpt}, {endpt[long_axis]}'

        # To ensure 3D triangles end up water-tight, need the thickness of the
        # filler lines must be >1 in at least one dimension.
        if all(thickness == 1):
            slopes = abs(endpt - basept)
            if all(eq(slopes, 0)):
                pass
            # Thickening along the axis with the largest slope results in the
            # fewest points being added, while still guaranteeing no holes.
            #TODO generalize this to more than 3 dimensional arrays. Just need
            #to find the axis with the max slope, other than long_axis
            elif slopes[long_axis - 1] > slopes[long_axis - 2]:
                thickness[long_axis - 1] += 1
            else:
                thickness[long_axis - 2] += 1

        drawline(image, basept, endpt, fill_value, thickness=thickness,
                 convention=convention, out_of_bounds=out_of_bounds)

    if n_steps == 0:
        draw_perpendicular_from_triangle_base(pt1_adjusted)
        return
    for i in range(n_steps + 1):  # runs if n_steps > 0
        startpt = pt1_adjusted + (i / n_steps) * vec_1to3_adj
        draw_perpendicular_from_triangle_base(startpt)

    drawline(image, pt1, pt2, value, thickness=thickness,
             convention=convention, out_of_bounds=out_of_bounds)
    drawline(image, pt2, pt3, value, thickness=thickness,
             convention=convention, out_of_bounds=out_of_bounds)
    drawline(image, pt3, pt1, value, thickness=thickness,
             convention=convention, out_of_bounds=out_of_bounds)


def drawcircle(image, center, perpendicular, radius, value,
               fill=False, spacing=1, **kwargs):
    """
    Draw a circle
    See drawpoint for all kwargs.
    """
    ndims = 3  #TODO code up logic for determining this from the input
    return_image = False
    if image is None and center is None:
        image = np.zeros((2*iceil(radius) + 1,)*ndims, dtype=np.uint8)
        center = (radius, radius, radius)
        kwargs['convention'] = 'center'
        return_image = True

    norm = lambda v: np.sqrt(sum(v**2))
    perpendicular = np.array(perpendicular)
    if eq(norm(perpendicular), 0):
        print('WARNING: Perpendicular had 0 length. No circle was drawn.')
    perpendicular = perpendicular / norm(perpendicular)

    ax = np.array((1,) + (0,) * (ndims-1))
    v0 = np.cross(ax, perpendicular)
    if eq(norm(v0), 0):
        v0 = np.cross(np.flip(ax), perpendicular)
    v1 = np.cross(v0, perpendicular)
    v0 = v0 / norm(v0)
    v1 = v1 / norm(v1)

    half_circumfrence = math.pi*radius
    num_pts = iceil(math.pi*radius/spacing) + 1
    angles = np.linspace(0, math.pi, num_pts)
    half1 = (np.outer(np.cos(angles), v0)
             + np.outer(np.sin(angles), v1)) * radius + center
    half2 = (np.outer(np.cos(angles), v0)
             - np.outer(np.sin(angles), v1)) * radius + center
    if fill:
        #import npimage # for testing
        for pt1, pt2 in zip(half1, half2):
            drawline(image, pt1, pt2, value, **kwargs)
            #print(f'Just drew {pt1}->{pt2}')  # for testing
            #npimage.imshow(image[0], mode='mpl')  # for testing
    else:
        draw = lambda pts: imset(image, pts, value, **kwargs)
        draw(half1)
        draw(half2)

    if return_image:
        return image


def drawsphere(*args, **kwargs):
    drawneighborhood(*args, **kwargs)


# TODO switch to kwargs
def drawneighborhood(image=None, distance=None, ndims=None, value=1,
                     center=None, voxel_size=None, metric='euclidian',
                     out_of_bounds='ignore'):
    """
    Draw a value into all pixels within a certain distance of a location.
    This is almost the same thing as drawpoint, but drawpoint's thickness and
    drawneighborhood's distance work differently.
    If you want an image made for you with size just large enough to fit the
    requested neighborhood, omit the image argument.
    """
    if ndims is None:
        try:
            ndims = len(distance)
        except:
            try:
                ndims = len(image.shape)
            except:
                if voxel_size is not None:
                    ndims = len(voxel_size)
                elif center is not None:
                    ndims = len(center)
                else:
                    raise Exception('Specify the dimensionality you want by'
                                    "passing 'ndims='")
            
    if distance is None and image is None:
        raise Exception('Must specify distance or image.')

    # TODO activate this block and test it out once multidimensional distances
    # can be accepted by get_voxels_within_distance
    if False:
        if distance is None:
            distance = np.array(image.shape) / 2
        try:
            if len(distance) != ndims:
                raise Exception('len(distance) must equal ndims')
        except:
            distance = np.array((distance,) * ndims)

        if center is None:
            center = distance
        else:
            center = np.array(center)
    else:
        if distance is None:
            if center is None:
                distance = (min(image.shape) - 1) // 2
                center = (np.array(image.shape) - 1) // 2
            else:
                dist_to_top_or_left = min(center)
                dist_to_bot_or_right = min(np.array(image.shape) - 1 - center)
                distance = min(dist_to_top_or_left, dist_to_bot_or_right)

        if center is None:
            center = np.array((distance,) * ndims)
        else:
            center = np.array(center)

    return_image = False
    if image is None:
        return_image = True
        if value >= 0 and value <= 255:
            image = np.zeros(ifloor(center + distance + 1), dtype=np.uint8)
        else:
            image = np.zeros(ifloor(center + distance + 1))

    #print(f'center={center}')
    #print(f'distance={distance}')
    #print(f'image.shape={image.shape}')

    voxels = get_voxels_within_distance(distance, center=center, ndims=ndims,
                                        voxel_size=voxel_size, metric=metric)
    imset(image, voxels, value, out_of_bounds=out_of_bounds)

    if return_image:
        return image


# --- Low-level wrappers for getting and setting values from arrays --- #
def imget(image, coords, convention='corner',
          out_of_bounds='ignore', verbose=True):
    """
    Given a numpy array, get a value from a particular coordinate or set of
    coordinates.
    By default, coordinates with negative indices are ignored, i.e. this
    function won't wrap negative indexes around to access the end of the array,
    despite numpy arrays normally wrapping negative indices. Refusing to wrap
    makes more sense in most graphics applications.
        out_of_bounds, string   : 'ignore' (default), 'wrap', or 'raise'
    """
    assert convention in ['corner', 'center']
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)
    ax = len(coords.shape) - 1

    if convention == 'corner':
        is_negative = coords < 0
        exceeds_dimensions = coords >= image.shape
    elif convention == 'center':
        is_negative = coords < -0.5
        exceeds_dimensions = coords + 0.5 >= image.shape

    is_out_of_bounds = np.logical_or(is_negative, exceeds_dimensions)
    if (is_out_of_bounds).any():
        def warn():
            print('WARNING: Some requested coordinates are out of bounds.'
                  ' The returned list of values will be shorter than the'
                  ' request list, and therefore the returned values will not'
                  ' match 1-to-1 with the requested coordinates.')
        if out_of_bounds == 'ignore':
            coords = coords[~is_out_of_bounds.any(axis=ax)]
            if verbose:
                print(f'verbose={verbose}')
                warn()
        elif out_of_bounds == 'wrap':
            coords = coords[~exceeds_dimensions.any(axis=ax)]
            if verbose:
                warn()
        elif out_of_bounds == 'raise':
            raise IndexError(f'Some coordinates out of bounds:\n'
                             '{coords[is_out_of_bounds]}')
        else:
            raise ValueError("out_of_bounds must be 'ignore', 'raise', or"
                             f"'wrap' but was {out_of_bounds}.")

    if convention == 'corner':
        return image[tuple(ifloor(coords).T)]
    elif convention == 'center':
        return image[tuple(iround(coords).T)]


def imset(image, coords, value, convention='corner', out_of_bounds='ignore'):
    """
    Given a numpy array, set a particular coordinate or set of coordinates to
    have a given value.
    By default, coordinates with negative indices are ignored, i.e. this
    function won't wrap negative indexes around to access the end of the array,
    despite numpy arrays normally wrapping negative indices. Refusing to wrap
    makes more sense in most graphics applications.
        out_of_bounds, string   : 'ignore' (default), 'wrap', or 'raise'
    """
    assert convention in ['corner', 'center']
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)

    if convention == 'corner':
        is_negative = coords < 0
        exceeds_dimensions = coords >= image.shape
    else:  # convention == 'center'
        is_negative = coords < -0.5
        exceeds_dimensions = coords + 0.5 >= image.shape

    is_out_of_bounds = np.logical_or(is_negative, exceeds_dimensions)
    if (is_out_of_bounds).any():
        if out_of_bounds == 'ignore':
            coords = coords[~is_out_of_bounds.any(axis=1)]
        elif out_of_bounds == 'wrap':
            coords = coords[~exceeds_dimensions.any(axis=1)]
        elif out_of_bounds == 'raise':
            raise IndexError(f'Some coordinates out of bounds:\n'
                             '{coords[is_out_of_bounds]}')
        else:
            raise ValueError("out_of_bounds must be 'ignore', 'raise', or"
                             f"'wrap' but was {out_of_bounds}.")

    if convention == 'corner':
        image[tuple(ifloor(coords).T)] = value
    else:  # convention == 'center'
        image[tuple(iround(coords).T)] = value


# --- Drawing shapes composed of many primitives, --- #
# --- with the shape description read from a text file --- #
# TODO switch to kwargs
def drawskeleton(image, swc_filename, value=255,
                 voxel_size=1, offset=0, convention='corner'):
    """
    Given an swc file specifying a set of nodes positioned in 3D space and
    the connections between them, draw that skeleton into a numpy array
    representing a 3D image volume.
    Works with swc files exported from CATMAID.
    Set convention='center' when using skeletons exported from Simple Neurite
    Tracer, since SNT uses the integers-at-voxel-centers convention.
    """
    outline_pts = np.genfromtxt(swc_filename)
    outline_pts[:, 2:5] = to_voxel_coordinates(outline_pts[:, 2:5],
                                               voxel_size, offset)

    n_lines = outline_pts[1:].shape[0]
    for pt in outline_pts[1:]:
        pt_idx = pt[0]
        parent_idx = pt[-1].astype(int)
        parent_pt = outline_pts[parent_idx-1, 2:5]
        pt = pt[2:5]
        print(f'{pt_idx-1:4.0f}/{n_lines}, {parent_pt} -> {pt}')
        drawline(image, parent_pt, pt, value, convention=convention)


def drawmesh(image, mesh_fn, value=255, verbose=True,
             voxel_size=1, offset=0, **kwargs):
    """
    Draw a mesh (set of triangles) into a numpy array.
    Currently supports only the stl file format.
    """
    from stl import mesh  # pip install numpy-stl
    mesh_data = mesh.Mesh.from_file(mesh_fn)
    v0 = to_voxel_coordinates(mesh_data.v0, voxel_size, offset)
    v1 = to_voxel_coordinates(mesh_data.v1, voxel_size, offset)
    v2 = to_voxel_coordinates(mesh_data.v2, voxel_size, offset)
    n = len(v0)
    if verbose:
        for i in range(n):
            print(f'Drawing triangle {i+1} of {n}')
            drawtriangle(image, v0[i], v1[i], v2[i], value, fill_value=value,
                    **kwargs)
    else:
        for i in range(n):
            drawtriangle(image, v0[i], v1[i], v2[i], value, fill_value=value,
                    **kwargs)


# --- Misc graphics operations --- #
suffixes = {1: 'st', 2: 'nd', 3: 'rd'}
def preprocess_polygon_vertices(*pts):
    """
    Given a list of polygon vertices, find the axis along which the polygon is
    longest, and order the points by their position along that axis.
    returns a tuple containing the ordered points plus an integer indicating
    the longest axis.
    Usage example:
        pt1, pt2, long_axis = preprocess_polygon_vertices(image, pt1, pt2)
    """
    # TODO TODO TODO make this aware of voxel sizes!! If voxel size is
    # anisotropic, the longest axis according to its physical space measurement
    # may not be the longest axis according to its voxel coordinates

    #assert type(image) == np.ndarray
    #n_dims = len(image.shape)

    pts = list(pts)  # Make mutable
    for i, pt in enumerate(pts):
    # This block tried to guess whether the user wants center or corner
    # convention, but now that the functions in this module explicitly include
    # convention as an argument, it's no longer needed
    #    if all(isint(pt)):
    #        pts[i] = np.array(pts[i]) + 0.5
    #        print(f'Adjusted {i+1}{suffixes.get(i+1, "th")} point to {pts[i]}'\
    #               ' so it represents the middle of its pixel')
    #    else:
        pts[i] = np.array(pts[i]).astype(np.float64)

    #    assert pts[i].shape == (n_dims,)

    longest_axis = None
    longest_axis_len = -1
    for pt1_idx in range(len(pts)):
        for pt2_idx in range(pt1_idx+1, len(pts)):
            pt1 = pts[pt1_idx]
            pt2 = pts[pt2_idx]
            #dif = ifloor(pt2) - ifloor(pt1)
            dif = pt2 - pt1
            is_long_axis = abs(dif) == max(abs(dif))
            long_axis = [ax for ax, longest in enumerate(is_long_axis)
                         if longest][0]
            #print(f'Pair {pt1_idx} {pt2_idx} -> long axis {long_axis}')
            if abs(dif[long_axis]) > longest_axis_len:
                longest_axis_len = abs(dif[long_axis])
                longest_axis = long_axis

    # Order so that the values on the long_axis are increasing
    #print(pts, longest_axis)
    pts = sorted(pts, key=lambda x: x[longest_axis])

    return tuple(pts + [longest_axis])


def thicken(pts, thickness=1, no_duplicates=True):
    """
    Given a list of points representing a shape to be drawn, thicken
    the shape by a given number of pixels and return the list of points
    comprising the thickened shape.
    """
    pts = np.array(pts)
    if len(pts.shape) == 1:
        pts = np.expand_dims(pts, axis=0)

    if thickness is 1:
        return pts

    try:
        iter(thickness)
    except:
        thickness = [thickness] * pts.shape[1]

    # TODO make an option for doing spherical thickening using
    # get_voxels_within_radius instead of the current cubic thickening. Still
    # need to code up logic in that function for different thicknesses in
    # different axes...
    #if mode == 'spherical':
        #offsets = get_voxels_within_distance(thickness)
    #else:
    offsets = np.array(list(itertools.product(*[range(-i//2 + 1, i//2 + 1)
                                       for i in thickness])))

    new_pts = np.concatenate([pts + offset for offset in offsets])


    if no_duplicates: #TODO test if the added time here is worth the savings
                      #when imsetting the resulting points
        new_pts = np.unique(new_pts, axis=0)

    return new_pts


def get_voxels_within_distance(distance, center=None, ndims=None,
                               voxel_size=None, metric='euclidian'):
    """
    Generates a list of voxel coordinates within a given distance of a given
    center point.
        center, tuple   : Voxel coordinate to be used as the center.
                            Defaults to the origin.
        distance, float : How far away from the center point to include.
        metric, string    : How to calculate distance. Options are:
                              euclidian (default)
                              manhattan (sum of coordinates)
                              chebyshev (max of coordinates)
    """
    assert metric in ['euclidian', 'manhattan', 'chebyshev']

    if ndims is None:
        if center != None:
            ndims = len(center)
        elif voxel_size != None:
            ndims = len(voxel_size)
        else:
            ndims = 3

    if center is None:
        center = (0,) * ndims
    if voxel_size is None:
        voxel_size = (1,) * ndims

    if len(center) != len(voxel_size):
        raise ValueError('These must be equal, but aren\'t: '
                         #f'ndims={ndims}\n'
                         f'len(center)={len(center)}, '
                         f'len(voxel_size)={len(voxel_size)}')

    #TODO rewrite this to be able to deal with non-integer centers but still
    #return the integer voxel locations within distance of center
    center = (center // np.array(voxel_size)).astype(int)
    max_offsets = (distance // np.array(voxel_size)).astype(int)
    # This approach starts with a list of candidate points filling the bounding
    # box of the true desired set of points, and then prunes away the points in
    # the bounding box that aren't within the distance. A faster algorithm
    # would use some closed form way of determining exactly the proper set of
    # points, without needing to iterate over and prune away extraneous points.
    candidate_pts = itertools.product(*[range(-m, m+1) for m in max_offsets])
    candidate_pts = np.array(list(candidate_pts), dtype=int)

    if metric == 'chebyshev':
        return candidate_pts + center

    norms = {
        'euclidian': lambda pt: (pt**2).sum(axis=1)**0.5,
        'manhattan': lambda pt: pt.sum(axis=1),
        'chebyshev': lambda pt: pt.max(axis=1)  # This won't actually get used
                                                # because it returns from above
    }
    d = norms[metric]

    candidate_pts_scaled = candidate_pts * voxel_size
    return candidate_pts[d(candidate_pts_scaled) <= distance] + center


def floodfill(image, seed, fill_value, fill_diagonally=False,
              debug_frequency=None, verbose=False, tolerance=1e-8):
    """
    This implementation is very slow and needs improvement
    """
    import npimage
    seed_value = image[seed]
    if verbose: print(f'seed_value={seed_value}')
    if fill_diagonally:
        neighbors = get_voxels_within_distance(1, ndims=len(image.shape),
                metric='chebyshev')
    else:
        neighbors = get_voxels_within_distance(1, ndims=len(image.shape))
        
    if verbose: print(f'neighbors={neighbors}')
    neighbors = neighbors[~(neighbors == 0).all(axis=1)]
    fill_mask = np.zeros(image.shape, dtype=bool)

    #TODO precompute which pixel values == seed_value and store that bool array
    #in memory, then access it during the loop
    #TODO try doing a 2D flood fill and confirm that I can get speeds about as
    #fast as imagej's ui flood fill - maybe depth first search is much faster?
    #TODO learn cython????

    def fill(wavefront, iteration):
        if verbose: print(f'iteration={iteration}, len(wavefront)={len(wavefront)}')
        #print(wavefront)
        #fill wavefront
        imset(fill_mask, wavefront, True)
        imset(image, wavefront, fill_value)
        #print(f'image:\n{image}')

        #add valid neighbors
        new_wavefront = []
        #for coord in wavefront:
        #    for neighbor in coord + neighbors:
        #        #print('1', neighbor)
        #        #print('2', (neighbor >= 0).all())
        #        #print('3', (neighbor < image.shape).all())
        #        #print('4', eq(imget(image, neighbor, verbose=False), seed_value, tolerance=tolerance))
        #        #print('5', imget(fill_mask, neighbor, verbose=False))
        #        if ((neighbor >= 0).all()
        #                and (neighbor < image.shape).all()
        #                and eq(imget(image, neighbor),
        #                       seed_value,
        #                       tolerance=tolerance)
        #                and not imget(fill_mask, neighbor)):
        #            #print(f'Appending {neighbor}')
        #            new_wavefront.append(neighbor)
        #        #print('\n')
        [[new_wavefront.append(neighbor) for neighbor in coord + neighbors
            if (neighbor >= 0).all()
            and (neighbor < image.shape).all()
            and eq(imget(image, neighbor), seed_value, tolerance=tolerance)
            and not imget(fill_mask, neighbor)]
        for coord in wavefront]

        if debug_frequency is not None and iteration % debug_frequency == 0:
            print(f'Writing iteration{iteration}.nrrd')
            npimage.numpy_to_imagefile(image, f'iteration{iteration}.nrrd')


        #recurseifneighbors
        if len(new_wavefront) > 0:
            new_wavefront = np.unique(new_wavefront, axis=0)
            #print(f'new wavefront:\n{new_wavefront}\n')
            #input()
            fill(new_wavefront, iteration + 1)

    fill([seed], 1)


# --- Convert between voxel coordinates and physical coordinates --- #
def to_physical_coordinates(voxel_coordinates, voxel_size, offset=0):
    """
    Convert voxel indexes to physical coordinates, using the given voxel size
    (in units of microns per voxel) and offset (indicating the physical
    coordinates of the origin voxel, in microns)
    """
    #TODO test this. Haven't yet run it.
    return voxel_coordinates * voxel_size + offset


def to_voxel_coordinates(physical_coordinates, voxel_size, offset=0):
    """
    Convert physical coordinates to voxel indices, using the given voxel size
    (in units of microns per voxel) and offset (indicating the physical
    coordinates of the origin voxel, in microns)
    Note that this will return non-integer voxel coordinates to maintain
    accuracy, and downstream functions will need to convert to int if these are
    to be used to index an array.
    """
    #print(f'physical_coordinates={physical_coordinates}')
    #print(f'offset={offset}')
    #print(f'voxel_size={voxel_size}')
    return (physical_coordinates - offset) / voxel_size



