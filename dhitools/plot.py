"""
Dfsu plotting functions

Authors: 
Robert Wall (original dhitools package),
Alex Waterhouse (expanded functionality)

This module contains various plotting functions that have been designed to
be able to handle triangular and quadrilateral mesh types.

Various plotting utility have also been added for ease of access when post-processing. 
"""

from . import _utils
from . import units
from . import config

import numpy as np
import geopandas as gpd
import datetime as dt
import os
import clr

# Plotting utilities
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.collections import PolyCollection
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from descartes import PolygonPatch
import seaborn as sns

plt.style.use("seaborn")
sns.set(style="white")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.size"] = "12"


def mesh_plot(x, y, element_table, ax=None, kwargs=None):
    """ 
    Triplot of the mesh 
    """
    if kwargs is None:
        kwargs = {}

    if ax is None:
        fig, ax = plt.subplots()
        plt.gca().set_aspect("equal")
    else:
        fig = plt.gcf()

    if element_table.shape[1] % 3 == 0:
        # If the mesh is triangular elements only

        # Subtract 1 from element table to align with Python indexing
        t = tri.Triangulation(x, y, element_table - 1)

        ax.triplot(t, **kwargs)

    elif element_table.shape[1] % 4 == 0:
        # If the mesh consists of quad and tri elements
        kwargs["facecolor"] = kwargs.pop("facecolor", "None")
        kwargs["edgecolor"] = kwargs.get("edgecolor", "black")

        xy = np.c_[x, y]
        verts = xy[element_table - 1]

        pc = PolyCollection(verts, **kwargs)

        ax.add_collection(pc)
        ax.autoscale()

    return fig, ax


def filled_mesh_plot(x, y, z, element_table, ax=None, kwargs=None):
    """ 
    Tricontourf of the mesh and input z
    """
    if kwargs is None:
        kwargs = {}
    kwargs["antialiased"] = kwargs.get("antialiased", "True")

    if ax is None:
        fig, ax = plt.subplots()
        plt.gca().set_aspect("equal")
    else:
        fig = plt.gcf()

    # Subtract 1 from element table to align with Python indexing
    t = tri.Triangulation(x, y, element_table - 1)

    tf = ax.tricontourf(t, z, **kwargs)

    # Remove bounding edge contour lines.
    for c in tf.collections:
        c.set_edgecolor("face")

    return fig, ax, tf


def geoimread(imname, tfwname):
    IM = {}
    IM["I"] = plt.imread(imname)

    with open(tfwname, "r") as WorldFile:
        XCellSize = float(WorldFile.readline())
        rot1 = WorldFile.readline()  # should be 0
        rot2 = WorldFile.readline()  # should be 0
        YCellSize = float(WorldFile.readline())
        WorldX = float(WorldFile.readline())
        WorldY = float(WorldFile.readline())

    Rows, Cols, _ = IM["I"].shape

    XMin = WorldX - (XCellSize / 2)
    YMax = WorldY - (YCellSize / 2)
    XMax = (WorldX + (Cols * XCellSize)) - (XCellSize / 2)
    YMin = (WorldY + (Rows * YCellSize)) - (YCellSize / 2)
    BBox = (XMin, XMax, YMin, YMax)

    IM["geo"] = {"XMin": XMin, "XMax": XMax, "YMin": YMin, "YMax": YMax, "BBox": BBox}

    return IM


def inpol(PolVerticies, X, Y):
    x, y = X.flatten(), Y.flatten()
    points = np.vstack((x, y)).T

    p = Path(PolVerticies)  # make a polygon
    grid = p.contains_points(points)

    # Mask with points inside a polygon
    return grid.reshape(X.shape)


def vert_colorbar(m, ax, label=None):
    """
    Vertical colorbar that matches the size of the axis.
    
    Requires: 
        m:      mappable object
        ax:     Axis to apply colorbar to
    
    Optional:
        label:  String colorbar label
    
    Returns:
        cb:     colorbar
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.075)
    cb = plt.colorbar(m, ax=ax, cax=cax, extend="neither")
    cb.ax.get_yaxis().labelpad = 15

    if label is not None:
        cb.set_label(label, rotation=270, fontweight="bold")

    return cb


def plot_shapefile(sf, plt_args, geom, ax):
    """
    Plot a background shapefile for SWAN Result pdict
    This will pdict all features in the shapefile.
    
    sf        : sf = shapefile.Reader(SHAPEFILE_FNAME)
    plt_args  : arguments to pass through to plt pdict(x,y,plt_args)
    
    """
    for shape in sf.shapes():
        shape_geo = shape.__geo_interface__
        if geom == "Polygon":
            ax.add_patch(PolygonPatch(shape_geo, **plt_args))
        else:
            tmp = list(zip(*shape_geo["coordinates"][:]))
            plt.plot(tmp[0], tmp[1], **plt_args)
    return


def fix_figure_margin(plot_margin=0.25):
    """
    Fix figure margins (remove whitespace) for printing images to bitmap/vector formats. 
    """
    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0 - plot_margin, x1 + plot_margin, y0 - plot_margin, y1 + plot_margin))
    return
