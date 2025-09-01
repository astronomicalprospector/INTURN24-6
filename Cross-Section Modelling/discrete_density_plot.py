
import itertools
import os
import warnings
from typing import Any, Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

from kuibit import grid_data as gd
from kuibit.cactus_grid_functions import BaseOneGridFunction

# UTILITIES


def setup_matplotlib(
    params: Optional[Dict[str, Any]] = None, rc_par_file: Optional[str] = None
) -> None:
    """Setup matplotlib with some reasonable defaults for better plots.

    If ``params`` is provided, add these parameters to matplotlib's settings
    (``params`` updates ``matplotlib.rcParams``).

    If ``rc_par_file`` is provided, first set the parameters reading the values
    from the ``rc_par_file``. (``params`` has the precedence over the parameters
    read from the file.)

    Matplotlib behaves differently on different machines. With this, we make
    sure that we set all the relevant paramters that we care of to the value we
    prefer. The default values are highly opinionated.

    :param params: Parameters to update matplotlib with.
    :type params: dict

    :param rc_par_file: File where to read parameters. The file has to use
                        matplotlib's configuration language. ``params``
                        overwrites the values set from this file, but this file
                        overrides the default values set in this function.
    :type rc_par_file: str

    """

    matplotlib.rcParams.update(
        {
            "lines.markersize": 4,
            "axes.labelsize": 16,
            "font.weight": "light",
            "font.size": 16,
            "legend.fontsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.formatter.limits": [-3, 3],
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "image.cmap": "inferno",
            "legend.fancybox": False,
            "legend.edgecolor": "inherit",
        }
    )

    if rc_par_file is not None:
        matplotlib.rc_file(rc_par_file)

    if params is not None:
        matplotlib.rcParams.update(params)


def preprocess_plot(func):
    """Decorator to set-up plot functions.

    When we plot anything, there is always some boilerplate that has to be
    executed. For example, we want to provide an axis keyword so that the user
    can specify where to plot, but if the keyword is not provided, we want to
    plot on the current figure.

    Essentially, this decorator sets default values. Why don't we do
    axis=plt.gca() then? The problem is that the default values are set when
    the function is defined, not when it is called. So, this will not work.

    This decorator takes care of everything.

    1. It handles the axis keyword setting it to plt.gca() if it was not
       provided.
    2. It handles the figure keyword setting it to plt.gcf() if it was not
       provided.

    func has to take as keyword arguments:
    1. 'axis=None', where the plot will be plot, or plt.gca() if None
    2. 'figure=None', where the plot will be plot, or plt.gcf() if None

    """

    def inner(*args, **kwargs):
        # Setdetault addes the key if it is not already there
        kwargs.setdefault("axis", plt.gca())
        kwargs.setdefault("figure", plt.gcf())
        return func(*args, **kwargs)

    return inner


def preprocess_plot_grid(func):
    """Decorator to set-up plot functions that plot grid data.

    This decorator extends :py:func:`~.preprocess_plot` for specific functions.

    1. It handles differt types to plot what intuitively one would want to
       plot.
    1a. If the data is a NumPy array with shape 2, just pass the data,
        otherwise raise an error
    1b. If the data is a NumPy array, just pass the data.
    1c. If data is :py:class:`~.UniformGridData`, pass the data and the
        coordinates.
    1d. If data is :py:class:`~.HierarchicalGridData`, read resample it to
        the given grid, then pass do 1c.
    1e. If data is a :py:class:`~.BaseOneGridFunction`, we read the iteration
        and pass to 1d.

    func has to take as keyword arguments (in addition to the ones in
    :py:func`~.preprocess_plot`):
    1. 'data'. data will be passed as a NumPy array, unless it is
               already so.
    2. 'coordinates=None'. coordinates will be passed as a list of NumPy
                           arrays, unless it is not None. Each NumPy
                           array is the coordinates along one axis.

    """

    @preprocess_plot
    def inner(data, *args, **kwargs):
        # The flow is: We check if data is BaseOneGridFunction or derived. If
        # yes, we read the requested iteration. Then, we check if data is
        # HierachicalGridData, if yes, we resample to UniformGridData. Then we
        # work with UniformGridData and handle coordinates, finally we work
        # with NumPy arrays, which is what we pass to the function.

        def attr_not_available(attr):
            """This is a helper function to see if the user passed an attribute
            or if the attribute is None
            """
            return attr not in kwargs or kwargs[attr] is None

        def default_or_kwargs(attr, default):
            """Return default if the attribute is not available in kwargs, otherwise return
            the attribute

            """
            if attr_not_available(attr):
                return default
            return kwargs[attr]

        if isinstance(data, BaseOneGridFunction):
            if attr_not_available("iteration"):
                raise TypeError(
                    "Data has multiple iterations, specify what do you want to plot"
                )

            # Overwrite data with HierarchicalGridData
            data = data[kwargs["iteration"]]

        if isinstance(data, gd.HierarchicalGridData):
            if attr_not_available("shape"):
                raise TypeError(
                    "The data must be resampled but the shape was not provided"
                )

            # If x0 or x1 are None, we use the ones of the grid
            x0 = default_or_kwargs("x0", data.x0)
            x1 = default_or_kwargs("x1", data.x1)
            resample = default_or_kwargs("resample", False)

            # Overwrite data with UniformGridData
            if data.is_masked():
                warnings.warn(
                    "Mask information will be lost with the resampling"
                )

            data = data.to_UniformGridData(
                shape=kwargs["shape"], x0=x0, x1=x1, resample=resample
            )

        if isinstance(data, gd.UniformGridData):
            # We check if the user has passed coordinates too.
            if "coordinates" in kwargs and kwargs["coordinates"] is not None:
                warnings.warn(
                    "Ignoring provided coordinates (data is UniformGridData)."
                    " To specify boundaries, use x0 and x1."
                )

            # If x0 or x1 are None, we use the ones of the grid
            x0 = default_or_kwargs("x0", data.x0)
            x1 = default_or_kwargs("x1", data.x1)
            # If x0 or x1 are provided, then we resample. So, we don't resample
            # only if x0 AND x1 are not provided.
            resampling = not (
                attr_not_available("x0") and attr_not_available("x1")
            )

            if resampling and attr_not_available("shape"):
                raise TypeError(
                    "The data must be resampled but the shape was not provided"
                )

            if resampling:
                resample = default_or_kwargs("resample", False)
                new_grid = gd.UniformGrid(shape=kwargs["shape"], x0=x0, x1=x1)

                if data.is_masked():
                    warnings.warn(
                        "Mask information will be lost with the resampling"
                    )

                data = data.resampled(
                    new_grid, piecewise_constant=(not resample)
                )

            kwargs["coordinates"] = data.coordinates_from_grid()
            # Overwrite data with NumPy array
            data = data.data_xyz

        if isinstance(data, np.ndarray) and data.ndim != 2:
            raise ValueError("Only 2-dimensional data can be plotted")

        # TODO: Check that coordinates are compatible with data

        # We remove what we don't need from kwargs, so that it is not
        # accidentally passed to the function
        def remove_attributes(*attributes):
            for attr in attributes:
                if attr in kwargs:
                    del kwargs[attr]

        remove_attributes("shape", "x0", "x1", "iteration", "resample")

        return func(data, *args, **kwargs)

    return inner

# GRID FUNCTIONS


def _vmin_vmax_extend(data, vmin=None, vmax=None):
    """Helper function to decide what to do with the colorbar (to extend it or not?)."""

    colorbar_extend = "neither"

    if vmin is None:
        vmin = data.min()

    if data.min() < vmin:
        colorbar_extend = "min"

    if vmax is None:
        vmax = data.max()

    if data.max() > vmax:
        if colorbar_extend == "min":
            colorbar_extend = "both"
        else:
            colorbar_extend = "max"

    return vmin, vmax, colorbar_extend


# All the difficult stuff is in preprocess_plot_grid
@preprocess_plot_grid
def _plot_grid(
    data,
    plot_type="color",
    figure=None,
    axis=None,
    coordinates=None,
    xlabel=None,
    ylabel=None,
    colorbar=False,
    label=None,
    logscale=False,
    vmin=None,
    vmax=None,
    aspect_ratio="equal",
    **kwargs,
):
    """Backend of the :py:func:`~.plot_color` and similar functions.

    The type of plot is specified by the variable ``plot_type``.

    Unknown arguments are passed to
    ``imshow`` if plot is color
    ``contourf`` if plot is contourf.
    ``contour`` if plot is contour.

    :param plot_type: Type of plot. It can be: 'color', 'contourf', 'contour'.
    :type plot_type: str
    """

    _known_plot_types = ("color", "contourf", "contour")

    if plot_type not in _known_plot_types:
        raise ValueError(
            f"Unknown plot_type {plot_type} (Options available {_known_plot_types})"
        )

    # Considering all the effort put in preprocess_plot_grid, we we can plot
    # as we were plotting normal NumPy arrays.

    if logscale:
        # We mask the values that are smaller or equal than 0
        data = np.ma.log10(data)

    vmin, vmax, colorbar_extend = _vmin_vmax_extend(data, vmin=vmin, vmax=vmax)

    # To implement vmin and vmax, we clamp the data to vmin and vmax instead of
    # using the options in matplotlib. This greatly simplifies handling things
    # like colormaps.
    data = np.clip(data, vmin, vmax)

    if aspect_ratio is not None:
        axis.set_aspect(aspect_ratio)

    if xlabel is not None:
        axis.set_xlabel(xlabel)
    if ylabel is not None:
        axis.set_ylabel(ylabel)

    if plot_type == "color":
        if coordinates is None:
            grid = None
        else:
            # We assume equally-spaced points.

            # TODO: (Refactoring)
            #
            # This is not a very pythonic way to write this...

            X, Y = coordinates

            dx, dy = X[1] - X[0], Y[1] - Y[0]
            grid = [
                X[0] - 0.5 * dx,
                X[-1] + 0.5 * dx,
                Y[0] - 0.5 * dy,
                Y[-1] + 0.5 * dy,
            ]

        image = axis.imshow(
            data, vmin=vmin, vmax=vmax, origin="lower", extent=grid, **kwargs
        )
    elif plot_type == "contourf":
        if coordinates is None:
            raise ValueError(
                f"You must provide the coordiantes with plot_type = {plot_type}"
            )
        levels = [0, 1, 2, 3, 3.5]
        colors = ('blue', 'yellow', 'green', 'red')
        image = axis.contourf(
            *coordinates, data, extend=colorbar_extend, levels = levels, colors = colors
        )
    elif plot_type == "contour":
        if coordinates is None:
            raise ValueError(
                f"You must provide the coordiantes with plot_type = {plot_type}"
            )
        # We need to pass the levels for the contours
        if "levels" not in kwargs:
            raise ValueError(
                f"You must provide the levels with plot_type = {plot_type}"
            )
        image = axis.contour(
            *coordinates, data, extend=colorbar_extend, **kwargs
        )

    if colorbar:
        plot_colorbar(image, figure=figure, axis=axis, label=label)

    return image


def plot_contourf(data, **kwargs):
    """Plot the given data drawing filled contours.

    You can pass (everything is processed by :py:func:`~.preprocess_plot_grid` so
    that at the end we have a 2D NumPy array):
    - A 2D NumPy array,
    - A :py:class:`~.UniformGridData`,
    - A :py:class:`~.HierarchicalGridData`,
    - A :py:class:`~.BaseOneGridFunction`.

    Depending on what you pass, you might need additional arguments.

    If you pass a :py:class:`~.BaseOneGridFunction`, you need also to pass
    ``iteration``, and ``shape``. If you pass
    :py:class:`~.HierarchicalGridData`, you also need to pass ``shape``. In all
    cases you can also pass ``x0`` and ``x1`` to define origin and corner of the
    grid. You can pass the option ``resample=True`` if you want to do bilinear
    resampling at the grid data level, otherwise, nearest neighbor resampling is
    done. When you pass the NumPy array, you also have to pass the
    ``coordinates``.

    All the unknown arguments are passed to ``contourf``.

    .. note

       Read the documentation for a concise table on what arguments are
       supported.

    :param data: Data that has to be plotted. The function expects a 2D NumPy
                 array, but the decorator :py:func:`~.preprocess_plot_grid`
                 allows it to take different kind of data.
    :type data: 2D NumPy array, or object that can be cast to 2D NumPy array.

    :param x0: Lowermost leftmost coordinate to plot. If passed, resampling will
               be performed.
    :type x0: 2D array or list

    :param x1: Uppermost rightmost coordinate to plot. If passed, resampling will
               be performed.
    :type x1: 2D array or list

    :param coordiantes: Coordinates to use for the plot. Used only if data is a
                        NumPy array.
    :type coordinates: 2D array or list

    :param shape: Resolution of the image. This parameter is used if resampling
                  is needed or requested.
    :type shape: tuple or list

    :param iteration: Iteration to plot. Relevant only if data is a
                      :py:class:`~.BaseOneGridData`.
    :type iteration: int

    :param resample: If resampling has to be done, do bilinear resampling at the
                     level of the grid data. If not passed, use nearest neighbors.
    :type resample: bool

    :param logscale: If True, take the log10 of the data before plotting.
    :type logscale: bool

    :param colorbar: If True, add a colorbar.
    :type colorbar: bool

    :param vmin: Remove all the data below this value. If logscale, this has to
                 be the log10.
    :type vmin: float
    :param vmax: Remove all the data above this value. If logscale, this has to
                 be the log10.
    :type vmax: float

    :param xlabel: Label of the x axis. If None (or not passed), no label is
                   placed.
    :type xlabel: str

    :param ylabel: Label of the y axis. If None (or not passed), no label is
                   placed.
    :type ylabel: str

    :param aspect_ratio: Aspect ratio of the plot, as passed to the function
                         ``set_aspect_ratio`` in matplotlib.
    :type aspect_ratio: str

    :param figure: If passed, plot on this figure. If not passed (or if None),
                   use the current figure.
    :type figure: ``matplotlib.pyplot.figure``

    :param axis: If passed, plot on this axis. If not passed (or if None), use
                 the current axis.
    :type axis: ``matplotlib.pyplot.axis``

    :param kwargs: All the unknown arguments are passed to ``imshow``.
    :type kwargs: dict

    """
    # This function is a convinence function around _plot_grid.
    return _plot_grid(data, plot_type="contourf", **kwargs)

@preprocess_plot
def plot_colorbar(
    mpl_artist,
    figure=None,
    axis=None,
    label=None,
    where="right",
    size="5%",
    pad=0.25,
    **kwargs,
):
    """Add a colorbar to an existing image.

    :param mpl_artist: Image from which to generate the colorbar.
    :type mpl_artist: ``matplotlib.cm.ScalarMappable``
    :param figure: If passed, plot on this figure. If not passed (or if None),
                   use the current figure.
    :type figure: ``matplotlib.pyplot.figure``

    :param axis: If passed, plot on this axis. If not passed (or if None), use
                 the current axis.
    :type axis: ``matplotlib.pyplot.axis``
    :param label: Label to place near the colorbar.
    :type label: str
    :param where: Where to place the colorbar (left, right, bottom, top).
    :type where: str
    :param size: Width of the colorbar with respect to ``axis``.
    :type size: float
    :param pad: Pad between the colorbar and ``axis``.
    :type pad: float

    """
    # The next two lines guarantee that the colorbar is the same size as
    # the plot. From https://stackoverflow.com/a/18195921
    divider = make_axes_locatable(axis)
    cax = divider.append_axes(where, size=size, pad=pad)
    cb = plt.colorbar(mpl_artist, cax=cax, **kwargs)
    if label is not None:
        cb.set_label(label)

    # When we draw a colorbar, that changes the selected axis. We do not
    # want that, so we select back the original one.
    plt.sca(axis)

    return cb