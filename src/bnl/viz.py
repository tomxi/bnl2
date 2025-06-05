"""Visualization utilities for segmentations."""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import librosa.display
import numpy as np
import itertools
from cycler import cycler
from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING


# Import Segmentation for type hinting only to avoid circular dependency
if TYPE_CHECKING:
    from .core import Segmentation  # pragma: no cover


def label_style_dict(labels, boundary_color="white", **kwargs):
    """
    Creates a mapping of labels to matplotlib style properties.

    Parameters
    ----------
    labels : list or ndarray
        List of labels (can be nested). Duplicates are processed once.
    boundary_color : str, default="white"
        Color for segment boundaries.
    **kwargs : dict
        Additional style properties to apply to all labels.

    Returns
    -------
    dict
        {label: {style_property: value}} mapping with keys like 'facecolor',
        'edgecolor', 'linewidth', 'hatch', and 'label'.
    """
    # Extract unique labels from potentially nested structure
    unique_labels = np.unique(np.concatenate([np.atleast_1d(l) for l in labels]))

    # More hatch patterns for more labels
    hatchs = ["", "..", "O.", "*", "xx", "xxO", "\\O", "oo", "\\"]
    more_hatchs = [h + "--" for h in hatchs]

    if len(unique_labels) <= 80:
        hatch_cycler = cycler(hatch=hatchs)
        fc_cycler = cycler(color=plt.get_cmap("tab10").colors)
        p_cycler = hatch_cycler * fc_cycler
    else:
        hatch_cycler = cycler(hatch=hatchs + more_hatchs)
        fc_cycler = cycler(color=plt.get_cmap("tab20").colors)
        # make it repeat...
        p_cycler = itertools.cycle(hatch_cycler * fc_cycler)

    # Create style mapping for each unique label
    seg_map = {}
    for lab, properties in zip(unique_labels, p_cycler):
        # Extract relevant style properties
        style = {
            k: v
            for k, v in properties.items()
            if k in ["color", "facecolor", "edgecolor", "linewidth", "hatch"]
        }

        # Convert color to facecolor to preserve edgecolor on rectangles
        if "color" in style:
            style["facecolor"] = style.pop("color")

        # Build final style dictionary
        seg_map[lab] = {
            "linewidth": 1,
            "edgecolor": boundary_color,
            "label": lab,
            **style,
            **kwargs,
        }
    return seg_map


def _plot_itvl_lbls(
    intervals: np.ndarray,
    labels: List[str],
    ax: plt.Axes,
    text: bool = True,
    style_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Internal helper to plot intervals and labels on a given Axes.

    Parameters
    ----------
    intervals : np.ndarray, shape=(n, 2)
        Segment intervals [start, end].
    labels : list of str
        Segment labels, must be same length as number of intervals.
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    text : bool, default=True
        Whether to display segment labels as text on the plot.
    style_map : dict, optional
        A precomputed mapping from labels to style properties.
        If None, it will be generated using `label_style_dict`.
    """
    ax.set_xlim(intervals[0][0], intervals[-1][-1])

    if style_map is None:
        style_map = label_style_dict(labels)
    transform = ax.get_xaxis_transform()

    for ival, lab in zip(intervals, labels):
        rect = ax.axvspan(ival[0], ival[1], ymin=0, ymax=1, **style_map[lab])
        if text:
            ann = ax.annotate(
                lab,
                xy=(ival[0], 1),
                xycoords=transform,
                xytext=(8, -10),
                textcoords="offset points",
                va="top",
                clip_on=True,
                bbox=dict(boxstyle="round", facecolor="white"),
            )
            ann.set_clip_path(rect)


def plot_segment(
    seg: "Segmentation",
    ax: Optional[plt.Axes] = None,
    text: bool = True,
    title: bool = True,
    ytick: str = "",
    time_ticks: bool = True,
    style_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a Segmentation object.

    Parameters
    ----------
    seg : bnl.core.Segmentation
        The Segmentation object to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    text : bool, default=True
        Whether to display segment labels as text on the plot.
    title : bool, default=True
        Whether to display a title on the axis.
    ytick : str, default=""
        Label for the y-axis. If empty, no label is shown.
    time_ticks : bool, default=True
        Whether to display time ticks and labels on the x-axis.
    style_map : dict, optional
        A precomputed mapping from labels to style properties.
        If None, it will be generated using `label_style_dict`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes.Axes
        The axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 0.6))  # short and wide
    else:
        fig = ax.figure

    # get extent of the segmentation
    t_start = seg.start
    t_end = seg.end

    # only plot if there are segments
    if len(seg) > 0:
        _plot_itvl_lbls(seg.itvls, seg.labels, ax, text=text, style_map=style_map)
    else:
        ax.text(
            0.5,
            0.5,
            "Empty Segmentation",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="gray",
        )

    if title and seg.name:
        ax.set_title(seg.name)
    ax.set_xlim(t_start, t_end)
    ax.set_ylim(0, 1)

    if time_ticks:
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_major_formatter(librosa.display.TimeFormatter())
        ax.set_xlabel("Time (s)")
    else:
        ax.set_xticks([])

    if ytick:
        ax.set_yticks([0.5])
        ax.set_yticklabels([ytick])
    else:
        ax.set_yticks([])

    return fig, ax
