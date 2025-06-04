"""Visualization utilities for segmentations."""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from librosa.display import TimeFormatter
import numpy as np
import itertools
from cycler import cycler
from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING


# Import Segment for type hinting only to avoid circular dependency
if TYPE_CHECKING:
    from .core import Segment


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
    # Find unique elements in labels. Labels can be list of arrays, list of labels, or a single array
    unique_labels = np.unique(
        np.concatenate([np.atleast_1d(np.asarray(l)) for l in labels])
    )
    # This modification ensures that even a single label is treated as a 1-dimensional array.

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

    # Create a mapping of labels to styles by cycling through the properties
    # and assigning them to the labels as they appear in the unique labels' ordering
    seg_map = dict()
    for lab, properties in zip(unique_labels, p_cycler):
        # set style according to p_cycler
        style = {
            k: v
            for k, v in properties.items()
            if k in ["color", "facecolor", "edgecolor", "linewidth", "hatch"]
        }
        # Swap color -> facecolor here so we preserve edgecolor on rects
        if "color" in style:
            style.setdefault("facecolor", style["color"])
            style.pop("color", None)
        seg_map[lab] = dict(linewidth=1, edgecolor=boundary_color)
        seg_map[lab].update(style)
        seg_map[lab].update(kwargs)
        seg_map[lab]["label"] = lab
    return seg_map


def _plot_intervals_and_labels(
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
    seg: "Segment",
    ax: Optional[plt.Axes] = None,
    text: bool = False,
    ytick: str = "",
    time_ticks: bool = True,
    style_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a Segment object.

    Parameters
    ----------
    seg : bnl.core.Segment
        The Segment object to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    text : bool, default=False
        Whether to display segment labels as text on the plot.
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

    # Set plot limits and content
    sorted_beta = seg._sorted_boundaries
    
    if seg.num_segments != 0:
        ax.set_xlim(sorted_beta[0], sorted_beta[-1])
        _plot_intervals_and_labels(
            seg.itvls, seg.labels, ax, text=text, style_map=style_map
        )
    else:
        # Handle empty segments
        start_time = sorted_beta[0] if sorted_beta else 0.0
        end_time = sorted_beta[-1] if len(sorted_beta) > 1 else start_time + 1.0
        if start_time == end_time:
            end_time = start_time + 1.0
        ax.set_xlim(start_time, end_time)
        ax.text(
            0.5, 0.5, "Empty Segment",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color="gray"
        )       

    # Apply common styling
    ax.set_ylim(0, 1)

    if time_ticks:
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_major_formatter(TimeFormatter())
        ax.set_xlabel("Time (s)")
    else:
        ax.set_xticks([])

    if ytick:
        ax.set_yticks([0.5])
        ax.set_yticklabels([ytick])
    else:
        ax.set_yticks([])

    return fig, ax