"""BNL: A Python library for hierarchical text segmentation and evaluation.

This library provides tools for working with hierarchical text segments.

Submodules
----------
core
    Core data structures and functionality.
viz
    Visualization utilities for segmentations.
data
    Data loading and management for musical structure datasets.
"""

__version__ = "0.1.0"

from .core import TimeSpan, Segmentation, seg_from_itvls, seg_from_brdys, Hierarchy
from . import viz
from . import data

# Explicitly import viz functions to make them available at package level
from .viz import plot_segment, label_style_dict

__all__ = [
    "TimeSpan",
    "Segmentation",
    "seg_from_itvls",
    "seg_from_brdys",
    "Hierarchy",
    "viz",
    "data",
    "plot_segment",
    "label_style_dict",
]
