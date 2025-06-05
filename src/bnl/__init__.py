"""BNL: A Python library for hierarchical text segmentation and evaluation.

This library provides tools for working with hierarchical text segments.

Submodules
----------
core
    Core data structures and functionality.
viz
    Visualization utilities for segmentations.
"""

__version__ = "0.1.0"

from .core import Segment, seg_from_itvls
from . import viz

# Explicitly import viz functions to make them available at package level
from .viz import plot_segment, label_style_dict

__all__ = [
    "Segment",
    "seg_from_itvls",
    "viz",
    "plot_segment",
    "label_style_dict",
]
