"""Data loading and management for BNL.

This module provides convenient access to musical structure datasets,
with primary support for SALAMI dataset, and future extensibility
for other datasets.

Examples
--------
>>> import bnl.data as data
>>> # Load a single SALAMI track
>>> track = data.slm.load_track(10)
>>> print(track.info)

>>> # Load multiple tracks
>>> tracks = data.slm.load_tracks([10, 100, 1000])

>>> # Get dataset info
>>> tids = data.slm.list_tids()
>>> print(f"Available tracks: {len(tids)}")
"""

from .base import get_config, set_config
from . import salami as slm

__all__ = [
    "get_config",
    "set_config",
    "slm",
]
