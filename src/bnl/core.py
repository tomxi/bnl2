"""Core data structures and constructors."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple
import numpy as np
from mir_eval.util import boundaries_to_intervals


@dataclass
class TimeSpan:
    """A labeled time span with start and end times.

    Parameters
    ----------
    start : float
        Start time in seconds.
    end : float
        End time in seconds.
    label : str, optional
        Label for this time span.

    Examples
    --------
    >>> span = TimeSpan(start=1.0, end=3.0, name='chorus')
    >>> span.end - span.start  # duration
    2.0
    """

    start: float = 0.0
    end: float = 0.0
    name: Optional[str] = None

    def __post_init__(self):
        if self.start > self.end:
            raise ValueError(
                f"Start time ({self.start}) must be less than end time ({self.end})"
            )

    def __repr__(self) -> str:
        label_str = f", label='{self.name}'" if self.name else "○"
        return f"TimeSpan([{self.start:.2f}, {self.end:.2f}], {label_str})"

    def __str__(self) -> str:
        label_str = f" ({self.name})" if self.name else "○"
        return f"[{self.start:.2f}-{self.end:.2f}s]{label_str}"


@dataclass
class Segmentation(TimeSpan):
    """A segmentation containing multiple time spans.

    Parameters
    ----------
    segments : List[TimeSpan]
        List of TimeSpan objects representing the segments.
        Must be sorted, non-overlapping, and contiguous.

    Examples
    --------
    >>> span1 = TimeSpan(start=0.0, end=2.0, name='A')
    >>> span2 = TimeSpan(start=2.0, end=5.0, name='B')
    >>> segmentation = Segmentation(segments=[span1, span2])
    """

    segments: List[TimeSpan] = field(default_factory=list)

    def __post_init__(self):
        # order the segments by start time
        self.segments = sorted(self.segments, key=lambda x: x.start)

        # Set start/end from segments if available
        if self.segments:
            object.__setattr__(self, "start", self.segments[0].start)
            object.__setattr__(self, "end", self.segments[-1].end)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> TimeSpan:
        return self.segments[idx]

    @property
    def labels(self) -> List[Optional[str]]:
        """Get labels from all segments.

        Returns
        -------
        List[Optional[str]]
            List of labels from each segment.
        """
        return [seg.name for seg in self.segments]

    @property
    def itvls(self) -> np.ndarray:
        """Get intervals as (start, end) pairs.

        Returns
        -------
        np.ndarray
            Array of interval start and end times, shape=(n_segments, 2).
        """
        if not self.segments:
            return np.array([])
        return np.array([[seg.start, seg.end] for seg in self.segments])

    @property
    def bdrys(self) -> List[float]:
        """Get all boundaries from the segmentation.

        Returns
        -------
        List[float]
            Sorted list of all boundary times.
        """
        if not self.segments:
            return []
        boundaries = [self.segments[0].start]
        boundaries.extend([seg.end for seg in self.segments])
        return boundaries

    def __repr__(self) -> str:
        dur = self.end - self.start
        return f"Segmentation({len(self)} segments, duration={dur:.2f}s)"

    def plot(
        self,
        ax=None,
        text: bool = True,
        title: bool = True,
        ytick: str = "",
        time_ticks: bool = True,
        style_map: Optional[Dict[str, Any]] = None,
    ):
        """Plot the segmentation boundaries and labels.

        This is a convenience wrapper around `bnl.viz.plot_segment`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        text : bool, default=False
            Whether to display segment labels as text on the plot.
        title : bool, default=True
            Whether to display a title on the axis.
        ytick : str, default=""
            Label for the y-axis. If empty, no label is shown.
        time_ticks : bool, default=True
            Whether to display time ticks and labels on the x-axis.
        style_map : dict, optional
            A precomputed mapping from labels to style properties.
            If None, it will be generated using `bnl.viz.label_style_dict`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object with the plot.

        See Also
        --------
        bnl.viz.plot_segment : The underlying plotting function.
        bnl.viz.label_style_dict : Function to generate style maps.
        """
        from .viz import plot_segment

        return plot_segment(
            self,
            ax=ax,
            text=text,
            title=title,
            ytick=ytick,
            time_ticks=time_ticks,
            style_map=style_map,
        )

    def __str__(self) -> str:
        if len(self) == 0:
            return "Segmentation(0 segments): []"

        if len(self) == 1:
            # For single segments, use compact format
            dur = self.end - self.start
            return f"Segmentation({len(self)} segments, duration={dur:.2f}s): {self.segments[0]}"
        else:
            # For multiple segments, use repr-style format
            dur = self.end - self.start
            return f"Segmentation({len(self)} segments, duration={dur:.2f}s)"


@dataclass
class Hierarchy(TimeSpan):
    """A hierarchical structure of segmentations.

    A Hierarchy is a TimeSpan containing multiple layers of Segmentation objects,
    representing different levels of segmentation from coarsest to finest.

    Parameters
    ----------
    layers : List[Segmentation]
        An ordered list of Segmentation objects, from coarsest to finest levels.

    """

    layers: List[Segmentation] = field(default_factory=list)

    def __post_init__(self):
        # order the layers by start time
        self.layers = sorted(self.layers, key=lambda x: x.start)

        # Set start/end from layers if available
        if self.layers:
            object.__setattr__(self, "start", self.layers[0].start)
            object.__setattr__(self, "end", self.layers[0].end)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, lvl_idx: int) -> Segmentation:
        return self.layers[lvl_idx]

    @property
    def itvls(self) -> List[np.ndarray]:
        """Get the intervals of the hierarchy.

        Returns
        -------
        list of np.ndarray
            A list of interval arrays for all levels.
        """
        return [lvl.itvls for lvl in self.layers]

    @property
    def labels(self) -> List[List[Optional[str]]]:
        """Get the labels of the hierarchy.

        Returns
        -------
        list of list of str
            A list of label lists for all levels.
        """
        return [lvl.labels for lvl in self.layers]

    @property
    def bdrys(self) -> List[List[float]]:
        """Get the boundaries of the hierarchy.

        Returns
        -------
        list of list of float
            A list of boundary lists for all levels.
        """
        return [lvl.bdrys for lvl in self.layers]

    def __repr__(self) -> str:
        dur = self.end - self.start if self.layers else 0.0
        return f"Hierarchy(depth={len(self)}, duration={dur:.2f}s)"

    def __str__(self) -> str:
        if len(self) == 0:
            return "Hierarchy(0 levels): []"

        dur = self.end - self.start if self.layers else 0.0
        if len(self) == 1:
            # For single layer, use compact format
            return f"Hierarchy({len(self)} levels, duration={dur:.2f}s): Level 0: {self.layers[0]}"
        else:
            # For multiple layers, use summary format
            return f"Hierarchy({len(self)} levels, duration={dur:.2f}s)"


def seg_from_itvls(intervals: np.ndarray, labels: List[str]) -> Segmentation:
    """Create segmentation from interval array."""
    time_spans = [
        TimeSpan(start=interval[0], end=interval[1], name=label)
        for interval, label in zip(intervals, labels)
    ]
    return Segmentation(segments=time_spans)


def seg_from_brdys(boundaries, labels: List[str]) -> Segmentation:
    """Create segmentation from boundaries."""
    intervals = boundaries_to_intervals(np.array(sorted(boundaries)))
    time_spans = [
        TimeSpan(start=interval[0], end=interval[1], name=label)
        for interval, label in zip(intervals, labels)
    ]
    return Segmentation(segments=time_spans)
