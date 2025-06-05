"""Core data structures and constructors."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Set, Tuple
import numpy as np
from mir_eval.util import boundaries_to_intervals


@dataclass
class Segment:
    """Labeled time intervals represented using boundaries (beta).

    Parameters
    ----------
    beta : set of float
        Set of boundary times in seconds.
    labels : list of str, optional
        Labels for each interval between boundaries.
        Length must be exactly one less than the number of boundaries.

    Examples
    --------
    >>> seg = Segment(beta={0.0, 1.0, 2.0}, labels=['A', 'B'])
    >>> seg.duration
    2.0
    """

    beta: Set[float]
    labels: Optional[List[str]] = None

    def __post_init__(self):
        # Ensure beta is a set (convert from other iterables if needed)
        if not isinstance(self.beta, set):
            self.beta = set(self.beta)

        # Handle empty boundaries
        if not self.beta:
            self.labels = []
            return

        # Generate labels if not provided
        if self.labels is None:
            # Generate labels from start times with 3 decimal places
            sorted_beta = self.boundaries
            self.labels = [f"{start:.3f}" for start in sorted_beta[:-1]]
        elif len(self.labels) != len(self.beta) - 1:
            raise ValueError(
                f"Number of labels ({len(self.labels)}) must be one less than "
                f"number of unique boundaries ({len(self.beta)})"
            )
        
    def __len__(self) -> int:
        """Get the number of segments in the segment.
        """
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[Tuple[float, float], str]:
        """Get the interval and label at the given index.
        """
        interval = self.itvls[idx]
        return (float(interval[0]), float(interval[1])), self.labels[idx]

    @property
    def boundaries(self) -> List[float]:
        """Get boundaries sorted in ascending order.

        Returns
        -------
        list of float
            Sorted list of boundary times.
        """
        return sorted(self.beta)

    @property
    def itvls(self) -> np.ndarray:
        """Get intervals as (start, end) pairs.

        Returns
        -------
        np.ndarray
            Array of interval start and end times, shape=(n_intervals, 2).
        """
        if not self.beta:
            return np.array([])
        return boundaries_to_intervals(np.array(self.boundaries))
    
    @property
    def duration(self) -> float:
        """Get the total duration of the segment.
        """
        return self.boundaries[-1] - self.boundaries[0] if len(self.boundaries) > 1 else 0.0

    def plot(
        self,
        ax=None,
        text: bool = False,
        ytick: str = "",
        time_ticks: bool = True,
        style_map: Optional[Dict[str, Any]] = None,
    ):
        """Plot the segment boundaries and labels.

        This is a convenience wrapper around `bnl.viz.plot_segment`.

        Parameters
        ----------
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
        from .viz import plot_segment  # matplotlib is now a mandatory dependency

        return plot_segment(
            self,
            ax=ax,
            text=text,
            ytick=ytick,
            time_ticks=time_ticks,
            style_map=style_map,
        )

    def __repr__(self) -> str:
        return f"Segment({len(self)} segments, total_duration={self.duration:.2f}s)"

    def __str__(self) -> str:
        if len(self) == 0:
            return "Segment(0 segments): []"

        segments_str = []
        for i in range(len(self)):
            (start, end), label = self[i]
            segments_str.append(f" [{start:.2f}-{end:.2f}s] {label}")
        return "\n".join([f"Segment({len(self)} segments):"] + segments_str)


@dataclass
class Hierarchy:
    """A hierarchical structure of segments.

    A Hierarchy is an ordered list of Segment objects, representing different
    levels of segmentation.

    Parameters
    ----------
    layers : List[Segment]
        An ordered list of Segment objects, from coarsest to finest levels.

    Examples
    --------
    >>> from bnl import Segment
    >>> seg1 = Segment(beta={0.0, 1.0, 2.0}, labels=['A', 'B'])
    >>> seg2 = Segment(beta={0.0, 0.5, 1.0, 1.5, 2.0}, labels=['a', 'b', 'c', 'd'])
    >>> hierarchy = Hierarchy(layers=[seg1, seg2])
    """
    layers: List[Segment]

    def __post_init__(self):
        if not isinstance(self.layers, list):
            raise TypeError("layers must be a list of Segment objects.")
        for i, segment in enumerate(self.layers):
            if not isinstance(segment, Segment):
                raise TypeError(f"Element at index {i} is not a Segment object.")

    def __len__(self) -> int:
        return len(self.layers)
    
    def __getitem__(self, lvl_idx: int) -> Segment:
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
    def labels(self) -> List[List[str]]:
        """Get the labels of the hierarchy.

        Returns
        -------
        list of list of str
            A list of label lists for all levels.
        """
        return [lvl.labels for lvl in self.layers]

    @property
    def beta(self) -> Set[float]:
        """Get the boundaries of the hierarchy.

        Returns
        -------
        set of float
            The union of boundary sets from all levels.
        """
        all_beta = set()
        for lvl in self.layers:
            all_beta.update(lvl.beta)
        return all_beta
        

    def __repr__(self) -> str:
        return f"Hierarchy(depth={len(self)})"

    def __str__(self) -> str:
        if len(self) == 0:
            return "Hierarchy(0 levels): []"
        
        lines = [f"Hierarchy({len(self)} levels):"]
        for i, lvl in enumerate(self.layers):
            lines.append(f"Level {i}: {lvl}")
        return "\n".join(lines)


def seg_from_itvls(intervals: np.ndarray, labels: List[str]) -> Segment:
    """Create segment from interval array.

    Parameters
    ----------
    intervals : np.ndarray [shape=(n, 2)]
        Array of interval start and end times.
        Each row should be [start_time, end_time] in seconds.
    labels : list of str [length=n]
        Label for each interval in `intervals`.
        Length must match the number of intervals.

    Returns
    -------
    Segment
        A new Segment instance with boundaries derived from the interval endpoints.

    Examples
    --------
    >>> intervals = np.array([[0.0, 1.0], [1.0, 2.5], [2.5, 3.0]])
    >>> labels = ['A', 'B', 'C']
    >>> seg = seg_from_itvls(intervals, labels)
    """
    boundaries = set(intervals.flatten())
    return Segment(beta=boundaries, labels=labels)
