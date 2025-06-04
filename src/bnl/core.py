"""Core data structures for text segmentation."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Set
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
        """Validate inputs and generate labels if needed."""
        # Ensure beta is a set (convert from list if needed)
        if not isinstance(self.beta, set):
            self.beta = set(self.beta)

        # Handle empty boundaries
        if not self.beta:
            self.labels = []
            return

        # Generate labels if not provided
        if self.labels is None:
            # Generate labels from start times with 3 decimal places
            sorted_beta = self._sorted_boundaries
            self.labels = [f"{start:.3f}" for start in sorted_beta[:-1]]
        elif len(self.labels) != len(self.beta) - 1:
            raise ValueError(
                f"Number of labels ({len(self.labels)}) must be one less than "
                f"number of unique boundaries ({len(self.beta)})"
            )

    @property
    def _sorted_boundaries(self) -> List[float]:
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
        return boundaries_to_intervals(np.array(self._sorted_boundaries))

    @property
    def duration(self) -> float:
        """Get total duration of the segment.

        Returns
        -------
        float
            Duration in seconds (last boundary minus first). Returns 0.0 if fewer than 2 boundaries.
        """
        if len(self.beta) < 2:
            return 0.0
        sorted_beta = self._sorted_boundaries
        return sorted_beta[-1] - sorted_beta[0]

    @property
    def num_segments(self) -> int:
        """Get number of segments (intervals with labels).

        Returns
        -------
        int
            Number of segments, which is one less than number of boundaries.
        """
        return max(0, len(self.beta) - 1)

    @classmethod
    def from_mir_eval(cls, intervals: np.ndarray, labels: List[str]) -> "Segment":
        """Create a Segment from mir_eval format.

        Parameters
        ----------
        intervals : np.ndarray [shape=(n, 2)]
            Array of interval start and end times.
            Each row should be [start_time, end_time] in seconds.
            Start times must be less than or equal to end times.
        labels : list of str [length=n]
            Label for each interval in `intervals`.
            Length must match the number of intervals.

        Returns
        -------
        Segment
            A new Segment instance with boundaries derived from the interval endpoints.

        Raises
        ------
        ValueError
            If any interval has a start time greater than its end time.
            If the number of labels doesn't match the number of intervals.

        """
        # Extract unique boundaries as a set
        boundaries = set(intervals.flatten())
        return cls(beta=boundaries, labels=labels)

    def __repr__(self) -> str:
        return f"Segment({self.num_segments} segments, total_duration={self.duration:.2f}s)"

    def __str__(self) -> str:
        if self.num_segments == 0:
            return "Segment(0 segments): []"

        segments = []
        for i, ((start, end), label) in enumerate(zip(self.itvls, self.labels)):
            segments.append(f"  {i}: [{start:.2f}-{end:.2f}s] {label}")
        return "\n".join([f"Segment({self.num_segments} segments):"] + segments)

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
