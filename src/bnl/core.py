"""Core data structures and constructors."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple
import numpy as np
from mir_eval.util import boundaries_to_intervals
import jams
import matplotlib.pyplot as plt

__all__ = ["TimeSpan", "Segmentation", "Hierarchy"]


@dataclass
class TimeSpan:
    """A labeled time span with start and end times.

    Parameters
    ----------
    start : float
        Start time in seconds.
    end : float
        End time in seconds.
    name : str, optional
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
        if self.name is None:
            self.name = str(self)

    def __str__(self) -> str:
        lab = self.name if self.name else ""
        return f"[{self.start:.1f}-{self.end:.1f}s]{lab}"

    def __repr__(self) -> str:
        return f"TimeSpan({self})"

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        text: bool = True,
        **style_map,
    ):
        """Plot the time span as axvspan on the given axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        text : bool, default=True
            Whether to display the annotation text.
        style_map : dict, optional
            A dictionary of style properties for the axvspan.
        """
        if ax is None:
            _, ax = plt.subplots()

        # Convert color to facecolor to preserve edgecolor, default edgecolor to white
        if "color" in style_map:
            style_map["facecolor"] = style_map.pop("color")
        style_map.setdefault("edgecolor", "white")

        rect = ax.axvspan(self.start, self.end, **style_map)

        # Get ymin/ymax for annotation positioning, default to 0/1 (bottom/top of axes)
        span_ymin = style_map.get("ymin", 0.0)  # get the bottom of the rect span
        span_ymax = style_map.get("ymax", 1.0)  # get the top of the rect span

        # Override ymin/ymax for axvspan if present in style_map
        axvspan_kwargs = {k: v for k, v in style_map.items() if k not in ["ymin", "ymax"]}
        rect = ax.axvspan(self.start, self.end, ymin=span_ymin, ymax=span_ymax, **axvspan_kwargs)

        if text:
            lab = str(self) if self.name == "" else self.name
            ann = ax.annotate(
                lab,
                xy=(self.start, span_ymax),  # Position text relative to the top of the span
                xycoords=ax.get_xaxis_transform(),
                xytext=(8, -10),
                textcoords="offset points",
                va="top",
                clip_on=True,
                bbox=dict(boxstyle="round", facecolor="white"),
            )
            ann.set_clip_path(rect)

        return ax.figure, ax


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
        # I should check that the segments are non-overlapping and contiguous.
        for i in range(len(self.segments) - 1):
            if self.segments[i].end != self.segments[i + 1].start:
                raise ValueError("Segments must be non-overlapping and contiguous.")

        # Set start/end from segments if available
        if self.segments:
            self.start = self.segments[0].start
            self.end = self.segments[-1].end

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
        return f"Segmentation({len(self)} segments over {dur:.2f}s)"

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
            label_text=text,
            title=title,
            ytick=ytick,
            time_ticks=time_ticks,
            style_map=style_map,
        )

    def __str__(self) -> str:
        if len(self) == 0:
            return "Segmentation(0 segments): []"

        dur = self.end - self.start
        return f"Segmentation({len(self)} segments over {dur:.2f}s)"

    @classmethod
    def from_intervals(
        cls,
        intervals: np.ndarray,
        labels: Optional[List[str]] = None,
        name: Optional[str] = None,
    ) -> "Segmentation":
        """Create segmentation from an interval array."""
        # Default labels is the interval string
        if labels is None:
            labels = [None] * len(intervals)

        time_spans = [
            TimeSpan(start=itvl[0], end=itvl[1], name=label)
            for itvl, label in zip(intervals, labels)
        ]
        return cls(segments=time_spans, name=name)

    @classmethod
    def from_boundaries(
        cls,
        boundaries: List[float],
        labels: Optional[List[str]] = None,
        name: Optional[str] = None,
    ) -> "Segmentation":
        """Create segmentation from a list of boundaries."""
        intervals = boundaries_to_intervals(np.array(sorted(boundaries)))
        return cls.from_intervals(intervals, labels, name)

    @classmethod
    def from_jams(cls, anno: jams.Annotation) -> "Segmentation":
        """Create segmentation from a JAMS annotation. (Not yet implemented)"""
        # TODO: Implement JAMS open_segment annotation parsing
        pass


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
        # Set start/end from layers if available
        if self.layers:
            self.start = self.layers[0].start
            self.end = self.layers[0].end

        for layer in self.layers:
            if layer.start != self.start or layer.end != self.end:
                raise ValueError("All layers must have the same start and end time.")

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
        return f"Hierarchy({len(self)} levels over {self.start:.2f}s-{self.end:.2f}s)"

    def __str__(self) -> str:
        if len(self) == 0:
            return "Hierarchy(0 levels)"

        return f"Hierarchy({len(self)} levels over {self.start:.2f}s-{self.end:.2f}s)"

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        label_text: bool = True,
        style_map: Optional[Dict[str, Any]] = None,
    ):
        """Plots the hierarchy.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        label_text : bool, default=True
            Whether to display segment labels as text on the plot.
        style_map : dict, optional
            A precomputed mapping from labels to style properties for all layers.
            If None, it will be generated using `bnl.viz.label_style_dict`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object with the plot.
        """
        from .viz import label_style_dict  # Moved import here

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if not self.layers:
            return fig, ax

        num_layers = len(self.layers)
        layer_height = 1.0 / num_layers

        # Generate a single style_map for all labels across all layers
        if style_map is None:
            all_labels = set()
            for layer in self.layers:
                for segment in layer.segments:
                    if segment.name:
                        all_labels.add(segment.name)
            style_map = label_style_dict(list(all_labels))

        y_tick_positions = []
        y_tick_labels = []

        for i, layer in enumerate(self.layers):
            ymin = 1.0 - (i + 1) * layer_height
            ymax = 1.0 - i * layer_height
            y_tick_positions.append(1.0 - (i + 0.5) * layer_height)
            y_tick_labels.append(layer.name if layer.name else f"Layer {i}")

            # Create a layer-specific style_map that includes ymin and ymax for each segment
            layer_style_map = {}
            for segment in layer.segments:
                seg_style = style_map.get(segment.name, {}).copy()
                seg_style["ymin"] = ymin
                seg_style["ymax"] = ymax
                layer_style_map[segment.name] = seg_style

            # For segments not in the global style_map (e.g. if style_map was provided by user)
            # still pass ymin/ymax. This logic might need refinement if segments can have no name
            # or if the provided style_map is not per-label but per-segment.
            # For now, we assume style_map is keyed by segment names.
            # A more robust approach might be to pass ymin/ymax directly to layer.plot()
            # if that method could forward it to individual TimeSpan.plot() calls.
            # However, Segmentation.plot calls viz.plot_segment, which calls TimeSpan.plot.
            # The current structure requires passing ymin/ymax via style_map to TimeSpan.plot.

            layer.plot(
                ax=ax,
                text=label_text,
                title=False,  # Disable title for individual layers
                time_ticks=(i == num_layers - 1),  # Enable time_ticks only for the last layer
                style_map=layer_style_map,
                ytick="", # Disable ytick for individual layers, handled by Hierarchy plot
            )

        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels(y_tick_labels)
        ax.set_ylim(0, 1)

        if self.name: # Add a title for the whole hierarchy if it has a name
            ax.set_title(self.name)

        return fig, ax

    @classmethod
    def from_jams(cls, anno) -> "Hierarchy":
        """Create hierarchy from a JAMS annotation. (Not yet implemented)"""
        # TODO: Implement JAMS multilevel annotation parsing
        pass
