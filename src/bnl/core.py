"""Core data structures for text segmentation."""
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from mir_eval.util import boundaries_to_intervals


@dataclass
class Segment:
    """Labeled time intervals represented using boundaries (beta).
    
    This class represents a sequence of labeled time intervals using boundary points.
    The boundaries are automatically deduplicated and sorted upon initialization.
    
    Parameters
    ----------
    beta : list of float
        List of boundary times in seconds.
        Will be automatically deduplicated and sorted in ascending order.
    labels : list of str, optional
        Labels for each interval between boundaries.
        If None, labels will be automatically generated from the start times.
        Length must be exactly one less than the number of boundaries.
    
    Attributes
    ----------
    beta : list of float
        Sorted list of unique boundary times.
    labels : list of str
        Labels for each interval between boundaries.
    
    Examples
    --------
    >>> # Create a segment with automatic label generation
    >>> seg = Segment(beta=[0.0, 1.0, 2.5, 4.0])
    >>> seg.labels
    ['0.000', '1.000', '2.500']
    
    >>> # Create a segment with custom labels
    >>> seg = Segment(beta=[0.0, 1.0, 2.0], labels=['A', 'B'])
    >>> seg.duration
    2.0
    """
    beta: List[float]
    labels: Optional[List[str]] = None
   
    def __post_init__(self):
        # """Validate inputs and generate labels if needed."""
        # Ensure beta is sorted and deduplicated
        self.beta = sorted(set(self.beta))
        
        # Handle empty boundaries
        if not self.beta:
            self.labels = []
            return
            
        # Generate labels if not provided
        if self.labels is None:
            # Generate labels from start times with 3 decimal places
            self.labels = [f"{start:.3f}" for start in self.beta[:-1]]
        elif len(self.labels) != len(self.beta) - 1:
            raise ValueError(
                f"Number of labels ({len(self.labels)}) must be one less than "
                f"number of unique boundaries ({len(self.beta)})"
            )
    
    @property
    def itvls(self) -> np.ndarray:
        """Convert boundaries to interval representation.
        
        Uses mir_eval's boundaries_to_intervals to convert the boundary points
        into (start, end) interval pairs.
        
        Returns
        -------
        intervals : np.ndarray [shape=(n_intervals, 2)]
            Array of interval start and end times.
            Each row corresponds to one interval as [start_time, end_time].
            Returns an empty array if there are fewer than 2 boundaries.
            
        Examples
        --------
        >>> seg = Segment(beta=[0.0, 1.0, 3.0])
        >>> seg.itvls
        array([[0., 1.],
               [1., 3.]])
        """
        if not self.beta:
            return np.array([])
        return boundaries_to_intervals(np.array(self.beta))

    @property
    def duration(self) -> float:
        """Get the total duration of the segment.
        
        Returns
        -------
        duration : float
            Total duration in seconds, calculated as the difference between
            the last and first boundary. Returns 0.0 if there are fewer than 2 boundaries.
            
        Examples
        --------
        >>> seg = Segment(beta=[0.5, 1.5, 3.0])
        >>> seg.duration
        2.5
        """
        if len(self.beta) < 2:
            return 0.0
        return self.beta[-1] - self.beta[0]

    @property
    def num_segments(self) -> int:
        """Return the number of segments."""
        return max(0, len(self.beta) - 1)

    @classmethod
    def from_mir_eval(cls, intervals: np.ndarray, labels: List[str]) -> 'Segment':
        """Create a Segment from mir_eval format.
        
        This is an alternative constructor that creates a Segment from the interval
        format used by the mir_eval library, where segments are represented as
        (start, end) time pairs.
        
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
            
        Examples
        --------
        >>> import numpy as np
        >>> intervals = np.array([[0.0, 1.0], [1.0, 2.5], [2.5, 3.0]])
        >>> labels = ['A', 'B', 'C']
        >>> seg = Segment.from_mir_eval(intervals, labels)
        >>> seg.beta
        [0.0, 1.0, 2.5, 3.0]
        >>> seg.labels
        ['A', 'B', 'C']
        """
        # Extract unique boundaries and sort them
        boundaries = sorted(set(intervals.flatten()))
        return cls(beta=boundaries, labels=labels)
    
    def __repr__(self) -> str:
        return f"Segment({self.num_segments} segments, total_duration={self.duration:.2f}s)"
    
    def __str__(self) -> str:
        if not self.beta:
            return "Segment(0 segments): []"
            
        segments = []
        for i, ((start, end), label) in enumerate(zip(self.itvls, self.labels)):
            segments.append(f"  {i}: [{start:.2f}-{end:.2f}s] {label}")
        return "\n".join([f"Segment({self.num_segments} segments):"] + segments)
