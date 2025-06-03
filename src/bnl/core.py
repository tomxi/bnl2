"""Core data structures for text segmentation."""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Segment:
    """Represents a segment of text with boundaries.
    
    Attributes:
        start: Start position of the segment
        end: End position of the segment (exclusive)
        text: The text content of the segment
        label: Optional label for the segment
    """
    start: int
    end: int
    text: str
    label: Optional[str] = None
    
    def __post_init__(self):
        if self.start >= self.end:
            raise ValueError("Start position must be before end position")
    
    def __len__(self):
        return self.end - self.start
    
    def __contains__(self, position: int) -> bool:
        """Check if a position is within this segment."""
        return self.start <= position < self.end
