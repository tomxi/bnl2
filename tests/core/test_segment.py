"""Tests for the Segment class."""
import pytest
from bnl.core import Segment


def test_segment_creation():
    """Test basic Segment creation."""
    seg = Segment(start=0, end=10, text="Hello world")
    assert seg.start == 0
    assert seg.end == 10
    assert seg.text == "Hello world"
    assert seg.label is None
    assert len(seg) == 10

def test_segment_with_label():
    """Test Segment creation with a label."""
    seg = Segment(start=5, end=15, text="Test segment", label="test_label")
    assert seg.label == "test_label"

def test_segment_invalid_boundaries():
    """Test Segment creation with invalid boundaries."""
    with pytest.raises(ValueError, match="Start position must be before end position"):
        Segment(start=10, end=5, text="Invalid")
    with pytest.raises(ValueError, match="Start position must be before end position"):
        Segment(start=5, end=5, text="Invalid")

def test_segment_contains():
    """Test the __contains__ method."""
    seg = Segment(start=10, end=20, text="Contains test")
    assert 10 in seg
    assert 15 in seg
    assert 19 in seg
    assert not (9 in seg)
    assert not (20 in seg)
