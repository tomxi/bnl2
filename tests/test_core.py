import pytest
import numpy as np

from bnl import Segment, seg_from_itvls, Hierarchy


def test_segment_creation_and_properties():
    """Test segment creation, properties, and core functionality."""
    # Basic creation
    seg = Segment(beta={0.0, 1.5, 3.0}, labels=["verse", "chorus"])
    assert seg.beta == {0.0, 1.5, 3.0}
    assert seg.labels == ["verse", "chorus"]
    
    # Properties
    assert seg.boundaries == [0.0, 1.5, 3.0]
    assert seg.duration == 3.0
    np.testing.assert_array_equal(seg.itvls, [[0.0, 1.5], [1.5, 3.0]])
    
    # New API: len and indexing
    assert len(seg) == 2
    interval, label = seg[0]
    assert interval == (0.0, 1.5) and label == "verse"
    
    # Iteration
    segments = list(seg)
    assert segments == [((0.0, 1.5), "verse"), ((1.5, 3.0), "chorus")]


def test_segment_edge_cases():
    """Test segment edge cases and error handling."""
    # Empty segment
    empty_seg = Segment(beta=set())
    assert len(empty_seg) == 0
    assert str(empty_seg) == "Segment(0 segments): []"
    
    # Auto-generated labels
    auto_seg = Segment(beta={0.0, 1.0, 2.0})
    assert auto_seg.labels == ["0.000", "1.000"]
    
    # Input conversion
    list_seg = Segment(beta=[0.0, 1.0, 2.0], labels=["A", "B"])
    assert isinstance(list_seg.beta, set)
    
    # Error handling - need too many labels for the boundaries
    with pytest.raises(ValueError, match="Number of labels"):
        Segment(beta={0.0, 1.0}, labels=["too", "many", "labels"])
    
    with pytest.raises(IndexError):
        auto_seg[5]


def test_hierarchy_creation_and_api():
    """Test hierarchy creation and new API functionality."""
    seg1 = Segment(beta={0.0, 2.0}, labels=["A"])
    seg2 = Segment(beta={0.0, 1.0, 2.0}, labels=["a", "b"])
    
    # Creation and basic properties
    hierarchy = Hierarchy([seg1, seg2])
    assert len(hierarchy) == 2
    assert hierarchy[0] == seg1
    assert hierarchy[1] == seg2
    
    # Properties
    assert len(hierarchy.itvls) == 2
    assert len(hierarchy.labels) == 2
    assert hierarchy.beta == {0.0, 1.0, 2.0}  # Union of boundaries
    
    # Iteration
    layers = list(hierarchy)
    assert layers == [seg1, seg2]
    
    # Type validation
    with pytest.raises(TypeError, match="layers must be a list"):
        Hierarchy("not a list")
    
    with pytest.raises(TypeError, match="not a Segment object"):
        Hierarchy([seg1, "not a segment"])


def test_seg_from_itvls():
    """Test interval-based segment creation."""
    intervals = np.array([[0.0, 1.0], [1.0, 2.5], [2.5, 3.0]])
    labels = ['A', 'B', 'C']
    seg = seg_from_itvls(intervals, labels)
    
    assert seg.beta == {0.0, 1.0, 2.5, 3.0}
    assert seg.labels == labels
    assert len(seg) == 3 

def test_str_repr():
    """Test the __str__ and __repr__ methods."""
    seg = Segment(beta={0.0, 1.0, 2.0, 3.5}, labels=["A", "B", "C"])
    
    # Test segment string representation (actual format without index numbers)
    expected_str = "Segment(3 segments):\n [0.00-1.00s] A\n [1.00-2.00s] B\n [2.00-3.50s] C"
    assert str(seg) == expected_str
    
    # Test segment repr
    assert repr(seg) == "Segment(3 segments, total_duration=3.50s)"
    
    # Test hierarchy
    hierarchy = Hierarchy([seg])
    expected_hierarchy_str = "Hierarchy(1 levels):\nLevel 0: " + expected_str
    assert str(hierarchy) == expected_hierarchy_str
    
    # Test hierarchy repr
    assert repr(hierarchy) == "Hierarchy(depth=1)"