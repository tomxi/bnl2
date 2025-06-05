import pytest
import numpy as np
import bnl


def test_segmentation_basic_init():
    """Test basic segmentation functionality via __init__."""
    seg = bnl.Segmentation(
        segments=[
            bnl.TimeSpan(start=0.0, end=1.5, name="verse"),
            bnl.TimeSpan(start=1.5, end=3.0, name="chorus"),
        ]
    )
    assert seg.bdrys == [0.0, 1.5, 3.0]
    assert seg.labels == ["verse", "chorus"]
    assert len(seg) == 2


def test_hierarchy_basic():
    """Test basic hierarchy functionality."""
    seg1 = bnl.Segmentation.from_boundaries([0.0, 2.0], ["A"])
    seg2 = bnl.Segmentation.from_boundaries([0.0, 1.0, 2.0], ["a", "b"])
    hierarchy = bnl.Hierarchy(layers=[seg1, seg2])
    assert len(hierarchy) == 2
    assert hierarchy[0] == seg1


def test_segmentation_from_intervals():
    """Test classmethod construction from intervals."""
    intervals = np.array([[0.0, 1.0], [1.0, 2.5], [2.5, 3.0]])
    labels = ["A", "B", "C"]
    seg = bnl.Segmentation.from_intervals(intervals, labels)
    assert seg.labels == labels


def test_segmentation_from_boundaries():
    """Test classmethod construction from boundaries."""
    boundaries = [1, 3, 5, 6]
    labels = ["vocals", "drums", "bass"]
    seg = bnl.Segmentation.from_boundaries(boundaries, labels)
    assert seg.bdrys == [1, 3, 5, 6]
    assert seg.labels == labels


def test_str_repr():
    """Test string representation of core classes."""
    seg = bnl.Segmentation(segments=[bnl.TimeSpan(start=0.0, end=1.0, name="A")])
    seg2 = bnl.Segmentation.from_intervals(
        np.array([[0.0, 0.5], [0.5, 1.0]]), ["B", "C"]
    )
    hierarchy = bnl.Hierarchy(layers=[seg, seg2])
    assert str(hierarchy) == "Hierarchy(2 levels, duration=1.00s)"
    assert repr(hierarchy) == "Hierarchy(depth=2, duration=1.00s)"

    assert str(seg) == "Segmentation(1 segments, duration=1.00s): [0.00-1.00s] (A)"
    assert repr(seg) == "Segmentation(1 segments, duration=1.00s)"
    assert str(seg2) == "Segmentation(2 segments, duration=1.00s)"
    assert repr(seg2) == "Segmentation(2 segments, duration=1.00s)"


def test_core_edge_cases_and_validation():
    """Test edge cases and validation for core classes."""
    # Test TimeSpan validation error
    with pytest.raises(ValueError):
        bnl.TimeSpan(start=2.0, end=1.0)

    # Test empty segmentation cases
    empty_seg = bnl.Segmentation()
    assert empty_seg.itvls.size == 0
    assert empty_seg.bdrys == []
    assert str(empty_seg) == "Segmentation(0 segments): []"

    # Test empty hierarchy cases
    empty_hierarchy = bnl.Hierarchy()
    assert str(empty_hierarchy) == "Hierarchy(0 levels): []"

    # Test TimeSpan without name
    unnamed_span = bnl.TimeSpan(start=1.0, end=2.0)
    assert "○" in str(unnamed_span)
    assert "○" in repr(unnamed_span)

    # Test hierarchy properties with empty layers
    assert empty_hierarchy.itvls == []
    assert empty_hierarchy.labels == []
    assert empty_hierarchy.bdrys == []
