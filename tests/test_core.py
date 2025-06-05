import pytest
import numpy as np

from bnl import Segmentation, TimeSpan, seg_from_itvls, seg_from_brdys, Hierarchy


def test_segmentation_basic():
    """Test basic segmentation functionality."""
    seg = Segmentation(
        segments=[
            TimeSpan(start=0.0, end=1.5, name="verse"),
            TimeSpan(start=1.5, end=3.0, name="chorus"),
        ]
    )
    assert seg.bdrys == [0.0, 1.5, 3.0]
    assert seg.labels == ["verse", "chorus"]
    assert len(seg) == 2


def test_hierarchy_basic():
    """Test basic hierarchy functionality."""
    seg1 = seg_from_brdys([0.0, 2.0], ["A"])
    seg2 = seg_from_brdys([0.0, 1.0, 2.0], ["a", "b"])
    hierarchy = Hierarchy(layers=[seg1, seg2])
    assert len(hierarchy) == 2
    assert hierarchy[0] == seg1


def test_seg_from_itvls():
    """Test interval-based construction."""
    intervals = np.array([[0.0, 1.0], [1.0, 2.5], [2.5, 3.0]])
    labels = ["A", "B", "C"]
    seg = seg_from_itvls(intervals, labels)
    assert seg.labels == labels


def test_seg_from_brdys():
    """Test boundary-based construction."""
    boundaries = [1, 3, 5, 6]
    labels = ["vocals", "drums", "bass"]
    seg = seg_from_brdys(boundaries, labels)
    assert seg.bdrys == [1, 3, 5, 6]
    assert seg.labels == labels


def test_str_repr():
    """Test string representation."""
    seg = Segmentation(segments=[TimeSpan(start=0.0, end=1.0, name="A")])
    seg2 = Segmentation(
        segments=[
            TimeSpan(start=0.0, end=0.5, name="B"),
            TimeSpan(start=0.5, end=1.0, name="C"),
        ]
    )
    hierarchy = Hierarchy(layers=[seg, seg2])
    # The actual output is just the simple format without detailed breakdown
    assert str(hierarchy) == "Hierarchy(2 levels, duration=1.00s)"
    assert repr(hierarchy) == "Hierarchy(depth=2, duration=1.00s)"

    assert str(seg) == "Segmentation(1 segments, duration=1.00s): [0.00-1.00s] (A)"
    assert repr(seg) == "Segmentation(1 segments, duration=1.00s)"
    assert str(seg2) == "Segmentation(2 segments, duration=1.00s)"
    assert repr(seg2) == "Segmentation(2 segments, duration=1.00s)"


def test_edge_cases_coverage():
    """Test edge cases for coverage completeness."""
    # Test TimeSpan validation error (line 35)
    with pytest.raises(ValueError):
        TimeSpan(start=2.0, end=1.0)

    # Test empty segmentation cases (lines 80, 103, 180)
    empty_seg = Segmentation()
    assert empty_seg.itvls.size == 0  # empty array
    assert empty_seg.bdrys == []
    assert str(empty_seg) == "Segmentation(0 segments): []"

    # Test empty hierarchy cases (line 232)
    empty_hierarchy = Hierarchy()
    assert str(empty_hierarchy) == "Hierarchy(0 levels): []"

    # Test TimeSpan without name (coverage for else branch in __str__ and __repr__)
    unnamed_span = TimeSpan(start=1.0, end=2.0)
    assert "○" in str(unnamed_span)
    assert "○" in repr(unnamed_span)

    # Test hierarchy properties with empty layers
    assert empty_hierarchy.itvls == []
    assert empty_hierarchy.labels == []
    assert empty_hierarchy.bdrys == []
