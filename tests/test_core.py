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
