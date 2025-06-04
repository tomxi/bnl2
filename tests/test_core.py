"""Tests for the Segment class."""

import numpy as np
import pytest
from bnl.core import Segment


def test_basic_segment_creation():
    """Test basic Segment creation with boundaries and labels."""
    # Test with explicit labels
    seg = Segment(beta=[0.0, 1.0, 2.5, 4.0], labels=["A", "B", "C"])
    assert seg.beta == {0.0, 1.0, 2.5, 4.0}
    assert seg.labels == ["A", "B", "C"]
    assert seg.duration == 4.0
    assert seg.num_segments == 3

    # Test automatic label generation
    seg = Segment(beta=[0.0, 1.0, 2.5, 4.0])
    assert seg.labels == ["0.000", "1.000", "2.500"]


def test_segment_properties():
    """Test Segment properties and methods."""
    seg = Segment(beta=[0.5, 1.5, 3.0], labels=["A", "B"])

    # Test duration
    assert seg.duration == 2.5

    # Test intervals
    intervals = seg.itvls
    assert isinstance(intervals, np.ndarray)
    assert intervals.shape == (2, 2)
    np.testing.assert_array_almost_equal(intervals, np.array([[0.5, 1.5], [1.5, 3.0]]))

    # Test string representation
    assert "Segment(2 segments)" in str(seg)
    assert "0: [0.50-1.50s] A" in str(seg)
    assert "1: [1.50-3.00s] B" in str(seg)
    assert repr(seg) == "Segment(2 segments, total_duration=2.50s)"


def test_from_mir_eval():
    """Test the from_mir_eval classmethod."""
    intervals = np.array([[0.0, 1.0], [1.0, 2.5], [2.5, 3.0]])
    labels = ["A", "B", "C"]
    seg = Segment.from_mir_eval(intervals, labels)

    assert seg.beta == {0.0, 1.0, 2.5, 3.0}
    assert seg.labels == ["A", "B", "C"]
    assert seg.num_segments == 3
    assert seg.duration == 3.0

    # Test with unsorted intervals
    intervals = np.array([[1.0, 2.5], [0.0, 1.0], [2.5, 3.0]])
    seg = Segment.from_mir_eval(intervals, labels)
    assert seg.beta == {0.0, 1.0, 2.5, 3.0}

    # Test with overlapping intervals (should deduplicate boundaries)
    intervals = np.array([[0.0, 1.0], [1.0, 2.5], [2.5, 3.0], [2.0, 2.5]])
    seg = Segment.from_mir_eval(intervals, ["A", "B", "C", "D"])
    assert seg.beta == {0.0, 1.0, 2.0, 2.5, 3.0}
    assert len(seg.labels) == 4  # Number of intervals


def test_edge_cases():
    """Test edge cases and error conditions."""
    # Empty segment
    seg = Segment(beta=[])
    assert seg.beta == set()
    assert seg.labels == []
    assert seg.duration == 0.0
    assert seg.num_segments == 0
    assert seg.itvls.size == 0
    assert str(seg) == "Segment(0 segments): []"
    assert repr(seg) == "Segment(0 segments, total_duration=0.00s)"

    # Single boundary (no intervals)
    seg = Segment(beta=[1.0])
    assert seg.beta == {1.0}
    assert seg.labels == []
    assert seg.duration == 0.0
    assert seg.num_segments == 0
    assert repr(seg) == "Segment(0 segments, total_duration=0.00s)"

    # Mismatched labels length
    with pytest.raises(
        ValueError,
        match=r"Number of labels \(2\) must be one less than number of unique boundaries \(4\)",
    ):
        Segment(beta=[0.0, 1.0, 2.0, 3.0], labels=["A", "B"])


def test_beta_input_gardening():
    """Test that beta is properly converted to a set."""
    # Test with list input
    seg = Segment(beta=[1.0, 2.0, 1.0, 3.0])  # duplicates should be removed
    assert seg.beta == {1.0, 2.0, 3.0}
    assert len(seg.beta) == 3
    
    # Test with set input
    seg = Segment(beta={1.0, 2.0, 3.0})
    assert seg.beta == {1.0, 2.0, 3.0}
    
    # Test with tuple input
    seg = Segment(beta=(1.0, 2.0, 3.0))
    assert seg.beta == {1.0, 2.0, 3.0}
