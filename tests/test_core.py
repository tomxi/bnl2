"""Tests for the Segment class."""

import numpy as np
import pytest
from bnl import Segment, seg_from_itvls


def test_basic_segment_creation():
    """Test basic Segment creation with boundaries and labels."""
    # Test with explicit labels
    seg = Segment(beta=[0.0, 1.0, 2.5, 4.0], labels=["A", "B", "C"])
    assert seg.beta == {0.0, 1.0, 2.5, 4.0}
    assert seg.labels == ["A", "B", "C"]
    expected_duration = (
        seg.boundaries[-1] - seg.boundaries[0] if len(seg.boundaries) > 1 else 0.0
    )
    assert expected_duration == 4.0
    assert max(0, len(seg.beta) - 1) == 3

    # Test automatic label generation
    seg = Segment(beta=[0.0, 1.0, 2.5, 4.0])
    assert seg.labels == ["0.000", "1.000", "2.500"]


def test_segment_properties():
    """Test Segment properties and methods."""
    seg = Segment(beta=[0.5, 1.5, 3.0], labels=["A", "B"])

    # Test duration
    expected_duration = (
        seg.boundaries[-1] - seg.boundaries[0] if len(seg.boundaries) > 1 else 0.0
    )
    assert expected_duration == 2.5

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


@pytest.mark.parametrize(
    "intervals_input, labels_input, expected_beta, expected_labels, expected_n_segments, expected_duration",
    [
        (  # Basic case
            np.array([[0.0, 1.0], [1.0, 2.5], [2.5, 3.0]]),
            ["A", "B", "C"],
            {0.0, 1.0, 2.5, 3.0},
            ["A", "B", "C"],
            3,
            3.0,
        ),
        (  # Unsorted intervals
            np.array([[1.0, 2.5], [0.0, 1.0], [2.5, 3.0]]),
            ["L1", "L2", "L3"],  # Labels for segments from sorted boundaries
            {0.0, 1.0, 2.5, 3.0},
            ["L1", "L2", "L3"],
            3,
            3.0,
        ),
        (  # Overlapping intervals, new boundaries created
            np.array([[0.0, 1.0], [1.0, 2.5], [2.5, 3.0], [2.0, 2.5]]),
            ["S1", "S2", "S3", "S4"],  # Labels for 4 new segments
            {0.0, 1.0, 2.0, 2.5, 3.0},
            ["S1", "S2", "S3", "S4"],
            4,
            3.0,
        ),
    ],
)
def test_from_itvls(
    intervals_input,
    labels_input,
    expected_beta,
    expected_labels,
    expected_n_segments,
    expected_duration,
):
    """Test the from_itvls classmethod with various interval configurations."""
    seg = Segment.from_itvls(intervals_input, labels_input)
    assert seg.beta == expected_beta
    assert seg.labels == expected_labels
    assert max(0, len(seg.beta) - 1) == expected_n_segments
    duration = (
        seg.boundaries[-1] - seg.boundaries[0] if len(seg.boundaries) > 1 else 0.0
    )
    assert duration == expected_duration


@pytest.mark.parametrize(
    "beta_input, expected_beta_set, expected_str, expected_repr",
    [
        (
            [],
            set(),
            "Segment(0 segments): []",
            "Segment(0 segments, total_duration=0.00s)",
        ),
        (
            [1.0],
            {1.0},
            "Segment(0 segments): []",
            "Segment(0 segments, total_duration=0.00s)",
        ),
    ],
)
def test_zero_segment_cases(beta_input, expected_beta_set, expected_str, expected_repr):
    """Test cases that result in zero segments (empty or single boundary)."""
    seg = Segment(beta=beta_input)
    assert seg.beta == expected_beta_set
    assert seg.labels == []
    duration = (
        seg.boundaries[-1] - seg.boundaries[0] if len(seg.boundaries) > 1 else 0.0
    )
    assert duration == 0.0
    assert max(0, len(seg.beta) - 1) == 0
    assert seg.itvls.size == 0
    assert str(seg) == expected_str
    assert repr(seg) == expected_repr


def test_mismatched_labels_error():
    """Test ValueError for mismatched number of labels and boundaries."""
    with pytest.raises(
        ValueError,
        match=r"Number of labels \(2\) must be one less than number of unique boundaries \(4\)",
    ):
        Segment(beta=[0.0, 1.0, 2.0, 3.0], labels=["A", "B"])


@pytest.mark.parametrize(
    "beta_input",
    [
        [1.0, 2.0, 1.0, 3.0],  # list with duplicates
        {1.0, 2.0, 3.0},  # set
        (1.0, 2.0, 3.0),  # tuple
    ],
)
def test_beta_input_conversion_to_set(beta_input):
    """Test that beta input is correctly converted to a set, removing duplicates."""
    seg = Segment(beta=beta_input)
    assert seg.beta == {1.0, 2.0, 3.0}
    # number of segments would be 2, labels auto-generated as ["1.000", "2.000"]
    # This test focuses on beta set conversion.


def test_seg_from_itvls_factory_function():
    """Test the seg_from_itvls factory function."""
    intervals = np.array([[0.0, 1.0], [1.0, 2.5], [2.5, 3.0]])
    labels = ["A", "B", "C"]

    # Test factory function creates correct segment
    seg = seg_from_itvls(intervals, labels)
    assert seg.beta == {0.0, 1.0, 2.5, 3.0}
    assert seg.labels == ["A", "B", "C"]

    # Test it produces same result as classmethod
    seg_classmethod = Segment.from_itvls(intervals, labels)
    assert seg.beta == seg_classmethod.beta
    assert seg.labels == seg_classmethod.labels
