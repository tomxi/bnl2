import pytest
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

from bnl import Segment, viz


@pytest.fixture(autouse=True)
def close_plots():
    """Close all matplotlib plots after each test to prevent state leakage."""
    yield
    plt.close("all")


class TestLabelStyleDict:
    def test_basic_functionality(self):
        """Test essential style generation functionality."""
        labels = ["A", "B", "A", "C"]
        styles = viz.label_style_dict(labels)

        # Basic structure checks
        assert isinstance(styles, dict)
        assert set(styles.keys()) == {"A", "B", "C"}

        # Essential style properties exist
        for label in ["A", "B", "C"]:
            assert "facecolor" in styles[label]
            assert "edgecolor" in styles[label]
            assert "label" in styles[label]

    def test_custom_boundary_color(self):
        """Test boundary color customization."""
        styles = viz.label_style_dict(["X"], boundary_color="blue")
        assert styles["X"]["edgecolor"] == "blue"


class TestSegmentPlotting:
    def test_basic_plotting(self):
        """Test core plotting functionality."""
        seg = Segment(beta=[0, 1, 2], labels=["X", "Y"])
        fig, ax = seg.plot(text=True, ytick="Test")

        # Basic return types
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        # Content verification
        assert len(ax.texts) == 2  # Text labels for segments
        text_contents = sorted([t.get_text() for t in ax.texts])
        assert text_contents == ["X", "Y"]

        # Ytick labels (not ylabel)
        assert ax.get_yticklabels()[0].get_text() == "Test"

    def test_time_formatting(self):
        """Test time axis formatting."""
        seg = Segment(beta=[0, 1], labels=["A"])

        # With time ticks
        fig, ax = viz.plot_segment(seg, time_ticks=True)
        assert isinstance(ax.xaxis.get_major_formatter(), librosa.display.TimeFormatter)

        # Without time ticks
        fig, ax = viz.plot_segment(seg, time_ticks=False)
        assert len(ax.get_xticks()) == 0

    def test_edge_cases(self):
        """Test edge cases for robust debugging."""
        # Empty segment
        seg = Segment(beta=[])
        fig, ax = viz.plot_segment(seg)
        # expect warning

        # Single boundary
        seg = Segment(beta=[5.0])
        fig, ax = viz.plot_segment(seg)

        # expect warning
        assert fig is not None


    def test_existing_axes(self):
        """Test plotting on existing axes."""
        seg = Segment(beta=[0, 1], labels=["A"])
        fig_orig, ax_orig = plt.subplots()
        fig, ax = viz.plot_segment(seg, ax=ax_orig)
        assert fig == fig_orig
        assert ax == ax_orig


class TestInternalPlotting:
    def test_intervals_and_labels(self):
        """Test internal plotting helper."""
        fig, ax = plt.subplots()
        intervals = np.array([[0.0, 1.0], [2.0, 3.0]])
        labels = ["A", "B"]

        viz._plot_itvl_lbls(intervals, labels, ax, text=True)
        assert len(ax.patches) > 0  # Some patches created
        assert len(ax.texts) == 2  # Text labels added

    def test_custom_styling(self):
        """Test custom style application."""
        fig, ax = plt.subplots()
        intervals = np.array([[0.0, 1.0]])
        labels = ["S1"]
        style_map = {"S1": {"facecolor": "purple", "alpha": 0.6}}

        viz._plot_itvl_lbls(intervals, labels, ax, style_map=style_map)
        patch = ax.patches[0]
        assert np.allclose(patch.get_facecolor()[:3], plt.cm.colors.to_rgb("purple"))
        assert patch.get_alpha() == 0.6
