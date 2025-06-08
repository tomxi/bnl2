import pytest
import numpy as np
import bnl
import matplotlib.pyplot as plt
import matplotlib.colors


def test_segmentation_basic_init():
    """Test basic segmentation functionality."""
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


def test_post_init_errors():
    """Test that __post_init__ raises errors for malformed objects."""
    # Test for non-contiguous segments
    with pytest.raises(
        ValueError, match="Segments must be non-overlapping and contiguous."
    ):
        bnl.Segmentation(segments=[bnl.TimeSpan(0, 1), bnl.TimeSpan(2, 3)])

    # Test for overlapping segments
    with pytest.raises(
        ValueError, match="Segments must be non-overlapping and contiguous."
    ):
        bnl.Segmentation(segments=[bnl.TimeSpan(0, 1.5), bnl.TimeSpan(1, 2)])

    # Test for inconsistent layer durations in Hierarchy
    with pytest.raises(
        ValueError, match="All layers must have the same start and end time."
    ):
        seg1 = bnl.Segmentation.from_boundaries([0, 2])
        seg2 = bnl.Segmentation.from_boundaries([0, 1, 3])
        bnl.Hierarchy(layers=[seg1, seg2])


@pytest.mark.parametrize(
    "constructor, data",
    [
        (bnl.Segmentation.from_intervals, np.array([[0.0, 1.0], [1.0, 2.5]])),
        (bnl.Segmentation.from_boundaries, [0.0, 1.0, 2.5]),
    ],
)
def test_segmentation_constructors(constructor, data):
    """Test Segmentation constructors with and without labels and names."""
    # Test with labels and a name
    seg1 = constructor(data, labels=["A", "B"], name="TestName")
    assert seg1.labels == ["A", "B"]
    assert seg1.name == "TestName"
    np.testing.assert_array_equal(seg1.itvls, np.array([[0.0, 1.0], [1.0, 2.5]]))

    # Test without labels (default labels)
    seg2 = constructor(data)
    assert seg2.labels == ["[0.0-1.0s]", "[1.0-2.5s]"]
    assert seg2[0] == bnl.TimeSpan(start=0.0, end=1.0, name="[0.0-1.0s]")


def test_str_repr():
    """Test string representation of core classes."""
    seg = bnl.Segmentation(segments=[bnl.TimeSpan(start=0.0, end=1.0, name="A")])
    seg2 = bnl.Segmentation.from_intervals(
        np.array([[0.0, 0.5], [0.5, 1.0]]), ["B", "C"]
    )
    hierarchy = bnl.Hierarchy(layers=[seg, seg2])
    assert str(hierarchy) == "Hierarchy(2 levels over 0.00s-1.00s)"
    assert repr(hierarchy) == "Hierarchy(2 levels over 0.00s-1.00s)"

    assert str(seg) == "Segmentation(1 segments over 1.00s)"
    assert repr(seg) == "Segmentation(1 segments over 1.00s)"
    assert str(seg2) == "Segmentation(2 segments over 1.00s)"
    assert repr(seg2) == "Segmentation(2 segments over 1.00s)"


def test_unimplemented_methods():
    """Cover unimplemented methods for test coverage."""
    bnl.Hierarchy.from_jams(None)
    bnl.Segmentation.from_jams(None)
    # bnl.Hierarchy(layers=[]).plot() # Removed this line, covered by test_hierarchy_plot_empty


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
    assert str(empty_hierarchy) == "Hierarchy(0 levels)"
    assert empty_hierarchy.itvls == []
    assert empty_hierarchy.labels == []
    assert empty_hierarchy.bdrys == []

    # Test TimeSpan without name
    unnamed_span = bnl.TimeSpan(start=1.0, end=2.0)
    assert str(unnamed_span) == "[1.0-2.0s][1.0-2.0s]"
    assert repr(unnamed_span) == "TimeSpan([1.0-2.0s][1.0-2.0s])"


def test_timespan_plot_full_coverage():
    """Test TimeSpan.plot method for full branch coverage."""
    # Test with a named span to cover default path for text and color handling
    span_named = bnl.TimeSpan(start=0.0, end=1.0, name="test_span")

    # Cover `if "color" in style_map:` branch and basic text
    fig1, ax1 = span_named.plot(color="blue")
    assert ax1.patches[0].get_facecolor() == (0.0, 0.0, 1.0, 1.0)
    assert ax1.texts[0].get_text() == "test_span"  # Ensure text is plotted
    plt.close(fig1)

    # Cover `if text:` branch when text is False
    fig2, ax2 = span_named.plot(text=False)
    assert len(ax2.texts) == 0
    plt.close(fig2)

    # Cover `style_map.setdefault("edgecolor", "white")` when no style is passed (and default text)
    fig3, ax3 = span_named.plot()
    assert ax3.patches[0].get_edgecolor() == (1.0, 1.0, 1.0, 1.0)
    assert ax3.texts[0].get_text() == "test_span"  # Ensure text is plotted
    plt.close(fig3)

    # Test with an unnamed span to cover `lab = self.name if self.name else str(self)`'s else branch
    span_unnamed = bnl.TimeSpan(start=0.0, end=1.0)  # name=None
    fig4, ax4 = span_unnamed.plot()  # text=True by default
    assert ax4.texts[0].get_text() == "[0.0-1.0s]"  # Expect str(self)
    plt.close(fig4)


# --- Tests for Hierarchy.plot() ---

def test_hierarchy_plot():
    """Test basic Hierarchy.plot() functionality."""
    seg1 = bnl.Segmentation.from_boundaries([0.0, 2.0, 4.0], ["A1", "B1"], name="Layer Alpha")
    seg2 = bnl.Segmentation.from_boundaries([0.0, 1.0, 3.0, 4.0], ["a2", "b2", "c2"], name="Layer Beta")
    hierarchy = bnl.Hierarchy(layers=[seg1, seg2], name="My Hierarchy")

    fig, ax = hierarchy.plot()

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "My Hierarchy"

    # Y-tick labels and positions
    expected_y_labels = ["Layer Alpha", "Layer Beta"]
    actual_y_labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert actual_y_labels == expected_y_labels

    num_layers = len(hierarchy.layers)
    layer_height = 1.0 / num_layers
    expected_y_ticks = [1.0 - (i + 0.5) * layer_height for i in range(num_layers)]
    np.testing.assert_allclose(ax.get_yticks(), expected_y_ticks)

    # X-axis label (time_ticks should be true for the last layer)
    assert ax.get_xlabel() == "Time (s)" # Default label from viz.plot_segment
    assert ax.get_ylim() == (0.0, 1.0)

    # Number of patches (axvspan objects)
    # Layer Alpha has 2 segments, Layer Beta has 3 segments
    assert len(ax.patches) == (len(seg1.segments) + len(seg2.segments))

    # Check if time_ticks are disabled for upper layers and enabled for the last one
    # This is indirectly tested by checking the x-axis label presence on the main ax,
    # as individual layer plots share the x-axis.
    # A more direct test would involve inspecting individual layer plot calls if possible,
    # but for now, the effect on the final plot is what matters.

    plt.close(fig)


def test_hierarchy_plot_no_name():
    """Test Hierarchy.plot() when Hierarchy has no name."""
    seg1 = bnl.Segmentation.from_boundaries([0.0, 2.0], ["A"], name="L1")
    hierarchy = bnl.Hierarchy(layers=[seg1]) # No name for hierarchy

    fig, ax = hierarchy.plot()
    assert ax.get_title() == ""
    plt.close(fig)


def test_hierarchy_plot_no_layer_names():
    """Test Hierarchy.plot() when layers have no names."""
    # Create segments without explicit names (they will get default "[start-end]s" names)
    ts1_layer1 = bnl.TimeSpan(0.0, 2.0)
    ts2_layer1 = bnl.TimeSpan(2.0, 4.0)
    seg1 = bnl.Segmentation(segments=[ts1_layer1, ts2_layer1]) # No name for layer

    ts1_layer2 = bnl.TimeSpan(0.0, 1.0)
    ts2_layer2 = bnl.TimeSpan(1.0, 4.0)
    seg2 = bnl.Segmentation(segments=[ts1_layer2, ts2_layer2]) # No name for layer

    hierarchy = bnl.Hierarchy(layers=[seg1, seg2], name="Test Hierarchy")

    fig, ax = hierarchy.plot()

    expected_y_labels = ["Layer 0", "Layer 1"]
    actual_y_labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert actual_y_labels == expected_y_labels
    plt.close(fig)


def test_hierarchy_plot_with_external_ax():
    """Test Hierarchy.plot() with an external Axes object."""
    seg1 = bnl.Segmentation.from_boundaries([0.0, 2.0], ["A"], name="L1")
    hierarchy = bnl.Hierarchy(layers=[seg1], name="External Ax Test")

    ext_fig, ext_ax = plt.subplots()
    fig, ax = hierarchy.plot(ax=ext_ax)

    assert fig is ext_fig
    assert ax is ext_ax
    assert ax.get_title() == "External Ax Test"
    expected_y_labels = ["L1"]
    actual_y_labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert actual_y_labels == expected_y_labels
    plt.close(fig)


def test_hierarchy_plot_with_style_map():
    """Test Hierarchy.plot() with a custom style_map."""
    seg = bnl.Segmentation.from_boundaries([0.0, 2.0, 4.0], ["SegmentX", "SegmentY"], name="Styled Layer")
    hierarchy = bnl.Hierarchy(layers=[seg], name="Style Test")

    custom_color = "purple"
    custom_style_map = {
        "SegmentX": {"facecolor": custom_color, "alpha": 0.7},
        "SegmentY": {"facecolor": "green"} # Will also get ymin/ymax added
    }

    fig, ax = hierarchy.plot(style_map=custom_style_map)

    # Find the patch corresponding to "SegmentX"
    # Patches are added per segment, so the first segment of the first layer is the first patch
    patch_segment_x = ax.patches[0]

    # Convert named color to RGBA for comparison
    expected_rgba = matplotlib.colors.to_rgba(custom_color, alpha=0.7)
    assert matplotlib.colors.same_color(patch_segment_x.get_facecolor(), expected_rgba)

    # Check that ymin/ymax were correctly applied from layer processing
    # For a single layer, ymin=0, ymax=1
    # The axvspan in TimeSpan.plot receives ymin/ymax directly now.
    # We need to check the properties of the patch itself if possible, or that it spans the ax.
    # ax.axvspan ymin, ymax are relative to axes height.
    # The patch extent can be checked via its Bbox.
    bbox = patch_segment_x.get_bbox() # This gives data coordinates for x, axes coordinates for y
    # For a single layer, ymin_ax_coord should be 0 and ymax_ax_coord should be 1.
    # However, ax.patches[i].get_y() and .get_height() might be more direct if ymin/ymax are in axes coords
    # Given that axvspan's ymin/ymax are in axes coordinates (0 to 1 relative to axes height),
    # and our Hierarchy.plot calculates them to span the layer's allocated region,
    # for a single layer, this region is the entire height of the axes.
    assert np.isclose(patch_segment_x._y, 0.0) # y (ymin in axes coords)
    assert np.isclose(patch_segment_x._height, 1.0) # height (ymax-ymin in axes coords)


    patch_segment_y = ax.patches[1]
    expected_rgba_y = matplotlib.colors.to_rgba("green")
    assert matplotlib.colors.same_color(patch_segment_y.get_facecolor(), expected_rgba_y)
    assert np.isclose(patch_segment_y._y, 0.0)
    assert np.isclose(patch_segment_y._height, 1.0)

    plt.close(fig)


def test_hierarchy_plot_empty():
    """Test Hierarchy.plot() with no layers."""
    hierarchy_named = bnl.Hierarchy(layers=[], name="Empty Test")
    fig_named, ax_named = hierarchy_named.plot()

    assert ax_named.get_title() == "Empty Test"
    assert len(ax_named.patches) == 0
    assert len(ax_named.texts) == 0 # No "Empty Hierarchy" text by default
    assert len(ax_named.get_xticks()) == 0 # No data, so likely no ticks by default unless set
    assert len(ax_named.get_yticks()) == 0
    plt.close(fig_named)

    hierarchy_unnamed = bnl.Hierarchy(layers=[])
    fig_unnamed, ax_unnamed = hierarchy_unnamed.plot()
    assert ax_unnamed.get_title() == ""
    assert len(ax_unnamed.patches) == 0
    assert len(ax_unnamed.texts) == 0
    assert len(ax_unnamed.get_xticks()) == 0
    assert len(ax_unnamed.get_yticks()) == 0
    plt.close(fig_unnamed)
