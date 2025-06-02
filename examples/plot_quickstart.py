"""
Quickstart Example
================

A simple example demonstrating basic usage of BNL.
"""

# %%
# Import required packages
from bnl.core.segment import Segment
from bnl.core.hierarchy import Hierarchy

# Create segments
seg1 = Segment(0, 10, "First segment")
seg2 = Segment(10, 20, "Second segment")

# Create a hierarchy
h = Hierarchy()
root = h.add_segment(Segment(0, 100, "Document"))
h.add_segment(seg1, parent=root)
h.add_segment(seg2, parent=root)

# Find a segment
found = h.find_segment(5)
print(f"Found segment: {found.segment.text}")

# %%
# You can also add more complex hierarchies
for i in range(3):
    h.add_segment(
        Segment(i*5, (i+1)*5, f"Child {i+1}"),
        parent=found
    )

# Now let's visualize the hierarchy
print("\nHierarchy structure:")
print(f"- {root.segment.text}")
for child in root.children:
    print(f"  - {child.segment.text}")
    for grandchild in child.children:
        print(f"    - {grandchild.segment.text}")
