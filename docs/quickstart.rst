Quick Start
===========

Installation
------------

.. code-block:: bash

    git clone https://github.com/tomxi/bnl.git
    cd bnl
    pip install -e ".[dev,docs]"

Basic Usage
-----------

.. code-block:: python

    import numpy as np
    from bnl import Segment, plot_segment

    # Create boundaries and labels
    boundaries = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
    labels = ['A', 'B', 'A', 'C']
    seg = Segment(boundaries, labels)

    # Access properties
    print(f"Duration: {seg.duration}")
    print(f"Number of segments: {seg.num_segments}")

    # Visualize
    fig, ax = plot_segment(seg, text=True)

From mir_eval Format
--------------------

.. code-block:: python

    # Convert from mir_eval-style data
    intervals = np.array([[0.0, 2.5], [2.5, 5.0], [5.0, 7.5], [7.5, 10.0]])
    labels = ['A', 'B', 'A', 'C']
    seg = Segment.from_mir_eval(intervals, labels)

Next Steps
----------

See :doc:`examples` for more detailed usage examples and :doc:`api/bnl` for complete API documentation. 