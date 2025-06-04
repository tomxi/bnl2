Quick Start
===========

Installation
------------

.. code-block:: bash

    git clone https://github.com/tomxi/bnl2.git
    cd bnl2
    pip install -e ".[dev,docs]"

Basic Usage
-----------

.. code-block:: python

    import numpy as np
    from bnl import Segment, plot_segment

    # Create boundaries and labels - boundaries as a set
    boundaries = {0.0, 2.5, 5.0, 7.5, 10.0}
    labels = ['A', 'B', 'A', 'C']
    seg = Segment(boundaries, labels)

    # Access properties
    print(f"Duration: {seg.duration}")
    print(f"Number of segments: {seg.num_segments}")

    # Visualize
    fig, ax = plot_segment(seg, text=True)


Next Steps
----------

See :doc:`examples` for more detailed usage examples and :doc:`api/bnl` for complete API documentation. 