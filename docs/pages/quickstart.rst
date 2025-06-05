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
    from bnl import seg_from_brdys, plot_segment

    # Create segmentation from boundaries
    boundaries = [0.0, 2.5, 5.0, 7.5, 10.0]
    labels = ['A', 'B', 'A', 'C']
    seg = seg_from_brdys(boundaries, labels)

    # Access properties
    print(f"Duration: {seg.end - seg.start}")
    print(f"Labels: {seg.labels}")
    print(f"Boundaries: {seg.bdrys}")

    # Visualize
    fig, ax = plot_segment(seg, text=True)


Next Steps
----------

See :doc:`examples` for more detailed usage examples and :doc:`../api/bnl` for complete API documentation. 