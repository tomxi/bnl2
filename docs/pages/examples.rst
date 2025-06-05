Examples
========

Music Structure Analysis
-------------------------

.. code-block:: python

    import numpy as np
    from bnl import Segment, plot_segment

    # Define song structure - boundaries as a set
    boundaries = {0.0, 15.2, 45.8, 78.3, 92.1, 120.0}
    labels = ['intro', 'verse', 'chorus', 'verse', 'outro']
    song = Segment(boundaries, labels)


Visualization
-------------

.. code-block:: python

    # Basic plot
    fig, ax = plot_segment(song, text=True, ytick="Song Structure")
    
    # Compare segmentations
    import matplotlib.pyplot as plt
    
    reference = Segment({0.0, 15.0, 45.0, 78.0, 92.0, 120.0}, 
                       ['intro', 'verse', 'chorus', 'verse', 'outro'])
    prediction = Segment({0.0, 16.5, 44.2, 80.1, 120.0}, 
                        ['intro', 'verse', 'chorus', 'outro'])
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 4))
    plot_segment(reference, ax=axes[0], text=True, ytick="Reference")
    plot_segment(prediction, ax=axes[1], text=True, ytick="Prediction")
    plt.tight_layout()
