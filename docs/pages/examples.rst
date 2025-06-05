Examples
========

Music Structure Analysis
-------------------------

.. code-block:: python

    import numpy as np
    from bnl import seg_from_brdys, plot_segment

    # Define song structure boundaries
    boundaries = [0.0, 15.2, 45.8, 78.3, 92.1, 120.0]
    labels = ['intro', 'verse', 'chorus', 'verse', 'outro']
    song = seg_from_brdys(boundaries, labels)


Visualization
-------------

.. code-block:: python

    # Basic plot
    fig, ax = plot_segment(song, text=True, ytick="Song Structure")
    
    # Compare segmentations
    import matplotlib.pyplot as plt
    
    reference = seg_from_brdys([0.0, 15.0, 45.0, 78.0, 92.0, 120.0], 
                               ['intro', 'verse', 'chorus', 'verse', 'outro'])
    prediction = seg_from_brdys([0.0, 16.5, 44.2, 80.1, 120.0], 
                                ['intro', 'verse', 'chorus', 'outro'])
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 4))
    plot_segment(reference, ax=axes[0], text=True, ytick="Reference")
    plot_segment(prediction, ax=axes[1], text=True, ytick="Prediction")
    plt.tight_layout()


Working with SALAMI Data
-------------------------

Loading and exploring the SALAMI dataset:

.. code-block:: python

    import bnl.data as data
    import matplotlib.pyplot as plt

    # Explore available tracks
    track_ids = data.slm.list_tids()
    print(f"Available SALAMI tracks: {len(track_ids)}")
    
    # Load a specific track
    track = data.slm.load_track('10')
    print(f"Track: {track}")
    
    # Access metadata
    info = track.info
    print(f"Artist: {info['artist']}")
    print(f"Title: {info['title']}")
    print(f"Duration: {info['duration']:.1f}s")
    
    # Access JAMS annotation data
    jams_obj = track.jams
    print(f"Annotations: {len(jams_obj.annotations)}")
    
    # Load multiple tracks for analysis
    sample_tracks = data.slm.load_tracks(['10', '100', '1000'])
    durations = [track.info['duration'] for track in sample_tracks]
    print(f"Sample durations: {durations}")

.. note::
   To use data loading functionality, ensure SALAMI dataset files are in ``~/data/``:
   
   - JAMS annotations: ``~/data/salami-jams/``
   - Audio files: ``~/data/salami/audio/``
