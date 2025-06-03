# BNL (Boundaries and Labels)

A Python library for hierarchical text segmentation and evaluation.
Rewritten with LLMs.

## Quick Start

```python
from bnl import Segment
```

## Documentation

The full documentation is available at [bnl.readthedocs.io](https://bnl.readthedocs.io)

## License
MIT

## Manual Documentation Build Instructions

Since you have a conda environment set up, here are the manual steps to build the documentation:

### Step 1: Build the documentation
```bash
pip install -r docs/requirements.txt
pip install -e .
cd docs
make clean
make html
```

### Step 2: View the documentation
```bash
open _build/html/index.html
```