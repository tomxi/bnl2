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

### Step 1: Activate your conda environment
```bash
conda activate py39  # or whatever your environment is named
```

### Step 2: Install documentation dependencies
```bash
pip install -r docs/requirements.txt
```

### Step 3: Install your project in editable mode
```bash
pip install -e .
``

### Step 4: Build the documentation
```bash
cd docs
make clean
make html
```

### Step 5: View the documentation
```bash
open _build/html/index.html
```

### Alternative: One-liner after conda activation
```bash
