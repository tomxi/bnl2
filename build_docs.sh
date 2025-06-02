#!/bin/bash
# Build the documentation and open it in the browser
conda deactivate
conda activate py39
pip install -r docs/requirements.txt && pip install -e . && cd docs && make clean && make html && open _build/html/index.html
