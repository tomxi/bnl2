#!/bin/bash
# Build the documentation and open it in the browser

conda activate py39
cd docs && make clean && make html && open _build/html/index.html
