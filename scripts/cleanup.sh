#!/bin/bash

# Remove empty directories
find . -type d -empty -delete

# Remove temporary files
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "__pycache__" -delete
find . -type f -name ".DS_Store" -delete

# Remove old structure scripts
rm -f setup_structure.sh
rm -f setup_ml_structure.sh
rm -f db/structure.sh
rm -f data/structure.sh
