#!/bin/bash

# Create main project structure
mkdir -p src/{data,models,preprocessing,training,evaluation,utils}
mkdir -p data/{raw,processed,interim}
mkdir -p models/{saved,configs}
mkdir -p notebooks
mkdir -p docs

# Create initial files
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/preprocessing/__init__.py
touch src/training/__init__.py
touch src/evaluation/__init__.py
touch src/utils/__init__.py

# Make the script executable
chmod +x setup_ml_structure.sh
