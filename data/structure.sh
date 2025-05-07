#!/bin/bash

# Create data directory structure
mkdir -p {raw,processed,interim,external}

# Create CSV templates in raw
touch raw/users.csv
touch raw/interactions.csv
touch raw/feedback.csv
touch raw/training_data.csv

# Create processed data folders
mkdir -p processed/{analytics,ml_ready,aggregated}

# Create interim data folders
mkdir -p interim/{temp,backup,validation}

# Create external data folders
mkdir -p external/{apis,third_party,references}

# Add gitkeep files to maintain structure
find . -type d -empty -exec touch {}/.gitkeep \;
