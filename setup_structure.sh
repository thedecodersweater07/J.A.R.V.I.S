#!/bin/bash

# Create main project structure
mkdir -p {data,models,notebooks,configs,logs}
mkdir -p data/{raw,processed,interim}
mkdir -p models/{saved,checkpoints}
mkdir -p configs/{env,model}

# Create config files
touch configs/env/{dev,prod,test}.yaml
touch configs/model/model_config.yaml
touch configs/logging_config.yaml

# Create initialization files
touch __init__.py
touch data/__init__.py
touch models/__init__.py

# Make the script executable
chmod +x setup_structure.sh
