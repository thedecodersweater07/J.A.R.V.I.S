#!/bin/bash

# Create deployment directory structure
mkdir -p {docker,kubernetes,scripts,config,monitoring}

# Docker setup
mkdir -p docker/{base,services,development}
touch docker/base/Dockerfile
touch docker/docker-compose.yml

# Kubernetes setup
mkdir -p kubernetes/{manifests,secrets,configmaps}
touch kubernetes/manifests/deployment.yaml
touch kubernetes/manifests/service.yaml
touch kubernetes/manifests/ingress.yaml

# Deployment scripts
mkdir -p scripts/{deploy,backup,maintenance}
touch scripts/deploy/deploy.sh
touch scripts/backup/backup.sh
touch scripts/maintenance/health_check.sh

# Configuration
mkdir -p config/{env,scaling}
touch config/env/{prod,staging,dev}.env

# Monitoring
mkdir -p monitoring/{prometheus,grafana,alerts}
touch monitoring/prometheus/prometheus.yml
touch monitoring/grafana/dashboards.json

# Make scripts executable
find . -name "*.sh" -exec chmod +x {} \;

# Add gitkeep files to maintain structure
find . -type d -empty -exec touch {}/.gitkeep \;
