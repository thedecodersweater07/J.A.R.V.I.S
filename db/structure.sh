#!/bin/bash

# Create database directory structure
mkdir -p {sql,nosql,vector,cache}

# SQL databases
mkdir -p sql/{main,analytics,archive}
touch sql/main/schema.sql
touch sql/main/init.sql

# NoSQL databases
mkdir -p nosql/{mongodb,redis,elasticsearch}
touch nosql/mongodb/indexes.js
touch nosql/redis/config.conf

# Vector databases
mkdir -p vector/{embeddings,indexes}
touch vector/config.yaml

# Cache structure
mkdir -p cache/{memory,disk}
touch cache/policy.json

# Add gitkeep files to maintain structure
find . -type d -empty -exec touch {}/.gitkeep \;
