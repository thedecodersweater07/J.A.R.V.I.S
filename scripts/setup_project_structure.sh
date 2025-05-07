#!/bin/bash

# Root structure
mkdir -p {core,ui,llm,ml,nlp,speech,security,db,config,utils,tests,docs}

# Core module
mkdir -p core/{brain,memory,command,logging}
mkdir -p core/brain/{cognitive,neural,consciousness}
mkdir -p core/memory/{short_term,long_term,indexing}
mkdir -p core/command/{parser,executor,feedback}

# UI module
mkdir -p ui/{screens,components,visual,input}
mkdir -p ui/screens/{base,chat,settings,data}
mkdir -p ui/components/{widgets,dialogs,menus}
mkdir -p ui/visual/{themes,assets,renderers}

# LLM module
mkdir -p llm/{models,training,inference,optimization,memory,knowledge}
mkdir -p llm/models/{architecture,checkpoints}
mkdir -p llm/training/{pipelines,data}
mkdir -p llm/inference/{pipeline,caching}

# ML module
mkdir -p ml/{models,training,evaluation,optimization}
mkdir -p ml/models/{supervised,unsupervised,reinforcement}
mkdir -p ml/training/{pipelines,optimizers}

# Database module
mkdir -p db/{sql,nosql,vector,cache}
mkdir -p db/sql/{migrations,queries,models}
mkdir -p db/nosql/{mongodb,redis}
mkdir -p db/vector/{embeddings,indexes}

# Data storage
mkdir -p data/{raw,processed,interim,external}
mkdir -p data/raw/{conversations,feedback,metrics}
mkdir -p data/processed/{analytics,training,validation}

# Config structure
mkdir -p config/{defaults,profiles,environments,secrets}
touch config/defaults/{core,llm,ml,ui}.json
touch config/main.json

# Documentation
mkdir -p docs/{api,guides,architecture}
touch docs/README.md

# Add gitkeep files
find . -type d -empty -exec touch {}/.gitkeep \;

# Make scripts executable
chmod +x scripts/*.sh
