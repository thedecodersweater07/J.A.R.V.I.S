#!/bin/bash

# =============================================================================
# NLP Model Training Script
# Trains natural language processing models with progress monitoring
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
MODEL_DIR="$PROJECT_ROOT/models"
LOG_DIR="$PROJECT_ROOT/logs"

# Training parameters
MODEL_TYPE="nlp"
BATCH_SIZE=32
EPOCHS=10
LEARNING_RATE=0.001
VALIDATION_SPLIT=0.2

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_DIR/nlp_training.log"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_DIR/nlp_training.log"
    exit 1
}

# Create required directories
create_directories() {
    log "Creating required directories..."
    mkdir -p "$MODEL_DIR/nlp" "$LOG_DIR" "$DATA_DIR/datasets/nlp_datasets"
}

# Check Python dependencies
check_dependencies() {
    log "Checking Python dependencies..."
    python3 -c "import torch, transformers, numpy, pandas" 2>/dev/null || {
        error "Missing required Python packages. Install with: pip install torch transformers numpy pandas"
    }
}

# Data validation
validate_data() {
    log "Validating training data..."
    if [[ ! -d "$DATA_DIR/datasets/nlp_datasets" ]]; then
        error "NLP datasets directory not found: $DATA_DIR/datasets/nlp_datasets"
    fi
    
    local data_files=$(find "$DATA_DIR/datasets/nlp_datasets" -name "*.txt" -o -name "*.json" -o -name "*.csv" | wc -l)
    if [[ $data_files -eq 0 ]]; then
        error "No training data files found in nlp_datasets directory"
    fi
    
    log "Found $data_files training data files"
}

# Start training process
start_training() {
    log "Starting NLP model training..."
    log "Configuration:"
    log "  - Model Type: $MODEL_TYPE"
    log "  - Batch Size: $BATCH_SIZE"
    log "  - Epochs: $EPOCHS"
    log "  - Learning Rate: $LEARNING_RATE"
    log "  - Validation Split: $VALIDATION_SPLIT"
    
    # Start Python training orchestrator
    python3 "$PROJECT_ROOT/core/model_trainer.py" \
        --model_type "$MODEL_TYPE" \
        --data_dir "$DATA_DIR/datasets/nlp_datasets" \
        --output_dir "$MODEL_DIR/nlp" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --validation_split "$VALIDATION_SPLIT" \
        --log_file "$LOG_DIR/nlp_training.log" || {
        error "Training failed. Check logs for details."
    }
}

# Monitor training progress
monitor_progress() {
    local log_file="$LOG_DIR/nlp_training.log"
    echo -e "${BLUE}Monitoring training progress...${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop monitoring (training will continue)${NC}"
    
    tail -f "$log_file" | while read line; do
        if [[ $line =~ "Epoch" ]]; then
            echo -e "${GREEN}$line${NC}"
        elif [[ $line =~ "Loss" ]]; then
            echo -e "${BLUE}$line${NC}"
        elif [[ $line =~ "ERROR" ]]; then
            echo -e "${RED}$line${NC}"
        fi
    done
}

# Main execution
main() {
    log "=== NLP Model Training Started ==="
    
    create_directories
    check_dependencies
    validate_data
    start_training
    
    log "=== NLP Model Training Completed ==="
    log "Model saved to: $MODEL_DIR/nlp/"
    log "Logs available at: $LOG_DIR/nlp_training.log"
}

# Handle command line arguments
case "${1:-}" in
    --monitor)
        monitor_progress
        ;;
    --help|-h)
        echo "Usage: $0 [--monitor|--help]"
        echo "  --monitor: Monitor training progress"
        echo "  --help:    Show this help message"
        ;;
    *)
        main "$@"
        ;;
esac