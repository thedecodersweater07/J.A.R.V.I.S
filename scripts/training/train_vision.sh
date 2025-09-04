#!/bin/bash

# =============================================================================
# Vision Model Training Script
# Trains computer vision models with GPU optimization
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
MODEL_DIR="$PROJECT_ROOT/models"
LOG_DIR="$PROJECT_ROOT/logs"

# Training parameters
MODEL_TYPE="vision"
BATCH_SIZE=16
EPOCHS=20
LEARNING_RATE=0.0001
IMAGE_SIZE=224
VALIDATION_SPLIT=0.2

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_DIR/vision_training.log"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_DIR/vision_training.log"
    exit 1
}

# Create required directories
create_directories() {
    log "Creating required directories..."
    mkdir -p "$MODEL_DIR/vision" "$LOG_DIR" "$DATA_DIR/datasets/vision_datasets"
}

# Check GPU availability
check_gpu() {
    log "Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi -L | wc -l)
        log "Found $gpu_count GPU(s)"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
            log "  GPU: $line"
        done
    else
        log "No GPU detected - training will use CPU (slower)"
    fi
}

# Check Python dependencies
check_dependencies() {
    log "Checking Python dependencies..."
    python3 -c "import torch, torchvision, PIL, numpy, opencv" 2>/dev/null || {
        error "Missing required Python packages. Install with: pip install torch torchvision pillow numpy opencv-python"
    }
}

# Data validation
validate_data() {
    log "Validating training data..."
    if [[ ! -d "$DATA_DIR/datasets/vision_datasets" ]]; then
        error "Vision datasets directory not found: $DATA_DIR/datasets/vision_datasets"
    fi
    
    local image_files=$(find "$DATA_DIR/datasets/vision_datasets" -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" | wc -l)
    if [[ $image_files -eq 0 ]]; then
        error "No image files found in vision_datasets directory"
    fi
    
    log "Found $image_files image files"
    
    # Check for labels file
    if [[ -f "$DATA_DIR/datasets/vision_datasets/labels.csv" ]]; then
        log "Found labels file: labels.csv"
    else
        log "Warning: No labels.csv found, using directory structure for labels"
    fi
}

# Disk space check
check_disk_space() {
    log "Checking available disk space..."
    local available_space=$(df "$MODEL_DIR" | tail -1 | awk '{print $4}')
    local space_gb=$((available_space / 1024 / 1024))
    
    if [[ $space_gb -lt 5 ]]; then
        error "Insufficient disk space. Need at least 5GB, have ${space_gb}GB"
    fi
    
    log "Available disk space: ${space_gb}GB"
}

# Start training process
start_training() {
    log "Starting Vision model training..."
    log "Configuration:"
    log "  - Model Type: $MODEL_TYPE"
    log "  - Batch Size: $BATCH_SIZE"
    log "  - Epochs: $EPOCHS"
    log "  - Learning Rate: $LEARNING_RATE"
    log "  - Image Size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
    log "  - Validation Split: $VALIDATION_SPLIT"
    
    # Start Python training orchestrator
    python3 "$PROJECT_ROOT/core/model_trainer.py" \
        --model_type "$MODEL_TYPE" \
        --data_dir "$DATA_DIR/datasets/vision_datasets" \
        --output_dir "$MODEL_DIR/vision" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --validation_split "$VALIDATION_SPLIT" \
        --image_size "$IMAGE_SIZE" \
        --log_file "$LOG_DIR/vision_training.log" || {
        error "Training failed. Check logs for details."
    }
}

# Monitor GPU usage during training
monitor_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${BLUE}GPU monitoring - Press Ctrl+C to stop${NC}"
        watch -n 5 nvidia-smi
    else
        echo -e "${YELLOW}No GPU monitoring available${NC}"
    fi
}

# Main execution
main() {
    log "=== Vision Model Training Started ==="
    
    create_directories
    check_gpu
    check_dependencies
    validate_data
    check_disk_space
    start_training
    
    log "=== Vision Model Training Completed ==="
    log "Model saved to: $MODEL_DIR/vision/"
    log "Logs available at: $LOG_DIR/vision_training.log"
}

# Handle command line arguments
case "${1:-}" in
    --monitor-gpu)
        monitor_gpu
        ;;
    --help|-h)
        echo "Usage: $0 [--monitor-gpu|--help]"
        echo "  --monitor-gpu: Monitor GPU usage during training"
        echo "  --help:        Show this help message"
        ;;
    *)
        main "$@"
        ;;
esac