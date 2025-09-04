#!/bin/bash

# =============================================================================
# Speech Model Training Script
# Trains speech recognition and synthesis models
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
MODEL_DIR="$PROJECT_ROOT/models"
LOG_DIR="$PROJECT_ROOT/logs"

# Training parameters
MODEL_TYPE="speech"
BATCH_SIZE=8
EPOCHS=15
LEARNING_RATE=0.0005
SAMPLE_RATE=16000
VALIDATION_SPLIT=0.2

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_DIR/speech_training.log"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_DIR/speech_training.log"
    exit 1
}

# Create required directories
create_directories() {
    log "Creating required directories..."
    mkdir -p "$MODEL_DIR/speech" "$LOG_DIR" "$DATA_DIR/datasets/speech_datasets"
}

# Check audio dependencies
check_audio_dependencies() {
    log "Checking audio processing capabilities..."
    
    # Check for system audio libraries
    if command -v ffmpeg &> /dev/null; then
        log "FFmpeg found: $(ffmpeg -version | head -n1)"
    else
        log "Warning: FFmpeg not found - install for better audio processing"
    fi
    
    # Check Python audio packages
    python3 -c "import librosa, torch, torchaudio, numpy" 2>/dev/null || {
        error "Missing required Python packages. Install with: pip install librosa torch torchaudio numpy"
    }
}

# Data validation
validate_audio_data() {
    log "Validating audio training data..."
    if [[ ! -d "$DATA_DIR/datasets/speech_datasets" ]]; then
        error "Speech datasets directory not found: $DATA_DIR/datasets/speech_datasets"
    fi
    
    local audio_files=$(find "$DATA_DIR/datasets/speech_datasets" -name "*.wav" -o -name "*.mp3" -o -name "*.flac" | wc -l)
    if [[ $audio_files -eq 0 ]]; then
        error "No audio files found in speech_datasets directory"
    fi
    
    log "Found $audio_files audio files"
    
    # Check for transcription files
    if [[ -f "$DATA_DIR/datasets/speech_datasets/transcriptions.txt" ]]; then
        log "Found transcription file: transcriptions.txt"
    else
        log "Warning: No transcriptions.txt found for supervised training"
    fi
    
    # Analyze audio file properties
    local sample_file=$(find "$DATA_DIR/datasets/speech_datasets" -name "*.wav" | head -n1)
    if [[ -n "$sample_file" ]]; then
        if command -v soxi &> /dev/null; then
            local duration=$(soxi -D "$sample_file" 2>/dev/null || echo "unknown")
            local channels=$(soxi -c "$sample_file" 2>/dev/null || echo "unknown")
            local rate=$(soxi -r "$sample_file" 2>/dev/null || echo "unknown")
            log "Sample audio properties - Duration: ${duration}s, Channels: $channels, Rate: ${rate}Hz"
        fi
    fi
}

# Check memory requirements
check_memory() {
    log "Checking system memory..."
    local total_mem=$(free -g | awk '/^Mem:/{print $2}')
    local available_mem=$(free -g | awk '/^Mem:/{print $7}')
    
    log "Total memory: ${total_mem}GB, Available: ${available_mem}GB"
    
    if [[ $available_mem -lt 4 ]]; then
        log "Warning: Low memory detected. Consider reducing batch size"
        BATCH_SIZE=4
        log "Automatically reduced batch size to $BATCH_SIZE"
    fi
}

# Pre-process audio data
preprocess_audio() {
    log "Starting audio preprocessing..."
    
    python3 -c "
import os
import librosa
import numpy as np
from pathlib import Path

data_dir = Path('$DATA_DIR/datasets/speech_datasets')
processed_dir = data_dir / 'processed'
processed_dir.mkdir(exist_ok=True)

audio_files = list(data_dir.glob('*.wav')) + list(data_dir.glob('*.mp3'))
print(f'Preprocessing {len(audio_files)} audio files...')

for i, audio_file in enumerate(audio_files):
    try:
        # Load and resample audio
        y, sr = librosa.load(str(audio_file), sr=$SAMPLE_RATE)
        
        # Save processed audio
        output_file = processed_dir / f'{audio_file.stem}_processed.npy'
        np.save(output_file, y)
        
        if (i + 1) % 100 == 0:
            print(f'Processed {i + 1}/{len(audio_files)} files')
            
    except Exception as e:
        print(f'Error processing {audio_file}: {e}')

print('Audio preprocessing completed')
" || log "Preprocessing completed with some warnings"
}

# Start training process
start_training() {
    log "Starting Speech model training..."
    log "Configuration:"
    log "  - Model Type: $MODEL_TYPE"
    log "  - Batch Size: $BATCH_SIZE"
    log "  - Epochs: $EPOCHS"
    log "  - Learning Rate: $LEARNING_RATE"
    log "  - Sample Rate: $SAMPLE_RATE Hz"
    log "  - Validation Split: $VALIDATION_SPLIT"
    
    # Start Python training orchestrator
    python3 "$PROJECT_ROOT/core/model_trainer.py" \
        --model_type "$MODEL_TYPE" \
        --data_dir "$DATA_DIR/datasets/speech_datasets" \
        --output_dir "$MODEL_DIR/speech" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --validation_split "$VALIDATION_SPLIT" \
        --sample_rate "$SAMPLE_RATE" \
        --log_file "$LOG_DIR/speech_training.log" || {
        error "Training failed. Check logs for details."
    }
}

# Monitor system resources
monitor_resources() {
    echo -e "${BLUE}System resource monitoring - Press Ctrl+C to stop${NC}"
    while true; do
        clear
        echo -e "${GREEN}=== System Resources ===${NC}"
        echo -e "${YELLOW}Memory Usage:${NC}"
        free -h
        echo -e "\n${YELLOW}CPU Usage:${NC}"
        top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print "CPU Usage: " 100 - $1 "%"}'
        echo -e "\n${YELLOW}Disk Usage:${NC}"
        df -h "$MODEL_DIR" | tail -1
        sleep 5
    done
}

# Main execution
main() {
    log "=== Speech Model Training Started ==="
    
    create_directories
    check_audio_dependencies
    validate_audio_data
    check_memory
    preprocess_audio
    start_training
    
    log "=== Speech Model Training Completed ==="
    log "Model saved to: $MODEL_DIR/speech/"
    log "Logs available at: $LOG_DIR/speech_training.log"
}

# Handle command line arguments
case "${1:-}" in
    --monitor-resources)
        monitor_resources
        ;;
    --preprocess-only)
        create_directories
        check_audio_dependencies
        validate_audio_data
        preprocess_audio
        log "Audio preprocessing completed"
        ;;
    --help|-h)
        echo "Usage: $0 [--monitor-resources|--preprocess-only|--help]"
        echo "  --monitor-resources: Monitor system resource usage"
        echo "  --preprocess-only:   Only preprocess audio data"
        echo "  --help:              Show this help message"
        ;;
    *)
        main "$@"
        ;;
esac