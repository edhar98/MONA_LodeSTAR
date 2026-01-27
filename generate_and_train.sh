#!/bin/bash

# Script to generate rod particles with different SNR values and launch training after each generation
# Usage: ./generate_and_train.sh [snr_start] [snr_end] [snr_step] [image_size]

# Default parameters
SNR_START=${1:-5}
SNR_END=${2:-30}
SNR_STEP=${3:-5}
IMAGE_SIZE=${4:-50}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Rod Particle Generation and Training Pipeline ===${NC}"
echo -e "${BLUE}SNR Range: ${SNR_START} to ${SNR_END} (step: ${SNR_STEP})${NC}"
echo -e "${BLUE}Image Size: ${IMAGE_SIZE}x${IMAGE_SIZE}${NC}"
echo ""

# Function to check if file exists with correct SNR
check_file_exists() {
    local snr=$1
    local file_path="data/Samples/Rod/Rod.jpg"
    local metadata_path="data/Samples/Rod/Rod_info.txt"
    
    # Check if both files exist
    if [ ! -f "$file_path" ] || [ ! -f "$metadata_path" ]; then
        return 1  # Files don't exist
    fi
    
    # Check if metadata contains the correct SNR
    if grep -q "SNR: $snr" "$metadata_path"; then
        return 0  # File exists with correct SNR
    else
        return 1  # File exists but wrong SNR
    fi
}

# Function to backup existing rod sample
backup_existing_sample() {
    local snr=$1
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_dir="data/Samples/Rod/backups"
    
    mkdir -p "$backup_dir"
    
    if [ -f "data/Samples/Rod/Rod.jpg" ]; then
        mv "data/Samples/Rod/Rod.jpg" "$backup_dir/Rod_${timestamp}_backup.jpg"
        echo -e "${YELLOW}Backed up existing rod sample${NC}"
    fi
    
    if [ -f "data/Samples/Rod/Rod_info.txt" ]; then
        mv "data/Samples/Rod/Rod_info.txt" "$backup_dir/Rod_${timestamp}_backup_info.txt"
        echo -e "${YELLOW}Backed up existing rod metadata${NC}"
    fi
}

# Function to generate rod particle with specific SNR
generate_rod_sample() {
    local snr=$1
    echo -e "${YELLOW}Generating rod particle with SNR=${snr}...${NC}"
    
    python src/generate_samples.py \
        --particle-type Rod \
        --snr $snr \
        --image-size $IMAGE_SIZE \
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully generated rod particle with SNR=${snr}${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to generate rod particle with SNR=${snr}${NC}"
        return 1
    fi
}

# Function to launch training process
launch_training() {
    local snr=$1
    echo -e "${YELLOW}Launching training process for SNR=${snr}...${NC}"
    echo -e "${BLUE}Training process would be launched here...${NC}"
    python src/train_single_particle.py --config src/config_debug.yaml
        
    echo -e "${GREEN}✓ Training completed for SNR=${snr}${NC}"
}

# Main loop
for snr in $(seq $SNR_START $SNR_STEP $SNR_END); do
    echo -e "${BLUE}--- Processing SNR=${snr} ---${NC}"
    
    # Check if file already exists with correct SNR
    if check_file_exists $snr; then
        echo -e "${YELLOW}Rod particle with SNR=${snr} already exists, skipping generation${NC}"
    else
        # Check if files exist but with wrong SNR
        if [ -f "data/Samples/Rod/Rod.jpg" ]; then
            echo -e "${YELLOW}Existing rod sample found with different SNR, backing up...${NC}"
            backup_existing_sample $snr
        fi
        
        # Generate rod particle
        if ! generate_rod_sample $snr; then
            echo -e "${RED}Failed to generate rod particle with SNR=${snr}, stopping pipeline${NC}"
            exit 1
        fi
    fi
    
    # Launch training process
    launch_training $snr
    
    echo -e "${GREEN}Completed processing SNR=${snr}${NC}"
    echo ""
done

echo -e "${GREEN}=== Pipeline Complete ===${NC}"
echo -e "${GREEN}Processed SNR values: $(seq $SNR_START $SNR_STEP $SNR_END | tr '\n' ' ')${NC}"
