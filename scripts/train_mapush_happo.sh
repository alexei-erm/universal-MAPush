#!/bin/bash

# HAPPO Training Script for MAPush
# This script provides easy commands to train MAPush with HAPPO algorithm

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}HAPPO Training for MAPush${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Activate mapush conda environment
if command -v conda &> /dev/null; then
    echo -e "${YELLOW}Activating mapush conda environment...${NC}"
    eval "$(conda shell.bash hook)"
    conda activate mapush
    echo -e "${GREEN}✓ Environment activated${NC}"
    echo ""
else
    echo -e "${YELLOW}Warning: conda not found. Make sure you're using the correct Python environment.${NC}"
    echo ""
fi

# Check if we're in the right directory
if [ ! -d "HARL" ]; then
    echo -e "${RED}Error: HARL directory not found!${NC}"
    echo "Please run this script from the MAPush root directory."
    exit 1
fi

# Default parameters
ALGO="happo"
ENV="mapush"
EXP_NAME="test"
OBJECT_TYPE="cuboid"
NUM_ENVS=10
NUM_STEPS=50000000
EPISODE_LENGTH=4000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --object_type)
            OBJECT_TYPE="$2"
            shift 2
            ;;
        --num_envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --num_steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --episode_length)
            EPISODE_LENGTH="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./train_mapush_happo.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --exp_name NAME        Experiment name (default: test)"
            echo "  --object_type TYPE     Object type: cuboid|cylinder|Tblock (default: cuboid)"
            echo "  --num_envs NUM         Number of parallel environments (default: 10)"
            echo "  --num_steps NUM        Total training steps (default: 50000000)"
            echo "  --episode_length NUM   Episode length in steps (default: 4000)"
            echo "  --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./train_mapush_happo.sh --exp_name my_experiment"
            echo "  ./train_mapush_happo.sh --object_type cylinder --num_envs 20"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Display configuration
echo -e "${YELLOW}Training Configuration:${NC}"
echo "  Algorithm: $ALGO"
echo "  Environment: $ENV"
echo "  Experiment Name: $EXP_NAME"
echo "  Object Type: $OBJECT_TYPE"
echo "  Parallel Environments: $NUM_ENVS"
echo "  Total Steps: $NUM_STEPS"
echo "  Episode Length: $EPISODE_LENGTH"
echo ""

# Confirm before starting
read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Navigate to HARL directory
cd HARL

# Start training
echo -e "${GREEN}Starting HAPPO training...${NC}"
echo ""

python examples/train.py \
    --algo $ALGO \
    --env $ENV \
    --exp_name "${EXP_NAME}" \
    --object_type "$OBJECT_TYPE" \
    --n_rollout_threads $NUM_ENVS \
    --num_env_steps $NUM_STEPS \
    --episode_length $EPISODE_LENGTH

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Results saved to: HARL/results/mapush/happo/${EXP_NAME}/"
    echo ""
    echo "To view training progress:"
    echo "  tensorboard --logdir HARL/results/mapush/happo/${EXP_NAME}"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Training failed!${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Check the error messages above for details."
    exit 1
fi
