#!/bin/bash

# Quick test script for HAPPO integration
# This ensures the correct conda environment is used

set -e

echo "========================================="
echo "HAPPO Integration Test"
echo "========================================="
echo ""

# Check if we're in conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda/Miniconda."
    exit 1
fi

# Activate the mapush environment
echo "Activating mapush environment..."
eval "$(conda shell.bash hook)"
conda activate mapush

# Verify Isaac Gym is available
python -c "import isaacgym; print('✓ Isaac Gym found')" || {
    echo "Error: Isaac Gym not found in mapush environment"
    echo "Please install Isaac Gym first:"
    echo "  cd isaac_gym/python && pip install -e ."
    exit 1
}

echo ""
echo "Running integration tests..."
echo ""

# Run the test
python test_mapush_harl.py

echo ""
echo "========================================="
echo "Test complete!"
echo "========================================="
