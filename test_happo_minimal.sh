#!/bin/bash

# Minimal test to isolate the issue

cd /home/gvlab/universal-MAPush/HARL

echo "Testing with minimal settings..."

python examples/train.py \
    --algo happo \
    --env mapush \
    --exp_name debug_test \
    --num_env_steps 100000

echo "Exit code: $?"
