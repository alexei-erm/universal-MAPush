# MAPush Repository - Comprehensive Summary

**Repository:** agnostic-MAPush
**Location:** `/home/gvlab/agnostic-MAPush`
**Purpose:** Multi-Agent Reinforcement Learning for Quadrupedal Collaborative Pushing
**Last Updated:** 2025-11-11

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Core Architecture](#core-architecture)
4. [Key Components](#key-components)
5. [Workflow & Usage](#workflow--usage)
6. [Important Files & Their Roles](#important-files--their-roles)
7. [Configuration System](#configuration-system)
8. [Training & Testing](#training--testing)
9. [Recent Work & Modifications](#recent-work--modifications)
10. [Common Operations](#common-operations)
11. [Troubleshooting Notes](#troubleshooting-notes)

---

## Project Overview

### Research Context
This is the implementation of a **hierarchical multi-agent reinforcement learning (MARL) framework** for multi-quadruped pushing tasks. The project enables multiple Unitree Go1 quadruped robots to collaboratively push objects (cuboids, T-blocks, cylinders) to target locations in Isaac Gym.

**Paper:** "Learning Multi-Agent Loco-Manipulation for Long-Horizon Quadrupedal Pushing"
**Website:** https://collaborative-mapush.github.io/
**Paper Link:** https://arxiv.org/pdf/2411.07104
**Based On:** [MQE](https://github.com/ziyanx02/multiagent-quadruped-environment)

### Key Features
- **Hierarchical Control:** Two-level controller architecture (mid-level and high-level)
- **Multi-Agent Coordination:** 2 quadruped robots working collaboratively
- **Isaac Gym Simulation:** GPU-accelerated physics simulation
- **Multiple Object Types:** Cuboid, T-block, and cylinder pushing
- **Domain Randomization:** Position, orientation, and friction randomization

---

## Repository Structure

```
agnostic-MAPush/
├── mqe/                          # Multi-Quadruped Environment (core engine)
│   ├── envs/
│   │   ├── base/                 # Base classes for environments
│   │   │   ├── legged_robot.py   # Core robot physics & control
│   │   │   ├── base_task.py      # Base task implementation
│   │   │   └── *_config.py       # Base configurations
│   │   ├── go1/                  # Unitree Go1 specific implementation
│   │   │   ├── go1.py            # Go1 robot class
│   │   │   └── go1_config.py     # Go1 base configuration
│   │   ├── npc/                  # Non-player characters (objects)
│   │   │   └── go1_object.py     # Go1 + object interaction
│   │   ├── configs/              # Task-specific configurations
│   │   │   ├── go1_push_mid_config.py    # Mid-level controller config
│   │   │   └── go1_push_upper_config.py  # High-level controller config
│   │   ├── wrappers/             # Environment wrappers for RL
│   │   │   ├── go1_push_mid_wrapper.py   # Mid-level observations/rewards
│   │   │   └── go1_push_upper_wrapper.py # High-level observations/rewards
│   │   └── field/                # Terrain and field configurations
│   └── utils/                    # Utility functions
│       └── terrain/              # Terrain generation
│
├── openrl_ws/                    # OpenRL workspace (training interface)
│   ├── train.py                  # Main training script
│   ├── test.py                   # Testing & evaluation script
│   ├── utils.py                  # Wrapper utilities for OpenRL
│   ├── update_config.py          # Config update helper
│   └── cfgs/                     # OpenRL algorithm configs
│       └── README_SAFE_CONFIG.md
│
├── task/                         # Task definitions for different objects
│   ├── cuboid/
│   │   ├── config.py             # Cuboid-specific configuration
│   │   └── train.sh              # Training/testing script for cuboid
│   ├── cylinder/
│   │   ├── config.py
│   │   └── train.sh
│   └── Tblock/
│       ├── config.py
│       └── train.sh
│
├── results/                      # Training results & checkpoints
│   ├── 10-15-23_cuboid/          # Example: Oct 15, 11PM cuboid run
│   ├── 11-06-17_cuboid/
│   └── 11-07-17_cuboid/
│   └── <mm-dd-hh_object>/
│       ├── checkpoints/          # Model checkpoints (every 10M-20M steps)
│       │   └── rl_model_*_steps/
│       │       └── module.pt
│       ├── events.out.tfevents.* # TensorBoard logs
│       ├── success_rate.txt      # Success rate evaluation results
│       └── task/                 # Copy of task config used
│
├── resources/                    # Assets & pretrained models
│   ├── robots/                   # Robot URDF files
│   ├── objects/                  # Object URDFs (cuboid, cylinder, etc.)
│   │   ├── cuboid/
│   │   ├── cylinder/
│   │   └── target.urdf
│   ├── actuator_nets/            # Pretrained low-level locomotion policies
│   ├── command_nets/             # Pretrained mid-level policies
│   └── goals_net/                # Pretrained high-level policies
│
├── docs/                         # Documentation & outputs
│   ├── video/                    # Recorded videos
│   └── gif/                      # GIF animations
│
├── helpers/                      # Helper scripts
│   ├── gpu_monitor.py
│   └── architecture_check.py
│
├── helpers_claude/               # This directory - for Claude sessions
│   └── claude_summary.md         # This file
│
├── log/                          # Temporary training logs
│   └── MQE/
│
├── script/                       # Legacy training scripts
│   └── utils/
│
└── [Documentation Files]
    ├── README.md                 # Main project README
    ├── MULTI_EPISODE_RECORDING.md    # Video recording guide
    ├── VNC_SETUP_STATUS.md           # Remote rendering setup
    └── TEST_CHECKPOINTS_README.md    # Checkpoint testing guide
```

---

## Core Architecture

### Three-Level Hierarchical Control

```
┌─────────────────────────────────────────────────────────┐
│              HIGH-LEVEL CONTROLLER                      │
│  (High-level Policy Network - go1push_upper)            │
│  - Plans subgoals for the object                        │
│  - Outputs: Intermediate target positions               │
│  - Observation: Global state (robots, box, final goal)  │
│  - Trained with: PPO in OpenRL                          │
└────────────────┬────────────────────────────────────────┘
                 │ Subgoal positions
                 ▼
┌─────────────────────────────────────────────────────────┐
│              MID-LEVEL CONTROLLER                       │
│  (Command Network - go1push_mid)                        │
│  - Coordinates robot movements to push object           │
│  - Outputs: Velocity commands (vx, vy, vyaw)            │
│  - Observation: Relative positions (robots↔box↔goal)    │
│  - Trained with: PPO/MAPPO in OpenRL                    │
└────────────────┬────────────────────────────────────────┘
                 │ Velocity commands
                 ▼
┌─────────────────────────────────────────────────────────┐
│              LOW-LEVEL CONTROLLER                       │
│  (Actuator Network - from walk-these-ways)              │
│  - Executes locomotion for each robot                   │
│  - Outputs: Joint positions/torques                     │
│  - Pretrained on locomotion tasks                       │
│  - NOT trained in this repo (loaded from resources/)    │
└─────────────────────────────────────────────────────────┘
```

### Environment Flow

```
┌──────────────────┐
│  Task Config     │ (task/cuboid/config.py)
│  - Object type   │
│  - Rewards       │
│  - Randomization │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Go1Object Env   │ (mqe/envs/npc/go1_object.py)
│  - Isaac Gym     │
│  - 2 Go1 robots  │
│  - 1 object      │
│  - 1 target area │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Task Wrapper    │ (mqe/envs/wrappers/go1_push_mid_wrapper.py)
│  - Observations  │ → Relative positions/distances
│  - Rewards       │ → Distance, collision, push rewards
│  - Actions       │ → Scale velocity commands
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  OpenRL Wrapper  │ (openrl_ws/utils.py)
│  - Multi-agent   │
│  - Numpy ↔ Torch │
│  - Batch rewards │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  PPO/MAPPO Agent │ (OpenRL)
│  - Training loop │
│  - Checkpointing │
└──────────────────┘
```

---

## Key Components

### 1. Environment Base Classes

**`mqe/envs/base/legged_robot.py`**
- Core physics simulation using Isaac Gym
- Robot state management (positions, velocities, orientations)
- Action application and physics stepping
- Rendering and video recording functionality
- **Important methods:**
  - `reset()`: Reset environment
  - `step()`: Apply actions and step physics
  - `_render_headless()`: Capture frames for video
  - `store_recording()`: Save recorded episodes

**`mqe/envs/go1/go1.py`**
- Unitree Go1 specific implementation
- Loads pretrained locomotion policy (actuator network)
- Joint control and leg dynamics
- Base configuration for Go1 robots

**`mqe/envs/npc/go1_object.py`**
- Combines robots with pushable objects
- Object physics and collision handling
- Target area management
- Multi-agent coordination logic

### 2. Task Wrappers

**`mqe/envs/wrappers/go1_push_mid_wrapper.py`**
- **Observation Space:** `[2 + 3*num_agents]` or `[3 + 3*num_agents]`
  - Distance and angle to goal
  - Relative positions of each robot to box
- **Action Space:** `[3]` per agent (vx, vy, vyaw)
- **Rewards:**
  - `target_reward`: Distance to goal decreased
  - `approach_reward`: Robots approaching box
  - `push_reward`: Box moving toward goal
  - `ocb_reward`: Optimal Circular Baseline reward
  - `collision_punishment`: Robots too close
  - `reach_target_reward`: Goal reached (+10)
  - `exception_punishment`: Termination conditions (-5)

**`mqe/envs/wrappers/go1_push_upper_wrapper.py`**
- High-level planner wrapper
- Outputs subgoal positions for mid-level controller
- Longer horizon planning

### 3. Configuration System

Configuration uses **class-based inheritance**:

```python
Go1Cfg (base)
  ↓
Go1PushMidCfg (mid-level)
  ↓
task/cuboid/config.py (task-specific)
```

**Key configuration classes:**
- `env`: Environment parameters (num_envs, num_agents, episode_length)
- `asset`: Robot and object URDF files, vertices
- `terrain`: Map size, walls, friction
- `command`: Velocity command configuration
- `control`: Control type ('C' for command)
- `termination`: Termination conditions (roll, pitch, collision)
- `rewards`: Reward scales
- `goal`: Goal positioning (static/random/received)
- `init_state`: Initial positions of robots and objects
- `domain_rand`: Randomization ranges (position, orientation, friction)

### 4. Training Infrastructure

**`openrl_ws/train.py`**
- Main training script
- Creates environment with `make_env()`
- Initializes PPONet and PPOAgent
- Sets up logging (TensorBoard, WandB)
- Checkpoint saving every 20M steps
- Moves results to `./results/<timestamp>_<object>/` after training

**`openrl_ws/test.py`**
- Evaluation script with multiple modes:
  - `viewer`: Visual rendering
  - `calculator`: Compute success rates
  - Video recording with `--record_video`
- **Recent modifications** (see MULTI_EPISODE_RECORDING.md):
  - Hardcoded seed control: `SEED = 5`
  - Multi-episode recording: `NUM_EPISODES = 3`
  - Monkey-patching for frame accumulation

**`task/<object>/train.sh`**
- Wrapper script for training and testing
- Updates `mqe/envs/configs/go1_push_mid_config.py` from `task/<object>/config.py`
- Training: `source task/cuboid/train.sh False`
- Testing: `source results/11-07-17_cuboid/task/train.sh True`

---

## Workflow & Usage

### Training Mid-Level Controller

```bash
# 1. Edit task configuration
vim task/cuboid/config.py

# 2. Start training (runs in background typically)
source task/cuboid/train.sh False

# What happens:
# - Updates mqe/envs/configs/go1_push_mid_config.py
# - Runs openrl_ws/train.py with specified parameters
# - Saves checkpoints to ./log/ during training
# - Moves final results to ./results/<mm-dd-hh>_cuboid/
# - Computes success rates for all checkpoints
```

**Training parameters** (in train.sh):
```bash
num_envs=500           # Number of parallel environments
num_steps=200000000    # Total training steps (200M)
checkpoint_freq=20000  # Save every 20M steps
algo=ppo               # or mappo for multi-agent
```

### Testing Mid-Level Controller

```bash
# 1. Navigate to saved results
cd results/11-07-17_cuboid/

# 2. Edit train.sh to select checkpoint
# Modify $filename variable to desired checkpoint

# 3. Run test
source ./task/train.sh True

# Add --record_video flag to record output
# Videos saved to docs/video/
```

### Training High-Level Controller

```bash
# 1. Ensure mid-level controller is trained
# Add checkpoint path to mqe/envs/configs/go1_push_upper_config.py
vim mqe/envs/configs/go1_push_upper_config.py
# Set: control.command_network_path = "/path/to/checkpoint/module.pt"

# 2. Start training
python ./openrl_ws/train.py \
    --algo ppo \
    --task go1push_upper \
    --train_timesteps 100000000 \
    --num_envs 500 \
    --use_tensorboard \
    --headless
```

### Testing High-Level Controller

```bash
python ./openrl_ws/test.py \
    --algo ppo \
    --task go1push_upper \
    --train_timesteps 100000000 \
    --num_envs 10 \
    --use_tensorboard \
    --checkpoint /path/to/checkpoint \
    --record_video
```

**Pretrained example:** `resources/goals_net` contains a pretrained high-level policy for 1.2m x 1.2m cube.

### Testing Multiple Checkpoints

```bash
# Use the automated checkpoint testing script
./test_checkpoints.sh ./results/10-15-23_cylinder 10 20 30 40 50 60 70 80 90 100

# Tests checkpoints at 10M, 20M, ..., 100M steps
# Results appended to results/<dir>/success_rate.txt
```

---

## Important Files & Their Roles

### Configuration Files

| File | Purpose |
|------|---------|
| `task/cuboid/config.py` | Task-specific settings (object, rewards, randomization) |
| `mqe/envs/configs/go1_push_mid_config.py` | Mid-level controller config (auto-updated from task config) |
| `mqe/envs/configs/go1_push_upper_config.py` | High-level controller config |
| `mqe/envs/go1/go1_config.py` | Base Go1 robot configuration |
| `mqe/envs/base/legged_robot_config.py` | Base legged robot configuration |

### Training Scripts

| File | Purpose |
|------|---------|
| `openrl_ws/train.py` | Main training entry point |
| `openrl_ws/test.py` | Testing and evaluation |
| `openrl_ws/utils.py` | Environment wrappers for OpenRL |
| `task/<object>/train.sh` | Task-specific training/testing wrapper |
| `openrl_ws/update_config.py` | Update config from task to mqe/envs/configs |

### Helper Scripts

| File | Purpose |
|------|---------|
| `test_checkpoints.sh` | Batch test multiple checkpoints |
| `helpers/gpu_monitor.py` | Monitor GPU usage during training |
| `helpers/architecture_check.py` | Verify network architecture |
| `resources/visualize.py` | Visualization utilities |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | Main project documentation |
| `MULTI_EPISODE_RECORDING.md` | Video recording implementation details |
| `VNC_SETUP_STATUS.md` | Remote rendering setup and troubleshooting |
| `TEST_CHECKPOINTS_README.md` | Checkpoint testing guide |

---

## Configuration System

### Hierarchical Configuration

The configuration system uses **class-based inheritance** for modularity:

```python
# Base configuration
class Go1Cfg:
    class env:
        num_agents = 1
    class terrain:
        map_size = [20.0, 20.0]

# Task-specific configuration
class Go1PushMidCfg(Go1Cfg):
    class env(Go1Cfg.env):
        num_agents = 2
    class terrain(Go1Cfg.terrain):
        map_size = [24.0, 24.0]
```

### Key Configuration Sections

#### Environment (`class env`)
```python
num_envs = 500              # Parallel environments
num_agents = 2              # Number of robots
num_npcs = 2                # Object + target area
episode_length_s = 20       # Episode duration
```

#### Asset (`class asset`)
```python
file_npc = "resources/objects/cuboid/SmallBox.urdf"
vertex_list = [[-0.6,-0.6], [0.6,-0.6], [0.6,0.6], [-0.6,0.6]]
npc_collision = True
fix_npc_base_link = False   # Object can move
```

#### Goal (`class goal`)
```python
static_goal_pos = False
random_goal_pos = True
random_goal_distance_from_init = [1.5, 3.0]
random_goal_theta_from_init = [0, 2*pi]
THRESHOLD = 1.0             # Success threshold (meters)
```

#### Domain Randomization (`class domain_rand`)
```python
random_base_init_state = True
init_base_pos_range = dict(r=[1.2, 1.3], theta=[0, 2*pi])
init_base_rpy_range = dict(y=[-0.01, 2*pi])  # Yaw randomization
init_npc_rpy_range = dict(y=[-0.01, 2*pi])   # Box yaw randomization
friction_range = [0.5, 0.6]
```

#### Rewards (`class rewards`)
```python
class scales:
    target_reward_scale = 0.00325
    approach_reward_scale = 0.00075
    collision_punishment_scale = -0.0025
    push_reward_scale = 0.0015
    ocb_reward_scale = 0.004
    reach_target_reward_scale = 10
    exception_punishment_scale = -5
```

---

## Training & Testing

### Training Pipeline

1. **Task Configuration** (`task/cuboid/config.py`)
   - Define object, rewards, randomization

2. **Update Environment Config** (automatic via train.sh)
   - `task/cuboid/config.py` → `mqe/envs/configs/go1_push_mid_config.py`

3. **Training** (`openrl_ws/train.py`)
   - Create environment with Isaac Gym
   - Initialize PPO agent with neural network
   - Train for specified steps (e.g., 200M)
   - Save checkpoints every 20M steps
   - Log to TensorBoard

4. **Checkpoint Testing** (automatic after training)
   - Test each checkpoint in calculator mode
   - Compute success rate over 300 environments
   - Save results to `success_rate.txt`

5. **Results Organization**
   - Move from `./log/` to `./results/<timestamp>_<object>/`
   - Include checkpoints, logs, configs

### Evaluation Metrics

**Success Rate:**
- Percentage of episodes where object reaches goal
- Threshold: 1.0 meter (configurable in `config.py`)

**Additional Metrics:**
- Finished time: Average episode duration for successful episodes
- Collision degree: Frequency of robot-robot collisions
- Collaboration degree: Measure of coordination quality

### Checkpoints Structure

```
results/11-07-17_cuboid/
├── checkpoints/
│   ├── rl_model_10000000_steps/
│   │   └── module.pt           # 10M steps checkpoint
│   ├── rl_model_20000000_steps/
│   │   └── module.pt           # 20M steps
│   ├── ...
│   └── rl_model_110000000_steps/
│       └── module.pt           # 110M steps (final)
├── events.out.tfevents.*       # TensorBoard logs
├── success_rate.txt            # Evaluation results
└── task/
    ├── config.py               # Configuration used
    └── train.sh                # Training script copy
```

---

## Recent Work & Modifications

### 1. Multi-Episode Video Recording (COMPLETED ✅)

**Implementation Date:** November 2024
**Documentation:** `MULTI_EPISODE_RECORDING.md`

**Problem Solved:**
- Original implementation only recorded one episode per video
- Seed control via command-line arguments didn't work through wrapper layers
- No way to record multiple consecutive episodes in one file

**Solution Implemented:**
- **Hardcoded configuration** in `openrl_ws/test.py`:
  ```python
  SEED = 5
  NUM_EPISODES = 3
  ```
- **Runtime monkey-patching** of recording methods
- **Frame accumulation** across episodes
- **Bypassing Gym's private attribute protection**

**Key Modifications:**
- `openrl_ws/test.py`: Lines 20-23 (config), 168-208 (monkey-patching)
- `mqe/envs/base/legged_robot.py`: Lines 1182, 1192, 1214-1224 (optional improvements)

**Usage:**
```bash
# Edit seed and episode count in openrl_ws/test.py
SEED = 42
NUM_EPISODES = 5

# Run test
source ./results/11-07-17_cuboid/task/train.sh True

# Output: docs/video/test_seed42_5eps.mp4
```

**Status:** ✅ Fully working when run locally
**Limitation:** Graphics rendering fails over plain SSH (see VNC setup below)

### 2. VNC Remote Rendering Setup (IN PROGRESS ⏳)

**Implementation Date:** November 2024
**Documentation:** `VNC_SETUP_STATUS.md`

**Problem:**
- Isaac Gym requires GLX (OpenGL) support for rendering
- Plain SSH causes GLFW initialization failure
- Rendering camera captures static/blank frames over SSH

**Approaches Tried:**
- ❌ xvfb-run: Lacks GLX extension
- ❌ Plain SSH with DISPLAY forwarding: GLFW still fails
- ❌ X11 forwarding: Too slow, GLFW still fails

**Current Status:**
- RealVNC Server installed but Service Mode requires physical login
- TigerVNC installed but command hijacked by RealVNC
- No VNC server actually listening on ports yet

**Next Steps:**
1. Try RealVNC Virtual Mode properly (`vncserver-virtuald`)
2. If fails, install TurboVNC + VirtualGL (best for GPU apps)
3. Verify port 5900-5910 is accessible and firewall allows VNC

**Current Workaround:** ✅ Run tests locally on desktop (works perfectly)

### 3. Checkpoint Testing Script (COMPLETED ✅)

**Files Added:**
- `test_checkpoints.sh`
- `test_checkpoints_example.sh`
- `TEST_CHECKPOINTS_README.md`

**Purpose:** Automate testing of multiple checkpoints in sequence

**Usage:**
```bash
./test_checkpoints.sh ./results/10-15-23_cylinder 10 20 30 40 50 60 70 80 90 100
```

**Features:**
- Tests multiple checkpoints automatically
- Continues on failure
- Validates checkpoint existence
- Appends results to `success_rate.txt`
- Summary statistics

---

## Common Operations

### Starting New Training

```bash
# 1. Choose object type
cd task/cuboid/  # or cylinder, Tblock

# 2. Edit configuration
vim config.py
# Adjust rewards, randomization, episode length, etc.

# 3. Start training
source train.sh False

# Monitor progress
tensorboard --logdir ./log/MQE/
```

### Resuming Training from Checkpoint

```bash
# Edit train.sh to add checkpoint path
--checkpoint /path/to/checkpoint/module.pt

# Then run training
source train.sh False
```

### Evaluating Specific Checkpoint

```bash
# 1. Go to results directory
cd results/11-07-17_cuboid/

# 2. Edit task/train.sh
# Set $filename to desired checkpoint:
filename="checkpoints/rl_model_50000000_steps/module.pt"

# 3. Run evaluation
source task/train.sh True
```

### Recording Video

```bash
# 1. Edit openrl_ws/test.py
SEED = 5
NUM_EPISODES = 3

# 2. Ensure record_video flag is set in train.sh
# Add --record_video to the test.py call

# 3. Run test
source ./results/11-07-17_cuboid/task/train.sh True

# 4. Check output
ls docs/video/test_seed5_3eps.mp4
```

### Checking Training Progress

```bash
# View TensorBoard logs
tensorboard --logdir ./results/11-07-17_cuboid/

# Check success rates
cat ./results/11-07-17_cuboid/success_rate.txt

# Monitor GPU usage
python helpers/gpu_monitor.py
```

### Changing Object Type

```bash
# Copy configuration from one object to another
cp task/cuboid/config.py task/my_new_object/config.py
vim task/my_new_object/config.py

# Update asset section:
class asset(Go1Cfg.asset):
    file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/my_new_object/object.urdf"
    vertex_list = [...]  # Define vertices

# Add URDF to resources
cp my_object.urdf resources/objects/my_new_object/

# Train
source task/my_new_object/train.sh False
```

---

## Troubleshooting Notes

### Common Issues

#### 1. ImportError: libpython3.8m.so.1.0
```bash
export LD_LIBRARY_PATH=/home/gvlab/miniconda3/envs/mapush/lib
# Or: sudo apt install libpython3.8
```

#### 2. Numpy version conflict with Isaac Gym
```bash
# Downgrade numpy
pip install numpy==1.19.5

# Or modify Isaac Gym source:
# In isaacgym/python/isaacgym/torch_utils.py
# Change 'np.float' to 'np.float32' in get_axis_params()
```

#### 3. Segmentation fault (core dumped) while rendering on A100/A800
**Solution:** Switch to GeForce graphics cards for rendering

#### 4. OpenRL callback import error
```bash
# Comment out in openrl/utils/callback/callback.py:
# from openrl.runners.common.base_agent import BaseAgent
```

#### 5. Video recording produces blank frames
**Cause:** Running over SSH without proper display access
**Solution:** Run locally or use VNC (see VNC_SETUP_STATUS.md)

#### 6. Checkpoint dimension mismatch
**Cause:** Checkpoint trained with different environment configuration
**Solution:** Ensure task configuration matches training configuration

#### 7. Training too slow
**Possible causes:**
- Too many environments (`num_envs` too high for GPU memory)
- Recording enabled during training
- TensorBoard logging too frequent

**Solutions:**
```bash
# Reduce number of environments
num_envs=200  # Instead of 500

# Disable recording during training
# In config.py:
record_video = False

# Reduce logging frequency
# In train.py, modify callback save_freq
```

### Environment Variables

**Required for Isaac Gym:**
```bash
export LD_LIBRARY_PATH=/home/gvlab/miniconda3/envs/mapush/lib:$LD_LIBRARY_PATH
```

**For rendering:**
```bash
export DISPLAY=:0
xhost +local:  # On desktop, to allow SSH access
```

**For debugging:**
```bash
export CUDA_LAUNCH_BLOCKING=1  # Synchronous CUDA for better error messages
```

---

## Development Context

### Python Environment
```
Conda environment: mapush
Python version: 3.8
```

**Key dependencies:**
- isaacgym (Preview 4)
- openrl
- gymnasium==0.29.1
- torch (CUDA-enabled)
- matplotlib
- tensorboardX
- pettingzoo

### Isaac Gym Details
- **Version:** Preview 4
- **Simulation:** GPU-accelerated physics
- **Rendering:** Requires GLX support
- **Backend:** GLFW (requires display for rendering camera)

### Directory Conventions
- Experiment names: `<mm-dd-hh>_<object>` (e.g., `11-07-17_cuboid`)
- Checkpoint names: `rl_model_<steps>_steps/module.pt`
- Video names: `test_seed<N>_<M>eps.mp4`

### Git Status
- Branch: `main`
- Status: Clean (no uncommitted changes)
- Remote: Likely private repository

---

## Quick Reference

### File Paths Cheat Sheet

```bash
# Training entry points
openrl_ws/train.py
task/cuboid/train.sh

# Testing entry points
openrl_ws/test.py

# Main configurations
task/cuboid/config.py
mqe/envs/configs/go1_push_mid_config.py
mqe/envs/configs/go1_push_upper_config.py

# Core environment
mqe/envs/base/legged_robot.py
mqe/envs/go1/go1.py
mqe/envs/npc/go1_object.py

# Wrappers
mqe/envs/wrappers/go1_push_mid_wrapper.py
mqe/envs/wrappers/go1_push_upper_wrapper.py
openrl_ws/utils.py

# Results
results/<timestamp>_<object>/checkpoints/
results/<timestamp>_<object>/success_rate.txt

# Videos
docs/video/
```

### Command Cheat Sheet

```bash
# Train mid-level
source task/cuboid/train.sh False

# Test mid-level
source results/11-07-17_cuboid/task/train.sh True

# Train high-level
python openrl_ws/train.py --algo ppo --task go1push_upper \
    --train_timesteps 100000000 --num_envs 500 --use_tensorboard --headless

# Test multiple checkpoints
./test_checkpoints.sh results/11-07-17_cuboid 10 20 30 40 50 60 70 80 90 100

# TensorBoard
tensorboard --logdir ./results/11-07-17_cuboid/

# Record video (edit test.py first)
source results/11-07-17_cuboid/task/train.sh True
```

### Configuration Quick Edits

**Increase episode length:**
```python
# In task/cuboid/config.py
class env(Go1Cfg.env):
    episode_length_s = 30  # Was 20
```

**Change reward weights:**
```python
# In task/cuboid/config.py
class rewards(Go1Cfg.rewards):
    class scales:
        push_reward_scale = 0.003  # Increase push reward
        collision_punishment_scale = -0.005  # Increase collision penalty
```

**Adjust randomization:**
```python
# In task/cuboid/config.py
class domain_rand(Go1Cfg.domain_rand):
    init_base_pos_range = dict(r=[1.5, 2.0], theta=[0, 2*pi])  # Further away
    friction_range = [0.3, 0.8]  # Wider friction range
```

---

## Summary for Future Sessions

**What This Repository Does:**
Multi-agent reinforcement learning for collaborative quadrupedal pushing. Two Unitree Go1 robots learn to push objects (cuboids, cylinders, T-blocks) to target locations using hierarchical control (low-level locomotion, mid-level coordination, high-level planning).

**Main Workflow:**
1. Configure task in `task/<object>/config.py`
2. Train mid-level controller with `source task/<object>/train.sh False`
3. Test checkpoints with `source results/<dir>/task/train.sh True`
4. (Optional) Train high-level controller with pretrained mid-level

**Key Recent Work:**
- ✅ Multi-episode video recording implemented (hardcoded seed + monkey-patching)
- ⏳ VNC remote rendering setup in progress (needed for SSH video recording)
- ✅ Automated checkpoint testing scripts

**Important Notes:**
- Must run locally or via VNC for video recording (SSH doesn't work for rendering)
- Checkpoints saved every 20M steps during training
- Configuration uses class-based inheritance (Go1Cfg → Go1PushMidCfg → task config)
- Results automatically moved to `./results/<timestamp>_<object>/` after training

**Common Commands:**
- Train: `source task/cuboid/train.sh False`
- Test: `source results/11-07-17_cuboid/task/train.sh True`
- Multi-checkpoint test: `./test_checkpoints.sh results/<dir> 10 20 30 ... 100`

---

**Last Updated:** 2025-11-11
**For:** Future Claude Code sessions
**Purpose:** Quick onboarding and reference for repository structure and workflows
