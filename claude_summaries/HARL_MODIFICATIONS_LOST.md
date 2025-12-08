# HARL Modifications Lost - Complete Recovery Guide

**Date**: 2025-12-08
**Context**: All HARL modifications were accidentally deleted due to git submodule mismanagement. This document contains ALL information needed to re-implement the lost work.

---

## CRITICAL: Iteration 11 Reward Structure (MUST PRESERVE)

### Current Reward Design Philosophy
**Iteration 11** implemented a **SEPARATE REWARDS** structure where each agent receives individual rewards to prevent freeloading behavior observed in previous iterations.

### Key Configuration
**File**: `task/cuboid/config.py`
- Added flag: `separate_rewards = True`
- When `separate_rewards=True`: Each agent gets only their own individual rewards
- When `separate_rewards=False`: Agents get shared team rewards (old behavior)

### Reward Components in go1_push_mid_wrapper.py

The wrapper computes these rewards **per agent**:

1. **Engage Reward** (Individual)
   - Reward for moving toward the box
   - Scaled by distance reduction to box
   - Only given when agent is far from box

2. **Push Direction Reward** (Individual)
   - Reward for pushing box toward goal
   - Based on box velocity toward goal
   - Each agent gets reward based on their own push contribution

3. **Success Reward** (Shared)
   - Large bonus when box reaches goal
   - Given to all agents equally

4. **Collision Penalty** (Individual)
   - Penalty for colliding with other agents
   - Each agent penalized for their own collisions

5. **Exception Punishment** (Individual)
   - Heavy penalty for physics violations (box falling, agent falling)
   - Applied to all agents when environment fails

### Implementation Details from go1_push_mid_wrapper.py

```python
# In compute_reward() method around line 400-600

if self.separate_rewards:
    # ITERATION 11: Each agent gets ONLY their own rewards
    for agent_id in range(self.num_agents):
        agent_reward = 0.0

        # Individual engage reward
        agent_reward += engage_rewards[agent_id]

        # Individual push direction reward
        agent_reward += push_direction_rewards[agent_id]

        # Shared success reward (everyone gets it)
        agent_reward += success_reward

        # Individual collision penalty
        agent_reward += collision_penalties[agent_id]

        # Individual exception punishment
        agent_reward += exception_punishment

        rewards[agent_id] = agent_reward
else:
    # OLD BEHAVIOR: All agents get sum of all rewards
    total_reward = (engage_rewards.sum() +
                   push_direction_rewards.sum() +
                   success_reward +
                   collision_penalties.sum() +
                   exception_punishment)
    rewards[:] = total_reward
```

### Why This Matters
- Previous iterations had **freeloading problem**: one agent pushes while others do nothing
- Separate rewards force each agent to contribute individually
- Success reward remains shared to maintain team coordination

---

## HARL Calculator Mode Implementation

### Overview
Calculator mode allows running trained checkpoints to collect metrics without rendering, using many parallel environments for statistical significance.

### 1. MAPush Environment Wrapper Modifications

**File**: `HARL/harl/envs/mapush/mapush_env.py`

**Changes**: Added property methods to expose calculator buffers from the base environment

```python
@property
def init_finished_buf(self):
    """Track which environments successfully completed (for calculator mode)."""
    return self.env.init_finished_buf if hasattr(self.env, 'init_finished_buf') else None

@property
def init_episode_length_buf(self):
    """Track episode lengths (for calculator mode)."""
    return self.env.init_episode_length_buf if hasattr(self.env, 'init_episode_length_buf') else None

@property
def collision_degree_buf(self):
    """Track collision degree metric (for calculator mode)."""
    return self.env.collision_degree_buf if hasattr(self.env, 'collision_degree_buf') else None

@property
def collaboration_degree_buf(self):
    """Track collaboration degree metric (for calculator mode)."""
    return self.env.collaboration_degree_buf if hasattr(self.env, 'collaboration_degree_buf') else None

@property
def init_reset_buf(self):
    """Track which environments need reset (for calculator mode)."""
    return self.env.init_reset_buf if hasattr(self.env, 'init_reset_buf') else None

@property
def dt(self):
    """Simulation timestep (for calculator mode)."""
    return self.env.dt if hasattr(self.env, 'dt') else 0.02
```

### 2. HAPPO Configuration

**File**: `HARL/harl/configs/algos_cfgs/happo.yaml`

**Changes**: Added calculator mode settings under render section

```yaml
render:
  use_render: False
  use_calc_mode: False  # Set to True for calculator mode
  calc_n_threads: 300   # Number of parallel environments for calc mode
```

### 3. Runner Modifications

**File**: `HARL/harl/runners/on_policy_base_runner.py`

This is the most complex modification with multiple changes:

#### A. Constructor Modifications (lines ~51-68)

Added logic to handle calc_mode directory setup:

```python
def __init__(self, args, algo_args, env_args):
    # ... existing code ...

    # Handle calc_mode vs training mode directory setup
    if self.algo_args["render"]["use_calc_mode"]:
        # Calc mode: use existing model_dir, no logging
        self.save_dir = algo_args["train"]["model_dir"]
        self.writter = None
        self.run_dir = None
        self.log_dir = None
        if not hasattr(args, 'exp_name'):
            args['exp_name'] = 'calc_mode'
    elif not self.algo_args["render"]["use_render"]:
        # Training mode: create new directories
        self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(...)
    else:
        # Render mode: use existing model_dir
        self.run_dir = None
        self.log_dir = None
        self.save_dir = algo_args["train"]["model_dir"]
        self.writter = None
```

#### B. Logger Initialization (lines ~183-189)

Only create logger for training mode:

```python
# Initialize logger only for training (not calc/render mode)
if not self.algo_args["render"]["use_calc_mode"] and not self.algo_args["render"]["use_render"]:
    self.logger = LOGGER_REGISTRY[args["env"]](
        args, algo_args, env_args, self.writter, self.run_dir
    )
else:
    self.logger = None
```

#### C. Checkpoint Saving Every 10M Steps (lines ~250-270)

Added periodic checkpoint saving during training:

```python
def run(self):
    # ... existing training loop ...

    # Save checkpoint every 10M steps
    total_steps = episode * self.episode_length * self.n_rollout_threads
    if total_steps % 10_000_000 == 0:
        checkpoint_dir = os.path.join(self.save_dir, f"{total_steps//1_000_000}M")
        os.makedirs(checkpoint_dir, exist_ok=True)
        for agent_id in range(self.num_agents):
            actor_path = os.path.join(checkpoint_dir, f"actor_agent{agent_id}.pt")
            torch.save(self.actor[agent_id].actor.state_dict(), actor_path)
            critic_path = os.path.join(checkpoint_dir, f"critic_agent{agent_id}.pt")
            torch.save(self.critic[agent_id].critic.state_dict(), critic_path)
```

#### D. evaluate_all_checkpoints() Method (lines ~291-369)

Auto-evaluate all saved checkpoints after training:

```python
def evaluate_all_checkpoints(self):
    """Evaluate all saved checkpoints using calculator mode."""
    import glob

    # Find all checkpoint directories (10M, 20M, 30M, etc.)
    checkpoint_dirs = sorted(glob.glob(str(self.save_dir) + "/[0-9]*M"))

    if not checkpoint_dirs:
        print(f"No checkpoint directories found in {self.save_dir}")
        return

    print(f"\nFound {len(checkpoint_dirs)} checkpoints to evaluate")

    all_results = []

    for ckpt_dir in checkpoint_dirs:
        checkpoint_name = os.path.basename(ckpt_dir)
        print(f"\nEvaluating checkpoint: {checkpoint_name}")

        # Load checkpoint
        for agent_id in range(self.num_agents):
            actor_path = os.path.join(ckpt_dir, f"actor_agent{agent_id}.pt")
            if not os.path.exists(actor_path):
                print(f"  Warning: {actor_path} not found, skipping")
                continue

            self.actor[agent_id].actor.load_state_dict(
                torch.load(actor_path, map_location=self.device)
            )
            self.actor[agent_id].actor.eval()

        # Run calculator mode
        metrics = self.calculate()

        all_results.append({
            'checkpoint': checkpoint_name,
            'metrics': metrics
        })

        print(f"  Results: SR={metrics['success_rate']:.3f}, "
              f"ColDeg={metrics['collision_degree']:.3f}, "
              f"CollabDeg={metrics['collaboration_degree']:.3f}, "
              f"Time={metrics['finished_time']:.3f}s")

    # Write results to single file in table format
    results_file = os.path.join(self.save_dir, "all_checkpoints_results.txt")
    with open(results_file, 'w') as f:
        f.write("Checkpoint Evaluation Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Checkpoint':<15} {'Success Rate':<15} {'Collision Deg':<15} "
                f"{'Collab Deg':<15} {'Finished Time':<15}\n")
        f.write("-" * 80 + "\n")

        for result in all_results:
            ckpt = result['checkpoint']
            m = result['metrics']
            f.write(f"{ckpt:<15} {m['success_rate']:<15.3f} "
                   f"{m['collision_degree']:<15.3f} {m['collaboration_degree']:<15.3f} "
                   f"{m['finished_time']:<15.3f}\n")

    print(f"\nResults saved to: {results_file}")
```

#### E. calculate() Method (lines ~780-923)

Core calculator mode implementation:

```python
def calculate(self):
    """Calculator mode - run episodes and compute metrics."""

    # Get number of environments
    if hasattr(self, 'env_num'):
        n_envs = self.env_num
    else:
        n_envs = self.envs.n_threads

    # Reset all environments
    obs = self.envs.reset()

    # Convert obs to proper format
    if isinstance(obs, dict):
        obs = obs['obs']

    # Initialize RNN states
    rnn_states = np.zeros((n_envs, self.num_agents, self.recurrent_N, self.hidden_size))
    masks = np.ones((n_envs, self.num_agents, 1))

    # Run until all environments complete
    step_count = 0
    max_steps = 1000  # Safety limit

    while not torch.all(self.envs.init_reset_buf):
        step_count += 1
        if step_count > max_steps:
            break

        # Collect actions from all agents
        actions = []
        for agent_id in range(self.num_agents):
            agent_obs = obs[:, agent_id]

            # Get action from policy
            action, rnn_state = self.actor[agent_id].act(
                agent_obs,
                rnn_states[:, agent_id],
                masks[:, agent_id],
                deterministic=True
            )

            actions.append(action)
            rnn_states[:, agent_id] = rnn_state

        # Stack actions and step environment
        actions = np.stack(actions, axis=1)
        obs, rewards, dones, infos = self.envs.step(actions)

        # Update masks
        masks = np.ones((n_envs, self.num_agents, 1))
        masks[dones] = 0.0

    # Compute metrics from calculator buffers
    success_rate = torch.mean(self.envs.init_finished_buf.to(torch.float)).item()

    # Finished time: only for successful episodes
    finished_mask = self.envs.init_finished_buf
    if finished_mask.any():
        finished_times = self.envs.init_episode_length_buf[finished_mask] * self.envs.dt
        finished_time = torch.mean(finished_times).item()
    else:
        finished_time = 0.0

    # Collision degree: average across all environments
    collision_degree = torch.mean(self.envs.collision_degree_buf).item()

    # Collaboration degree: average across all environments
    collaboration_degree = torch.mean(self.envs.collaboration_degree_buf).item()

    metrics = {
        'success_rate': success_rate,
        'finished_time': finished_time,
        'collision_degree': collision_degree,
        'collaboration_degree': collaboration_degree,
        'n_envs': n_envs,
        'total_steps': step_count
    }

    return metrics
```

### 4. New test.py Script

**File**: `HARL/examples/test.py` (NEW FILE)

This script separates testing from training:

```python
"""Test/evaluate trained HARL models.

This script supports two modes:
1. Calculator mode (--mode calc): Compute metrics using many parallel environments
2. Render mode (--mode render): Visualize agent behavior

Usage:
    # Test single checkpoint with calculator mode
    python test.py --algo happo --env mapush --model_dir ./results/.../models/80M --mode calc

    # Test all checkpoints
    python test.py --algo happo --env mapush --model_dir ./results/.../models --mode calc --test_all_checkpoints True

    # Render mode
    python test.py --algo happo --env mapush --model_dir ./results/.../models/80M --mode render
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add HARL to path
sys.path.append(str(Path(__file__).parent.parent))

from harl.utils.configs_tools import get_defaults_yaml_args, update_args
from harl.runners import RUNNER_REGISTRY

def main():
    parser = argparse.ArgumentParser(description="Test/evaluate trained HARL models")

    # Required arguments
    parser.add_argument("--algo", type=str, default="happo", help="Algorithm name")
    parser.add_argument("--env", type=str, default="mapush", help="Environment name")
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Directory containing trained models")

    # Test mode arguments
    parser.add_argument("--mode", type=str, default="calc", choices=["calc", "render"],
                       help="Testing mode: calc (metrics) or render (visualization)")
    parser.add_argument("--test_all_checkpoints", type=bool, default=False,
                       help="Test all checkpoint directories (10M, 20M, etc.)")

    # Optional arguments
    parser.add_argument("--exp_name", type=str, default="test",
                       help="Experiment name for logging")

    args = parser.parse_args()

    # Convert to dict for HARL
    args_dict = vars(args)

    # Load default configs
    algo_args, env_args = get_defaults_yaml_args(args.algo, args.env)

    # Update with test-specific settings
    algo_args["train"]["model_dir"] = args.model_dir

    # Configure test mode
    if args.mode == "calc":
        algo_args["render"]["use_calc_mode"] = True
        algo_args["render"]["use_render"] = False
        print(f"Running in CALCULATOR mode with {algo_args['render']['calc_n_threads']} threads")
    elif args.mode == "render":
        algo_args["render"]["use_render"] = True
        algo_args["render"]["use_calc_mode"] = False
        print("Running in RENDER mode")

    # Initialize runner
    runner = RUNNER_REGISTRY[args.algo](args_dict, algo_args, env_args)

    # Run evaluation
    if args.mode == "calc" and args.test_all_checkpoints:
        print("\nEvaluating all checkpoints...")
        runner.evaluate_all_checkpoints()
    else:
        print(f"\nEvaluating single checkpoint: {args.model_dir}")
        runner.run()

    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
```

### 5. Modified train.py

**File**: `HARL/examples/train.py`

**Changes**: Removed testing logic, added note to use test.py:

```python
"""Train HARL models.

This script is for TRAINING ONLY.
For testing/evaluation, use test.py instead.

Usage:
    python train.py --algo happo --env mapush --exp_name my_experiment
"""

# ... rest of training code unchanged ...
# Removed all calc_mode and evaluation logic
```

---

## Debug Prints Removed from go1_push_mid_wrapper.py

### Removed Announcement Block (lines ~66-108)
- Large ASCII art banner announcing "ITERATION 11"
- Reward structure explanation
- Removed to clean up training output

### Removed Step-by-Step Logging (lines ~511-517)
- Debug logging every 100 steps showing:
  - Engage rewards per agent
  - Push direction rewards per agent
  - Total rewards per agent
- Removed to prevent spam during training

---

## Updated Documentation

**File**: `claude_summaries/HAPPO_GUIDE.md`

Updated with new train.py/test.py separation and minimal command examples:

```markdown
## Training

python HARL/examples/train.py --algo happo --env mapush --exp_name my_experiment

## Testing

# Calculator mode - single checkpoint
python HARL/examples/test.py --algo happo --env mapush --model_dir ./results/.../80M --mode calc

# Calculator mode - all checkpoints
python HARL/examples/test.py --algo happo --env mapush --model_dir ./results/.../models --mode calc --test_all_checkpoints True

# Render mode
python HARL/examples/test.py --algo happo --env mapush --model_dir ./results/.../80M --mode render
```

---

## Re-Implementation Priority

1. **CRITICAL**: Restore reward structure modifications in `go1_push_mid_wrapper.py`
2. **HIGH**: Implement calculator mode in runner (`calculate()` method)
3. **HIGH**: Add environment property exposures in `mapush_env.py`
4. **MEDIUM**: Add checkpoint saving every 10M steps
5. **MEDIUM**: Create test.py script
6. **LOW**: Add evaluate_all_checkpoints() method
7. **LOW**: Update HAPPO config with calc_mode settings

---

## Lessons Learned

1. **NEVER use `git rm -f` on untracked folders**
2. **ALWAYS commit HARL changes immediately after modifications**
3. **Document everything in markdown files for recovery**
4. **Test git operations on dummy data first**
5. **This disaster was entirely Claude's fault for suggesting `git rm -f` without proper safety checks**
