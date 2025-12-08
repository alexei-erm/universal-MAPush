# Calculator Mode Implementation Progress

**Date**: 2025-12-08
**Status**: PARTIALLY COMPLETE
**Branch**: happo-reward-design

---

## What Was Completed

### 1. ✅ test.py Script Created

**Location**: `HARL/examples/test.py`

**Features**:
- Calculator mode (--mode calc): Compute metrics using many parallel environments
- Render mode (--mode render): Visualize agent behavior
- Test all checkpoints flag (--test_all_checkpoints True)
- Properly loads models from --model_dir

**Usage**:
```bash
# Test single checkpoint with calculator mode
python HARL/examples/test.py --algo happo --env mapush --model_dir ./results/.../models/80M --mode calc

# Test all checkpoints
python HARL/examples/test.py --algo happo --env mapush --model_dir ./results/.../models --mode calc --test_all_checkpoints True

# Render mode
python HARL/examples/test.py --algo happo --env mapush --model_dir ./results/.../models/80M --mode render
```

### 2. ✅ calc_mode Configuration Added

**Location**: `HARL/harl/configs/algos_cfgs/happo.yaml`

**Added lines**:
```yaml
render:
  use_render: False
  render_episodes: 10
  use_calc_mode: False  # NEW - enables calculator mode
  calc_n_threads: 300   # NEW - number of parallel envs for metrics
```

### 3. ✅ Constructor Modified (PARTIALLY)

**Location**: `HARL/harl/runners/on_policy_base_runner.py` (lines 49-66)

**What was changed**:
```python
# Only init directories for training mode (not calc or render)
use_calc_mode = self.algo_args["render"].get("use_calc_mode", False)
if not self.algo_args["render"]["use_render"] and not use_calc_mode:  # train mode only
    self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
        args["env"],
        env_args,
        args["algo"],
        args["exp_name"],
        algo_args["seed"]["seed"],
        logger_path=algo_args["logger"]["log_dir"],
    )
    save_config(args, algo_args, env_args, self.run_dir)
else:
    # In calc or render mode, don't create directories
    self.run_dir = None
    self.log_dir = None
    self.save_dir = None
    self.writter = None
```

**Status**: This prevents directory creation in calc mode, which is correct.

---

## What Still Needs To Be Done

### 1. ❌ Finish Constructor Modifications

**Location**: `HARL/harl/runners/on_policy_base_runner.py` (line ~173)

**What to do**: Make logger initialization conditional

**Current code** (line 173-175):
```python
self.logger = LOGGER_REGISTRY[args["env"]](
    args, algo_args, env_args, self.num_agents, self.writter, self.run_dir
)
```

**Need to change to**:
```python
# Only create logger for training mode (not calc/render)
use_calc_mode = self.algo_args["render"].get("use_calc_mode", False)
if not use_calc_mode and not self.algo_args["render"]["use_render"]:
    self.logger = LOGGER_REGISTRY[args["env"]](
        args, algo_args, env_args, self.num_agents, self.writter, self.run_dir
    )
else:
    self.logger = None
```

**Why**: Prevents logger errors when writter/run_dir are None in calc mode.

### 2. ❌ Implement calculate() Method

**Location**: Add to `HARL/harl/runners/on_policy_base_runner.py` (after line 775, end of file)

**Full implementation** (from `HARL_MODIFICATIONS_LOST.md` lines 780-923):

```python
@torch.no_grad()
def calculate(self):
    """Run calculator mode to compute metrics using many parallel environments."""
    print("Starting calculator mode...")

    # Get calculator-specific number of threads
    calc_n_threads = self.algo_args["render"].get("calc_n_threads", 300)
    print(f"Using {calc_n_threads} parallel environments for calculator mode")

    # Reset environment
    obs, share_obs, available_actions = self.envs.reset()

    # Initialize RNN states and masks
    rnn_states = np.zeros(
        (calc_n_threads, self.num_agents, self.recurrent_n, self.rnn_hidden_size),
        dtype=np.float32,
    )
    masks = np.ones((calc_n_threads, self.num_agents, 1), dtype=np.float32)

    # Track which environments have finished
    finished_envs = np.zeros(calc_n_threads, dtype=bool)

    # Access calculator buffers from environment
    init_finished_buf = self.envs.init_finished_buf
    finished_time_buf = self.envs.finished_time_buf
    collision_degree_buf = self.envs.collision_degree_buf
    collaboration_degree_buf = self.envs.collaboration_degree_buf

    if init_finished_buf is None:
        print("WARNING: Environment does not expose calculator buffers!")
        print("Calculator mode requires init_finished_buf, finished_time_buf, etc.")
        return

    step = 0
    max_steps = self.algo_args["train"]["episode_length"]

    print(f"Running for max {max_steps} steps...")

    while not finished_envs.all() and step < max_steps:
        # Collect actions from all agents
        actions_collector = []
        for agent_id in range(self.num_agents):
            actions, temp_rnn_state = self.actor[agent_id].act(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                masks[:, agent_id],
                available_actions[:, agent_id] if available_actions[0] is not None else None,
                deterministic=True,
            )
            rnn_states[:, agent_id] = _t2n(temp_rnn_state)
            actions_collector.append(_t2n(actions))

        actions = np.array(actions_collector).transpose(1, 0, 2)

        # Step environment
        obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

        # Update finished environments
        finished_envs = finished_envs | init_finished_buf.cpu().numpy()

        # Reset masks for done agents
        masks = np.ones((calc_n_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        step += 1

        if step % 50 == 0:
            num_finished = finished_envs.sum()
            print(f"Step {step}/{max_steps}: {num_finished}/{calc_n_threads} environments finished")

    # Compute final metrics
    print("\nComputing metrics...")

    # Success rate: percentage of environments that finished successfully
    success_rate = init_finished_buf.float().mean().item()

    # Average time to finish (only for successful episodes)
    successful_mask = init_finished_buf
    if successful_mask.any():
        avg_finish_time = finished_time_buf[successful_mask].float().mean().item()
    else:
        avg_finish_time = 0.0

    # Average collision degree
    avg_collision = collision_degree_buf.float().mean().item()

    # Average collaboration degree
    avg_collaboration = collaboration_degree_buf.float().mean().item()

    print("\n" + "="*60)
    print("CALCULATOR MODE RESULTS")
    print("="*60)
    print(f"Success Rate:          {success_rate*100:.2f}%")
    print(f"Avg Finish Time:       {avg_finish_time:.2f} steps")
    print(f"Avg Collision Degree:  {avg_collision:.4f}")
    print(f"Avg Collaboration:     {avg_collaboration:.4f}")
    print("="*60)

    return {
        'success_rate': success_rate,
        'avg_finish_time': avg_finish_time,
        'avg_collision': avg_collision,
        'avg_collaboration': avg_collaboration,
    }
```

**Why needed**: This is what actually runs the calculator mode evaluation.

### 3. ❌ Implement evaluate_all_checkpoints() Method

**Location**: Add to `HARL/harl/runners/on_policy_base_runner.py` (after calculate() method)

**Full implementation** (from `HARL_MODIFICATIONS_LOST.md` lines 291-369):

```python
def evaluate_all_checkpoints(self):
    """Evaluate all checkpoint directories (10M, 20M, 30M, etc.) and save results."""
    import os
    from pathlib import Path

    # Get the base models directory
    model_dir = Path(self.algo_args["train"]["model_dir"])

    # If model_dir points to a specific checkpoint (e.g., "80M"), go up one level
    if model_dir.name.endswith('M'):
        model_dir = model_dir.parent

    # Find all checkpoint directories
    checkpoint_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.endswith('M')],
                            key=lambda x: int(x.name[:-1]))  # Sort by number (10M, 20M, etc.)

    if not checkpoint_dirs:
        print(f"No checkpoint directories found in {model_dir}")
        return

    print(f"\nFound {len(checkpoint_dirs)} checkpoints to evaluate:")
    for d in checkpoint_dirs:
        print(f"  - {d.name}")

    results = []

    for checkpoint_dir in checkpoint_dirs:
        print(f"\n{'='*60}")
        print(f"Evaluating checkpoint: {checkpoint_dir.name}")
        print(f"{'='*60}")

        # Update model_dir to point to this checkpoint
        self.algo_args["train"]["model_dir"] = str(checkpoint_dir)

        # Load the checkpoint
        self.restore()

        # Run calculator mode
        metrics = self.calculate()

        if metrics is not None:
            results.append({
                'checkpoint': checkpoint_dir.name,
                **metrics
            })

    # Save results to file
    results_file = model_dir / "all_checkpoints_results.txt"
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EVALUATION RESULTS FOR ALL CHECKPOINTS\n")
        f.write("="*80 + "\n\n")

        # Write table header
        f.write(f"{'Checkpoint':<12} {'Success Rate':<15} {'Avg Finish Time':<18} {'Avg Collision':<16} {'Avg Collaboration':<18}\n")
        f.write("-"*80 + "\n")

        # Write results
        for result in results:
            f.write(f"{result['checkpoint']:<12} "
                   f"{result['success_rate']*100:>6.2f}%         "
                   f"{result['avg_finish_time']:>8.2f} steps     "
                   f"{result['avg_collision']:>10.4f}      "
                   f"{result['avg_collaboration']:>10.4f}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"\n\nResults saved to: {results_file}")
    print("\nAll checkpoints evaluated!")
```

**Why needed**: This allows testing all saved checkpoints at once.

### 4. ❌ Add Checkpoint Auto-Saving

**Location**: Modify `HARL/harl/runners/on_policy_base_runner.py` run() method (around line 265)

**Current code** (lines 260-266):
```python
# eval
if episode % self.algo_args["train"]["eval_interval"] == 0:
    if self.algo_args["eval"]["use_eval"]:
        self.prep_rollout()
        self.eval()
    self.save()
```

**Change to**:
```python
# eval
if episode % self.algo_args["train"]["eval_interval"] == 0:
    if self.algo_args["eval"]["use_eval"]:
        self.prep_rollout()
        self.eval()
    self.save()

# Checkpoint auto-saving every 10M steps
total_steps = episode * self.algo_args["train"]["episode_length"] * self.algo_args["train"]["n_rollout_threads"]
checkpoint_interval = 10_000_000  # 10M steps
if total_steps % checkpoint_interval == 0 or (episode > 1 and (total_steps - self.algo_args["train"]["episode_length"] * self.algo_args["train"]["n_rollout_threads"]) // checkpoint_interval < total_steps // checkpoint_interval):
    checkpoint_name = f"{total_steps // 1_000_000}M"
    checkpoint_dir = os.path.join(os.path.dirname(self.save_dir), checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save models
    for agent_id in range(self.num_agents):
        policy_actor = self.actor[agent_id].actor
        torch.save(
            policy_actor.state_dict(),
            os.path.join(checkpoint_dir, f"actor_agent{agent_id}.pt"),
        )
    policy_critic = self.critic.critic
    torch.save(
        policy_critic.state_dict(),
        os.path.join(checkpoint_dir, "critic_agent.pt")
    )
    if self.value_normalizer is not None:
        torch.save(
            self.value_normalizer.state_dict(),
            os.path.join(checkpoint_dir, "value_normalizer.pt"),
        )

    print(f"\n{'='*60}")
    print(f"Checkpoint saved: {checkpoint_name} ({total_steps:,} steps)")
    print(f"Location: {checkpoint_dir}")
    print(f"{'='*60}\n")
```

**Why needed**: Automatically saves checkpoints every 10M steps for later evaluation.

### 5. ❌ Update close() Method

**Location**: `HARL/harl/runners/on_policy_base_runner.py` (lines 765-776)

**Current code**:
```python
def close(self):
    """Close environment, writter, and logger."""
    if self.algo_args["render"]["use_render"]:
        self.envs.close()
    else:
        self.envs.close()
        if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
            self.eval_envs.close()
        self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
        self.writter.close()
        self.logger.close()
```

**Change to**:
```python
def close(self):
    """Close environment, writter, and logger."""
    if self.algo_args["render"]["use_render"]:
        self.envs.close()
    else:
        self.envs.close()
        if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
            self.eval_envs.close()
        # Only close writter/logger if they exist (not in calc mode)
        if self.writter is not None:
            self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
            self.writter.close()
        if self.logger is not None:
            self.logger.close()
```

**Why needed**: Prevents errors when closing None objects in calc mode.

### 6. ❌ Update run() Method for Calculator Mode

**Location**: `HARL/harl/runners/on_policy_base_runner.py` (lines 179-184)

**Current code**:
```python
def run(self):
    """Run the training (or rendering) pipeline."""
    if self.algo_args["render"]["use_render"] is True:
        self.render()
        return
    print("start running")
    self.warmup()
```

**Change to**:
```python
def run(self):
    """Run the training (or rendering) pipeline."""
    if self.algo_args["render"]["use_render"] is True:
        self.render()
        return

    # Handle calculator mode
    use_calc_mode = self.algo_args["render"].get("use_calc_mode", False)
    if use_calc_mode:
        self.calculate()
        return

    print("start running")
    self.warmup()
```

**Why needed**: Makes run() call calculate() when in calc mode.

---

## Summary of Changes Needed

**File**: `HARL/harl/runners/on_policy_base_runner.py`

1. **Line 173**: Make logger initialization conditional (STARTED BUT NOT COMPLETED)
2. **Line 179**: Add calc_mode check to run() method
3. **Line 265**: Add checkpoint auto-saving after eval block
4. **Line 773**: Update close() to handle None writter/logger
5. **Line 776** (end of file): Add calculate() method (~150 lines)
6. **After calculate()**: Add evaluate_all_checkpoints() method (~80 lines)

**Total lines to add**: ~250 lines
**Total modifications**: 4 small edits to existing code

---

## Quick Testing After Implementation

```bash
# Test calculator mode on single checkpoint
python HARL/examples/test.py --algo happo --env mapush \
  --model_dir ./results/mapush/cuboid_go1push_mid/happo/separate_rewards_test11/seed-00001-2025-12-08-10-07-02/models/80M \
  --mode calc

# Test all checkpoints
python HARL/examples/test.py --algo happo --env mapush \
  --model_dir ./results/mapush/cuboid_go1push_mid/happo/separate_rewards_test11/seed-00001-2025-12-08-10-07-02/models \
  --mode calc --test_all_checkpoints True
```

---

## Files Reference

- **Full implementation code**: `HARL_MODIFICATIONS_LOST.md`
- **Reward structure (PRESERVE)**: `ITERATION_11_REWARD_STRUCTURE.md`
- **Integration status**: `HAPPO_INTEGRATION_COMPLETE.md`
- **This document**: `CALCULATOR_MODE_PROGRESS.md`

---

## CRITICAL NOTES FOR NEXT SESSION

1. **Separate rewards MUST be preserved**: `use_per_agent_rewards=True` in `task/cuboid/config.py`
2. **share_param: False** in `happo.yaml` - DO NOT CHANGE
3. All implementation code is in `HARL_MODIFICATIONS_LOST.md` - use it as reference
4. Constructor modification was STARTED but NOT COMPLETED - line 173 needs conditional logger
5. Total work remaining: ~6 edits to one file, most of it is adding new methods at end
