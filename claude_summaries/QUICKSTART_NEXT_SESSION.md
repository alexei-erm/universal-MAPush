# Quickstart for Next Claude Session

**Date**: 2025-12-08
**Current Branch**: happo-reward-design
**Task**: Complete calculator mode implementation

---

## Context Summary (30 seconds)

1. **HARL was deleted** in disaster - all modifications lost
2. **Recovery complete**: HARL cloned, MAPush integrated, separate rewards verified
3. **Current task**: Implement calculator mode (test.py + runner methods)
4. **Status**: 50% done - test.py exists, runner needs 6 modifications

---

## What You Need to Complete

**ONE FILE TO MODIFY**: `HARL/harl/runners/on_policy_base_runner.py`

**6 EDITS NEEDED**:

### Edit 1: Make logger conditional (Line 173)
```python
# FIND THIS (line 168-175):
if self.algo_args["train"]["use_valuenorm"] is True:
    self.value_normalizer = ValueNorm(1, device=self.device)
else:
    self.value_normalizer = None

self.logger = LOGGER_REGISTRY[args["env"]](
    args, algo_args, env_args, self.num_agents, self.writter, self.run_dir
)

# REPLACE WITH:
if self.algo_args["train"]["use_valuenorm"] is True:
    self.value_normalizer = ValueNorm(1, device=self.device)
else:
    self.value_normalizer = None

# Only create logger for training mode (not calc/render)
use_calc_mode = self.algo_args["render"].get("use_calc_mode", False)
if not use_calc_mode and not self.algo_args["render"]["use_render"]:
    self.logger = LOGGER_REGISTRY[args["env"]](
        args, algo_args, env_args, self.num_agents, self.writter, self.run_dir
    )
else:
    self.logger = None
```

### Edit 2: Add calc mode to run() (Line 179-184)
```python
# FIND THIS (line 179-184):
def run(self):
    """Run the training (or rendering) pipeline."""
    if self.algo_args["render"]["use_render"] is True:
        self.render()
        return
    print("start running")
    self.warmup()

# REPLACE WITH:
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

### Edit 3: Add checkpoint auto-saving (After line 265)
```python
# FIND THIS (line 260-267):
# eval
if episode % self.algo_args["train"]["eval_interval"] == 0:
    if self.algo_args["eval"]["use_eval"]:
        self.prep_rollout()
        self.eval()
    self.save()

self.after_update()

# INSERT BETWEEN self.save() AND self.after_update():
# Checkpoint auto-saving every 10M steps
import os
total_steps = episode * self.algo_args["train"]["episode_length"] * self.algo_args["train"]["n_rollout_threads"]
checkpoint_interval = 10_000_000  # 10M steps
# Check if we just crossed a 10M boundary
if episode > 1:
    prev_total_steps = (episode - 1) * self.algo_args["train"]["episode_length"] * self.algo_args["train"]["n_rollout_threads"]
    if total_steps // checkpoint_interval > prev_total_steps // checkpoint_interval:
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

self.after_update()
```

### Edit 4: Update close() to handle None (Line 765-776)
```python
# FIND THIS (line 765-776):
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

# REPLACE WITH:
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

### Edit 5: Add calculate() method (After line 776, end of file)

**FULL CODE** - Copy from `HARL_MODIFICATIONS_LOST.md` lines 780-923 or use this:

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

### Edit 6: Add evaluate_all_checkpoints() method (After calculate())

**FULL CODE** - Copy from `HARL_MODIFICATIONS_LOST.md` lines 291-369 or use this:

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

---

## Testing Commands After Implementation

```bash
# Test calculator mode on single checkpoint
python HARL/examples/test.py --algo happo --env mapush \
  --model_dir ./results/mapush/cuboid_go1push_mid/happo/separate_rewards_test11/seed-00001-2025-12-08-10-07-02/models/80M \
  --mode calc

# Expected output: Success rate, finish time, collision, collaboration metrics

# Test all checkpoints
python HARL/examples/test.py --algo happo --env mapush \
  --model_dir ./results/mapush/cuboid_go1push_mid/happo/separate_rewards_test11/seed-00001-2025-12-08-10-07-02/models \
  --mode calc --test_all_checkpoints True

# Expected: Evaluates all checkpoints, creates all_checkpoints_results.txt
```

---

## Files Already Complete

✅ `HARL/examples/test.py` - Testing script
✅ `HARL/harl/configs/algos_cfgs/happo.yaml` - Has use_calc_mode and calc_n_threads
✅ `HARL/harl/envs/mapush/mapush_env.py` - Exposes calculator buffers
✅ Constructor partial modification (lines 49-66) - Handles directory creation

---

## Critical Reminders

1. **DO NOT CHANGE REWARDS**: `use_per_agent_rewards=True` in `task/cuboid/config.py`
2. **DO NOT CHANGE**: `share_param: False` in happo.yaml
3. **Reference file**: All code is in `HARL_MODIFICATIONS_LOST.md`
4. **One file to edit**: `HARL/harl/runners/on_policy_base_runner.py`
5. **Total work**: 6 edits, most are adding new methods at the end

---

## Estimated Time: 15 minutes

Just make the 6 edits listed above, test, done.
