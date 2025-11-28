# HAPPO Quick Fix Guide

**Problem:** HAPPO not learning (0% success rate after 20M steps)
**Solution:** Apply these exact configuration changes

---

## Fix 1: Increase Reward Scales (10x)

**File:** `task/cuboid/config.py`
**Lines:** 99-106

**CHANGE FROM:**
```python
class rewards(Go1Cfg.rewards):
    expanded_ocb_reward = False
    class scales:
        target_reward_scale = 0.00325
        approach_reward_scale = 0.00075
        collision_punishment_scale = -0.0025
        push_reward_scale = 0.0015
        ocb_reward_scale = 0.004
        reach_target_reward_scale = 10
        exception_punishment_scale = -5
```

**CHANGE TO:**
```python
class rewards(Go1Cfg.rewards):
    expanded_ocb_reward = False
    class scales:
        target_reward_scale = 0.0325                # 10x
        approach_reward_scale = 0.0075              # 10x
        collision_punishment_scale = -0.025         # 10x
        push_reward_scale = 0.015                   # 10x
        ocb_reward_scale = 0.04                     # 10x
        reach_target_reward_scale = 100             # 10x
        exception_punishment_scale = -50            # 10x
```

---

## Fix 2: Increase Mini-Batches & Learning Rate

**File:** `HARL/harl/configs/algos_cfgs/happo.yaml`

### Change 1: Learning Rate (Line 68)
```yaml
# BEFORE
lr: 0.0005

# AFTER
lr: 0.001
```

### Change 2: Critic Learning Rate (Line 69)
```yaml
# BEFORE
critic_lr: 0.0005

# AFTER
critic_lr: 0.001
```

### Change 3: Actor Mini-Batches (Line 90)
```yaml
# BEFORE
actor_num_mini_batch: 1

# AFTER
actor_num_mini_batch: 4
```

### Change 4: Critic Mini-Batches (Line 92)
```yaml
# BEFORE
critic_num_mini_batch: 1

# AFTER
critic_num_mini_batch: 4
```

### Change 5: Entropy Coefficient (Line 94)
```yaml
# BEFORE
entropy_coef: 0.01

# AFTER
entropy_coef: 0.03
```

### Change 6: Discount Factor (Line 104)
```yaml
# BEFORE
gamma: 0.99

# AFTER
gamma: 0.97
```

---

## Testing Command

After making changes, run a 5M step test:

```bash
conda activate mapush
cd /home/gvlab/universal-MAPush/HARL

python examples/train.py \
    --algo happo \
    --env mapush \
    --exp_name happo_fixed_v1 \
    --n_rollout_threads 10 \
    --num_env_steps 5000000
```

---

## Monitor Progress

### TensorBoard
```bash
tensorboard --logdir HARL/results/mapush/cuboid_go1push_mid/happo/happo_fixed_v1/
```

### What to Watch

**Success Indicators (within 5M steps):**
- ✅ Success rate > 5% (was 0%)
- ✅ Episode reward > -10 (was -19.9)
- ✅ Approach reward > 0 (was -0.151)
- ✅ Distance to target < 1.5m (was 2.2m)

**Failure Indicators:**
- ❌ Success rate still 0%
- ❌ Episode reward still < -15
- ❌ Approach reward still negative

---

## If Still Not Learning

Try more aggressive changes:

### Option A: Even Higher Rewards (100x)
```python
target_reward_scale = 0.325      # 100x original
approach_reward_scale = 0.075    # 100x original
# etc.
```

### Option B: Shorter Episodes
```yaml
# In HARL/harl/configs/envs_cfgs/mapush.yaml
episode_length: 2000  # Was 4000
```

### Option C: Curriculum Learning
Start with easier task:
- Closer starting distance
- Larger success threshold (2.0m instead of 1.0m)

---

## Full Training (if test successful)

Once 5M test shows improvement:

```bash
python examples/train.py \
    --algo happo \
    --env mapush \
    --exp_name happo_fixed_full \
    --n_rollout_threads 20 \
    --num_env_steps 50000000
```

Expected results at 50M steps:
- Success rate: >50%
- Episode reward: >0
- Distance to goal: <1.0m consistently

---

## Quick Copy-Paste Commands

```bash
# Backup original configs
cp task/cuboid/config.py task/cuboid/config.py.backup
cp HARL/harl/configs/algos_cfgs/happo.yaml HARL/harl/configs/algos_cfgs/happo.yaml.backup

# Edit files
vim task/cuboid/config.py          # Multiply all scales by 10
vim HARL/harl/configs/algos_cfgs/happo.yaml  # Apply 6 changes listed above

# Run test
cd HARL
python examples/train.py --algo happo --env mapush --exp_name happo_fixed_v1 \
    --n_rollout_threads 10 --num_env_steps 5000000

# Monitor
tensorboard --logdir results/mapush/
```

---

## Verification Checklist

Before running:
- [ ] Reward scales multiplied by 10 in config.py
- [ ] lr changed to 0.001
- [ ] actor_num_mini_batch changed to 4
- [ ] critic_num_mini_batch changed to 4
- [ ] entropy_coef changed to 0.03
- [ ] gamma changed to 0.97
- [ ] Backup configs saved

After 1M steps:
- [ ] Success rate increasing (even if small)
- [ ] Episode reward trending upward
- [ ] Approach reward becoming less negative
- [ ] TensorBoard shows clear learning signal

---

**Good luck! The fixes should show improvement within the first million steps.**
