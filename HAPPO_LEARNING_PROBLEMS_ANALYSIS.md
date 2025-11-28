# HAPPO Learning Problems Analysis - MAPush Task

**Date:** 2025-11-28
**Training Run:** 20M steps (quick_test)
**Task:** Cuboid pushing with 2 quadrupeds

---

## Executive Summary

**🚨 CRITICAL FINDING: The HAPPO algorithm is NOT learning the MAPush task effectively.**

- **Success Rate:** 0.00% throughout 20M steps (should be >50% by this point)
- **Episode Rewards:** Getting WORSE over time (-15.2 → -19.9)
- **Distance to Target:** Stuck at ~2.2m (threshold is 1.0m)
- **Agents:** Not reaching the goal even once

---

## 1. Detailed Metrics Analysis

### 1.1 Task Performance (CRITICAL)

| Metric | Initial | Final | Expected | Status |
|--------|---------|-------|----------|--------|
| **Success Rate** | 0.00% | 0.00% | >50% | ❌ FAIL |
| **Episode Reward** | -15.2 | -19.9 | >0 | ❌ WORSE |
| **Distance to Target** | 2.26m | 2.21m | <1.0m | ⚠️ STUCK |
| **Collision Rate** | 5.42% | 3.10% | <10% | ✅ OK |

**Key Insight:** Agents are maintaining a constant distance (~2.2m) from the target and never reaching it. This suggests they've learned a **local optimum** behavior pattern that avoids penalties but doesn't achieve the goal.

### 1.2 Reward Components Breakdown

| Component | Initial | Final | Scale | Contribution |
|-----------|---------|-------|-------|--------------|
| Distance to Target | -0.0735 | -0.0730 | 0.00325 | Main task signal |
| Approach to Box | -0.0750 | -0.1510 | 0.00075 | Getting WORSE |
| Collision Punishment | -0.0451 | -0.0379 | -0.0025 | Improving |
| Reach Target Bonus | 0.0000 | 0.0004 | 10.0 | Rarely triggered |
| Push Reward | 0.0000 | 0.0002 | 0.0015 | Minimal |
| OCB Reward | -0.0012 | -0.0013 | 0.004 | Negative (bad) |
| Exception Punishment | -0.0037 | -0.0043 | -5.0 | Getting worse |

**Key Insights:**
1. **Approach to Box reward is DECLINING** (-0.075 → -0.151): Agents moving AWAY from the box over time!
2. **Push reward near zero**: Agents not pushing the box at all
3. **Reach target bonus almost never triggered**: 0.0004 over millions of steps
4. **OCB reward is negative**: Agents not in optimal circular positions around box

### 1.3 Agent-Specific Learning Signals

| Metric | Agent 0 | Agent 1 | Healthy Range | Status |
|--------|---------|---------|---------------|--------|
| **Policy Loss** | -0.000564 | -0.003078 | -0.01 to -0.1 | ⚠️ TOO SMALL |
| **Entropy** | 1.50 → 1.69 | 1.50 → 1.69 | 0.5 - 2.0 | ✅ OK |
| **Gradient Norm** | 0.32 | 0.32 | 0.1 - 2.0 | ✅ OK |
| **Importance Ratio** | 1.0000 | 1.0000 | 0.8 - 1.2 | ✅ OK |

**Key Insights:**
1. **Policy loss is VERY SMALL**: -0.0006 and -0.003 are tiny values, suggesting weak learning signal
2. **Entropy is INCREASING**: Good for exploration, but suggests policy hasn't converged to useful behavior
3. **Gradients are healthy**: Not vanishing or exploding, so backprop is working
4. **Importance ratios perfect**: On-policy learning is stable (no distribution shift)

### 1.4 Critic (Value Function) Metrics

| Metric | Initial | Final | Status |
|--------|---------|-------|--------|
| **Value Loss** | 0.245 | 0.073 | ✅ Converging |
| **Critic Grad Norm** | 0.46 ± 0.43 | - | ✅ Stable |
| **Avg Step Reward** | -0.016 | -0.014 | ⚠️ Negative |

**Key Insight:** The critic is learning properly (value loss decreasing), but it's learning to predict negative rewards because that's all it sees. The value function is working, but the policy is not improving.

---

## 2. Root Cause Analysis

### 2.1 PRIMARY PROBLEM: Reward Signal Too Weak/Sparse

**Evidence:**
- Policy loss magnitude: -0.0006 to -0.003 (should be ~-0.01 to -0.1)
- Reach target bonus triggered: 0.0004 times per step (almost never)
- Push reward: 0.0002 per step (almost never)
- No positive rewards being generated

**Why this matters:**
```python
# Current reward scales from config.py
target_reward_scale = 0.00325      # Very small
approach_reward_scale = 0.00075    # Very small
push_reward_scale = 0.0015         # Very small
reach_target_reward_scale = 10     # Large but never triggered

# With discount factor γ=0.99 over 4000 steps:
# Discounted reward after 100 steps: 0.00325 * 0.99^100 ≈ 0.0012
# The signal becomes extremely weak very quickly!
```

The rewards are too small to drive learning, especially with:
- 4000-step episodes
- Gamma = 0.99 (heavy discounting)
- Distance to goal ~2.2m (never reaching the big +10 bonus)

### 2.2 SECONDARY PROBLEM: HAPPO-Specific Issues

**1. Single Mini-Batch Training**
```yaml
actor_num_mini_batch: 1  # Only 1 update per epoch!
ppo_epoch: 5             # 5 epochs = only 5 gradient steps per rollout
```

For comparison, standard PPO uses 4-8 mini-batches. With complex multi-agent coordination, 1 mini-batch is likely insufficient.

**2. Factor Batch Computation**

From `HARL/harl/algorithms/actors/happo.py:47`:
```python
factor_batch,  # This is the HAPPO-specific multi-agent credit assignment
```

This factor is used to weight advantages for each agent based on other agents' actions. If this computation is incorrect for the 2-agent pushing task, it could severely hurt learning.

**Need to verify:** How is `factor_batch` computed? Is it appropriate for cooperative pushing?

**3. Action Aggregation**
```yaml
action_aggregation: prod  # Product of action probabilities
```

For 2 agents with continuous 3D actions (vx, vy, vyaw), this means:
- Agent 0: 3 action dimensions
- Agent 1: 3 action dimensions
- Combined probability is product of 6 probability densities

This can lead to very small importance weights and numerical instability.

### 2.3 HYPERPARAMETER ISSUES

| Parameter | Current | Typical | Impact |
|-----------|---------|---------|--------|
| Learning rate | 0.0005 | 0.001-0.003 | Too conservative |
| Entropy coef | 0.01 | 0.01-0.05 | Reasonable but could increase |
| Mini-batches | 1 | 4-8 | Way too few updates |
| Hidden sizes | [128, 128] | [256, 256] | May be too small |
| GAE lambda | 0.95 | 0.95-0.99 | OK |
| Episode length | 4000 | - | Very long for sparse rewards |

---

## 3. Comparison with Baseline (PPO/MAPPO)

The OpenRL PPO training (in `results/10-15-23_cuboid/`) achieved:
- Success rates >70% by 100M steps
- Positive episode rewards
- Effective box pushing behavior

**Key differences:**
1. **Different RL framework**: OpenRL vs HARL
2. **Different algorithm**: PPO/MAPPO vs HAPPO
3. **Likely different hyperparameters**: Need to compare training scripts

**Action needed:** Extract OpenRL training hyperparameters to compare.

---

## 4. Specific Issues Identified

### Issue 1: Approach Reward is NEGATIVE and DECLINING
```
Approach to Box: -0.075 → -0.151 (getting worse!)
```

This means agents are moving AWAY from the box over time. This is learned behavior - they discovered that staying away from the box reduces collision punishment and doesn't significantly hurt the small distance reward.

### Issue 2: Agents Stuck in "Safe" Local Optimum

The agents have learned a policy that:
- ✅ Avoids collisions (3.1% collision rate)
- ✅ Maintains ~2.2m from target (not too far)
- ❌ Never approaches the box
- ❌ Never pushes the box
- ❌ Never reaches the goal

This is a **local optimum** - the policy found a behavior pattern that:
- Gets moderate negative rewards (-19.9 per episode)
- Is stable (variance is low: ±2.1)
- Doesn't improve toward the actual goal

### Issue 3: Exploration Not Effective

Despite entropy INCREASING (1.24 → 1.69), the agents aren't discovering better strategies. This suggests:
- Random exploration alone isn't enough
- Need curriculum learning or shaped rewards
- Initial policy might be stuck in bad region of policy space

### Issue 4: Reward Discounting Too Aggressive

With γ=0.99 and 4000-step episodes:
- Horizon = 1/(1-0.99) = 100 steps
- After 100 steps, rewards are discounted by 0.99^100 = 0.366
- After 500 steps, rewards are discounted by 0.99^500 = 0.0067

The tiny reward scales (0.00325, 0.00075) become essentially zero after discounting.

---

## 5. Recommended Solutions (Prioritized)

### 🔴 HIGH PRIORITY (Must Fix)

#### 1. Increase Reward Scales (10-100x)
```python
# In task/cuboid/config.py
class rewards(Go1Cfg.rewards):
    class scales:
        target_reward_scale = 0.0325          # Was 0.00325 (10x increase)
        approach_reward_scale = 0.0075        # Was 0.00075 (10x increase)
        push_reward_scale = 0.015             # Was 0.0015 (10x increase)
        collision_punishment_scale = -0.025   # Was -0.0025 (10x)
        ocb_reward_scale = 0.04               # Was 0.004 (10x)
        reach_target_reward_scale = 100       # Was 10 (10x increase)
        exception_punishment_scale = -50      # Was -5 (10x)
```

**Why:** Current rewards are 10-100x too small for the long episode length and heavy discounting.

#### 2. Increase Mini-Batches
```yaml
# In HARL/harl/configs/algos_cfgs/happo.yaml
algo:
  actor_num_mini_batch: 4  # Was 1
  critic_num_mini_batch: 4  # Was 1
```

**Why:** 1 mini-batch = 1 gradient update per rollout is far too few for complex tasks.

#### 3. Increase Learning Rate
```yaml
model:
  lr: 0.001         # Was 0.0005 (2x increase)
  critic_lr: 0.001  # Was 0.0005 (2x increase)
```

**Why:** Tiny policy losses suggest we need stronger gradient steps.

### 🟡 MEDIUM PRIORITY (Should Fix)

#### 4. Reduce Episode Length or Adjust Gamma
```yaml
# Option A: Reduce gamma to increase effective horizon
algo:
  gamma: 0.97  # Was 0.99 (less aggressive discounting)

# Option B: Shorten episodes (in mapush.yaml)
episode_length: 2000  # Was 4000 (shorter episodes)
```

**Why:** Current combination makes future rewards disappear.

#### 5. Increase Network Capacity
```yaml
model:
  hidden_sizes: [256, 256]  # Was [128, 128]
```

**Why:** 2-agent coordination with 8D observations may need more capacity.

#### 6. Increase Entropy Coefficient
```yaml
algo:
  entropy_coef: 0.03  # Was 0.01
```

**Why:** Need more exploration to escape local optimum.

### 🟢 LOW PRIORITY (Nice to Have)

#### 7. Check HAPPO Factor Computation

Need to inspect the code that computes `factor_batch` in the HAPPO buffer:
```bash
# Find where factor is computed
grep -r "factor" HARL/harl/algorithms/ HARL/harl/runners/
```

Verify it's appropriate for 2-agent cooperative pushing.

#### 8. Try Different Action Aggregation
```yaml
algo:
  action_aggregation: mean  # Was prod
```

**Why:** Product can cause numerical issues with continuous multi-dimensional actions.

#### 9. Add Curriculum Learning

Start with:
- Shorter distances to target
- Smaller box
- Gradually increase difficulty

---

## 6. Testing Plan

### Phase 1: Quick Validation (5M steps each)

Test these configurations to find best improvement:

**Config A: Increase Rewards Only**
- 10x all reward scales
- Everything else same

**Config B: Increase Mini-Batches Only**
- actor_num_mini_batch: 4
- Everything else same

**Config C: Both Rewards + Mini-Batches**
- 10x rewards
- 4 mini-batches

**Config D: Full Recommended**
- 10x rewards
- 4 mini-batches
- lr: 0.001
- gamma: 0.97

### Phase 2: Long Run (50M steps)

Use best configuration from Phase 1 for full training run.

**Success criteria:**
- Success rate >10% by 10M steps
- Success rate >50% by 50M steps
- Episode rewards trending positive
- Distance to target decreasing below 1.0m

---

## 7. Code Locations for Fixes

### Reward Scales
```bash
vim /home/gvlab/universal-MAPush/task/cuboid/config.py
# Edit lines 99-106 (rewards.scales)
```

### HAPPO Hyperparameters
```bash
vim /home/gvlab/universal-MAPush/HARL/harl/configs/algos_cfgs/happo.yaml
# Edit:
# - Line 68: lr
# - Line 69: critic_lr
# - Line 94: entropy_coef
# - Line 90: actor_num_mini_batch
# - Line 92: critic_num_mini_batch
# - Line 104: gamma
```

### Episode Length
```bash
vim /home/gvlab/universal-MAPush/HARL/harl/configs/envs_cfgs/mapush.yaml
# Edit line 9: episode_length
```

---

## 8. Next Steps

1. ✅ Analysis complete - problems identified
2. ⏳ Implement Config D (full recommended changes)
3. ⏳ Run 5M step test with Config D
4. ⏳ Monitor tensorboard for improvements:
   - Success rate should increase
   - Episode rewards should trend positive
   - Approach reward should become positive
5. ⏳ If successful, run 50M step full training
6. ⏳ Compare with OpenRL baseline

---

## 9. Summary for Quick Reference

### Main Problems
1. **Rewards too small** (10-100x too weak)
2. **Too few gradient updates** (1 mini-batch per epoch)
3. **Learning rate too conservative** (0.0005 vs should be 0.001+)
4. **Agents stuck in local optimum** (avoid box = avoid penalties)

### Quick Fix Command
```bash
# Edit reward scales (multiply by 10)
vim task/cuboid/config.py

# Edit HAPPO config
vim HARL/harl/configs/algos_cfgs/happo.yaml
# Change:
#   actor_num_mini_batch: 1 → 4
#   lr: 0.0005 → 0.001
#   gamma: 0.99 → 0.97
#   entropy_coef: 0.01 → 0.03

# Run test
cd HARL
python examples/train.py --algo happo --env mapush --exp_name fixed_config \
    --n_rollout_threads 10 --num_env_steps 5000000
```

### Expected Improvements
- Episode rewards: -19.9 → >-10 → >0
- Success rate: 0% → >5% → >10%
- Approach reward: -0.151 → >0
- Distance to target: 2.2m → <1.5m → <1.0m

---

**Analysis Complete: 2025-11-28**
