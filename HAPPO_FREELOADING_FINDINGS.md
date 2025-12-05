# HAPPO Freeloading Problem - Session Findings

**Date**: 2025-12-04
**Branch**: `happo-reward-design`
**Key Discovery**: One agent pushes, one agent abandons task (freeloading behavior)

---

## Executive Summary

HAPPO training with 500 parallel envs reached 30M steps but showed degrading performance (-15 → -20 reward). Visual inspection revealed the root cause: **one agent consistently pushes the box while the other agent leaves in a random direction and ignores the task entirely**.

This freeloading behavior explains why:
- HAPPO fails to learn despite correct hyperparameters
- Performance briefly improves at 13M (both agents cooperate by chance) then degrades
- Rewards show contradictory patterns (OCB/push increase while distance/approach decrease)

---

## Training Timeline

### Configuration Used (envs500-test run)
```yaml
# HAPPO settings
n_rollout_threads: 500
episode_length: 200
share_param: False         # ← KEY: Separate networks per agent
lr: 0.0005
entropy_coef: 0.01
gamma: 0.99
hidden_sizes: [128, 128]   # Reduced from 256x256 for memory
use_linear_lr_decay: False
```

### Performance Over Time
```
0-15M steps:   -15 (relatively stable)
13M steps:     Reach target reward spike (FIRST SUCCESS!)
15-30M steps:  -15 → -20 (accelerating degradation)

Reward component changes (20-30M):
  ✅ OCB reward increasing      → Better positioning
  ✅ Push reward increasing     → More pushing
  ❌ Distance to target worse   → Box moving wrong direction
  ❌ Approach to box worse      → Agents getting farther
  ❌ Collision punishment worse → More collisions
```

---

## Visual Observation in Viewer Mode

### Setup
```bash
# Loaded 30M step checkpoint
# Rendered with n_threads: 1, headless: False
python examples/train.py --algo happo --env mapush --use_render True \
  --model_dir results/.../models
```

### What We Saw (Consistent Pattern)

**Agent 1:**
- Positions near/behind the box
- Attempts to push (not always correct direction)
- Stays engaged with task
- Gets positive OCB and push rewards

**Agent 2:**
- Immediately moves away in a straight line
- Ignores the box and target completely
- Random direction each episode
- Avoids approach penalty, freeloads on shared rewards

**Result:**
- Single agent cannot effectively push box alone
- No coordination between agents
- Task fails consistently

---

## Root Cause Analysis

### The Freeloading Problem

**Reward Structure:**
```python
# From go1_push_mid_wrapper.py

# SHARED rewards (both agents get same):
reward[:, :] += distance_reward.repeat(1, num_agents)  # Distance to target
reward[:, :] += push_reward.repeat(1, num_agents)      # Box velocity > 0.1
reward[:, :] += reach_target_reward.repeat(1, num_agents)  # Success bonus

# PER-AGENT rewards:
reward[:, i] += approach_reward_i  # Distance to box (negative)
reward[:, i] += ocb_reward_i       # Optimal circular baseline
reward[:, i] += collision_punishment  # Agent-agent collision
```

**Why Freeloading Emerges:**

**Agent 1's Optimization:**
```
Try: Push box
  → Get: Approach penalty (-0.02) + OCB reward (+0.01) + Shared rewards (+0.05)
  → Net: +0.04
  → Policy: "Pushing is good, keep doing it"
```

**Agent 2's Optimization:**
```
Try: Push box
  → Get: Approach penalty (-0.02) + Collision penalty (-0.01) + Shared rewards (+0.05)
  → Net: +0.02

Try: Leave area
  → Get: No approach penalty (0) + No collision (0) + Shared rewards (+0.05)
  → Net: +0.05 (BETTER!)
  → Policy: "Leaving is optimal, Agent 1 will do the work"
```

### Why Separate Networks Enable This

**HAPPO: `share_param: False`**
```
Agent 1: network_1(obs) → "Push box"
Agent 2: network_2(obs) → "Leave area"

Each network independently optimizes its own rewards
→ Networks can diverge into asymmetric strategies
→ Freeloading is a stable local optimum for Agent 2
```

**MAPPO: `share_param: True`**
```
Both agents: shared_network(obs) → Same policy

If network outputs "push" for Agent 1, outputs "push" for Agent 2 too
→ Can't learn asymmetric freeloading strategy
→ Forced to find symmetric cooperation (eventually)
```

---

## Why MAPPO Works Despite Same Reward Structure

**MAPPO eventually learns (100M steps) because:**

1. **Parameter sharing prevents divergence**
   - Both agents must use the same policy
   - Can't have one push while other leaves (same network!)
   - Only symmetric strategies possible

2. **Eventually finds symmetric cooperation**
   - Through massive exploration over 100M steps
   - Random chance eventually produces "both push"
   - This symmetric strategy gets reinforced
   - Takes forever but eventually converges

3. **Freeloading isn't expressible**
   - Shared network can't encode "Agent 1 push, Agent 2 leave"
   - Any policy applies to both agents equally
   - Asymmetric equilibrium is impossible

---

## Why HAPPO Fails

**With separate networks:**

1. **Can express asymmetric strategies**
   - Agent 1's network: "I'll push"
   - Agent 2's network: "I'll freeload"
   - This is a valid policy for separate networks

2. **Freeloading is locally optimal**
   - Given Agent 1 pushes, Agent 2 leaving maximizes Agent 2's reward
   - Given Agent 2 leaves, Agent 1 pushing is better than both leaving
   - **Nash equilibrium:** Neither agent has incentive to change
   - More training time won't fix this!

3. **Factors ≈ 1.0 don't help**
   - HAPPO factors only adjust importance ratios
   - Doesn't change that rewards are shared
   - Doesn't prevent freeloading strategy

---

## Reward Design Problems

### Problem 1: Push Reward Doesn't Check Direction

```python
# Line 358 in go1_push_mid_wrapper.py
push_reward[torch.norm(box_velocity) > 0.1] = push_reward_scale
```

**Issue:** Rewards ANY box movement, even if:
- Moving away from target
- Moving sideways
- Spinning in circles

**Effect:** Agents learn to make box move, not move box toward target

---

### Problem 2: OCB Reward Poorly Defined

```python
# Lines 374-384
target_direction = (target_pos - box_pos) / norm(target_pos - box_pos)
normal_vector = calc_normal_vector(agent_pos - box_pos)
ocb_reward = dot(target_direction, normal_vector)
```

**Issue:** Rewards positioning on "target side" of box, but:
- Doesn't ensure effective pushing angle
- Can be positive even if agent pushes sideways
- No check if agent actually contributes force

**Effect:** Agents position "correctly" but push ineffectively

---

### Problem 3: Shared Rewards Enable Freeloading

```python
reward[:, :] += shared_reward.repeat(1, num_agents)
```

**Issue:** Both agents get the same reward regardless of contribution

**Effect:** Agent 2 can get full shared rewards while doing nothing

---

## Contradictory Metrics Explained

**Observation (20-30M steps):**
```
OCB reward ↑ + Push reward ↑  BUT  Distance ↓ + Approach ↓
```

**Explanation:**
```
Agent 1:
  - Positions behind box           → OCB ↑
  - Pushes (any direction)         → Push ↑
  - Alone, can't push effectively  → Distance ↓

Agent 2:
  - Leaves the area                → Approach less negative
  - Doesn't collide                → Collision ↓
  - Freeloads shared rewards       → No incentive to help

Net Result:
  - Good individual metrics (OCB, push)
  - Bad task metrics (distance, approach)
  - No cooperation → Task fails
```

---

## Brief Success at 13M Steps

**What happened:**
- Random exploration led both agents to cooperate momentarily
- Both positioned and pushed together
- Box moved toward target → Reach target reward spiked!
- This was the FIRST time task succeeded

**Why it didn't persist:**
- High entropy (0.01) kept exploration active
- Freeloading strategy eventually re-emerged
- Agent 2's network learned leaving was locally better
- Performance degraded 13M → 30M steps

---

## Comparison: MAPPO vs HAPPO

| Feature | MAPPO | HAPPO |
|---------|-------|-------|
| **Parameter Sharing** | True | False |
| **Networks** | 1 shared | 2 separate |
| **Can Freeload** | No | Yes |
| **Symmetric Policies** | Required | Optional |
| **Convergence** | Slow (100M) | Fails (stuck in asymmetric equilibrium) |
| **Memory Usage** | Lower | Higher |
| **Exploration** | Limited (same policy) | Rich (different policies) |

---

## Why More Training Won't Help HAPPO

**The freeloading equilibrium is stable:**

```
Agent 1: "If I change to leaving, both agents leave → Worse rewards"
         → Keep pushing (even though suboptimal)

Agent 2: "If I change to pushing, collision penalty increases → Worse rewards"
         → Keep leaving (locally optimal given Agent 1 pushes)

Result: Neither agent has incentive to change
        → Training longer reinforces this bad equilibrium
        → 50M, 100M, 200M steps won't fix it!
```

**Evidence from run:**
- Performance degrading (not improving) over time
- Degradation accelerating (15-30M worse than 0-15M)
- Agent strategies becoming MORE entrenched

---

## Memory Issues Encountered

**Problem:** OOM with 500 envs × 200 steps × 2 networks (256×256)

**Solution:** Reduced network to 128×128
```yaml
hidden_sizes: [128, 128]  # Was [256, 256]
```

**Result:** 7750MB / 8000MB usage (manageable)

**Note:** Parameter sharing would save ~50% memory (MAPPO uses 1 network)

---

## Test Run: HAPPO with share_param=True

**Hypothesis:** With parameter sharing, HAPPO should behave like MAPPO

**Prediction:**
- Both agents will do similar things (can't diverge)
- Either both push or both leave
- Should eventually learn cooperation (like MAPPO at 100M)
- Defeats purpose of HAPPO but validates the analysis

**Status:** Currently running to verify hypothesis

---

## Next Steps

### Option 1: Per-Agent Rewards (Recommended)
- Make all rewards based on individual contribution
- Prevents freeloading
- Preserves HAPPO's heterogeneous capability
- See: `HAPPO_PER_AGENT_REWARD_DESIGN.md`

### Option 2: Cooperation Bonuses
- Add explicit reward when both agents engage
- Punish when one agent is far away
- Requires careful tuning

### Option 3: Use MAPPO Instead
- Enable `share_param: True` in HAPPO
- Loses heterogeneous learning capability
- Essentially becomes MAPPO

---

## Key Insights

1. **Shared rewards + separate networks = freeloading**
   - Each agent optimizes independently
   - No incentive for cooperation
   - Stable bad equilibrium emerges

2. **Parameter sharing enforces symmetry**
   - MAPPO works because it can't express asymmetric strategies
   - Takes long time but eventually finds symmetric cooperation
   - Not ideal but works

3. **Reward design matters for multi-agent**
   - Individual contribution must be rewarded
   - Can't rely on shared team rewards alone
   - Need explicit cooperation incentives

4. **Visual inspection is crucial**
   - Metrics showed contradictions
   - Only by watching agents did we understand the problem
   - Always verify learned behavior visually!

---

## Files Modified

1. **HARL/harl/configs/algos_cfgs/happo.yaml**
   - Set `n_rollout_threads: 500`
   - Set `hidden_sizes: [128, 128]`
   - Set `entropy_coef: 0.01`

2. **HARL/harl/utils/envs_tools.py**
   - Changed render n_threads: 64 → 1
   - Allows viewing single environment

3. **openrl_ws/cfgs/ppo.yaml**
   - Fixed `hidden_sizes` → `hidden_size` and `layer_N`

---

## Conclusion

HAPPO fails to learn MAPush not because of hyperparameters or algorithm issues, but because **the reward structure enables a freeloading strategy** with separate agent networks. One agent learns to push (ineffectively alone), the other learns to abandon the task and collect free rewards.

MAPPO works (slowly) because parameter sharing prevents this asymmetric strategy. However, this defeats the purpose of heterogeneous agent learning.

**Solution:** Redesign rewards to be per-agent based on individual contribution. See annex document for detailed proposal.

---

**Analysis Complete**: 2025-12-04
**Branch**: happo-reward-design
