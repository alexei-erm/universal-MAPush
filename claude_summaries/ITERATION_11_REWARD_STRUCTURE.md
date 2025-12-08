# Iteration 11: Separate Rewards Structure (CRITICAL REFERENCE)

**Date**: 2025-12-08
**Status**: ACTIVE - This is the current reward design
**Purpose**: Prevent freeloading in HAPPO by giving agents individual rewards

---

## Overview

Iteration 11 implements **separate rewards** where each agent receives only their own individual contribution rewards, not the sum of all agents' rewards. This prevents the freeloading problem where one agent does all the work while others idle.

## Configuration Flag

**File**: `task/cuboid/config.py`

```python
class rewards(Go1Cfg.rewards):
    # Enable per-agent reward mode for HAPPO (prevents freeloading)
    # Set to False for MAPPO/shared networks (default, backward compatible)
    # Set to True for HAPPO with share_param=False (separate networks)
    use_per_agent_rewards = True
```

**Critical**: This flag **MUST be True** for HAPPO training with separate networks!

---

## Reward Components

### 1. Push Direction Reward (Individual) - **PRIMARY DRIVER**

**Scale**: `per_agent_push_reward_scale = 0.15` (ITERATION 8: AMPLIFIED 50X)

**Purpose**: Reward each agent for their individual contribution to pushing the box toward goal

**How it works**:
- Each agent gets reward proportional to how much THEY pushed the box toward goal
- Based on proximity to box and push alignment
- **This is the DOMINANT reward** - must be larger than all others

**Threshold**: `push_contact_threshold = 1.0m` - Agent must be within this distance to get push credit

### 2. Directional Progress Reward (Individual)

**Scale**: `directional_progress_scale = 0.05` (ITERATION 11: REDUCED from 0.15)

**Purpose**: Reward box movement toward goal (attributed to nearby agents)

**How it works**:
- Computes box velocity toward goal
- Credits agents within `progress_contribution_radius = 1.5m` of box
- Reduced in Iter 11 because it was causing freeloading (agents could get reward without pushing)

### 3. Engagement Bonus (Individual)

**Scale**: `engagement_bonus_scale = 0.02` (ITERATION 7: RESTORED)

**Purpose**: Encourage agents to stay engaged with the box

**How it works**:
- Small bonus for being within `engagement_bonus_radius = 1.5m` of box
- Keeps agents from wandering away

### 4. Cooperation Bonus (Individual)

**Scale**: `cooperation_bonus_scale = 0.01` (ITERATION 7: RESTORED)

**Purpose**: Encourage both agents to work near box together

**How it works**:
- Given when both agents are within `cooperation_radius = 2.0m` of box
- Promotes teamwork

### 5. Success Reward (Shared)

**Scale**: `reach_target_reward_scale = 2.0` (ITERATION 5: REDUCED from 10.0)

**Purpose**: Big bonus when task succeeds

**How it works**:
- Given to ALL agents equally when box reaches goal
- Shared because success is a team achievement
- Reduced from 10.0 to prevent reward hacking (agents ignoring other rewards to only chase this)

### 6. Collision Penalty (Individual)

**Scale**: `collision_punishment_scale = -0.0025`

**Purpose**: Penalize agents for colliding with each other

**How it works**:
- Each agent penalized for their own collisions
- Scales with inverse distance (closer = worse)

### 7. Exception Punishment (Shared)

**Scale**: `exception_punishment_scale = -5`

**Purpose**: Heavy penalty for environment failures

**How it works**:
- Applied when box falls, agent falls, or physics breaks
- All agents get this penalty (environment failed, not individual fault)

### 8. Anti-Blocking Rewards (ITERATION 9 - Individual)

**Blocking Penalty Scale**: `blocking_penalty_scale = 0.05`
**Same Side Bonus Scale**: `same_side_bonus_scale = 0.02`

**Purpose**: Prevent agents from blocking each other's push paths

**How it works**:
- Penalty if agent is between box and goal
- Bonus when both agents on correct (push) side of box

---

## Critical Implementation Details

### Reward Computation Logic (go1_push_mid_wrapper.py)

```python
if self.use_per_agent_rewards:
    # ITERATION 11: Each agent gets ONLY their own rewards
    for agent_id in range(self.num_agents):
        agent_reward = 0.0

        # Individual rewards (computed per-agent)
        agent_reward += push_direction_rewards[agent_id]        # PRIMARY
        agent_reward += directional_progress_rewards[agent_id]
        agent_reward += engagement_bonuses[agent_id]
        agent_reward += cooperation_bonuses[agent_id]
        agent_reward += blocking_penalties[agent_id]
        agent_reward += same_side_bonuses[agent_id]
        agent_reward += collision_penalties[agent_id]

        # Shared rewards (everyone gets same amount)
        agent_reward += success_reward  # When box reaches goal
        agent_reward += exception_punishment  # When environment fails

        rewards[agent_id] = agent_reward
else:
    # OLD BEHAVIOR: All agents get sum of all rewards (CAUSES FREELOADING!)
    total_reward = (push_direction_rewards.sum() +
                   directional_progress_rewards.sum() +
                   engagement_bonuses.sum() +
                   cooperation_bonuses.sum() +
                   blocking_penalties.sum() +
                   same_side_bonuses.sum() +
                   success_reward +
                   collision_penalties.sum() +
                   exception_punishment)
    rewards[:] = total_reward  # All agents get same total
```

### Why This Prevents Freeloading

**Old behavior (use_per_agent_rewards=False)**:
- Agent 0 pushes box: +1.5 reward
- Agent 1 does nothing: +0.0 reward
- BOTH agents receive: +1.5 reward (sum of all agents)
- **Result**: Agent 1 learns to freeload!

**New behavior (use_per_agent_rewards=True)**:
- Agent 0 pushes box: +1.5 reward
- Agent 1 does nothing: +0.0 reward
- Agent 0 receives: +1.5 reward
- Agent 1 receives: +0.0 reward
- **Result**: Agent 1 must push to get reward!

---

## Reward Scale Hierarchy (MUST MAINTAIN)

**Order of magnitude** (from strongest to weakest):

1. **Push Direction**: `0.15` - **MUST DOMINATE ALL OTHERS**
2. **Directional Progress**: `0.05`
3. **Engagement Bonus**: `0.02`
4. **Same Side Bonus**: `0.02`
5. **Cooperation Bonus**: `0.01`
6. **Success Reward**: `2.0` (but only once per episode)
7. **Exception Punishment**: `-5.0` (but rare)

**Critical**: Push direction reward MUST be larger than sum of other continuous rewards, otherwise agents will optimize for easier rewards (engagement, cooperation) instead of actual pushing.

---

## Iteration History Context

### Why Iteration 11 Changed from Iteration 10

**Problem in Iter 10**: `directional_progress_scale = 0.15` was too high
- Agents could get large rewards just by being near box as it moved
- Led to "passive freeloading" - one agent pushes, other stands nearby
- Both agents got similar directional progress rewards

**Solution in Iter 11**: Reduced to `0.05`
- Push direction reward (0.15) now clearly dominates
- Agents must actively push, not just stand nearby
- Reduces freeloading incentive

### Why We Can't Go Back to Shared Rewards

Shared rewards (use_per_agent_rewards=False) were tested in Iterations 1-4 and consistently produced:
- One agent learning to push correctly
- Other agent(s) learning to idle or wander
- Success rate plateauing at 50-60% instead of 90%+

---

## Configuration Reference

**File**: `task/cuboid/config.py` (lines 97-136)

```python
class rewards(Go1Cfg.rewards):
    use_per_agent_rewards = True  # MUST BE TRUE FOR HAPPO

    class scales:
        # Shared scales (used when flag=False)
        target_reward_scale = 0.00325
        approach_reward_scale = 0.00075
        collision_punishment_scale = -0.0025
        push_reward_scale = 0.0015
        ocb_reward_scale = 0.004
        reach_target_reward_scale = 2.0
        exception_punishment_scale = -5

        # Per-agent scales (used when flag=True) - CURRENT ACTIVE SCALES
        per_agent_approach_reward_scale = 0.0  # DISABLED
        per_agent_push_reward_scale = 0.15     # PRIMARY DRIVER
        engagement_bonus_scale = 0.02
        cooperation_bonus_scale = 0.01
        directional_progress_scale = 0.05      # REDUCED IN ITER 11

        # Thresholds
        push_contact_threshold = 1.0
        progress_contribution_radius = 1.5
        positioning_engagement_radius = 1.5
        engagement_bonus_radius = 1.5
        cooperation_radius = 2.0
        blocking_penalty_scale = 0.05
        blocking_radius = 2.0
        blocking_alignment_threshold = 0.3
        same_side_bonus_scale = 0.02
        same_side_alignment_threshold = -0.3
```

---

## Implementation Checklist

When re-implementing this in HARL:

- [ ] Ensure `use_per_agent_rewards = True` in config.py
- [ ] Verify reward computation uses separate_rewards logic in wrapper
- [ ] Check that push_reward_scale (0.15) > sum of other continuous rewards
- [ ] Confirm success_reward is shared (all agents get it)
- [ ] Confirm exception_punishment is shared (all agents penalized)
- [ ] Verify directional_progress_scale = 0.05 (NOT 0.15)
- [ ] Test that freeloading does NOT occur (both agents should push)

---

## Testing Success Criteria

**Good training run** (Iteration 11 working correctly):
- Success rate increases steadily to 85%+
- Both agents learn to push (not just one)
- Episode reward increases steadily
- Collision degree stays low (<0.5)
- Collaboration degree high (>0.7)

**Bad training run** (freeloading detected):
- Success rate plateaus at 50-60%
- Render shows one agent pushing, one idle
- Episode reward plateaus early
- One agent's policy stops improving

---

## Related Files

- **Config**: `task/cuboid/config.py` (lines 97-136)
- **Reward Logic**: `mqe/envs/wrappers/go1_push_mid_wrapper.py` (lines 300-450)
- **HAPPO Config**: `HARL/harl/configs/algos_cfgs/happo.yaml`

---

## Emergency Reference

**If training shows freeloading**:
1. Check `use_per_agent_rewards = True` in config
2. Check `per_agent_push_reward_scale = 0.15` (must dominate)
3. Check reward computation uses `if self.use_per_agent_rewards:` path
4. Verify HAPPO is using `share_param: False` in happo.yaml

**If agents won't approach box**:
1. Check `engagement_bonus_scale = 0.02` (not 0.0)
2. Check `cooperation_bonus_scale = 0.01` (not 0.0)
3. These were disabled in Iter 6 and MUST be restored in Iter 7+

**If success rate is low but agents push**:
1. May need to increase `directional_progress_scale` slightly
2. May need to tune `push_contact_threshold`
3. May need to increase `success_reward`
