# Iteration 8 Summary

**Duration**: 83M steps | **Result**: ❌ FAILED (0% Success Rate)

## Key Changes from Iter7
- Push reward: 0.05 → 0.15 (3X increase, 50X from baseline)
- Push:Bonus ratio: 1.67X → 5X
- All other settings preserved from Iter7

## Results
```
Episode Reward:    -5.12 → +12.59  (looks good but...)
Success Rate:      0%              (ZERO successes!)
Distance to Target: 2.24m → 2.06m  (only 18cm improvement)
Push Reward:       -0.01 → +3.13   (agents ARE pushing)
```

## Training Dynamics (CONCERNING)
```
Agent 0 Entropy:   4.27 → 9.53     (EXPLODED!)
Agent 1 Entropy:   4.27 → 6.72     (High but stable)
Agent 0 Grad Norm: 0.68 → 0.02     (DEAD)
Agent 1 Grad Norm: 0.56 → 0.06     (Nearly dead)
Critic Grad Norm:  6.90 → 0.03     (DEAD)
```

## Visual Observation (CRITICAL FINDING)

**Agent 1 (Good):**
- Positions behind box correctly
- Pushes toward goal
- Frequently succeeds when alone or unopposed
- Has learned the task!

**Agent 0 (Problematic):**
- Positions symmetrically OPPOSITE to Agent 1
- This means: BETWEEN box and goal!
- Sometimes pushes backwards (against Agent 1)
- Often just stays near box, moves with it passively
- Effectively BLOCKING the task!

## The Symmetric Positioning Problem

```
         TARGET
            ↓
     ┌─────────────┐
     │   AGENT 0   │  ← BLOCKING! (between box and target)
     │     ↓       │
     │   ┌───┐     │
     │   │BOX│     │
     │   └───┘     │
     │     ↑       │
     │   AGENT 1   │  ← PUSHING correctly
     └─────────────┘
```

**Why this happens:**

1. **OCB Reward Bug**: The OCB (Optimal Circular Baseline) reward measures alignment between agent's normal vector and target direction. Agent 0 on the "wrong" side ALSO gets positive OCB because its normal vector points toward target!

2. **Engagement Bonus Agnostic**: Just rewards being near box, doesn't care about positioning relative to target

3. **Cooperation Bonus Naive**: Only checks if both are near box, not if both are on correct side

4. **Push Reward Exploited**: Agent 0 near box + box moving = gets positive push reward even if Agent 1 is doing all the work!

## Root Cause Analysis

### OCB Reward Formula (Lines 717-724):
```python
# Agent position relative to box (in box frame)
gf_pos = base_pos[:, i, :2] - box_pos[:, :2]
# Compute normal vector from box surface
normal_vector = calc_normal_vector_for_obc_reward(vertex_list, box_relative_pos)
# Reward = dot(target_direction, normal_vector)
ocb_reward = torch.sum(target_direction * normal_vector, dim=1)
```

**Problem**: Normal vector points OUTWARD from box surface. Agent on wrong side has normal pointing toward target too!

```
          TARGET (↑)
             │
    Agent0 ──┼── normal→  (positive dot product!)
             │
         ┌───┴───┐
         │  BOX  │
         └───┬───┘
             │
    Agent1 ──┼── normal→  (positive dot product!)
             │
```

Both agents get POSITIVE OCB reward! No penalty for blocking position!

### Push Contribution Formula (Lines 634-658):
```python
agent_to_box = box_pos - base_pos  # Vector from agent to box
force_direction = normalize(agent_to_box)
alignment = dot(force_direction, target_direction)
contribution = alignment * box_speed * scale
```

**Problem**: If Agent 0 is in front of box and box is moving toward it (pushed by Agent 1), the alignment is NEGATIVE but box_speed is POSITIVE. However, Agent 0 isn't actually pushing - it's being pushed into!

The `in_contact` check (distance < 1.0m) is too generous. Agent 0 can be "in contact" while being pushed by the box, not pushing the box.

## Why Iter8 Failed Despite 5X Push:Bonus Ratio

1. **Symmetric positioning is a stable equilibrium**:
   - Both agents get engagement bonus (+0.02)
   - Both agents get cooperation bonus (+0.01)
   - Both agents get positive OCB reward
   - Agent 0 freeloads on Agent 1's pushing

2. **Agent 0 entropy exploded (→9.53)**:
   - Learned that position doesn't matter much
   - Random actions still get bonuses
   - Stopped learning meaningful policy

3. **Agent 1 carries the team**:
   - Learned to push correctly
   - Rewards are dominated by its contribution
   - But can't overcome Agent 0's blocking

## Lessons Learned

1. **OCB reward has a fundamental flaw**: Rewards ANY position with normal pointing toward target, including blocking positions

2. **"Near box" bonuses are direction-agnostic**: Need to reward being on the CORRECT side of box

3. **Push contribution is too permissive**: Agents can claim credit without actually applying useful force

4. **Separate networks diverged asymmetrically**: One learned to push, one learned to block (local optimum)

---

**Key Insight**: The reward structure allows an agent to collect maximum rewards by:
1. Staying near box (engagement ✓)
2. Being on any side (OCB ✓ even if blocking)
3. Being near other agent (cooperation ✓)
4. Not actually helping (push reward is weak relative to penalties avoided)

**Takeaway**: Must add EXPLICIT penalty for blocking position and reward for same-side positioning.
