# Iteration 9 Proposal: Fix Blocking Behavior

**Date**: 2025-12-07
**Branch**: `happo-reward-design`
**Problem**: Agent 0 positions between box and goal, blocking Agent 1's push

---

## Executive Summary

Iter8 revealed that one agent learned to push correctly while the other learned to position symmetrically opposite - **between the box and goal** - effectively blocking progress. This is a stable local optimum because current rewards don't penalize blocking positions.

**Solution**: Add explicit **blocking penalty** and **same-side bonus** to break the symmetric equilibrium.

---

## The Problem Visualized

### Current Behavior (Iter8)
```
                    GOAL
                      ↓
              ┌───────────────┐
              │               │
              │   AGENT 0     │  ← Gets: engagement(+) + OCB(+) + coop(+)
              │      ↓        │     = POSITIVE reward for BLOCKING!
              │   ┌─────┐     │
              │   │ BOX │ → → │  (trying to move toward goal)
              │   └─────┘     │
              │      ↑        │
              │   AGENT 1     │  ← Actually pushing, doing all the work
              │               │
              └───────────────┘
```

### Desired Behavior (Iter9)
```
                    GOAL
                      ↓
              ┌───────────────┐
              │               │
              │               │
              │   ┌─────┐     │
              │   │ BOX │ → → │  → → → GOAL
              │   └─────┘     │
              │    ↑   ↑      │
              │  AG0   AG1    │  ← Both pushing from SAME side!
              │               │
              └───────────────┘
```

---

## Root Cause Analysis

### 1. OCB Reward Flaw

**Current OCB Logic** (simplified):
```python
normal_vector = get_surface_normal(agent_pos_relative_to_box)
ocb_reward = dot(target_direction, normal_vector)
```

**Problem**: Surface normals point OUTWARD from all sides. An agent on the blocking side (between box and goal) has a normal vector that ALSO points toward the goal!

```
          GOAL (↑ target_direction)
            │
   Agent0 ──●── normal→ (points toward goal = POSITIVE ocb!)
            │
        ┌───┴───┐
        │  BOX  │
        └───┬───┘
            │
   Agent1 ──●── normal→ (points toward goal = POSITIVE ocb!)
```

**Result**: Both agents get positive OCB, even though Agent 0 is blocking!

### 2. Engagement Bonus is Position-Agnostic

```python
bonus = 1.0 - distance_to_box / radius  # Only considers DISTANCE, not DIRECTION
```

Agent 0 in front of box gets same engagement bonus as Agent 1 behind box.

### 3. Cooperation Bonus Doesn't Check Positioning

```python
all_engaged = (distances < cooperation_radius).all(dim=1)
```

Just checks if both are near box. Doesn't check if they're positioned to actually cooperate.

### 4. Push Contribution Has Loophole

The push reward checks alignment with target direction, but:
- Agent 0 in front gets NEGATIVE alignment (good, should be penalized)
- BUT: The engagement + OCB + cooperation bonuses OUTWEIGH this penalty
- Net result: Agent 0 still gets positive total reward!

---

## Proposed Fixes

### Fix 1: BLOCKING PENALTY (Critical)

**Purpose**: Explicitly penalize agents positioned between box and goal.

**Logic**:
```python
def _compute_blocking_penalty(self, base_pos, box_pos, target_pos):
    """Penalize agents that are between box and goal (blocking position)."""
    blocking_penalty = torch.zeros((num_envs, num_agents), device=device)

    # Vector from box to target
    box_to_target = target_pos[:, :2] - box_pos[:, :2]
    box_to_target_norm = box_to_target / (torch.norm(box_to_target, dim=1, keepdim=True) + 1e-6)

    for i in range(num_agents):
        # Vector from box to agent
        box_to_agent = base_pos[:, i, :2] - box_pos[:, :2]
        distance_to_box = torch.norm(box_to_agent, dim=1)
        box_to_agent_norm = box_to_agent / (distance_to_box.unsqueeze(1) + 1e-6)

        # Dot product: positive if agent is in front of box (toward goal)
        # negative if agent is behind box (away from goal)
        alignment = torch.sum(box_to_agent_norm * box_to_target_norm, dim=1)

        # Agent is blocking if:
        # 1. alignment > 0 (in front of box, toward goal)
        # 2. within blocking radius (close enough to obstruct)
        is_blocking = (alignment > 0.3) & (distance_to_box < blocking_radius)

        # Penalty proportional to how directly in front
        penalty = -blocking_scale * alignment * (1.0 - distance_to_box / blocking_radius)
        blocking_penalty[:, i] = torch.where(is_blocking, penalty, torch.zeros_like(penalty))

    return blocking_penalty
```

**Parameters**:
```python
blocking_radius = 2.5  # Distance from box to consider blocking
blocking_penalty_scale = 0.05  # Strong penalty (2.5X engagement bonus)
```

**Effect**:
- Agent between box and goal: **-0.05** per step
- Agent behind box: **0** (no penalty)
- Breaks symmetric equilibrium!

### Fix 2: SAME-SIDE BONUS (Cooperation Enhancement)

**Purpose**: Reward agents for being on the same (correct) side of the box.

**Logic**:
```python
def _compute_same_side_bonus(self, base_pos, box_pos, target_pos):
    """Bonus when both agents are on the pushing side (behind box)."""

    # Vector from box to target
    box_to_target = target_pos[:, :2] - box_pos[:, :2]
    box_to_target_norm = box_to_target / (torch.norm(box_to_target, dim=1, keepdim=True) + 1e-6)

    # Check each agent's side
    agent_sides = []
    for i in range(num_agents):
        box_to_agent = base_pos[:, i, :2] - box_pos[:, :2]
        box_to_agent_norm = box_to_agent / (torch.norm(box_to_agent, dim=1, keepdim=True) + 1e-6)
        alignment = torch.sum(box_to_agent_norm * box_to_target_norm, dim=1)

        # Agent is on "push side" if alignment < -0.3 (behind box relative to goal)
        is_push_side = alignment < -0.3
        agent_sides.append(is_push_side)

    # Both on push side = same-side bonus
    both_on_push_side = agent_sides[0] & agent_sides[1]

    same_side_bonus = torch.zeros((num_envs, num_agents), device=device)
    same_side_bonus[both_on_push_side] = same_side_scale

    return same_side_bonus
```

**Parameters**:
```python
same_side_bonus_scale = 0.02  # Equal to engagement bonus
```

**Effect**:
- Both behind box: **+0.02** each (on top of other rewards)
- One blocking: **0** (no bonus)
- Incentivizes coordinated positioning!

### Fix 3: MODIFY OCB REWARD (Optional Enhancement)

**Purpose**: Only give OCB reward to agents on the correct side.

**Change**:
```python
def _compute_positioning_reward(self, base_pos, box_pos, target_pos, box_rpy, target_direction):
    # ... existing OCB calculation ...

    # NEW: Check if agent is on push side (not blocking)
    box_to_agent = base_pos[:, i, :2] - box_pos[:, :2]
    box_to_target = target_pos[:, :2] - box_pos[:, :2]
    side_alignment = torch.sum(
        box_to_agent / (torch.norm(box_to_agent, dim=1, keepdim=True) + 1e-6) *
        box_to_target / (torch.norm(box_to_target, dim=1, keepdim=True) + 1e-6),
        dim=1
    )

    # Only give OCB reward if on push side (alignment < 0)
    is_push_side = side_alignment < 0
    ocb_reward = torch.where(is_push_side, ocb_reward, torch.zeros_like(ocb_reward))
```

**Effect**:
- Agent behind box: Gets OCB reward as before
- Agent in front (blocking): Gets ZERO OCB reward
- Fixes the fundamental OCB flaw

### Fix 4: REDUCE ENGAGEMENT BONUS RADIUS

**Purpose**: Make engagement bonus more focused on actual contact position.

**Change**:
```python
engagement_bonus_radius = 2.0 → 1.2  # Tighter radius
```

**Effect**:
- Agents must be closer to box to get engagement reward
- Less reward for "hovering" near box without contributing

---

## Complete Reward Structure for Iter9

### Per-Agent Rewards
```python
reward[agent_i] = (
    # Existing (keep)
    + push_contribution        # 0.15 scale (DOMINANT)
    + progress_reward          # Per-agent attribution
    + positioning_reward       # OCB (MODIFIED: only push-side)
    + engagement_bonus         # 0.02 (REDUCED radius)
    + cooperation_bonus        # 0.01 (keep)
    + collision_punishment     # -0.0025 (keep)

    # NEW for Iter9
    + blocking_penalty         # -0.05 (NEW: penalize blocking)
    + same_side_bonus          # +0.02 (NEW: reward coordination)
)
```

### Expected Reward Breakdown

**Agent on PUSH side (correct)**:
```
Push contribution:   +0.15 (if pushing)
Engagement:          +0.02
Cooperation:         +0.01
Same-side bonus:     +0.02
OCB reward:          +0.004
Blocking penalty:    0.00
─────────────────────────────
Total per step:      +0.20 to +0.25
```

**Agent on BLOCKING side (wrong)**:
```
Push contribution:   -0.15 (negative alignment!)
Engagement:          +0.02
Cooperation:         0.00 (other agent on different side)
Same-side bonus:     0.00
OCB reward:          0.00 (FIXED: no reward for blocking)
Blocking penalty:    -0.05
─────────────────────────────
Total per step:      -0.18 to -0.20
```

**The difference**: +0.40 per step advantage for being on push side!

---

## Config Changes for Iter9

```python
# task/cuboid/config.py

class rewards(Go1Cfg.rewards):
    use_per_agent_rewards = True

    class scales:
        # EXISTING (keep from Iter8)
        per_agent_push_reward_scale = 0.15
        engagement_bonus_scale = 0.02
        cooperation_bonus_scale = 0.01
        reach_target_reward_scale = 2.0
        collision_punishment_scale = -0.0025
        exception_punishment_scale = -5

        # MODIFIED for Iter9
        engagement_bonus_radius = 1.2          # Was 2.0, now tighter
        positioning_engagement_radius = 1.5    # Was 2.0, now tighter

        # NEW for Iter9
        blocking_penalty_scale = 0.05          # NEW: penalize blocking
        blocking_radius = 2.5                  # NEW: detection radius
        same_side_bonus_scale = 0.02           # NEW: reward same-side
        same_side_threshold = -0.3             # NEW: alignment threshold
```

---

## Implementation Plan

### Step 1: Add new methods to go1_push_mid_wrapper.py

```python
def _compute_blocking_penalty(self, base_pos, box_pos, target_pos):
    """NEW: Penalize agents positioned between box and goal."""
    # Implementation as described above

def _compute_same_side_bonus(self, base_pos, box_pos, target_pos):
    """NEW: Bonus when both agents on push side."""
    # Implementation as described above
```

### Step 2: Modify existing methods

1. **`_compute_positioning_reward`**: Add push-side check before giving OCB reward
2. **`_compute_engagement_bonus`**: Use new tighter radius

### Step 3: Update step() function

```python
# After existing bonuses...
if self.use_per_agent_rewards:
    # NEW: Blocking penalty
    blocking_penalty = self._compute_blocking_penalty(base_pos, box_pos, target_pos)
    for i in range(self.num_agents):
        reward[:, i] += blocking_penalty[:, i]

    # NEW: Same-side bonus
    same_side_bonus = self._compute_same_side_bonus(base_pos, box_pos, target_pos)
    for i in range(self.num_agents):
        reward[:, i] += same_side_bonus[:, i]
```

### Step 4: Add logging

```python
self.reward_buffer["blocking_penalty"] = blocking_penalty.sum().cpu()
self.reward_buffer["same_side_bonus"] = same_side_bonus.sum().cpu()
```

---

## Expected Behavior

### Early Training (0-5M)
- Both agents explore randomly
- Blocking penalty kicks in when agent goes to wrong side
- Agents learn: front = bad, back = good

### Mid Training (5-20M)
- Both agents converge to push side
- Same-side bonus reinforces coordination
- Push reward becomes dominant signal

### Late Training (20-50M)
- Coordinated pushing emerges
- Success rate increases (target: >20%)
- No blocking behavior observed

---

## Success Criteria

### @ 10M Steps
- [ ] Both agents primarily on push side (visual confirmation)
- [ ] Blocking penalty near zero (logged metric)
- [ ] Same-side bonus consistently positive

### @ 30M Steps
- [ ] Success rate > 5%
- [ ] Distance to target improving consistently
- [ ] No symmetric blocking behavior

### @ 50M Steps
- [ ] Success rate > 15-20%
- [ ] Stable coordinated pushing
- [ ] Reward > +15 (above Iter8's +12.5)

---

## Risk Analysis

### Risk 1: Both agents go to wrong side
**Mitigation**: Push reward (0.15) is strong; pushing from wrong side = negative reward

### Risk 2: Agents collide on same side
**Mitigation**: Existing collision punishment (-0.0025) handles this

### Risk 3: Blocking penalty too strong
**Mitigation**: Scale (0.05) is moderate; can reduce if needed

### Risk 4: Same-side bonus creates camping
**Mitigation**: Push reward dominates; must actively push to maximize reward

---

## Comparison Table

| Aspect | Iter8 | Iter9 |
|--------|-------|-------|
| Blocking penalty | None | -0.05 |
| Same-side bonus | None | +0.02 |
| OCB for blocking agents | Yes | No |
| Engagement radius | 2.0m | 1.2m |
| Expected behavior | One blocks, one pushes | Both push from same side |

---

## Training Command

```bash
cd /home/gvlab/universal-MAPush/HARL
python examples/train.py --algo happo --env mapush \
    --exp_name iteration9_anti_blocking \
    --n_rollout_threads 500 \
    --num_env_steps 100000000 \
    --lr 0.005 \
    --episode_length 200
```

**Validation print to look for**:
```
[ITERATION 9 - ANTI-BLOCKING]
blocking_penalty_scale:    0.05  ← NEW
same_side_bonus_scale:     0.02  ← NEW
```

---

## Summary

**The Core Fix**: Add explicit penalty for positioning between box and goal, and bonus for both agents being on the correct (push) side.

**Why It Will Work**:
1. Blocking penalty breaks symmetric equilibrium
2. Same-side bonus incentivizes coordination
3. OCB fix removes reward for wrong-side positioning
4. Combined with existing push reward, creates strong directional gradient

**Confidence Level**: 85% HIGH

This directly addresses the observed behavior (one agent blocking) with targeted penalties and bonuses.

---

**Proposal Complete**: 2025-12-07
**Ready for Implementation**: Yes
