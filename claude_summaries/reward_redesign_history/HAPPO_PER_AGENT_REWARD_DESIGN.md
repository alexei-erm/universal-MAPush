# HAPPO Per-Agent Reward Design Proposal

**Date**: 2025-12-04
**Branch**: `happo-reward-design`
**Purpose**: Eliminate freeloading by making rewards individual-contribution-based

---

## Overview

This document proposes changes to the MAPush reward structure to make HAPPO learn effectively without one agent freeloading. The key principle: **each agent is rewarded only for its own contribution to the task**.

---

## Current vs Proposed Reward Structure

### Current (Enables Freeloading)

```python
# SHARED rewards (both get same):
distance_to_target_reward  # Both get reward for box progress
push_reward                # Both get reward if box velocity > 0.1
reach_target_reward        # Both get success bonus

# PER-AGENT rewards:
approach_to_box_reward     # Distance to box (negative)
ocb_reward                 # Optimal circular baseline
collision_punishment       # Agent-agent collision
```

**Problem:** Agent 2 can freeload on Agent 1's distance/push/reach rewards while avoiding approach penalty.

### Proposed (Prevents Freeloading)

```python
# ALL rewards become PER-AGENT based on contribution:
individual_push_contribution_reward    # Agent's force on box toward target
individual_positioning_reward          # Agent positioned to help
individual_progress_reward             # Agent caused box to move toward target
cooperation_bonus                      # Both agents contributing (optional)
penalty_for_abandonment               # Agent far from task area

# REMOVE shared rewards entirely
```

---

## Detailed Reward Design

### 1. Individual Push Contribution Reward

**Purpose:** Reward each agent for the force it applies to the box in the direction of the target.

**Current Push Reward (Broken):**
```python
# Line 358 in go1_push_mid_wrapper.py
push_reward = (norm(box_velocity) > 0.1) * scale  # Shared, no direction check
```

**Proposed:**
```python
# For each agent:
agent_to_box_vector = box_pos - agent_pos
agent_contact = (norm(agent_to_box_vector) < contact_threshold)  # e.g., 0.5m

if agent_contact:
    # Compute force direction (from agent to box)
    force_direction = normalize(agent_to_box_vector)

    # Compute target direction (from box to target)
    target_direction = normalize(target_pos - box_pos)

    # Reward alignment between force direction and target direction
    alignment = dot(force_direction, target_direction)  # [-1, 1]

    # Only reward if pushing TOWARD target (alignment > 0)
    if alignment > 0:
        push_contribution[agent_i] = alignment * scale * norm(box_velocity)
    else:
        push_contribution[agent_i] = 0  # Pushing wrong direction
else:
    push_contribution[agent_i] = 0  # Not in contact
```

**Effect:**
- Agent only rewarded when in contact with box
- Only rewarded when pushing toward target (not sideways/backward)
- Reward magnitude scales with alignment and box velocity
- No freeloading: Must actually touch and push correctly

**Scale:** `push_contribution_scale = 0.01`

---

### 2. Individual Progress Reward

**Purpose:** Reward each agent proportional to how much IT caused the box to move toward target.

**Current Distance Reward (Shared):**
```python
# Line 329
distance_reward = scale * (past_distance - current_distance)
reward[:, :] += distance_reward.repeat(1, num_agents)  # Both get same
```

**Proposed:**
```python
# Compute box progress
box_progress = past_distance - current_distance  # Positive if closer

# Attribute progress to agents based on their contribution
for agent_i in range(num_agents):
    agent_to_box = box_pos - agent_pos[agent_i]
    agent_distance = norm(agent_to_box)

    # Agent must be near box to claim progress
    if agent_distance < contribution_radius:  # e.g., 1.5m
        # Compute how much this agent's position contributed
        agent_force_direction = normalize(agent_to_box)
        target_direction = normalize(target_pos - box_pos)
        alignment = dot(agent_force_direction, target_direction)

        # Weight by proximity (closer = more contribution)
        proximity_weight = max(0, 1 - agent_distance / contribution_radius)

        # Only positive contributions count
        contribution_weight = max(0, alignment) * proximity_weight

        # Distribute progress reward based on contribution
        progress_reward[agent_i] = contribution_weight * box_progress * scale
    else:
        progress_reward[agent_i] = 0  # Too far to contribute
```

**Effect:**
- Only agents near the box can claim progress
- Credit is proportional to positioning and alignment
- If only one agent is near, it gets all credit
- If both agents are near and pushing, credit is shared fairly
- Agent far away gets zero progress reward (no freeloading!)

**Scale:** `progress_reward_scale = 0.03`

---

### 3. Individual Positioning Reward (Improved OCB)

**Purpose:** Reward positioning behind box relative to target, but only if agent is engaged.

**Current OCB Reward (Poorly Defined):**
```python
# Lines 374-384
target_direction = normalize(target_pos - box_pos)
normal_vector = calc_normal_vector(agent_pos - box_pos)
ocb_reward[agent_i] = dot(target_direction, normal_vector) * scale
```

**Problems:**
- Gives reward even if agent is far from box
- Doesn't check if agent is actually helping

**Proposed:**
```python
for agent_i in range(num_agents):
    agent_to_box = box_pos - agent_pos[agent_i]
    agent_distance = norm(agent_to_box)

    # Only reward if agent is near the box
    if agent_distance < engagement_radius:  # e.g., 2.0m
        # Compute target direction
        target_direction = normalize(target_pos - box_pos)

        # Compute agent's direction relative to box
        agent_direction = normalize(agent_to_box)

        # Reward if agent is behind box (pushing toward target)
        alignment = dot(agent_direction, target_direction)

        # Compute proximity bonus (closer = better)
        proximity_bonus = max(0, 1 - agent_distance / engagement_radius)

        # Only reward good positioning
        if alignment > 0.3:  # At least 30° angle toward target
            positioning_reward[agent_i] = alignment * proximity_bonus * scale
        else:
            positioning_reward[agent_i] = 0
    else:
        positioning_reward[agent_i] = 0  # Too far, no positioning reward
```

**Effect:**
- Only rewards agents that are near the box and positioned correctly
- Proximity bonus encourages getting closer
- Alignment requirement ensures pushing from correct side
- Far agents get zero (no freeloading!)

**Scale:** `positioning_reward_scale = 0.005`

---

### 4. Engagement Requirement

**Purpose:** Penalize agents that abandon the task.

**Proposed:**
```python
for agent_i in range(num_agents):
    agent_to_box = box_pos - agent_pos[agent_i]
    agent_distance = norm(agent_to_box)

    # Penalize if agent is too far from action
    if agent_distance > abandonment_threshold:  # e.g., 3.0m
        abandonment_penalty[agent_i] = -abandonment_scale * (agent_distance - abandonment_threshold)
    else:
        abandonment_penalty[agent_i] = 0
```

**Effect:**
- Agents that leave the task area get increasingly negative reward
- Linear penalty beyond threshold
- Encourages staying engaged even if not directly pushing

**Scale:** `abandonment_scale = -0.002`

---

### 5. Cooperation Bonus (Optional)

**Purpose:** Explicitly reward when BOTH agents are contributing.

**Proposed:**
```python
# Check if both agents are engaged
agent0_engaged = (norm(box_pos - agent_pos[0]) < engagement_radius)
agent1_engaged = (norm(box_pos - agent_pos[1]) < engagement_radius)

if agent0_engaged and agent1_engaged:
    # Both agents near box
    cooperation_bonus_both = cooperation_scale
else:
    cooperation_bonus_both = 0

# Give bonus to both if both engaged
reward[0] += cooperation_bonus_both
reward[1] += cooperation_bonus_both
```

**Effect:**
- Gives extra reward when both agents work together
- Encourages coordination
- Symmetric (both get same bonus, so no freeloading issue)

**Scale:** `cooperation_scale = 0.01` (optional, can start without this)

---

### 6. Success Bonus (Keep Shared)

**Purpose:** Large reward when task succeeds.

**Current:**
```python
reward[finished_buf, :] += reach_target_reward_scale  # 10.0, shared
```

**Proposed:** Keep as is (shared is fine for success)
```python
reward[finished_buf, :] += reach_target_reward_scale  # 10.0, shared
```

**Reasoning:**
- Success is a team achievement
- By the time they succeed, both must have contributed
- Large magnitude makes it worth cooperating
- Shared success reward is common in MARL

---

### 7. Remove Approach Reward

**Current:**
```python
# Line 338
approach_reward[i] = (-(distance_to_box + 0.5)**2) * scale
```

**Proposed:** Remove entirely

**Reasoning:**
- Now covered by positioning reward (rewards being near and well-positioned)
- The negative approach reward was part of the freeloading problem
- Replacing with engagement requirement (abandonment penalty)

---

## Complete Reward Function

### Per-Agent Rewards
```python
for agent_i in range(num_agents):
    reward[agent_i] = (
        push_contribution[agent_i]      # Force applied toward target
        + progress_reward[agent_i]      # Caused box to move toward target
        + positioning_reward[agent_i]   # Well-positioned behind box
        + abandonment_penalty[agent_i]  # Penalty for leaving
        + cooperation_bonus             # Both agents engaged (optional)
        + collision_punishment[agent_i] # Agent-agent collision (keep)
    )
```

### Shared Rewards (Only Success)
```python
if task_success:
    reward[:] += reach_target_reward_scale  # 10.0
```

### Exception Punishment (Keep)
```python
if exception:
    reward[:] += exception_punishment_scale  # -5.0
```

---

## Reward Scales (Proposed)

```python
# task/cuboid/config.py
class rewards(Go1Cfg.rewards):
    class scales:
        # New per-agent rewards
        push_contribution_scale = 0.01
        progress_reward_scale = 0.03
        positioning_reward_scale = 0.005
        abandonment_scale = -0.002
        cooperation_scale = 0.01  # Optional

        # Keep existing
        collision_punishment_scale = -0.0025
        reach_target_reward_scale = 10.0
        exception_punishment_scale = -5.0

        # REMOVE (replaced by above)
        # target_reward_scale = 0.00325  # DELETE
        # approach_reward_scale = 0.00075  # DELETE
        # push_reward_scale = 0.0015  # DELETE
        # ocb_reward_scale = 0.004  # DELETE
```

**Relative magnitudes:**
- Progress (0.03) > Push (0.01) > Positioning (0.005)
- Emphasizes making box move toward target
- Success bonus (10.0) is dominant (requires cooperation to achieve)

---

## Implementation Locations

### File: `mqe/envs/wrappers/go1_push_mid_wrapper.py`

**Lines to modify:**
- **Line 69-73:** Add new reward buffer entries
- **Line 315-321:** Keep exception punishment as is
- **Line 323-331:** Replace with new progress reward (per-agent)
- **Line 333-341:** Replace with abandonment penalty
- **Line 343-353:** Replace with push contribution reward (per-agent)
- **Line 355-360:** Delete old push reward
- **Line 362-387:** Replace with new positioning reward (per-agent)

**New functions to add:**
```python
def compute_push_contribution(self, agent_pos, box_pos, target_pos, box_velocity):
    """Compute how much each agent contributes to pushing box toward target."""
    # Implementation as described above

def compute_progress_attribution(self, agent_pos, box_pos, target_pos, progress):
    """Attribute box progress to agents based on proximity and positioning."""
    # Implementation as described above

def compute_positioning_reward(self, agent_pos, box_pos, target_pos):
    """Reward agents for being well-positioned behind box."""
    # Implementation as described above

def compute_abandonment_penalty(self, agent_pos, box_pos):
    """Penalize agents that are far from the task."""
    # Implementation as described above
```

---

## Expected Behavior Changes

### Before (Freeloading)

**Agent 1:**
- Pushes box (ineffectively alone)
- Gets: push (shared) + progress (shared) + approach (negative)
- Net: Moderate reward

**Agent 2:**
- Leaves area
- Gets: push (shared from Agent 1) + progress (shared from Agent 1) + no approach penalty
- Net: HIGHER reward than Agent 1 (freeloading!)

**Result:** Agent 2 learns to leave, Agent 1 tries alone, task fails

---

### After (No Freeloading)

**Agent 1 (if pushes alone):**
- Gets: push contribution + progress + positioning
- Net: Positive but moderate (task hard alone)

**Agent 2 (if leaves):**
- Gets: abandonment penalty + no contribution rewards
- Net: NEGATIVE (leaving is punished!)

**Result:** Agent 2 learns leaving is bad, both agents stay engaged

**Agent 1 + Agent 2 (if both push):**
- Both get: push contribution + progress share + positioning + cooperation bonus
- Net: HIGHER than solo attempt
- Plus: Task succeeds → +10 success bonus

**Result:** Both agents learn cooperation is optimal, task succeeds!

---

## Transition Strategy

### Phase 1: Implement Core Changes (Recommended Start)
1. Replace distance reward with per-agent progress reward
2. Replace push reward with per-agent push contribution
3. Replace OCB with improved positioning reward
4. Remove approach reward, add abandonment penalty

**Test with:** 5-10M steps, check if both agents stay engaged

### Phase 2: Tune Scales
1. Adjust relative magnitudes based on Phase 1 results
2. Ensure no single reward dominates
3. Check that cooperation emerges

**Test with:** 10-20M steps, check task success rate

### Phase 3: Add Cooperation Bonus (If Needed)
1. Only if Phase 2 shows slow cooperation emergence
2. Add explicit bonus for both agents being engaged

**Test with:** 20-50M steps, compare with/without bonus

---

## Validation Metrics

### Per-Agent Metrics to Log
```python
# Add to tensorboard logging:
"agent0/push_contribution"
"agent0/progress_reward"
"agent0/positioning_reward"
"agent0/abandonment_penalty"
"agent0/distance_to_box"

"agent1/push_contribution"
"agent1/progress_reward"
"agent1/positioning_reward"
"agent1/abandonment_penalty"
"agent1/distance_to_box"
```

### Success Indicators
```
✅ Both agents have positive push_contribution (not just one)
✅ Both agents have low abandonment_penalty (both engaged)
✅ Distance_to_box similar for both agents (both near box)
✅ Success rate increasing over time
✅ No single agent dominating all contribution metrics
```

### Failure Indicators
```
❌ One agent has zero push_contribution consistently
❌ One agent has high abandonment_penalty (leaving)
❌ Large difference in distance_to_box between agents
❌ Success rate remains at 0%
❌ One agent's total reward significantly higher (still freeloading somehow)
```

---

## Potential Issues and Solutions

### Issue 1: Both Agents Collide Constantly

**Symptom:** High collision punishment for both agents

**Solution:**
- Increase collision_punishment_scale (-0.0025 → -0.005)
- Add "spacing bonus" when agents are on opposite sides of box
- Reduce cooperation_bonus if it's encouraging too much proximity

---

### Issue 2: Both Agents Abandon Task

**Symptom:** Both have high abandonment penalty, low engagement

**Solution:**
- Increase abandonment_scale magnitude (-0.002 → -0.005)
- Increase positioning_reward_scale (0.005 → 0.01)
- Check if other rewards are too small

---

### Issue 3: Agents Push from Wrong Sides

**Symptom:** Agents engaged but box moves sideways/backward

**Solution:**
- Increase alignment requirement in push_contribution (only reward alignment > 0.5)
- Add "anti-alignment penalty" for pushing wrong direction
- Increase progress_reward_scale (most important signal)

---

### Issue 4: One Agent Still Freeloads

**Symptom:** One agent still has zero contribution, other works alone

**Solution:**
- Verify abandonment penalty is actually being applied
- Increase abandonment_scale magnitude
- Add cooperation bonus (requires both engaged)
- Check if some reward is still shared unintentionally

---

## Expected Training Timeline

**With per-agent rewards:**

```
0-2M steps:
  - Both agents explore randomly
  - Low rewards, high variance
  - Both try approaching box (no freeloading!)

2-10M steps:
  - Agents learn approaching box is good
  - Start getting positioning rewards
  - Occasional pushes in right direction
  - Low but increasing success rate (~1-5%)

10-30M steps:
  - Both agents consistently approach and push
  - Coordination improves
  - Success rate climbing (~10-30%)
  - No freeloading observed

30-50M steps:
  - Refined pushing strategies
  - High success rate (~50-70%)
  - Stable cooperation
  - Performance comparable to MAPPO
```

**Much faster than MAPPO's 100M steps** because:
- No time wasted on freeloading strategies
- Clear individual incentives from the start
- Both agents learn useful policies simultaneously

---

## Code Skeleton

```python
# In go1_push_mid_wrapper.py, step() function

# After computing base positions, box state, target state...

# 1. Compute per-agent push contribution
push_contribution = self._compute_push_contribution(
    base_pos, box_pos, target_pos, box_velocity
)
reward[:, 0] += push_contribution[0]
reward[:, 1] += push_contribution[1]

# 2. Compute per-agent progress reward
box_progress = past_distance - current_distance
progress_reward = self._compute_progress_attribution(
    base_pos, box_pos, target_pos, box_progress
)
reward[:, 0] += progress_reward[0]
reward[:, 1] += progress_reward[1]

# 3. Compute per-agent positioning reward
positioning_reward = self._compute_positioning_reward(
    base_pos, box_pos, target_pos
)
reward[:, 0] += positioning_reward[0]
reward[:, 1] += positioning_reward[1]

# 4. Compute abandonment penalty
abandonment_penalty = self._compute_abandonment_penalty(
    base_pos, box_pos
)
reward[:, 0] += abandonment_penalty[0]
reward[:, 1] += abandonment_penalty[1]

# 5. Optional: Cooperation bonus
if cooperation_enabled:
    cooperation_bonus = self._compute_cooperation_bonus(base_pos, box_pos)
    reward[:, :] += cooperation_bonus

# 6. Keep existing: collision, success, exception
# ... (as before)

# Update reward buffer for logging
self.reward_buffer["agent0_push_contribution"] += push_contribution[0].sum().cpu()
self.reward_buffer["agent1_push_contribution"] += push_contribution[1].sum().cpu()
# ... (similar for all per-agent rewards)
```

---

## Summary

**Key Changes:**
1. ✅ All main rewards become per-agent contribution-based
2. ✅ Abandonment penalty prevents leaving
3. ✅ No shared rewards except success (can't freeload)
4. ✅ Clear incentive for both agents to engage and contribute

**Expected Outcome:**
- Both agents stay engaged with task
- Learn complementary pushing strategies
- Success rate increases over 30-50M steps
- No freeloading behavior
- Faster convergence than MAPPO (no parameter sharing needed)

**Next Step:** Implement Phase 1 changes and run 5-10M step test

---

**Proposal Complete**: 2025-12-04
**Branch**: happo-reward-design
