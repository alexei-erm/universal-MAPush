# Iteration 9 Summary

**Duration**: ~42M steps (stopped early) | **Result**: ⚠️ PARTIAL SUCCESS

## Key Changes from Iter8
- Added `blocking_penalty_scale = 0.05` (penalize being between box and goal)
- Added `same_side_bonus_scale = 0.02` (reward both agents on push side)
- Modified OCB reward to only apply to agents on push side
- Tightened `engagement_bonus_radius`: 2.0 → 1.5m

## Results
```
Episode Reward:    -5.31 → +12.89 @ 42M
Success Rate:      0%
Distance to Target: 2.24m → 2.17m (only 7cm improvement)
```

## Training Dynamics
```
Agent0 Entropy:   4.27 → 5.99  (stable, NOT exploding like Iter8's 9.53)
Agent1 Entropy:   4.27 → 5.46  (stable)
Agent0 Grad Norm: 0.59 → 0.04  (with intermittent spikes to 0.25)
Agent1 Grad Norm: 0.50 → 0.04  (declining)
```

## Visual Observation (BREAKTHROUGH!)

**What Changed from Iter8**:
- ✅ BOTH agents now on push side (no more blocking!)
- ✅ Both agents pushing together
- ❌ Pushing in random direction, NOT toward goal
- ❌ 0% success rate

**The blocking penalty worked!** Agent0 no longer positions between box and goal.

## The New Problem

Agents learned:
1. ✅ Stay near box (engagement)
2. ✅ Stay on push side (blocking penalty)
3. ✅ Push together (cooperation)
4. ❌ Push toward goal (missing signal!)

They push the box, but in random directions. The box moves, but not toward the target.

## Root Cause Analysis

The `push_contribution` reward was flawed:

```python
# What we had:
force_direction = agent_to_box  # Agent POSITION relative to box
alignment = dot(force_direction, target_direction)
contribution = alignment * box_speed * scale
```

This rewarded agent **position** (being behind box), not actual **push direction**.

An agent positioned correctly behind the box could push sideways and still get rewarded because:
- Their position vector points toward target ✓
- Box is moving (speed > 0) ✓
- But box velocity could be perpendicular to target!

## Gradient Spike Analysis

Observed intermittent spikes in Agent0 grad norm:
```
40.5M: 0.22
41.5M: 0.27
42.0M: 0.25
```

These spikes indicate the blocking penalty is working - when Agent0 wanders into blocking position, it gets penalized and has to learn "don't do that", causing gradient updates.

## Comparison: Iter8 vs Iter9

| Metric @ 40M | Iter8 | Iter9 | Verdict |
|--------------|-------|-------|---------|
| Reward | +12.40 | +12.77 | Similar |
| Agent0 Entropy | 5.69 | 5.44 | Iter9 more stable |
| Agent1 Entropy | 6.94 | 5.96 | Iter9 more stable |
| Distance | 2.06m | 2.17m | Similar |
| Blocking | Yes | No | **FIXED** |
| Push Direction | Random | Random | Same problem |

## What Iter10 Needs

1. **Restore directional_progress reward** (was disabled in Iter6)
   - Shared reward when box moves toward goal
   - Both agents get same signal for team success

2. **Fix push_contribution to use actual box velocity**
   - Reward based on where box is ACTUALLY moving
   - Not where agent is positioned

---

**Key Insight**: Iter9 solved the positioning problem (both on push side). Now need to solve the direction problem (push toward goal).

**Takeaway**: Blocking penalty was the right fix for Iter8's problem. Now need velocity-based rewards for Iter9's problem.
