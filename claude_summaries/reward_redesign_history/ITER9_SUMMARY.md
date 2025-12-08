# ITERATION 9 - ANTI-BLOCKING

**Date**: 2025-12-07
**Problem**: Agent0 positions between box and goal, blocking Agent1's push

## Changes from Iter8

### NEW Rewards
```python
blocking_penalty_scale = 0.05       # Penalize being between box and goal
same_side_bonus_scale = 0.02        # Reward both agents on push side
```

### Modified Rewards
```python
engagement_bonus_radius = 1.5       # Was 2.0 (tighter)
positioning_engagement_radius = 1.5 # Was 2.0 (tighter)

# OCB reward now only given to agents on PUSH SIDE (not blocking side)
```

## Reward Breakdown

**Agent on PUSH side (correct):**
```
Push contribution:   +0.15 (if pushing toward goal)
Engagement:          +0.02
Cooperation:         +0.01
Same-side bonus:     +0.02  ← NEW
OCB reward:          +0.004
Blocking penalty:    0.00
─────────────────────────────
Total per step:      +0.20 to +0.25
```

**Agent on BLOCKING side (wrong):**
```
Push contribution:   -0.15 (negative alignment!)
Engagement:          +0.02
Cooperation:         0.00 (other agent different side)
Same-side bonus:     0.00  ← NOT GIVEN
OCB reward:          0.00  ← FIXED: no reward for blocking
Blocking penalty:    -0.05 ← NEW PENALTY
─────────────────────────────
Total per step:      -0.18 to -0.20
```

**Advantage for correct positioning: +0.40 per step!**

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

## Validation

Look for this print at startup:
```
================================================================================
[ITERATION 9 - ANTI-BLOCKING]
================================================================================
use_per_agent_rewards:    True
push_reward_scale:        0.15
reach_target_scale:       2.0
engagement_bonus_scale:   0.02
cooperation_bonus_scale:  0.01
blocking_penalty_scale:   0.05  ← NEW: Penalize blocking position
same_side_bonus_scale:    0.02  ← NEW: Reward same-side positioning

ITERATION 9: Anti-blocking fix
  Iter8 problem: Agent0 positioned BETWEEN box and goal (blocking)
  Iter9 fix: Blocking penalty (-0.05) + Same-side bonus (+0.02)
  Expected: Both agents push from same side
================================================================================
```

## Success Criteria

**@ 10M Steps:**
- [ ] Both agents primarily on push side
- [ ] Blocking penalty near zero in logs
- [ ] Same-side bonus consistently positive

**@ 30M Steps:**
- [ ] Success rate > 5%
- [ ] No symmetric blocking behavior visually

**@ 50M Steps:**
- [ ] Success rate > 15-20%
- [ ] Reward > +15 (exceeding Iter8's +12.5)
