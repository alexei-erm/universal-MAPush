# Iteration 11 Proposal: Reduced Shared Reward

**Date**: 2025-12-08
**Problem**: Iter10 works but shared reward (0.15) might enable freeloading

---

## Iter10 Success Recap

✅ Both agents push toward goal
✅ Actual task successes observed
✅ Distance improving (2.24m → 1.96m)
✅ reach_target rewards in 152/200 checkpoints

## The Concern

Shared `directional_progress` reward:
```python
# Both agents get this when box moves toward goal:
reward = (old_distance - new_distance) * 0.15
```

**Freeloading risk**:
- Agent A pushes hard → box moves → +0.15 for BOTH
- Agent B realizes: "I get reward even if I don't push"
- Agent B becomes passive freeloader

## Iter11 Change

```python
directional_progress_scale = 0.05  # Was 0.15
```

**New ratio**:
- Per-agent push_contribution: 0.15 (requires contact + box moving toward goal)
- Shared directional_progress: 0.05

**3:1 ratio** means individual contribution dominates.

## Expected Behavior

- Agents still get directional signal (box→goal = good)
- But must actively contribute to maximize reward
- Less incentive to coast on partner's work

## Training Command

```bash
cd /home/gvlab/universal-MAPush/HARL
python examples/train.py --algo happo --env mapush \
    --exp_name iteration11_reduced_shared \
    --n_rollout_threads 500 \
    --num_env_steps 100000000 \
    --lr 0.005 \
    --episode_length 200
```

## Success Criteria

| Metric | Iter10 | Iter11 Target |
|--------|--------|---------------|
| Distance | 1.96m | < 1.8m |
| Successes | Yes | More consistent |
| Freeloading | Unknown | Reduced |

## Validation Print

```
[ITERATION 11 - REDUCED SHARED REWARD]
directional_progress:     0.05  ← REDUCED from 0.15
```
