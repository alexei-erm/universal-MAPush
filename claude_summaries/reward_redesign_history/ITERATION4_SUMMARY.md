# Iteration 4 Summary

**Duration**: Not completed (superseded) | **Result**: ⏭️ SKIPPED

## Key Changes
- Removed `clamp(min=0.0)` from push reward
- Added debug logging for bonuses
- Allow negative push rewards (penalty for wrong direction)

## Why Created
**Iteration 3 problem**: Agents push in OPPOSITE direction
**Root cause**: `clamp(min=0.0)` prevented penalty for wrong direction
```python
# Before: push away = 0 reward (no penalty)
# After:  push away = negative reward (penalty!)
```

## Why Skipped
- Iteration 3 revealed `use_per_agent_rewards = False` (config bug)
- All Iter 2-3 improvements were disabled
- Decided to fix config first before fixing push direction

## What Should Have Happened
```
Push toward target:  +0.003 reward
Push away from target: -0.003 penalty (not 0!)
Result: Learn correct direction
```

## Lesson Learned
**Always verify config is actually loaded!**
- Check startup prints
- Don't assume flags are set correctly
- Config bugs invalidate entire experiment

**Takeaway**: Good diagnosis, but config bug meant the whole reward system wasn't working as designed.
