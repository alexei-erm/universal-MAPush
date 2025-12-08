# Iteration 3 Summary

**Duration**: 100M steps | **Result**: ⚠️ INVALID - Config Error

## Key Changes
- Lower entropy_coef: 0.01 → 0.005
- Increase engagement: 0.005 → 0.02 (4X)
- Add cooperation bonus: 0.01
- Lower GAE lambda: 0.95 → 0.90

## Results
```
Final Reward:   +9.82 (first positive ever!)
Success Rate:   0%
Distance:       2.35m (no progress)
Entropy:        11.4 @ 100M (still exploded)
```

## The Paradox
- ✅ Rewards went positive (+9.82)
- ❌ 0% success rate (task never solved)
- ❌ Box pushed AWAY from target
- ❌ One agent freeloads, one pushes wrong direction

## Critical Discovery: **CONFIG BUG**
```python
use_per_agent_rewards = False  # ❌ SHOULD BE TRUE!
```

**This disabled EVERYTHING:**
- ❌ Cooperation bonus never activated
- ❌ Engagement bonus never activated
- ❌ Per-agent push attribution disabled
- ❌ Ran with old Iteration 1 code

## Why Rewards Went Positive (Reward Hacking)
1. Random box movement → near target → +reach_target
2. Any pushing → +push_reward (no direction check)
3. Avoiding exceptions → less penalty
4. **Result**: +9.82 without solving task!

## Other Issues Found
1. **Reward buffer never resets** → shows cumulative averages
2. **Entropy still exploded** → 0.005 still too low
3. **Push reward has clamp(min=0)** → no penalty for wrong direction
4. **reach_target_scale = 10.0** → too generous

## Critical Fixes for Iteration 4
**MUST DO:**
1. Set `use_per_agent_rewards = True` in config
2. Remove `clamp(min=0.0)` from push reward
3. Increase `entropy_coef` to 0.01+
4. Reduce `reach_target_scale` to 1.0

**VERIFY:**
- Print config at startup
- Check cooperation/engagement bonuses > 0 in logs
- Monitor entropy < 7.0

**Takeaway**: Iteration 3 was an invalid experiment. Must re-run with correct configuration.
