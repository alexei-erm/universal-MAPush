# Iteration 2 Summary

**Duration**: 92.5M steps | **Result**: ❌ FAILED

## Key Changes
- Disabled approach penalty (0.0)
- Increased push reward 2X (0.003)
- Added engagement bonus (0.005)
- Relaxed push threshold (1.0m)

## Results
```
Best Reward:    -1.84 @ 35M steps
Final Reward:   -2.61 @ 92M steps
Success Rate:   0%
Entropy:        4.3 → 16.8 (EXPLODED)
```

## What Happened
- ✅ Good progress 0-35M steps (-5.47 → -1.84)
- ❌ Policy degraded after 35M
- ❌ Entropy explosion (never converged)
- ❌ Agents found bad equilibrium:
  - Agent A: Push randomly (wrong direction)
  - Agent B: Leave area (freeload)

## Root Problems
1. **Entropy too high** (0.01 coef) → random exploration forever
2. **Weak engagement** (0.005) → agents still leave
3. **No cooperation bonus** → no teamwork incentive
4. **No directional push** → any pushing rewarded
5. **GAE λ too high** (0.95) → poor credit assignment

## Fixes for Iteration 3
- Lower entropy_coef: 0.01 → 0.005
- Increase engagement: 0.005 → 0.02 (4X)
- Add cooperation bonus: 0.01
- Lower GAE lambda: 0.95 → 0.90

**Takeaway**: Per-agent rewards showed promise but need tighter exploration control and cooperation incentives.
