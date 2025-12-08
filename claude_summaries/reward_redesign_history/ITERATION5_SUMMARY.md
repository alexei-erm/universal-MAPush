# Iteration 5 Summary

**Duration**: 32M steps | **Result**: 🎉 BREAKTHROUGH!

## Key Changes
- Reduced success reward: 10.0 → 2.0
- Added directional progress reward: 0.01
- Added validation prints (catch config errors)

## Results
```
Episode Reward: +10.20 @ 30M
Success Rate:   ~1% (first successes!)
Visual:         BOTH robots push together!
```

## The Breakthrough
**"Both robots push together ALMOST EVERY EPISODE!"**

✅ Freeloading SOLVED (both engaged)
✅ Abandonment SOLVED (both stay)
✅ Coordination SOLVED (both push)
❌ Direction WRONG (random/opposite sides)

## Why It Worked
**Success reward reduction (10.0 → 2.0):**
- Forced learning per-step behaviors
- Less reward hacking from lucky successes

**Engagement + cooperation bonuses:**
- Both agents stay near box (0.02 + 0.01)
- Consistent coordination emerged

## The Problem
**Direction signal TOO WEAK:**
```
Engagement + cooperation: +0.03 (ALWAYS positive)
Push direction:           ±0.003 (tiny!)

Result: Optimize "stay near box" not "push correctly"
Ratio: Proximity 10X stronger than direction!
```

## Key Insight
"Both pushing together" = HUGE progress!
Just need to teach them WHERE to push.

**Takeaway**: Major breakthrough in coordination, but direction signal needs amplification.
