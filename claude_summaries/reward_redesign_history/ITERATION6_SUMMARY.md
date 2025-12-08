# Iteration 6 Summary

**Duration**: 100M steps | **Result**: ❌ CATASTROPHIC FAILURE

## Key Changes
- REMOVED engagement bonus: 0.02 → 0.0
- REMOVED cooperation bonus: 0.01 → 0.0
- AMPLIFIED push reward: 0.003 → 0.02 (6X)

## Results
```
Episode Reward: -1.95 (stayed NEGATIVE entire 100M!)
vs Iter 5:      -12.65 points WORSE
Visual:         Both agents ABANDON and RUN AWAY
```

| Milestone | Iter 5 | Iter 6 | Difference |
|-----------|--------|--------|------------|
| 1M        | +3.80  | -6.23  | -10.03     |
| 30M       | +10.20 | -2.20  | -12.40     |
| Final     | +10.70 | -1.95  | -12.65     |

## The Fatal Mistake
**I removed engagement bonuses thinking agents "learned" it:**
```python
engagement_bonus_scale = 0.0  # CATASTROPHIC!
cooperation_bonus_scale = 0.0 # CATASTROPHIC!
```

## Why It Failed
**The wrong assumption:**
- "Agents internalized engagement in Iter 5"
- "Can remove 'training wheels' now"

**The reality:**
- RL agents optimize for CURRENT rewards
- Remove incentive → behavior disappears
- No engagement bonus → no reason to approach box
- Only penalties left → agents learn "stay away = safe"

## The Vicious Cycle
```
No engagement → Explore randomly
→ No box contact → No push rewards
→ Only penalties → Avoid task
→ Complete abandonment
```

## Critical Lesson
**RL agents DON'T "internalize" behaviors!**
- They optimize for current reward signal
- Past training doesn't create "habits"
- Engagement bonuses weren't "training wheels"
- They're THE FOUNDATION!

**The paradox:**
- Tried to simplify → Made it worse
- Removed complexity → Lost essential signals
- "Focus on task" → Task abandoned entirely

**Takeaway**: Don't remove what works - amplify what's missing! Can't build direction on top of nothing.
