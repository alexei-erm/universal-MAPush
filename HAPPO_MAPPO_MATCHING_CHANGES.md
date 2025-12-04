# HAPPO Configuration Modified to Match MAPPO Settings

**Date**: 2025-12-01
**Purpose**: Remove algorithmic differences between HAPPO and MAPPO (except network capacity)

---

## Files Modified

### 1. `/HARL/harl/models/base/distributions.py`
**Modified DiagGaussian class to support exp() std transformation like MAPPO**

**Changes**:
- Added `use_exp_std` parameter (default: False for backward compatibility)
- When `use_exp_std=True`: Uses `exp(log_std)` like MAPPO (unbounded)
- When `use_exp_std=False`: Uses `sigmoid(log_std/x)*y` like original HAPPO (bounded to [0, 0.5])
- Initialization changes:
  - With `use_exp_std=True`: Initializes log_std to 0 → std = exp(0) = 1.0
  - With `use_exp_std=False`: Initializes log_std to std_x_coef → std = sigmoid(1.0)*0.5 = 0.366

**Lines modified**: 58-104

---

### 2. `/HARL/harl/configs/algos_cfgs/happo.yaml`
**Updated hyperparameters to match MAPPO**

**Backup created**: `happo_mappo_matched.yaml`

| Parameter | Original HAPPO | MAPPO-Matched | Reason |
|-----------|---------------|---------------|---------|
| **episode_length** | 1000 | **200** | Match MAPPO episode rollout length |
| **use_feature_normalization** | True | **False** | MAPPO doesn't use input LayerNorm |
| **lr** | 0.001 | **0.0005** | Match MAPPO learning rate (2x slower) |
| **critic_lr** | 0.001 | **0.0005** | Match MAPPO critic learning rate |
| **use_exp_std** | (not present) | **True** | Use exp() instead of sigmoid() for std |
| **gamma** | 0.97 | **0.99** | Match MAPPO discount factor |
| **hidden_sizes** | [256, 256] | **[256, 256]** | KEPT as requested (MAPPO uses [64]) |

---

## What's Now Identical Between HAPPO and MAPPO

✅ **Output parameterization**: Both use `std = exp(log_std)`
✅ **Initial exploration**: Both start with std = 1.0
✅ **Std dev bounds**: Both unbounded (can learn any positive std)
✅ **Feature normalization**: Both disabled
✅ **Learning rates**: Both 0.0005 (actor and critic)
✅ **Episode length**: Both 200 steps
✅ **Discount factor**: Both 0.99
✅ **Activation**: Both ReLU
✅ **Initialization**: Both orthogonal

---

## What Remains Different

### Network Capacity (Intentional)
- **MAPPO**: 1 hidden layer of 64 units (~5K parameters)
- **HAPPO**: 2 hidden layers of 256 units (~68K parameters)

**Why kept different**: Per user request to maintain HAPPO's larger network capacity

### Algorithm-Specific Features
- **HAPPO**: Uses heterogeneous actor updates (per-agent optimization)
- **MAPPO**: Uses homogeneous actor updates (shared optimization)

These are core algorithmic differences that define HAPPO vs MAPPO.

---

## Verification

### Expected Behavior After Changes

**Before modifications**:
```python
# HAPPO
std = sigmoid(log_std / 1.0) * 0.5  # Bounded [0, 0.5]
initial_std = sigmoid(1.0) * 0.5 = 0.366
```

**After modifications**:
```python
# HAPPO (with use_exp_std=True)
std = exp(log_std)  # Unbounded [0, ∞)
initial_std = exp(0) = 1.0
```

**MAPPO (reference)**:
```python
# MAPPO
std = exp(log_std)  # Unbounded [0, ∞)
initial_std = exp(0) = 1.0
```

Now HAPPO and MAPPO have **identical output distributions**!

---

## Testing the Changes

### Quick Test (5 min)
```bash
cd /home/gvlab/universal-MAPush/HARL
conda activate mapush

python examples/train.py --algo happo --env mapush --exp_name test_matched \
    --n_rollout_threads 5 --num_env_steps 1000000
```

### Verify Config Loaded
Check training logs for:
```
episode_length: 200
lr: 0.0005
use_exp_std: True
use_feature_normalization: False
```

### Compare with Original HAPPO
To test original HAPPO behavior, temporarily change in `happo.yaml`:
```yaml
use_exp_std: False  # Back to sigmoid std
```

---

## Expected Impact on Learning

### With Matched Settings

**Advantages**:
1. **Fair comparison**: Algorithmic differences (HAPPO vs MAPPO) not confounded by hyperparameter differences
2. **Higher exploration**: std=1.0 instead of 0.366 (2.7x more initial exploration)
3. **Unbounded std**: Can learn very explorative or very exploitative policies
4. **Faster convergence**: Larger network capacity (256x256) with same exploration as MAPPO

**Potential Issues**:
1. **More unstable**: Unbounded std can grow very large if not controlled
2. **May need tuning**: Larger network + high exploration might be too much
3. **Different optimal settings**: 256x256 network might prefer different lr/entropy_coef

### Recommendations

If training becomes unstable with these settings:
1. Reduce `entropy_coef` from 0.03 to 0.01
2. Add gradient clipping (already enabled with max_grad_norm: 10.0)
3. Try `use_linear_lr_decay: True` to reduce lr over time
4. Monitor std dev during training - if it grows >5.0, consider bounds

---

## Rollback Instructions

### To restore original HAPPO settings:

```bash
cd /home/gvlab/universal-MAPush/HARL

# Restore distributions.py (remove use_exp_std support)
git checkout harl/models/base/distributions.py

# Restore happo.yaml
git checkout harl/configs/algos_cfgs/happo.yaml
```

Or manually change in `happo.yaml`:
```yaml
episode_length: 1000
use_feature_normalization: True
lr: 0.001
critic_lr: 0.001
use_exp_std: False  # Or remove this line
gamma: 0.97
```

---

## Summary

The modified HAPPO configuration now matches MAPPO in all key hyperparameters except network capacity (kept at 256x256 as requested). This creates a fair comparison where any performance differences are due to:

1. **Network capacity**: HAPPO has 13x more parameters
2. **Algorithm**: HAPPO uses heterogeneous actor updates vs MAPPO's homogeneous updates

All other confounding factors (exploration, learning rate, episode length, feature normalization, output distribution) have been eliminated.

**Status**: ✅ Ready for fair HAPPO vs MAPPO comparison experiments
