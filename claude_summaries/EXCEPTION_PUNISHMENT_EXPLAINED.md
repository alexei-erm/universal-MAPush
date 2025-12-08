# Exception Punishment Reward - Explained

**Date**: 2025-12-04
**Purpose**: Explain what the `exception_punishment` reward component does

---

## Quick Summary

**`exception_punishment`** is a **penalty** (negative reward, typically -5) applied when an episode terminates early due to **termination conditions** being violated. It punishes behaviors that cause the robots or task to fail.

**Value**: `-5` (configured in `task/cuboid/config.py`)

**When it's applied**: When `exception_buf` or `value_exception_buf` is True for an environment

---

## What Triggers Exception Punishment?

The exception punishment is triggered by **two types of exceptions**:

### 1. **Value Exceptions** (`value_exception_buf`)

Triggered when observations contain **NaN or Inf** values (numerical errors):

```python
# From go1_push_mid_wrapper.py:272-273
self.value_exception_buf = torch.isnan(obs).any(dim=2).any(dim=1) \
                         | torch.isinf(obs).any(dim=2).any(dim=1)
```

**What this means**:
- If any observation value becomes `NaN` (not a number) or `Inf` (infinity)
- Usually indicates a simulation error or physics instability
- Episode is immediately terminated and reset
- Agents receive -5 reward

**Typical causes**:
- Physics explosion (robots flying off)
- Division by zero in calculations
- Extreme velocities or positions

---

### 2. **Termination Exceptions** (`exception_buf`)

Triggered when **termination conditions** are violated. From `task/cuboid/config.py:83-89`:

```python
class termination(Go1Cfg.termination):
    check_obstacle_conditioned_threshold = False
    z_wave_kwargs = dict(threshold= 0.35)  # Z position change > 0.35m
    collision_kwargs = dict(threshold= 0.25)  # Robot distance < 0.25m
    termination_terms = [
        "roll",      # Robot rolls over
        "pitch",     # Robot pitches too much
        "z_wave",    # Robot/box falls or jumps
        "collision", # Robots collide with each other
    ]
```

Each termination condition:

#### a) **Roll** - Robot flipped sideways
```python
# Check if robot roll angle exceeds threshold
torch.abs(roll) > threshold
```
- Robot has fallen over on its side
- **Threshold**: Default from base config

#### b) **Pitch** - Robot tipped forward/backward
```python
# Check if robot pitch angle exceeds threshold
torch.abs(pitch) > threshold
```
- Robot has fallen forward or backward
- **Threshold**: Default from base config

#### c) **Z-Wave** - Vertical position change too large
```python
# Check if z-position changed more than 0.35m from initial
torch.abs(current_z - initial_z) > 0.35
```
- Robot jumped, fell, or was launched upward
- Box fell through floor or was thrown
- **Threshold**: 0.35 meters

**Applies to**:
- All robots (checks each one)
- All objects (box and target area)

#### d) **Collision** - Robots too close together
```python
# Check if distance between robots < 0.25m
torch.norm(robot1_pos - robot2_pos) < 0.25
```
- The two robots have collided or are overlapping
- **Threshold**: 0.25 meters (25cm apart)
- This is DIFFERENT from `collision_punishment` reward!

**Difference from collision_punishment**:
| Metric | Collision (termination) | Collision Punishment (reward) |
|--------|------------------------|-------------------------------|
| **Threshold** | 0.25m | Used in reward calculation |
| **Effect** | Episode ends immediately | Small negative reward each step |
| **Penalty** | -5 (exception_punishment) | -0.0025 per step |
| **Purpose** | Prevent severe collisions | Encourage spacing |

---

## How It Works in the Code

### Step 1: Check Termination Conditions

In `legged_robot_field.py:146-201`:

```python
# Initialize exception buffer
self.exception_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

# Check each termination condition
if "roll" in termination_terms:
    self.exception_buf |= (roll_angle_exceeded)

if "pitch" in termination_terms:
    self.exception_buf |= (pitch_angle_exceeded)

if "z_wave" in termination_terms:
    self.exception_buf |= (z_position_change_exceeded)

if "collision" in termination_terms:
    self.exception_buf |= (robots_too_close)

# Mark these environments for reset
self.reset_buf |= self.exception_buf
```

### Step 2: Apply Punishment Reward

In `go1_push_mid_wrapper.py:315-321`:

```python
# Calculate exception punishment
if self.exception_punishment_scale != 0:
    # Apply to environments with termination exceptions
    reward[self.exception_buf, :] += self.exception_punishment_scale  # -5

    # Apply to environments with value exceptions (NaN/Inf)
    reward[self.value_exception_buf, :] += self.exception_punishment_scale  # -5

    # Track for logging
    self.reward_buffer["exception_punishment"] += self.exception_punishment_scale * \
            (self.exception_buf.sum().item() + self.value_exception_buf.sum().item())
```

### Step 3: Reset Environment

```python
# Environments with exception_buf=True are immediately reset
self.reset_buf |= self.exception_buf
self.reset_buf |= self.value_exception_buf

# Episode terminates early, agents start fresh
```

---

## Example Scenarios

### Scenario 1: Robots Collide (Good Outcome)
```
Time  | Robot Distance | Collision? | Exception? | Reward
------|---------------|------------|------------|--------
0     | 2.0m          | No         | No         | 0
10    | 1.0m          | No         | No         | -0.0025 (collision_punishment)
20    | 0.5m          | No         | No         | -0.0025
30    | 0.3m          | No         | No         | -0.0025
40    | 0.26m         | No         | No         | -0.0025
50    | 0.35m         | No         | No         | -0.0025 (agents moving apart!)
60    | 0.50m         | No         | No         | 0 (safe distance)
```
**Result**: Agents got close but didn't cross 0.25m threshold. Small penalties encourage spacing.

### Scenario 2: Robots Collide Hard (Bad Outcome)
```
Time  | Robot Distance | Collision? | Exception? | Reward
------|---------------|------------|------------|--------
0     | 2.0m          | No         | No         | 0
10    | 1.0m          | No         | No         | -0.0025
20    | 0.5m          | No         | No         | -0.0025
30    | 0.24m         | YES!       | YES!       | -5 (exception_punishment!)
      |               |            |            | EPISODE RESET
```
**Result**: Agents crossed 0.25m threshold, episode terminated with -5 penalty.

### Scenario 3: Robot Falls Over
```
Time  | Roll Angle | Pitch Angle | Exception? | Reward
------|-----------|-------------|------------|--------
0     | 0°        | 0°          | No         | 0
10    | 5°        | 3°          | No         | Normal rewards
20    | 15°       | 8°          | No         | Normal rewards
30    | 45°       | 12°         | YES!       | -5 (exception_punishment!)
      |           |             |            | EPISODE RESET
```
**Result**: Robot tipped over 45°, exceeded roll threshold, episode terminated.

### Scenario 4: Numerical Instability
```
Time  | Box Position | Observation | Exception? | Reward
------|-------------|-------------|------------|--------
0     | [12, 0, 0.3]| Valid       | No         | 0
10    | [11, 0, 0.3]| Valid       | No         | Normal rewards
20    | [10, 0, NaN]| NaN!        | YES!       | -5 (exception_punishment!)
      |             |             |            | EPISODE RESET
```
**Result**: Physics simulation became unstable, NaN value detected, episode terminated.

---

## Why Is This Important?

### 1. **Safety Constraint**
- Prevents the policy from learning dangerous behaviors
- Forces robots to maintain stability (no flipping, falling)
- Encourages gentle interactions (no hard collisions)

### 2. **Simulation Stability**
- Catches numerical errors before they corrupt training
- Prevents physics explosions from accumulating
- Keeps training data clean

### 3. **Learning Signal**
- **Large negative reward (-5)** is a strong deterrent
- Much larger than per-step rewards (0.003-0.01 range)
- Agents quickly learn to avoid these failure modes

---

## Configuration

```python
# In task/cuboid/config.py

# Termination conditions (what triggers exception_buf)
class termination(Go1Cfg.termination):
    termination_terms = [
        "roll",      # Enable/disable by adding/removing from list
        "pitch",
        "z_wave",
        "collision",
        # "far_away",  # Not enabled for mid-level task
    ]
    z_wave_kwargs = dict(threshold=0.35)      # Vertical movement limit
    collision_kwargs = dict(threshold=0.25)   # Collision distance limit

# Reward scale (how much penalty to apply)
class rewards(Go1Cfg.rewards):
    class scales:
        exception_punishment_scale = -5  # Penalty when episode terminates
```

---

## Metrics in TensorBoard

You can see exception punishment in TensorBoard under:
```
rewards/exception_punishment
```

**What the value means**:
- **0.0**: No exceptions (good!)
- **-0.001**: ~0.02% of steps triggered exceptions (very rare)
- **-0.01**: ~0.2% of steps triggered exceptions (occasional)
- **-0.1**: ~2% of steps triggered exceptions (concerning)
- **< -0.5**: Frequent failures (policy unstable)

**From HAPPO training analysis**:
```
exception_punishment: -0.0037 → -0.0043 (getting worse over time!)
```

This means:
- Initially: 0.074% of steps triggered exceptions
- Finally: 0.086% of steps triggered exceptions
- **Interpretation**: Policy is becoming slightly MORE unstable over time
- This contributes to the learning problems (agents learning unsafe behaviors)

---

## Summary

**Exception Punishment**:
- ❌ **-5 penalty** when episode terminates early
- 🛑 **Triggered by**: Roll, Pitch, Z-wave, Collision violations, NaN/Inf values
- 🎯 **Purpose**: Prevent unsafe behaviors and simulation instability
- 📊 **Typical value**: -0.003 to -0.01 per step (0.06-0.2% failure rate)
- ⚠️ **In HAPPO**: -0.004 and increasing (slight instability)

**Key Insight**: This is a **terminal reward** - only applied when the episode ends early due to failure. It's different from per-step penalties like `collision_punishment` which apply continuously.

---

**Document Complete**: 2025-12-04
