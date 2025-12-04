# MAPPO vs HAPPO Implementation Analysis

**Date:** 2025-11-29
**Purpose:** Understand the differences between MAPPO and HAPPO in HARL and verify correctness

---

## Executive Summary

Both MAPPO and HAPPO are **correctly implemented** in HARL. The key difference is:

- **MAPPO**: Standard multi-agent PPO - agents updated independently or with shared parameters
- **HAPPO**: Heterogeneous-Agent PPO - sequential agent updates with **credit assignment factor**

The HAPPO implementation uses a **factor** that accounts for policy changes of previously updated agents, which is the core innovation of the algorithm.

---

## 1. Algorithm Comparison

### 1.1 MAPPO (Multi-Agent PPO)

**File:** `HARL/harl/algorithms/actors/mappo.py`

**Core Update Rule:**
```python
# Line 70-71: Standard PPO clipped objective
surr1 = imp_weights * adv_targ
surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

# Line 75-76: Policy loss
policy_action_loss = (
    -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
).sum() / active_masks_batch.sum()
```

**Key Characteristics:**
- ✅ No factor term - standard PPO loss
- ✅ Agents can be updated independently or with parameter sharing
- ✅ Uses importance sampling weights but no cross-agent credit assignment

### 1.2 HAPPO (Heterogeneous-Agent PPO)

**File:** `HARL/harl/algorithms/actors/happo.py`

**Core Update Rule:**
```python
# Line 71-75: PPO clipped objective with factor
surr1 = imp_weights * adv_targ
surr2 = (
    torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
    * adv_targ
)

# Line 78-81: Policy loss WITH FACTOR (key difference!)
policy_action_loss = (
    -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True)
    * active_masks_batch
).sum() / active_masks_batch.sum()
```

**Key Characteristics:**
- ✅ **Factor term** multiplies the policy gradient
- ✅ Sequential agent updates (fixed or random order)
- ✅ Factor computed from previously updated agents' policy changes

---

## 2. The "Factor" Computation (HAPPO's Innovation)

### 2.1 Where Factor is Computed

**File:** `HARL/harl/runners/on_policy_ha_runner.py` (Lines 16-124)

```python
# Line 16-23: Initialize factor to ones
factor = np.ones(
    (
        self.algo_args["train"]["episode_length"],
        self.algo_args["train"]["n_rollout_threads"],
        1,
    ),
    dtype=np.float32,
)

# Sequential update loop (Lines 47-125)
for agent_id in agent_order:
    # Line 52-54: Current agent saves the factor
    self.actor_buffer[agent_id].update_factor(factor)

    # Line 66-83: Get old action log probs BEFORE update
    old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(...)

    # Line 86-93: UPDATE THE AGENT
    actor_train_info = self.actor[agent_id].train(...)

    # Line 96-113: Get new action log probs AFTER update
    new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(...)

    # Line 116-124: UPDATE FACTOR for next agent
    factor = factor * _t2n(
        getattr(torch, self.action_aggregation)(
            torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
        ).reshape(...)
    )
```

### 2.2 Factor Interpretation

The factor represents **how much the joint action probability changed** due to previous agents' policy updates:

```
factor_after_agent_i = factor_after_agent_(i-1) × exp(new_log_prob_i - old_log_prob_i)
                      = factor_after_agent_(i-1) × (new_prob_i / old_prob_i)
```

**For 2 agents (our case):**
1. Agent 0 updates with factor = 1 (no previous updates)
2. Factor updated: `factor = π_new^0(a_0) / π_old^0(a_0)`
3. Agent 1 updates with this factor, accounting for Agent 0's policy change

**Mathematical Meaning:**
- Factor weights the advantage by the **change in joint action probability**
- This provides proper credit assignment when agents are updated sequentially
- Prevents over-updating when previous agents' policies have already changed

---

## 3. Configuration Differences

### 3.1 MAPPO Config (`mappo.yaml`)

```yaml
hidden_sizes: [128, 128]
lr: 0.0005
critic_lr: 0.0005
actor_num_mini_batch: 1
critic_num_mini_batch: 1
entropy_coef: 0.01
gamma: 0.99
share_param: True        # Can share parameters
fixed_order: True        # Not really used
```

### 3.2 HAPPO Config (`happo.yaml`)

```yaml
hidden_sizes: [256, 256]  # LARGER network (was modified)
lr: 0.001                 # HIGHER learning rate (was modified)
critic_lr: 0.001
actor_num_mini_batch: 2   # MORE mini-batches (was modified)
critic_num_mini_batch: 2
entropy_coef: 0.03        # HIGHER entropy (was modified)
gamma: 0.97               # LOWER gamma (was modified)
share_param: False        # Separate actors
fixed_order: False        # Random update order
```

**Note:** The HAPPO config has already been modified based on the analysis in `HAPPO_LEARNING_PROBLEMS_ANALYSIS.md`.

---

## 4. Runner Differences

### 4.1 MAPPO Runner

**File:** `HARL/harl/runners/on_policy_ma_runner.py`

```python
def train(self):
    # Compute advantages (same as HAPPO)
    advantages = returns - value_preds

    # Update actors (two modes)
    if self.share_param:
        # Mode 1: Parameter sharing - single network for all agents
        actor_train_info = self.actor[0].share_param_train(...)
    else:
        # Mode 2: Separate networks - independent updates
        for agent_id in range(self.num_agents):
            actor_train_info = self.actor[agent_id].train(...)

    # Update critic
    critic_train_info = self.critic.train(...)
```

**Key Points:**
- ❌ No factor computation
- ❌ No sequential updates with credit assignment
- ✅ Can use parameter sharing for efficiency
- ✅ Simpler, faster training loop

### 4.2 HAPPO Runner

**File:** `HARL/harl/runners/on_policy_ha_runner.py`

```python
def train(self):
    # Initialize factor = 1
    factor = np.ones(...)

    # Compute advantages (same as MAPPO)
    advantages = returns - value_preds

    # Sequential agent updates with factor tracking
    for agent_id in agent_order:
        self.actor_buffer[agent_id].update_factor(factor)
        old_log_probs = self.actor[agent_id].evaluate_actions(...)
        actor_train_info = self.actor[agent_id].train(...)
        new_log_probs = self.actor[agent_id].evaluate_actions(...)

        # Update factor for next agent
        factor *= exp(new_log_probs - old_log_probs)

    # Update critic
    critic_train_info = self.critic.train(...)
```

**Key Points:**
- ✅ Factor-based credit assignment
- ✅ Sequential updates (order matters)
- ✅ More complex but theoretically sounder for heterogeneous agents
- ❌ Cannot use parameter sharing (defeats the purpose)

---

## 5. Is the Implementation Correct?

### 5.1 MAPPO: ✅ CORRECT

**Evidence:**
1. Matches standard PPO loss with multi-agent batching
2. Importance sampling weights computed correctly
3. Advantage normalization appropriate for multi-agent
4. Parameter sharing implementation follows best practices

**Reference:** The implementation matches the MAPPO paper (Yu et al., 2021)

### 5.2 HAPPO: ✅ CORRECT

**Evidence:**
1. Factor computation matches HAPPO paper (Kuba et al., 2021)
2. Sequential update with proper credit assignment
3. Factor multiplied with PPO objective as specified in paper
4. Action aggregation (prod/mean) properly handled

**Reference:** The implementation matches the HAPPO paper equations

**Specific verification:**
```python
# From paper: L(θ_i) = E[min(r_i(θ) A, clip(r_i(θ), 1-ε, 1+ε) A) × factor_i]
# From code (happo.py:78-81):
policy_action_loss = (
    -torch.sum(factor_batch * torch.min(surr1, surr2), ...)  # ✅ Matches!
)
```

---

## 6. Why Might HAPPO Not Be Learning?

### 6.1 Configuration Issues (Being Addressed)

The HAPPO config has been modified, but let's verify the changes are appropriate:

| Parameter | MAPPO | HAPPO (old) | HAPPO (new) | Status |
|-----------|-------|-------------|-------------|---------|
| `lr` | 0.0005 | 0.0005 | 0.001 | ✅ Good |
| `mini_batch` | 1 | 1 | 2 | ⚠️ Still low |
| `entropy_coef` | 0.01 | 0.01 | 0.03 | ✅ Good |
| `gamma` | 0.99 | 0.99 | 0.97 | ✅ Good |
| `hidden_sizes` | [128,128] | [128,128] | [256,256] | ✅ Good |

### 6.2 Potential Algorithm-Specific Issues

**Issue 1: Action Aggregation with Continuous Actions**

Both MAPPO and HAPPO use `action_aggregation: prod`:
```python
imp_weights = getattr(torch, self.action_aggregation)(
    torch.exp(action_log_probs - old_action_log_probs),
    dim=-1,
    keepdim=True,
)
```

For continuous 3D actions (vx, vy, vyaw) per agent:
- `action_log_probs.shape = (batch, 3)`
- With `prod`: multiply 3 probability densities together
- This can create very small or very large values → numerical instability

**Potential Fix:** Try `action_aggregation: mean` instead

**Issue 2: Factor Accumulation**

With 2 agents:
```
factor = 1.0 × (π_new^0 / π_old^0)
```

If Agent 0's policy changes significantly:
- Factor could become very small (if π_new << π_old)
- Factor could become very large (if π_new >> π_old)
- This affects Agent 1's gradient scale

**Potential Problem:** Factor might be scaling gradients inappropriately

### 6.3 Task-Specific Issues

The core issue might not be the algorithm implementation but rather:

1. **Reward scales too small** (10-100x too weak) - HIGHEST PRIORITY
2. **Episode too long** (1000 steps) with heavy discounting
3. **Sparse reward signal** - agents rarely reach target

These issues would affect **both MAPPO and HAPPO**, but HAPPO might be more sensitive due to:
- Sequential updates (one agent updates then affects the other)
- Factor-based credit assignment (small rewards → small factors → weak gradients)

---

## 7. Recommended Next Steps

### Step 1: Test MAPPO on MAPush ✅ DO THIS FIRST

**Why:** If MAPPO also fails, the problem is NOT algorithm-specific

**Command:**
```bash
cd HARL
python examples/train.py \
    --algo mappo \
    --env mapush \
    --exp_name mappo_baseline \
    --n_rollout_threads 10 \
    --num_env_steps 5000000
```

**What to look for:**
- If MAPPO learns: Problem is HAPPO-specific (factor computation or config)
- If MAPPO fails: Problem is environment/reward/config (not algorithm)

### Step 2: Apply Reward Scaling

**Edit:** `task/cuboid/config.py` (multiply all rewards by 10x or 100x)

**Test both:**
```bash
# MAPPO with scaled rewards
python examples/train.py --algo mappo --env mapush --exp_name mappo_10x_rewards

# HAPPO with scaled rewards
python examples/train.py --algo happo --env mapush --exp_name happo_10x_rewards
```

### Step 3: Try Different Action Aggregation (HAPPO only)

**Edit:** `HARL/harl/configs/algos_cfgs/happo.yaml`
```yaml
action_aggregation: mean  # Was prod
```

**Test:**
```bash
python examples/train.py --algo happo --env mapush --exp_name happo_mean_agg
```

### Step 4: Debug Factor Values

Add logging to see if factor is becoming too small/large:

**Edit:** `HARL/harl/runners/on_policy_ha_runner.py` (Line 124)
```python
factor = factor * _t2n(...)
print(f"Agent {agent_id} - Factor: mean={factor.mean():.4f}, min={factor.min():.4f}, max={factor.max():.4f}")
```

---

## 8. Summary

### What We Know

1. ✅ **MAPPO implementation is correct** - standard multi-agent PPO
2. ✅ **HAPPO implementation is correct** - matches paper with factor-based credit assignment
3. ⚠️ **HAPPO config has been modified** but may need further tuning
4. ❓ **Unknown: Does MAPPO learn on MAPush?** - this will tell us if the issue is algorithm or environment

### Key Differences: MAPPO vs HAPPO

| Aspect | MAPPO | HAPPO |
|--------|-------|-------|
| **Update Order** | Independent/Parallel | Sequential |
| **Credit Assignment** | None (standard PPO) | Factor-based |
| **Parameter Sharing** | Supported | Not used |
| **Complexity** | Simpler | More complex |
| **Theory** | Works for homogeneous agents | Works for heterogeneous agents |
| **Gradient Scaling** | Standard | Scaled by factor |

### Most Likely Issues (Ranked)

1. 🔴 **Reward scales too small** (affects both algorithms equally)
2. 🟡 **HAPPO-specific: Factor accumulation** (might scale gradients poorly)
3. 🟡 **Action aggregation with continuous actions** (numerical issues)
4. 🟢 **Mini-batch count still too low** (but 2 is better than 1)

### Next Action

**RUN MAPPO FIRST** to isolate whether this is:
- Algorithm problem (HAPPO-specific)
- Environment problem (affects both)

Then proceed based on results.

---

**Analysis Complete: 2025-11-29**
