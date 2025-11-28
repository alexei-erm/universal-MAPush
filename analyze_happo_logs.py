#!/usr/bin/env python3
"""
Analyze HAPPO training logs from TensorBoard summary.json
"""
import json
import numpy as np
from pathlib import Path

# Path to the training run
LOG_DIR = Path("/home/gvlab/universal-MAPush/HARL/results/mapush/cuboid_go1push_mid/happo/quick_test/seed-00001-2025-11-27-18-33-17/logs")
SUMMARY_FILE = LOG_DIR / "summary.json"

def load_summary():
    """Load the summary.json file"""
    with open(SUMMARY_FILE, 'r') as f:
        data = json.load(f)
    return data

def extract_metric(data, metric_name):
    """Extract a specific metric from the summary data"""
    for key in data.keys():
        if metric_name in key:
            values = data[key]
            # Format: [[timestamp, step, value], ...]
            steps = [v[1] for v in values]
            vals = [v[2] for v in values]
            return steps, vals
    return None, None

def analyze_metrics():
    """Analyze all metrics and provide insights"""
    data = load_summary()

    print("="*80)
    print("HAPPO Training Analysis - 20M Steps")
    print("="*80)
    print()

    # 1. Episode Rewards
    print("1. EPISODE REWARDS")
    print("-" * 80)
    steps, rewards = extract_metric(data, "train_episode_rewards")
    if rewards:
        rewards_arr = np.array(rewards)
        print(f"Total updates: {len(rewards)}")
        print(f"Initial reward (first 10 avg): {np.mean(rewards[:10]):.3f}")
        print(f"Final reward (last 10 avg): {np.mean(rewards[-10:]):.3f}")
        print(f"Best reward: {np.max(rewards):.3f} at step {steps[np.argmax(rewards)]}")
        print(f"Worst reward: {np.min(rewards):.3f} at step {steps[np.argmin(rewards)]}")
        print(f"Mean reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")

        # Check for improvement
        first_quarter = np.mean(rewards[:len(rewards)//4])
        last_quarter = np.mean(rewards[3*len(rewards)//4:])
        improvement = last_quarter - first_quarter
        print(f"Improvement (last 25% vs first 25%): {improvement:.3f}")
        if improvement < 1.0:
            print("⚠️  WARNING: Very little improvement in episode rewards!")
    print()

    # 2. Task Performance Metrics
    print("2. TASK PERFORMANCE METRICS")
    print("-" * 80)

    # Success Rate
    steps, success_rate = extract_metric(data, "success_rate/success_rate")
    if success_rate:
        sr_arr = np.array(success_rate)
        print(f"Success Rate:")
        print(f"  Initial (first 10 avg): {np.mean(success_rate[:10])*100:.2f}%")
        print(f"  Final (last 10 avg): {np.mean(success_rate[-10:])*100:.2f}%")
        print(f"  Best: {np.max(success_rate)*100:.2f}%")
        print(f"  Mean: {np.mean(success_rate)*100:.2f}% ± {np.std(success_rate)*100:.2f}%")
        if np.mean(success_rate[-10:]) < 0.05:
            print("  ❌ CRITICAL: Success rate < 5% - agents not learning task!")

    # Distance to Target
    steps, distance = extract_metric(data, "distance_to_target/distance_to_target")
    if distance:
        print(f"\nDistance to Target (meters):")
        print(f"  Initial (first 10 avg): {np.mean(distance[:10]):.3f}")
        print(f"  Final (last 10 avg): {np.mean(distance[-10:]):.3f}")
        print(f"  Best: {np.min(distance):.3f}")
        print(f"  Mean: {np.mean(distance):.3f} ± {np.std(distance):.3f}")
        if np.mean(distance[-10:]) > 3.0:
            print("  ⚠️  WARNING: Distance still > 3m - not getting close to target!")

    # Collision Rate
    steps, collision = extract_metric(data, "collision_rate/collision_rate")
    if collision:
        print(f"\nCollision Rate:")
        print(f"  Initial (first 10 avg): {np.mean(collision[:10])*100:.2f}%")
        print(f"  Final (last 10 avg): {np.mean(collision[-10:])*100:.2f}%")
        print(f"  Mean: {np.mean(collision)*100:.2f}% ± {np.std(collision)*100:.2f}%")
        if np.mean(collision[-10:]) > 0.5:
            print("  ⚠️  WARNING: High collision rate - agents interfering with each other!")
    print()

    # 3. Reward Components
    print("3. REWARD COMPONENTS BREAKDOWN")
    print("-" * 80)

    reward_components = {
        "distance_to_target": "Distance to Target Reward",
        "approach_to_box": "Approach to Box Reward",
        "collision_punishment": "Collision Punishment",
        "reach_target": "Reach Target Bonus",
        "push_reward": "Push Reward",
        "ocb_reward": "OCB Positioning Reward",
        "exception_punishment": "Exception Punishment"
    }

    for key, name in reward_components.items():
        steps, vals = extract_metric(data, f"rewards/{key}")
        if vals:
            vals_arr = np.array(vals)
            print(f"{name}:")
            print(f"  Initial: {np.mean(vals[:10]):.4f}, Final: {np.mean(vals[-10:]):.4f}")
            print(f"  Mean: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print()

    # 4. Agent-Specific Metrics
    print("4. AGENT-SPECIFIC LEARNING METRICS")
    print("-" * 80)

    for agent_id in [0, 1]:
        print(f"\nAgent {agent_id}:")

        # Policy Loss
        steps, policy_loss = extract_metric(data, f"agent{agent_id}/policy_loss/agent{agent_id}/policy_loss")
        if policy_loss:
            pl_arr = np.array(policy_loss)
            print(f"  Policy Loss: {np.mean(pl_arr):.6f} ± {np.std(pl_arr):.6f}")
            if abs(np.mean(pl_arr)) < 0.0001:
                print(f"    ⚠️  WARNING: Very small policy loss - might not be learning!")

        # Entropy
        steps, entropy = extract_metric(data, f"agent{agent_id}/dist_entropy/agent{agent_id}/dist_entropy")
        if entropy:
            ent_arr = np.array(entropy)
            print(f"  Entropy: {np.mean(ent_arr):.4f} ± {np.std(ent_arr):.4f}")
            print(f"    Initial: {np.mean(entropy[:10]):.4f}, Final: {np.mean(entropy[-10:]):.4f}")
            if np.mean(entropy[-10:]) < 0.1:
                print(f"    ⚠️  WARNING: Very low entropy - policy may have collapsed!")

        # Gradient Norm
        steps, grad_norm = extract_metric(data, f"agent{agent_id}/actor_grad_norm/agent{agent_id}/actor_grad_norm")
        if grad_norm:
            gn_arr = np.array(grad_norm)
            print(f"  Gradient Norm: {np.mean(gn_arr):.4f} ± {np.std(gn_arr):.4f}")
            if np.mean(gn_arr) < 0.001:
                print(f"    ⚠️  WARNING: Very small gradients - vanishing gradient problem!")
            elif np.mean(gn_arr) > 10.0:
                print(f"    ⚠️  WARNING: Very large gradients - exploding gradient problem!")

        # Ratio (importance sampling weight)
        steps, ratio = extract_metric(data, f"agent{agent_id}/ratio/agent{agent_id}/ratio")
        if ratio:
            ratio_arr = np.array(ratio)
            print(f"  Importance Ratio: {np.mean(ratio_arr):.4f} ± {np.std(ratio_arr):.4f}")
            if np.mean(ratio_arr) < 0.5 or np.mean(ratio_arr) > 2.0:
                print(f"    ⚠️  WARNING: Ratio far from 1.0 - policy changing too fast!")
    print()

    # 5. Critic Metrics
    print("5. CRITIC (VALUE FUNCTION) METRICS")
    print("-" * 80)

    # Value Loss
    steps, value_loss = extract_metric(data, "critic/value_loss/critic/value_loss")
    if value_loss:
        vl_arr = np.array(value_loss)
        print(f"Value Loss:")
        print(f"  Initial: {np.mean(value_loss[:10]):.4f}, Final: {np.mean(value_loss[-10:]):.4f}")
        print(f"  Mean: {np.mean(vl_arr):.4f} ± {np.std(vl_arr):.4f}")
        if np.mean(vl_arr) > 100.0:
            print(f"  ⚠️  WARNING: Very high value loss - value function not converging!")

    # Critic Gradient Norm
    steps, critic_grad = extract_metric(data, "critic/critic_grad_norm/critic/critic_grad_norm")
    if critic_grad:
        cg_arr = np.array(critic_grad)
        print(f"Critic Gradient Norm:")
        print(f"  Mean: {np.mean(cg_arr):.4f} ± {np.std(cg_arr):.4f}")
        if np.mean(cg_arr) > 10.0:
            print(f"  ⚠️  WARNING: Large critic gradients - value function unstable!")

    # Average Step Rewards
    steps, step_rewards = extract_metric(data, "critic/average_step_rewards/critic/average_step_rewards")
    if step_rewards:
        sr_arr = np.array(step_rewards)
        print(f"Average Step Rewards:")
        print(f"  Initial: {np.mean(step_rewards[:10]):.4f}, Final: {np.mean(step_rewards[-10:]):.4f}")
        print(f"  Mean: {np.mean(sr_arr):.4f} ± {np.std(sr_arr):.4f}")
    print()

    # 6. Summary and Recommendations
    print("="*80)
    print("DIAGNOSIS AND RECOMMENDATIONS")
    print("="*80)

    # Calculate key indicators
    success_improving = False
    reward_improving = False

    if success_rate:
        sr_improvement = np.mean(success_rate[-len(success_rate)//4:]) - np.mean(success_rate[:len(success_rate)//4])
        success_improving = sr_improvement > 0.05

    if rewards:
        reward_improvement = np.mean(rewards[-len(rewards)//4:]) - np.mean(rewards[:len(rewards)//4])
        reward_improving = reward_improvement > 1.0

    print(f"\n📊 Overall Learning Status:")
    print(f"  - Success rate improving: {'✅ Yes' if success_improving else '❌ No'}")
    print(f"  - Rewards improving: {'✅ Yes' if reward_improving else '❌ No'}")

    if not success_improving and not reward_improving:
        print("\n🚨 CRITICAL ISSUES DETECTED:")
        print("  The agents are NOT learning the task effectively!")
        print("\n💡 LIKELY PROBLEMS:")

        # Check various issues
        if entropy and np.mean(entropy[-10:]) < 0.5:
            print("  1. LOW ENTROPY - Policy has low exploration")
            print("     → Increase entropy_coef from 0.01 to 0.05 or 0.1")

        if policy_loss and abs(np.mean(policy_loss)) < 0.001:
            print("  2. SMALL POLICY LOSS - Weak learning signal")
            print("     → Check if rewards are too sparse or small")
            print("     → Consider increasing learning rate from 0.0005 to 0.001")

        if distance and np.mean(distance[-10:]) > 3.0:
            print("  3. NOT REACHING TARGET - Agents stay far from goal")
            print("     → Rewards may be too sparse")
            print("     → Check if approach_to_box and distance rewards are strong enough")

        if collision and np.mean(collision[-10:]) > 0.3:
            print("  4. HIGH COLLISIONS - Poor coordination")
            print("     → HAPPO factor computation may be incorrect")
            print("     → Check multi-agent credit assignment")

        print("\n🔧 RECOMMENDED ACTIONS:")
        print("  1. Increase exploration: entropy_coef = 0.05 (currently 0.01)")
        print("  2. Increase learning rate: lr = 0.001 (currently 0.0005)")
        print("  3. More mini-batches: actor_num_mini_batch = 4 (currently 1)")
        print("  4. Check reward scales in task/cuboid/config.py")
        print("  5. Verify HAPPO factor computation is correct for 2-agent task")
        print("  6. Try larger network: hidden_sizes = [256, 256] (currently [128, 128])")
    else:
        print("\n✅ Learning is progressing, but may be slow")
        print("   Consider tuning hyperparameters for faster convergence")

    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_metrics()
