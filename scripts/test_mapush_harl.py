"""Test script to verify MAPush environment integration with HARL."""

# CRITICAL: Import isaacgym BEFORE any other imports that use torch
import isaacgym

import sys
import os

# Add HARL to path
harl_path = os.path.join(os.path.dirname(__file__), "HARL")
sys.path.insert(0, harl_path)

def test_environment_import():
    """Test if MAPush environment can be imported."""
    print("=" * 60)
    print("TEST 1: Import MAPush environment")
    print("=" * 60)
    try:
        from harl.envs.mapush.mapush_env import MAPushEnv
        from harl.envs.mapush.mapush_logger import MAPushLogger
        print("✓ Successfully imported MAPushEnv and MAPushLogger")
        return True
    except Exception as e:
        print(f"✗ Failed to import: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_instantiation():
    """Test if MAPush environment can be created."""
    print("\n" + "=" * 60)
    print("TEST 2: Instantiate MAPush environment")
    print("=" * 60)
    try:
        from harl.envs.mapush.mapush_env import MAPushEnv

        env_args = {
            "n_threads": 10,  # Isaac Gym needs at least 4-10 environments
            "task": "go1push_mid",
            "object_type": "cuboid",
            "headless": True,
        }

        print(f"Creating environment with args: {env_args}")
        env = MAPushEnv(env_args)
        print(f"✓ Environment created successfully")
        print(f"  - n_threads: {env.n_threads}")
        print(f"  - n_agents: {env.n_agents}")
        print(f"  - observation_space: {env.observation_space}")
        print(f"  - action_space: {env.action_space}")

        return True, env
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_environment_reset(env):
    """Test if environment can be reset."""
    print("\n" + "=" * 60)
    print("TEST 3: Reset environment")
    print("=" * 60)
    try:
        obs, share_obs, available_actions = env.reset()
        print(f"✓ Environment reset successfully")
        print(f"  - obs shape: {obs.shape}")
        print(f"  - share_obs shape: {share_obs.shape}")
        print(f"  - obs dtype: {obs.dtype}")
        print(f"  - obs min/max: [{obs.min():.2f}, {obs.max():.2f}]")

        # Verify shapes
        expected_obs_shape = (env.n_threads, env.n_agents, 8)  # 8 = 2 + 3*2
        if obs.shape == expected_obs_shape:
            print(f"  ✓ Observation shape matches expected: {expected_obs_shape}")
        else:
            print(f"  ✗ Observation shape mismatch!")
            print(f"    Expected: {expected_obs_shape}")
            print(f"    Got: {obs.shape}")
            return False

        return True
    except Exception as e:
        print(f"✗ Failed to reset environment: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_step(env):
    """Test if environment can step."""
    print("\n" + "=" * 60)
    print("TEST 4: Step environment")
    print("=" * 60)
    try:
        import numpy as np

        # Reset first
        obs, share_obs, available_actions = env.reset()

        # Create random actions
        actions = np.random.uniform(-1, 1, size=(env.n_threads, env.n_agents, 3))
        print(f"Taking step with actions shape: {actions.shape}")

        obs, share_obs, rewards, dones, infos, available_actions = env.step(actions)

        print(f"✓ Environment step successful")
        print(f"  - obs shape: {obs.shape}")
        print(f"  - rewards shape: {rewards.shape}")
        print(f"  - dones shape: {dones.shape}")
        print(f"  - rewards dtype: {rewards.dtype}")
        print(f"  - dones dtype: {dones.dtype}")
        print(f"  - infos length: {len(infos)}")
        print(f"  - rewards range: [{rewards.min():.4f}, {rewards.max():.4f}]")

        # Verify shapes
        expected_reward_shape = (env.n_threads, env.n_agents, 1)
        expected_done_shape = (env.n_threads, env.n_agents, 1)

        if rewards.shape == expected_reward_shape:
            print(f"  ✓ Rewards shape matches expected: {expected_reward_shape}")
        else:
            print(f"  ✗ Rewards shape mismatch!")
            print(f"    Expected: {expected_reward_shape}")
            print(f"    Got: {rewards.shape}")
            return False

        if dones.shape == expected_done_shape:
            print(f"  ✓ Dones shape matches expected: {expected_done_shape}")
        else:
            print(f"  ✗ Dones shape mismatch!")
            print(f"    Expected: {expected_done_shape}")
            print(f"    Got: {dones.shape}")
            return False

        if dones.dtype == bool:
            print(f"  ✓ Dones dtype is boolean")
        else:
            print(f"  ✗ Dones dtype is not boolean: {dones.dtype}")
            return False

        return True
    except Exception as e:
        print(f"✗ Failed to step environment: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_steps(env, num_steps=10):
    """Test multiple environment steps."""
    print("\n" + "=" * 60)
    print(f"TEST 5: Run {num_steps} steps")
    print("=" * 60)
    try:
        import numpy as np

        obs, share_obs, available_actions = env.reset()

        rewards_list = []
        for i in range(num_steps):
            actions = np.random.uniform(-1, 1, size=(env.n_threads, env.n_agents, 3))
            obs, share_obs, rewards, dones, infos, available_actions = env.step(actions)
            rewards_list.append(rewards.mean())

            if i % 5 == 0:
                print(f"  Step {i}: mean reward = {rewards.mean():.4f}")

        print(f"✓ Successfully ran {num_steps} steps")
        print(f"  - Average reward over all steps: {np.mean(rewards_list):.4f}")

        return True
    except Exception as e:
        print(f"✗ Failed during multi-step test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("MAPush-HARL Integration Test Suite")
    print("="*60 + "\n")

    # Test 1: Import
    if not test_environment_import():
        print("\n✗ FAILED: Cannot proceed without successful import")
        return

    # Test 2: Instantiation
    success, env = test_environment_instantiation()
    if not success:
        print("\n✗ FAILED: Cannot proceed without successful instantiation")
        return

    # Test 3: Reset
    if not test_environment_reset(env):
        print("\n✗ FAILED: Reset test failed")
        env.close()
        return

    # Test 4: Step
    if not test_environment_step(env):
        print("\n✗ FAILED: Step test failed")
        env.close()
        return

    # Test 5: Multiple steps
    if not test_multiple_steps(env, num_steps=10):
        print("\n✗ FAILED: Multi-step test failed")
        env.close()
        return

    # Cleanup
    print("\n" + "=" * 60)
    print("Closing environment...")
    print("=" * 60)
    env.close()

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nMAPush environment is successfully integrated with HARL.")
    print("You can now train with:")
    print("  cd HARL")
    print("  python examples/train.py --algo happo --env mapush --exp_name test")


if __name__ == "__main__":
    main()
