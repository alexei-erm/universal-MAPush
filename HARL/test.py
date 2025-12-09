"""Test script for HAPPO/MAPPO with viewer and calculator modes."""
import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add HARL to path
harl_path = Path(__file__).parent
if str(harl_path) not in sys.path:
    sys.path.insert(0, str(harl_path))

from harl.utils.configs_tools import get_defaults_yaml_args, update_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test HAPPO/MAPPO models')

    # Mode selection
    parser.add_argument('--mode', type=str, required=True, choices=['viewer', 'calculator'],
                        help='Test mode: viewer (render episodes) or calculator (compute metrics)')

    # Model loading
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing model checkpoints (e.g., ./results/.../models/10M)')
    parser.add_argument('--algo', type=str, default='happo', choices=['happo', 'mappo'],
                        help='Algorithm to test')
    parser.add_argument('--env', type=str, default='mapush',
                        help='Environment name')

    # Viewer mode settings
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of episodes to render/evaluate')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')

    # Calculator mode settings
    parser.add_argument('--num_threads', type=int, default=500,
                        help='Number of parallel environments for calculator mode')

    return parser.parse_args()


def load_model(checkpoint_dir, args, algo_args, env_args):
    """Load model from checkpoint."""
    print(f"\n{'='*60}")
    print(f"Loading checkpoint from: {checkpoint_dir}")
    print(f"{'='*60}\n")

    # Set model_dir in algo_args
    algo_args["train"]["model_dir"] = checkpoint_dir

    # Import runner
    from harl.runners import RUNNER_REGISTRY

    # Create runner with model loading
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)

    return runner


def viewer_mode(args):
    """Run viewer mode - render episodes with visualization."""
    print(f"\n{'='*60}")
    print(f"VIEWER MODE: Rendering {args.num_episodes} episodes")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}\n")

    # Load configs
    algo_args, env_args = get_defaults_yaml_args(args.algo, args.env)

    # Override settings for viewer mode
    algo_args["render"]["use_render"] = True
    algo_args["render"]["render_episodes"] = args.num_episodes
    algo_args["seed"]["seed"] = args.seed

    # Ensure we're using the right environment settings
    if args.env == "mapush":
        env_args["headless"] = False  # Show visualization
        env_args["task"] = "cuboid_go1push_mid"

    # Convert to dict format
    args_dict = {
        "algo": args.algo,
        "env": args.env,
        "exp_name": "test_viewer"
    }

    # Load model
    runner = load_model(args.checkpoint_dir, args_dict, algo_args, env_args)

    # Run rendering
    print("Starting rendering...\n")
    runner.run()

    print(f"\n{'='*60}")
    print(f"Viewer mode completed!")
    print(f"{'='*60}\n")


def calculator_mode(args):
    """Run calculator mode - compute metrics without visualization."""
    print(f"\n{'='*60}")
    print(f"CALCULATOR MODE: Evaluating on {args.num_threads} parallel environments")
    print(f"Episodes: {args.num_episodes}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}\n")

    # Load configs
    algo_args, env_args = get_defaults_yaml_args(args.algo, args.env)

    # Override settings for calculator mode
    algo_args["render"]["use_render"] = False
    algo_args["render"]["use_calc_mode"] = True
    algo_args["render"]["calc_n_threads"] = args.num_threads
    algo_args["seed"]["seed"] = args.seed

    # Ensure we're using the right environment settings
    if args.env == "mapush":
        env_args["headless"] = True  # No visualization
        env_args["task"] = "cuboid_go1push_mid"
        env_args["n_threads"] = args.num_threads

    # Convert to dict format
    args_dict = {
        "algo": args.algo,
        "env": args.env,
        "exp_name": "test_calculator"
    }

    # Load model
    runner = load_model(args.checkpoint_dir, args_dict, algo_args, env_args)

    # Run evaluation
    print("Starting evaluation...\n")

    # Track metrics
    total_episodes = 0
    successful_episodes = 0
    episode_rewards = []
    episode_lengths = []
    collision_degrees = []
    collaboration_degrees = []

    # Reset environment
    obs, share_obs, available_actions = runner.envs.reset()

    # Run episodes
    episode_count = np.zeros(args.num_threads)
    current_episode_reward = np.zeros(args.num_threads)
    current_episode_length = np.zeros(args.num_threads)

    max_steps_per_episode = 1000  # Safety limit
    step = 0

    with torch.no_grad():
        while total_episodes < args.num_episodes:
            step += 1

            # Get actions from policy
            actions_collector = []
            for agent_id in range(runner.num_agents):
                actor = runner.actor[agent_id]
                actor.prep_rollout()

                # Convert to tensor
                obs_agent = torch.tensor(obs[:, agent_id], dtype=torch.float32).to(runner.device)
                rnn_states = torch.zeros((args.num_threads, 1, 64), dtype=torch.float32).to(runner.device)
                masks = torch.ones((args.num_threads, 1), dtype=torch.float32).to(runner.device)

                # Get action
                action, _, _ = actor.get_actions(obs_agent, rnn_states, masks,
                                                  available_actions=None, deterministic=True)
                actions_collector.append(action)

            # Stack actions
            actions = np.stack([a.cpu().numpy() for a in actions_collector], axis=1)

            # Step environment
            obs, share_obs, rewards, dones, infos, available_actions = runner.envs.step(actions)

            # Update metrics
            current_episode_reward += rewards[:, 0, 0]  # Sum over first agent
            current_episode_length += 1

            # Check for done episodes
            for env_idx in range(args.num_threads):
                if dones[env_idx, 0] and episode_count[env_idx] < args.num_episodes:
                    episode_count[env_idx] += 1
                    total_episodes += 1

                    # Store episode metrics
                    episode_rewards.append(current_episode_reward[env_idx])
                    episode_lengths.append(current_episode_length[env_idx])

                    # Check success (if environment has finished buffer)
                    if hasattr(runner.envs, 'init_finished_buf'):
                        if runner.envs.init_finished_buf[env_idx]:
                            successful_episodes += 1

                    # Store additional metrics if available
                    if hasattr(runner.envs, 'collision_degree_buf'):
                        collision_degrees.append(runner.envs.collision_degree_buf[env_idx].item())
                    if hasattr(runner.envs, 'collaboration_degree_buf'):
                        collaboration_degrees.append(runner.envs.collaboration_degree_buf[env_idx].item())

                    # Reset counters for this environment
                    current_episode_reward[env_idx] = 0
                    current_episode_length[env_idx] = 0

                    # Progress update
                    if total_episodes % 10 == 0:
                        print(f"Progress: {total_episodes}/{args.num_episodes} episodes completed")

            # Safety check
            if step > max_steps_per_episode * args.num_episodes:
                print("Warning: Max steps reached, stopping evaluation")
                break

    # Compute statistics
    success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0.0
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
    std_reward = np.std(episode_rewards) if episode_rewards else 0.0
    avg_length = np.mean(episode_lengths) if episode_lengths else 0.0

    # Print results
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total Episodes:        {total_episodes}")
    print(f"Successful Episodes:   {successful_episodes}")
    print(f"Success Rate:          {success_rate*100:.2f}%")
    print(f"Average Reward:        {avg_reward:.3f} ± {std_reward:.3f}")
    print(f"Average Episode Length: {avg_length:.1f} steps")

    if collision_degrees:
        print(f"Collision Degree:      {np.mean(collision_degrees):.3f} ± {np.std(collision_degrees):.3f}")
    if collaboration_degrees:
        print(f"Collaboration Degree:  {np.mean(collaboration_degrees):.3f} ± {np.std(collaboration_degrees):.3f}")

    print(f"{'='*60}\n")

    # Save results to file
    results_file = os.path.join(os.path.dirname(args.checkpoint_dir), "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Checkpoint: {args.checkpoint_dir}\n")
        f.write(f"Total Episodes: {total_episodes}\n")
        f.write(f"Successful Episodes: {successful_episodes}\n")
        f.write(f"Success Rate: {success_rate*100:.2f}%\n")
        f.write(f"Average Reward: {avg_reward:.3f} ± {std_reward:.3f}\n")
        f.write(f"Average Episode Length: {avg_length:.1f} steps\n")
        if collision_degrees:
            f.write(f"Collision Degree: {np.mean(collision_degrees):.3f} ± {np.std(collision_degrees):.3f}\n")
        if collaboration_degrees:
            f.write(f"Collaboration Degree: {np.mean(collaboration_degrees):.3f} ± {np.std(collaboration_degrees):.3f}\n")

    print(f"Results saved to: {results_file}")


def main():
    """Main function."""
    args = parse_args()

    # Verify checkpoint directory exists
    if not os.path.exists(args.checkpoint_dir):
        print(f"Error: Checkpoint directory does not exist: {args.checkpoint_dir}")
        sys.exit(1)

    # Check if checkpoint files exist
    required_files = [f"actor_agent{i}.pt" for i in range(2)]  # Assuming 2 agents
    required_files.append("critic_agent.pt")

    missing_files = []
    for f in required_files:
        if not os.path.exists(os.path.join(args.checkpoint_dir, f)):
            missing_files.append(f)

    if missing_files:
        print(f"Warning: Some checkpoint files missing: {missing_files}")
        print("Continuing anyway...\n")

    # Run appropriate mode
    if args.mode == 'viewer':
        viewer_mode(args)
    elif args.mode == 'calculator':
        calculator_mode(args)
    else:
        print(f"Error: Unknown mode '{args.mode}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
