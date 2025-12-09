"""Test/evaluate trained HARL models.

This script supports two modes:
1. Calculator mode (--mode calc): Compute metrics using many parallel environments
2. Render mode (--mode render): Visualize agent behavior

Usage:
    # Test single checkpoint with calculator mode
    python test.py --algo happo --env mapush --model_dir ./results/.../models/80M --mode calc

    # Test all checkpoints
    python test.py --algo happo --env mapush --model_dir ./results/.../models --mode calc --test_all_checkpoints True

    # Render mode
    python test.py --algo happo --env mapush --model_dir ./results/.../models/80M --mode render
"""

import argparse
import json
import sys
from pathlib import Path

from harl.utils.configs_tools import get_defaults_yaml_args, update_args


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test/evaluate trained HARL models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
        ],
        help="Algorithm name"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="mapush",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lag",
            "mapush",
        ],
        help="Environment name"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing trained models"
    )

    # Test mode arguments
    parser.add_argument(
        "--mode",
        type=str,
        default="calc",
        choices=["calc", "render"],
        help="Testing mode: calc (metrics) or render (visualization)"
    )
    parser.add_argument(
        "--test_all_checkpoints",
        type=lambda x: str(x).lower() == 'true',
        default=False,
        help="Test all checkpoint directories (10M, 20M, etc.)"
    )

    # Optional arguments
    parser.add_argument(
        "--exp_name",
        type=str,
        default="test",
        help="Experiment name for logging"
    )

    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict

    # Load default configs
    algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])

    # Update with test-specific settings
    algo_args["train"]["model_dir"] = args["model_dir"]
    update_args(unparsed_dict, algo_args, env_args)

    # Configure test mode
    if args["mode"] == "calc":
        algo_args["render"]["use_calc_mode"] = True
        algo_args["render"]["use_render"] = False
        # Use calc_n_threads for calculator mode
        calc_threads = unparsed_dict.get("calc_n_threads", algo_args["render"]["calc_n_threads"])
        algo_args["train"]["n_rollout_threads"] = calc_threads
        print(f"Running in CALCULATOR mode with {calc_threads} threads")
    elif args["mode"] == "render":
        algo_args["render"]["use_render"] = True
        algo_args["render"]["use_calc_mode"] = False
        # Use render_episodes for render mode
        render_eps = unparsed_dict.get("render_episodes", algo_args["render"]["render_episodes"])
        algo_args["render"]["render_episodes"] = render_eps
        print(f"Running in RENDER mode ({render_eps} episodes)")

    # Import Isaac Gym for mapush/dexhands
    if args["env"] == "dexhands" or args["env"] == "mapush":
        import isaacgym

    # Disable eval mode for Isaac Gym environments
    if args["env"] == "dexhands" or args["env"] == "mapush":
        algo_args["eval"]["use_eval"] = False

    # Initialize runner
    from harl.runners import RUNNER_REGISTRY
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)

    # Run evaluation
    if args["mode"] == "calc" and args["test_all_checkpoints"]:
        print("\nEvaluating all checkpoints...")
        runner.evaluate_all_checkpoints()
    else:
        print(f"\nEvaluating single checkpoint: {args['model_dir']}")
        runner.run()

    print("\nEvaluation complete!")
    runner.close()


if __name__ == "__main__":
    main()
