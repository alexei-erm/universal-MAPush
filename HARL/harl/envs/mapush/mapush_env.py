"""MAPush Environment Wrapper for HARL."""
import sys
import os
from pathlib import Path
import numpy as np
import torch


class MAPushEnv:
    """Environment wrapper for MAPush Isaac Gym environment."""

    def __init__(self, args):
        """Initialize MAPush environment.

        Args:
            args: Dictionary with environment configuration
                - task: task name (e.g., 'cuboid_go1push_mid')
                - headless: whether to run without rendering
                - n_threads: number of parallel environments
        """
        self.args = args

        # Add universal-MAPush directory to Python path
        # HARL is in universal-MAPush/HARL, so go up 3 levels
        mapush_root = Path(__file__).parent.parent.parent.parent.parent
        if str(mapush_root) not in sys.path:
            sys.path.insert(0, str(mapush_root))

        # Import MAPush environment (lazy import to avoid unnecessary dependencies)
        from task.cuboid.config import Go1PushMidCfg
        from mqe.envs.wrappers.go1_push_mid_wrapper import Go1PushMidWrapper
        from mqe.envs.go1.go1_env import Go1Env

        # Create environment configuration
        self.cfg = Go1PushMidCfg()

        # Override config with args
        if 'headless' in args:
            self.cfg.env.headless = args['headless']
        if 'n_threads' in args:
            self.cfg.env.num_envs = args['n_threads']

        # Create base environment and wrap it
        base_env = Go1Env(self.cfg)
        self.env = Go1PushMidWrapper(base_env)

        # Get environment properties
        self.n_threads = self.cfg.env.num_envs
        self.n_agents = self.cfg.env.num_agents

        # Set observation and action spaces
        self.observation_space = [self.env.observation_space] * self.n_agents
        self.share_observation_space = [self.env.observation_space] * self.n_agents
        self.action_space = [self.env.action_space] * self.n_agents

        # Continuous action space
        self.discrete = False

    def reset(self):
        """Reset all environments.

        Returns:
            obs: Local observations for each agent [n_envs, n_agents, obs_dim]
            share_obs: Shared observations (same as obs for now)
            available_actions: None for continuous action space
        """
        obs = self.env.reset()

        # obs shape from wrapper: [n_envs, n_agents, obs_dim]
        # Convert to numpy and split by agent
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()

        # HARL expects: list of [n_envs, obs_dim] arrays (one per agent)
        obs_list = [obs[:, i, :] for i in range(self.n_agents)]

        # For MAPush, shared observation is same as local observation
        share_obs_list = obs_list

        return obs_list, share_obs_list, self.get_avail_actions()

    def step(self, actions):
        """Execute one step for all agents.

        Args:
            actions: List of action arrays [n_agents, n_envs, action_dim]

        Returns:
            obs: Local observations
            share_obs: Shared observations
            rewards: Rewards for each agent
            dones: Done flags
            infos: Info dictionaries
            available_actions: None for continuous
        """
        # actions is list of [n_envs, action_dim] arrays (one per agent)
        # Convert to [n_envs, n_agents, action_dim] tensor
        actions_array = np.stack(actions, axis=1)  # [n_envs, n_agents, action_dim]
        actions_tensor = torch.from_numpy(actions_array).to(self.env.device)

        # Step environment
        obs, rewards, dones, infos = self.env.step(actions_tensor)

        # Convert to numpy
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(dones, torch.Tensor):
            dones = dones.cpu().numpy()

        # Split by agent
        # obs: [n_envs, n_agents, obs_dim] -> list of [n_envs, obs_dim]
        obs_list = [obs[:, i, :] for i in range(self.n_agents)]
        share_obs_list = obs_list

        # rewards: [n_envs, n_agents] -> list of [n_envs, 1]
        rewards_list = [rewards[:, i:i+1] for i in range(self.n_agents)]

        # dones: [n_envs] -> expand to [n_envs] for each agent
        # All agents done at same time in MAPush
        dones_list = [dones for _ in range(self.n_agents)]

        # infos: single dict -> list of dicts (one per agent)
        infos_list = [infos for _ in range(self.n_agents)]

        return (obs_list, share_obs_list, rewards_list, dones_list,
                infos_list, self.get_avail_actions())

    def get_avail_actions(self):
        """Get available actions (None for continuous action space)."""
        return None

    def close(self):
        """Close the environment."""
        # Isaac Gym doesn't need explicit close
        pass

    def render(self):
        """Render the environment."""
        # Isaac Gym handles rendering automatically based on headless flag
        pass

    def seed(self, seed):
        """Set random seed."""
        # Isaac Gym uses cfg.seed
        pass

    # Calculator mode properties (from lost implementation)
    @property
    def init_finished_buf(self):
        """Track which environments successfully completed (for calculator mode)."""
        return self.env.init_finished_buf if hasattr(self.env, 'init_finished_buf') else None

    @property
    def init_episode_length_buf(self):
        """Track episode lengths (for calculator mode)."""
        return self.env.init_episode_length_buf if hasattr(self.env, 'init_episode_length_buf') else None

    @property
    def collision_degree_buf(self):
        """Track collision degree metric (for calculator mode)."""
        return self.env.collision_degree_buf if hasattr(self.env, 'collision_degree_buf') else None

    @property
    def collaboration_degree_buf(self):
        """Track collaboration degree metric (for calculator mode)."""
        return self.env.collaboration_degree_buf if hasattr(self.env, 'collaboration_degree_buf') else None

    @property
    def init_reset_buf(self):
        """Track which environments need reset (for calculator mode)."""
        return self.env.init_reset_buf if hasattr(self.env, 'init_reset_buf') else None

    @property
    def dt(self):
        """Simulation timestep (for calculator mode)."""
        return self.env.dt if hasattr(self.env, 'dt') else 0.02
