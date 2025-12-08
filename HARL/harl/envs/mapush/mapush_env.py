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
        from mqe.envs.utils import make_mqe_env
        from argparse import Namespace
        from isaacgym import gymapi

        # Create mock Isaac Gym args object
        gym_args = Namespace(
            sim_device='cuda:0',
            pipeline='gpu',
            graphics_device_id=0,
            physics_engine=gymapi.SIM_PHYSX,  # Use gymapi constant
            num_threads=0,
            subscenes=0,
            slices=0,
            use_gpu=True,
            use_gpu_pipeline=True,
            device='cuda:0',
            task=args.get('task', 'go1push_mid'),
            resume=False,
            experiment_name='harl',
            run_name='harl_run',
            load_run=-1,
            checkpoint=-1,
            headless=args.get('headless', True),
            horovod=False,
            rl_device='cuda:0',
            num_envs=args.get('n_threads', 20),
            seed=1,
            max_iterations=0,
            record_video=False
        )

        # Get task name from args or use default
        task_name = args.get('task', 'go1push_mid')

        # Create environment using MAPush's make_mqe_env function
        self.env, self.cfg = make_mqe_env(task_name, gym_args)

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
        # Convert to numpy
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()

        # HARL expects numpy arrays of shape [n_envs, n_agents, obs_dim]
        # For MAPush, shared observation is same as local observation
        share_obs = obs.copy()

        return obs, share_obs, self.get_avail_actions()

    def step(self, actions):
        """Execute one step for all agents.

        Args:
            actions: Actions array [n_envs, n_agents, action_dim]

        Returns:
            obs: Local observations [n_envs, n_agents, obs_dim]
            share_obs: Shared observations [n_envs, n_agents, obs_dim]
            rewards: Rewards for each agent [n_envs, n_agents, 1]
            dones: Done flags [n_envs, n_agents]
            infos: Info dictionaries [n_envs]
            available_actions: None for continuous
        """
        # actions is [n_envs, n_agents, action_dim]
        # Convert to tensor
        actions_tensor = torch.from_numpy(actions).to(self.env.device)

        # Step environment
        obs, rewards, dones, infos = self.env.step(actions_tensor)

        # Convert to numpy
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(dones, torch.Tensor):
            dones = dones.cpu().numpy()

        # HARL expects:
        # obs: [n_envs, n_agents, obs_dim]
        # rewards: [n_envs, n_agents, 1]
        # dones: [n_envs, n_agents]
        share_obs = obs.copy()

        # Reshape rewards to [n_envs, n_agents, 1] if needed
        if rewards.ndim == 2:
            rewards = rewards[:, :, np.newaxis]

        # Expand dones to [n_envs, n_agents] if it's [n_envs]
        if dones.ndim == 1:
            dones = np.tile(dones[:, np.newaxis], (1, self.n_agents))

        # infos: HARL expects [n_envs] list, where each element is:
        # - EP state_type: a dict with agent info in keys like 0, 1, etc.
        # - FP state_type: a list of dicts (one per agent)
        # MAPush returns a single dict, so we create list of dicts for each env
        # For EP mode, create dict with agent indices as keys
        infos_list = []
        for env_idx in range(self.n_threads):
            env_info = {}
            for agent_id in range(self.n_agents):
                env_info[agent_id] = {}  # Empty dict for each agent
            infos_list.append(env_info)

        return (obs, share_obs, rewards, dones, infos_list, self.get_avail_actions())

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
