"""Logger for MAPush environment."""
import torch
from harl.common.base_logger import BaseLogger


class MAPushLogger(BaseLogger):
    """Logger for MAPush training."""

    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir, env=None):
        """Initialize MAPush logger."""
        super().__init__(args, algo_args, env_args, num_agents, writter, run_dir)

        # Store environment reference for accessing reward buffers
        self.env = env

        # MAPush-specific metrics
        self.success_count = 0
        self.episode_count = 0

        # Cumulative success tracking
        self.total_episodes = 0
        self.successful_episodes = 0
        self.cumulative_success_rate = 0.0

    def get_task_name(self):
        """Get the task name for logging."""
        return self.env_args.get("task", "cuboid_go1push_mid")

    def eval_init(self):
        """Initialize evaluation metrics."""
        self.total_num_episodes = 0
        self.eval_episode_rewards = []
        self.one_episode_rewards = [0 for _ in range(self.num_agents)]
        self.success_count = 0

    def eval_per_step(self, rewards, dones, infos):
        """Update per-step evaluation metrics."""
        for i in range(self.num_agents):
            self.one_episode_rewards[i] += rewards[i].mean()

        # Check if episode is done
        if dones[0].any():
            self.total_num_episodes += dones[0].sum()
            # Store episode reward
            self.eval_episode_rewards.append(sum(self.one_episode_rewards))
            self.one_episode_rewards = [0 for _ in range(self.num_agents)]

            # Track successes if info contains finished_buf
            if 'finished_buf' in infos[0]:
                self.success_count += infos[0]['finished_buf'].sum().item()

    def eval_thread_done(self, total_num_steps):
        """Called when evaluation thread is done."""
        pass

    def eval_log(self, total_num_steps):
        """Log evaluation metrics."""
        eval_avg_rew = sum(self.eval_episode_rewards) / len(self.eval_episode_rewards) if len(self.eval_episode_rewards) > 0 else 0

        print(f"Evaluation at step {total_num_steps}:")
        print(f"  Average episode reward: {eval_avg_rew:.3f}")

        if self.total_num_episodes > 0:
            success_rate = self.success_count / self.total_num_episodes
            print(f"  Success rate: {success_rate:.3f}")

            if self.writter is not None:
                self.writter.add_scalar("eval_success_rate", success_rate, total_num_steps)

        if self.writter is not None:
            self.writter.add_scalar("eval_average_episode_reward", eval_avg_rew, total_num_steps)

    def train_init(self):
        """Initialize training metrics."""
        self.total_num_steps = 0
        self.last_battles_game = 0
        self.last_battles_won = 0

    def train_per_step(self, rewards, dones, infos):
        """Update per-step training metrics."""
        # Track episodes for cumulative success rate
        if dones[0].any():
            for env_idx in range(len(infos)):
                if dones[env_idx, 0]:  # Episode done for this environment
                    self.total_episodes += 1
                    # Check if successful (from info dict or finished_buf)
                    if 'finished_buf' in infos[env_idx]:
                        if infos[env_idx]['finished_buf']:
                            self.successful_episodes += 1
                    self.cumulative_success_rate = self.successful_episodes / max(1, self.total_episodes)

    def log_train(self, actor_train_infos, critic_train_info):
        """Log training metrics (called by base episode_log)."""
        # Call parent implementation for actor/critic logging
        super().log_train(actor_train_infos, critic_train_info)

        # Now add our custom reward component logging
        # (parent already logs actor/critic stuff)

        # Log cumulative success rate
        if self.writter is not None:
            self.writter.add_scalar(
                "metrics/cumulative_success_rate",
                self.cumulative_success_rate,
                self.total_num_steps
            )

        # Log reward components from environment
        if self.env is not None and hasattr(self.env, 'env') and hasattr(self.env.env, 'reward_buffer'):
            reward_buffer = self.env.env.reward_buffer
            step_count = max(1, reward_buffer.get('step_count', 1))

            # List of all reward components to log (using actual keys from wrapper)
            components = [
                'push_reward',
                'engagement_bonus',
                'cooperation_bonus',
                'directional_progress',
                'reach_target_reward',
                'exception_punishment',
                'blocking_penalty',
                'same_side_bonus',
                'approach_to_box_reward',
                'distance_to_target_reward',
                'collision_punishment',
                'ocb_reward',
            ]

            # Log each component
            for component in components:
                if component in reward_buffer:
                    value = reward_buffer[component]
                    # Convert tensor to float if needed
                    if hasattr(value, 'item'):
                        value = value.item()
                    avg_value = float(value) / step_count
                    if self.writter is not None:
                        self.writter.add_scalar(
                            f"rewards/{component}",
                            avg_value,
                            self.total_num_steps
                        )

            # Log instantaneous success rate
            if hasattr(self.env.env, 'init_finished_buf'):
                success_rate = self.env.env.init_finished_buf.float().mean().item()
                if self.writter is not None:
                    self.writter.add_scalar(
                        "metrics/success_rate",
                        success_rate,
                        self.total_num_steps
                    )

            # Log distance to target
            if hasattr(self.env.env, 'distance_to_target'):
                avg_distance = self.env.env.distance_to_target.mean().item()
                if self.writter is not None:
                    self.writter.add_scalar(
                        "metrics/distance_to_target",
                        avg_distance,
                        self.total_num_steps
                    )

    def eval_init_all(self):
        """Initialize all evaluation metrics."""
        self.eval_init()
