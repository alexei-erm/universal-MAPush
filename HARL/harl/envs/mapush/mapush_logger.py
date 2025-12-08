"""Logger for MAPush environment."""
import torch
from harl.common.base_logger import BaseLogger


class MAPushLogger(BaseLogger):
    """Logger for MAPush training."""

    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        """Initialize MAPush logger."""
        super().__init__(args, algo_args, env_args, num_agents, writter, run_dir)

        # MAPush-specific metrics
        self.success_count = 0
        self.episode_count = 0

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
        # Track per-agent rewards if available
        pass

    def train_log(self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer):
        """Log training metrics."""
        # Get total steps
        self.total_num_steps = actor_buffer.step * self.algo_args["train"]["n_rollout_threads"] * self.algo_args["train"]["episode_length"]

        # Log actor training info (if available)
        if len(actor_train_infos) > 0 and actor_train_infos[0] is not None:
            for agent_id in range(self.num_agents):
                if actor_train_infos[agent_id] is not None:
                    for key in actor_train_infos[agent_id]:
                        if self.writter is not None:
                            self.writter.add_scalar(
                                f"agent{agent_id}/train_{key}",
                                actor_train_infos[agent_id][key],
                                self.total_num_steps
                            )

        # Log critic training info
        if critic_train_info is not None:
            for key in critic_train_info:
                if self.writter is not None:
                    self.writter.add_scalar(
                        f"critic/train_{key}",
                        critic_train_info[key],
                        self.total_num_steps
                    )

        # Log average episode reward from buffer
        if hasattr(actor_buffer, 'rewards'):
            avg_reward = actor_buffer.rewards.mean()
            if self.writter is not None:
                self.writter.add_scalar(
                    "train_average_reward",
                    avg_reward,
                    self.total_num_steps
                )

    def eval_init_all(self):
        """Initialize all evaluation metrics."""
        self.eval_init()
