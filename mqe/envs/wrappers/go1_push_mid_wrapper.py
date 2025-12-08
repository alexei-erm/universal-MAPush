import gym
from gym import spaces
import numpy
import torch
from copy import copy,deepcopy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper

from isaacgym.torch_utils import *

# tensor type
def rotation_matrix_2D(theta):
    theta = theta.float()
    cos_theta = torch.cos(theta)  
    sin_theta = torch.sin(theta)  

    rotation_matrices = torch.stack([
        torch.stack([cos_theta, -sin_theta], dim=1),
        torch.stack([sin_theta, cos_theta], dim=1)
    ], dim=1)

    return rotation_matrices

def euler_to_quaternion_tensor(euler_angles):
    roll = euler_angles[:, 0]
    pitch = euler_angles[:, 1]
    yaw = euler_angles[:, 2]

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    quaternion = torch.stack([qx, qy, qz, qw], dim=1)
    return quaternion

def normalize_rpy(box_rpy):

    box_rpy = box_rpy % (2 * torch.pi)
    
    return box_rpy

class Go1PushMidWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        if getattr(self.cfg.goal, "general_dist",False):
            self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(3 + 3 * self.num_agents,), dtype=float)
            pass
        else:
            self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(2 + 3 * self.num_agents,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[0.5, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)
        
        # for hard setting of reward scales (not recommended)

        # NEW: Read per-agent reward mode flag (default False for backward compatibility)
        self.use_per_agent_rewards = getattr(self.cfg.rewards, "use_per_agent_rewards", False)

        # ITERATION 11: Validation prints
        print(f"\n{'='*80}")
        print(f"[ITERATION 11 - REDUCED SHARED REWARD]")
        print(f"{'='*80}")
        print(f"use_per_agent_rewards:    {self.use_per_agent_rewards}")

        # Select reward scales based on mode (flag-dependent for backward compatibility)
        if self.use_per_agent_rewards:
            # ITERATION 2: Use per-agent scales (modified values)
            self.approach_reward_scale = getattr(self.cfg.rewards.scales, 'per_agent_approach_reward_scale', 0.0)
            self.push_reward_scale = getattr(self.cfg.rewards.scales, 'per_agent_push_reward_scale', 0.0030)
        else:
            # Original shared scales (backward compatible)
            self.approach_reward_scale = self.cfg.rewards.scales.approach_reward_scale
            self.push_reward_scale = self.cfg.rewards.scales.push_reward_scale

        # These scales are shared between both modes
        self.target_reward_scale = self.cfg.rewards.scales.target_reward_scale
        self.reach_target_reward_scale = self.cfg.rewards.scales.reach_target_reward_scale
        self.collision_punishment_scale = self.cfg.rewards.scales.collision_punishment_scale
        self.ocb_reward_scale = self.cfg.rewards.scales.ocb_reward_scale
        self.exception_punishment_scale = self.cfg.rewards.scales.exception_punishment_scale

        # ITERATION 10: Print reward scales for verification
        print(f"push_reward_scale:        {self.push_reward_scale}")
        print(f"reach_target_scale:       {self.reach_target_reward_scale}")
        if self.use_per_agent_rewards:
            engagement = getattr(self.cfg.rewards.scales, 'engagement_bonus_scale', 'NOT_FOUND')
            cooperation = getattr(self.cfg.rewards.scales, 'cooperation_bonus_scale', 'NOT_FOUND')
            blocking = getattr(self.cfg.rewards.scales, 'blocking_penalty_scale', 'NOT_FOUND')
            same_side = getattr(self.cfg.rewards.scales, 'same_side_bonus_scale', 'NOT_FOUND')
            directional = getattr(self.cfg.rewards.scales, 'directional_progress_scale', 'NOT_FOUND')
            print(f"engagement_bonus_scale:   {engagement}")
            print(f"cooperation_bonus_scale:  {cooperation}")
            print(f"blocking_penalty_scale:   {blocking}")
            print(f"same_side_bonus_scale:    {same_side}")
            print(f"directional_progress:     {directional}  ← REDUCED from 0.15 to reduce freeloading")
            print()
            print("ITERATION 11: Reduced shared reward")
            print("  Iter10 SUCCESS: Both agents push toward goal!")
            print("  Iter11 tweak: directional_progress 0.15 → 0.05 (less freeloading risk)")
            print("  Per-agent push (0.15) now 3X stronger than shared (0.05)")
        print(f"{'='*80}\n")

        # ITERATION 5: Track previous box position for directional progress
        self.prev_box_pos = None

        self.reward_buffer = {
            "distance_to_target_reward": 0,
            "exception_punishment": 0,
            "approach_to_box_reward": 0,
            "collision_punishment":0,
            "reach_target_reward":0,
            "push_reward":0,
            "ocb_reward":0,
            "directional_progress": 0,  # ITERATION 5: NEW
            "blocking_penalty": 0,      # ITERATION 9: NEW
            "same_side_bonus": 0,       # ITERATION 9: NEW
            "step_count": 0,
            "success_count": 0,         # ITERATION 10 FIX: Track cumulative successes
            "episode_count": 0,         # ITERATION 10 FIX: Track total episodes
        }

    def _init_extras(self, obs):
        return
        # self.gate_pos = obs.env_info["gate_deviation"]
        # self.gate_pos[:, 0] += self.BarrierTrack_kwargs["init"]["block_length"] + self.BarrierTrack_kwargs["gate"]["block_length"] / 2
        # self.gate_pos = self.gate_pos.unsqueeze(1).repeat(1, self.num_agents, 1)
        # self.gate_distance = self.gate_pos.reshape(-1, 2)[:, 0]


    def calc_normal_vector_for_obc_reward(self, vertex_list, pos_tensor):
        pos_tensor = pos_tensor.to(self.device)
        vertices = torch.tensor(vertex_list, device=self.device).float()
        num_vertices = vertices.shape[0]

        edges = torch.roll(vertices, -1, dims=0) - vertices
        vp = pos_tensor[:, None, :] - vertices[None, :, :]

        edges_expanded = edges[None, :, :].repeat(pos_tensor.shape[0], 1, 1)
        edge_lengths = torch.norm(edges_expanded, dim=2, keepdim=True)
        edge_unit = edges_expanded / edge_lengths
        edge_normals = torch.stack([-edge_unit[:,:,1], edge_unit[:,:,0]], dim=2)

        cross_prod = torch.abs(vp[:,:,0] * edge_unit[:,:,1] - vp[:,:,1] * edge_unit[:,:,0])
        dot_product1 = (vp * edges_expanded).sum(dim=2)
        dot_product2 = (torch.roll(vp, -1, dims=1) * edges_expanded).sum(dim=2)

        on_segment = (dot_product1 >= 0) & (dot_product2 <= 0)
        dist_to_line = torch.where(on_segment, cross_prod, torch.tensor(float('inf'), device=self.device))

        dist_to_vertex1 = torch.norm(vp, dim=2)
        dist_to_vertex2 = torch.norm(pos_tensor[:, None, :] - torch.roll(vertices, -1, dims=0)[None, :, :], dim=2)

        min_dist_each_edge, indices = torch.min(torch.stack([dist_to_line, dist_to_vertex1, dist_to_vertex2], dim=-1), dim=2)
        min_dist, indices = torch.min(min_dist_each_edge,dim=1)
        selected_normals = edge_normals[0][indices]

        return selected_normals
    
    def reset(self,next_target_pos=None):
        if getattr(self.cfg.goal, "received_goal_pos",False):
            if next_target_pos == None:
                pass
                # raise ValueError("next_target_pos is required when received_goal_pos is True")
            self.next_target_pos = next_target_pos

        obs_buf = self.env.reset()

        # get agent state
        base_pos = deepcopy(obs_buf.base_pos) 
        base_rpy = deepcopy(obs_buf.base_rpy) 
        # get box state and target pos
        npc_pos = self.root_states_npc[:, :3].reshape(self.num_envs, self.num_npcs, -1)
        box_pos = npc_pos[:,0,:] - self.env.env_origins
        target_pos = npc_pos[:,1,:] - self.env.env_origins 
        box_qyaternion = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0 , 3:7]
        box_rpy = torch.stack(get_euler_xyz(box_qyaternion), dim=1)
        target_qyaternion = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 1 , 3:7]
        target_rpy = torch.stack(get_euler_xyz(target_qyaternion), dim=1)

        # rotate box state and target pos to agent's local state
        box_pos = box_pos.repeat_interleave(self.num_agents, dim=0)
        target_pos = target_pos.repeat_interleave(self.num_agents, dim=0)
        box_rpy = box_rpy.repeat_interleave(self.num_agents, dim=0)
        target_rpy = target_rpy.repeat_interleave(self.num_agents, dim=0)
        rotated_box_pos = torch.stack([(box_pos[:, 0] - base_pos[:, 0]) * torch.cos(-base_rpy[:, 2]) - (box_pos[:, 1] - base_pos[:, 1]) * torch.sin(-base_rpy[:, 2]),
                                       (box_pos[:, 0] - base_pos[:, 0]) * torch.sin(-base_rpy[:, 2]) + (box_pos[:, 1] - base_pos[:, 1]) * torch.cos(-base_rpy[:, 2]),
                                      box_pos[:, 2]], dim=1)
        rotated_target_pos = torch.stack([(target_pos[:, 0] - base_pos[:, 0]) * torch.cos(-base_rpy[:, 2]) - (target_pos[:, 1] - base_pos[:, 1]) * torch.sin(-base_rpy[:, 2]),
                                          (target_pos[:, 0] - base_pos[:, 0]) * torch.sin(-base_rpy[:, 2]) + (target_pos[:, 1] - base_pos[:, 1]) * torch.cos(-base_rpy[:, 2]),
                                         target_pos[:, 2]], dim=1)
        rotated_box_rpy = deepcopy(box_rpy)
        rotated_box_rpy[:,2] = box_rpy[:,2] - base_rpy[:,2]
        rotated_box_rpy = normalize_rpy(rotated_box_rpy)
        rotated_target_rpy = deepcopy(target_rpy)
        rotated_target_rpy[:,2] = target_rpy[:,2] - base_rpy[:,2]
        rotated_target_rpy = normalize_rpy(rotated_target_rpy)
        rotated_box_pos = rotated_box_pos.reshape([self.env.num_envs, self.env.num_agents, -1])
        rotated_box_rpy = rotated_box_rpy.reshape([self.env.num_envs, self.env.num_agents, -1])
        rotated_target_pos = rotated_target_pos.reshape([self.env.num_envs, self.env.num_agents, -1])
        rotated_target_rpy = rotated_target_rpy.reshape([self.env.num_envs, self.env.num_agents, -1])

        # rotate other agents' state to agent's local state
        base_pos = base_pos.reshape([self.env.num_envs, self.env.num_agents, -1])
        base_rpy = base_rpy.reshape([self.env.num_envs, self.env.num_agents, -1])
        base_info = torch.cat([base_pos, base_rpy], dim=2)
        all_base_info = []
        if self.num_agents != 1:
            for i in range(1, self.env.num_agents):
                other_base_info = deepcopy(torch.roll(base_info, i, dims=1))
                # roate other agents' state to agent's local state
                other_base_pos = torch.stack([(other_base_info[:, :, 0] - base_pos[:, :, 0]) * torch.cos(-base_rpy[:, :, 2]) - (other_base_info[:, :, 1] - base_pos[:, :, 1]) * torch.sin(-base_rpy[:, :, 2]),
                                              (other_base_info[:, :, 0] - base_pos[:, :, 0]) * torch.sin(-base_rpy[:, :, 2]) + (other_base_info[:, :, 1] - base_pos[:, :, 1]) * torch.cos(-base_rpy[:, :, 2]),
                                              other_base_info[:, :, 2]], dim=2)
                other_base_rpy = deepcopy(other_base_info[:, :, 3:6])
                other_base_rpy[:, :, 2] = other_base_info[:, :, 5] - base_rpy[:, :, 2]
                other_base_rpy = normalize_rpy(other_base_rpy)
                other_base_info = torch.cat([other_base_pos[:,:,:2], other_base_rpy[:,:,2].unsqueeze(2)], dim=2)
                all_base_info.append(other_base_info)
            all_base_info = torch.cat(all_base_info, dim=2)

        if getattr(self.cfg.goal, "general_dist", False):
            obs = torch.cat([rotated_target_pos[:,:,:2], rotated_target_rpy[:,:,2].unsqueeze(2), rotated_box_pos[:,:,:2], rotated_box_rpy[:,:,2].unsqueeze(2), all_base_info], dim=2)
        else:
            if all_base_info == []:
                obs = torch.cat([rotated_target_pos[:,:,:2], rotated_box_pos[:,:,:2], rotated_box_rpy[:,:,2].unsqueeze(2)], dim=2)
            else:
                obs = torch.cat([rotated_target_pos[:,:,:2], rotated_box_pos[:,:,:2], rotated_box_rpy[:,:,2].unsqueeze(2), all_base_info], dim=2)
        self.last_box_state = None
        return obs

    def step(self, action, next_target_pos=None):
        if next_target_pos is not None:
            assert next_target_pos.shape == (self.num_envs, 3)
            assert self.cfg.generalize_obsersation.rotate_obs
            assert self.cfg.goal.received_goal_pos

        if getattr(self.cfg.goal, "received_goal_pos",False):
            if next_target_pos is None:
                raise ValueError("next_target_pos is required when received_goal_pos is True")
            self.env.next_target_pos = next_target_pos

        action = torch.clip(action, -1.0, 1.0)
        if getattr(self.cfg.goal, "received_goal_pos",False):
            if torch.any(self.env.stop_buf):
                action[self.env.stop_buf] = torch.tensor([0., 0., 0.], device=self.device).repeat(self.stop_buf.sum().item(), self.num_agents, 1)
        # set static action
        # action = torch.tensor([[1.0, 0.0, 0.0]], device="cuda").repeat(self.num_envs, 1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))

        # get agent state
        base_pos = deepcopy(obs_buf.base_pos) 
        base_rpy = deepcopy(obs_buf.base_rpy) 
        # get box state and target pos
        npc_pos = self.root_states_npc[:, :3].reshape(self.num_envs, self.num_npcs, -1)
        box_pos = npc_pos[:,0,:] - self.env.env_origins
        target_pos = npc_pos[:,1,:] - self.env.env_origins 
        box_qyaternion = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0 , 3:7]
        box_rpy = torch.stack(get_euler_xyz(box_qyaternion), dim=1)
        target_qyaternion = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 1 , 3:7]
        target_rpy = torch.stack(get_euler_xyz(target_qyaternion), dim=1)

        # rotate box state and target pos to agent's local state
        box_pos = box_pos.repeat_interleave(self.num_agents, dim=0)
        target_pos = target_pos.repeat_interleave(self.num_agents, dim=0)
        box_rpy = box_rpy.repeat_interleave(self.num_agents, dim=0)
        target_rpy = target_rpy.repeat_interleave(self.num_agents, dim=0)
        rotated_box_pos = torch.stack([(box_pos[:, 0] - base_pos[:, 0]) * torch.cos(-base_rpy[:, 2]) - (box_pos[:, 1] - base_pos[:, 1]) * torch.sin(-base_rpy[:, 2]),
                                       (box_pos[:, 0] - base_pos[:, 0]) * torch.sin(-base_rpy[:, 2]) + (box_pos[:, 1] - base_pos[:, 1]) * torch.cos(-base_rpy[:, 2]),
                                      box_pos[:, 2]], dim=1)
        rotated_target_pos = torch.stack([(target_pos[:, 0] - base_pos[:, 0]) * torch.cos(-base_rpy[:, 2]) - (target_pos[:, 1] - base_pos[:, 1]) * torch.sin(-base_rpy[:, 2]),
                                          (target_pos[:, 0] - base_pos[:, 0]) * torch.sin(-base_rpy[:, 2]) + (target_pos[:, 1] - base_pos[:, 1]) * torch.cos(-base_rpy[:, 2]),
                                         target_pos[:, 2]], dim=1)
        rotated_box_rpy = deepcopy(box_rpy)
        rotated_box_rpy[:,2] = box_rpy[:,2] - base_rpy[:,2]
        rotated_box_rpy = normalize_rpy(rotated_box_rpy)
        rotated_target_rpy = deepcopy(target_rpy)
        rotated_target_rpy[:,2] = target_rpy[:,2] - base_rpy[:,2]
        rotated_target_rpy = normalize_rpy(rotated_target_rpy)
        rotated_box_pos = rotated_box_pos.reshape([self.env.num_envs, self.env.num_agents, -1])
        rotated_box_rpy = rotated_box_rpy.reshape([self.env.num_envs, self.env.num_agents, -1])
        rotated_target_pos = rotated_target_pos.reshape([self.env.num_envs, self.env.num_agents, -1])
        rotated_target_rpy = rotated_target_rpy.reshape([self.env.num_envs, self.env.num_agents, -1])

        # rotate other agents' state to agent's local state
        base_pos = base_pos.reshape([self.env.num_envs, self.env.num_agents, -1])
        base_rpy = base_rpy.reshape([self.env.num_envs, self.env.num_agents, -1])
        base_info = torch.cat([base_pos, base_rpy], dim=2)
        all_base_info = []
        if self.num_agents != 1:
            for i in range(1, self.env.num_agents):
                other_base_info = deepcopy(torch.roll(base_info, i, dims=1))
                # roate other agents' state to agent's local state
                other_base_pos = torch.stack([(other_base_info[:, :, 0] - base_pos[:, :, 0]) * torch.cos(-base_rpy[:, :, 2]) - (other_base_info[:, :, 1] - base_pos[:, :, 1]) * torch.sin(-base_rpy[:, :, 2]),
                                              (other_base_info[:, :, 0] - base_pos[:, :, 0]) * torch.sin(-base_rpy[:, :, 2]) + (other_base_info[:, :, 1] - base_pos[:, :, 1]) * torch.cos(-base_rpy[:, :, 2]),
                                              other_base_info[:, :, 2]], dim=2)
                other_base_rpy = deepcopy(other_base_info[:, :, 3:6])
                other_base_rpy[:, :, 2] = other_base_info[:, :, 5] - base_rpy[:, :, 2]
                other_base_rpy = normalize_rpy(other_base_rpy)
                other_base_info = torch.cat([other_base_pos[:,:,:2], other_base_rpy[:,:,2].unsqueeze(2)], dim=2)
                all_base_info.append(other_base_info)
            all_base_info = torch.cat(all_base_info, dim=2)

        if getattr(self.cfg.goal, "general_dist", False):
            obs = torch.cat([rotated_target_pos[:,:,:2], rotated_target_rpy[:,:,2].unsqueeze(2), rotated_box_pos[:,:,:2], rotated_box_rpy[:,:,2].unsqueeze(2), all_base_info], dim=2)
        else:
            if all_base_info == []:
                obs = torch.cat([rotated_target_pos[:,:,:2], rotated_box_pos[:,:,:2], rotated_box_rpy[:,:,2].unsqueeze(2)], dim=2)
            else:
                obs = torch.cat([rotated_target_pos[:,:,:2], rotated_box_pos[:,:,:2], rotated_box_rpy[:,:,2].unsqueeze(2), all_base_info], dim=2)

        # get env_id which should be reseted, because of nan or inf in obs and reward
        self.value_exception_buf = torch.isnan(obs).any(dim=2).any(dim=1) \
                                | torch.isinf(obs).any(dim=2).any(dim=1) \
                                
        # remove nan and inf in obs and reward
        obs[torch.isnan(obs)] = 0
        obs[torch.isinf(obs)] = 0

        # calculate reward
        box_state = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0]
        target_state = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 1]
        npc_pos = self.root_states_npc[:, :3].reshape(self.num_envs, self.num_npcs, -1)
        box_pos = npc_pos[:,0,:] - self.env.env_origins
        target_pos = npc_pos[:,1,:] - self.env.env_origins 
        box_qyaternion = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0 , 3:7]
        box_rpy = torch.stack(get_euler_xyz(box_qyaternion), dim=1)
        target_qyaternion = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 1 , 3:7]
        target_rpy = torch.stack(get_euler_xyz(target_qyaternion), dim=1)

        base_pos = obs_buf.base_pos # (env_num, agent_num, 3)
        base_vel = obs_buf.lin_vel # (env_num, agent_num, 3)
        base_rpy = obs_buf.base_rpy # (env_num, agent_num, 3)
        base_pos = base_pos.reshape([self.env.num_envs, self.env.num_agents, -1])
        base_vel = base_vel.reshape([self.env.num_envs, self.env.num_agents, -1])
        base_rpy = base_rpy.reshape([self.env.num_envs, self.env.num_agents, -1])

        # occlude nan or inf
        box_pos[torch.isnan(box_pos)] = 0
        box_pos[torch.isinf(box_pos)] = 0
        target_pos[torch.isnan(target_pos)] = 0
        target_pos[torch.isinf(target_pos)] = 0
        base_pos[torch.isnan(base_pos)] = 0
        base_pos[torch.isinf(base_pos)] = 0
        box_rpy[torch.isnan(box_rpy)] = 0
        box_rpy[torch.isinf(box_rpy)] = 0

        self.reward_buffer["step_count"] += 1
        reward = torch.zeros([self.env.num_envs, self.num_agents], device=self.env.device)

        # calculate reach target reward and set finish task termination
        if self.reach_target_reward_scale != 0:
            reward[self.finished_buf, :] += self.reach_target_reward_scale
            self.reward_buffer["reach_target_reward"] += self.reach_target_reward_scale * self.finished_buf.sum().item()

        # ITERATION 10 FIX: Track cumulative success rate (only for HARL/per-agent mode)
        if self.use_per_agent_rewards:
            # Count successes (finished_buf = reached goal)
            self.reward_buffer["success_count"] += self.finished_buf.sum().item()
            # Count episode endings (reset_buf includes finished + exceptions + timeouts)
            self.reward_buffer["episode_count"] += self.reset_buf.sum().item()
        
        # calculate exception punishment
        if self.exception_punishment_scale != 0:
            reward[self.exception_buf, :] += self.exception_punishment_scale
            reward[self.value_exception_buf, :] += self.exception_punishment_scale
            # reward[self.time_out_buf, :] += self.exception_punishment_scale
            self.reward_buffer["exception_punishment"] += self.exception_punishment_scale * \
                    (self.exception_buf.sum().item()+self.value_exception_buf.sum().item())

        # calculate distance from current_box_pos to target_box_pos reward
        if self.target_reward_scale != 0:
            if self.last_box_state is None:
                self.last_box_state = copy(box_state)
            past_distance = self.env.dist_calculator.cal_dist(self.last_box_state, target_state)
            distance = self.env.dist_calculator.cal_dist(box_state, target_state)

            if self.use_per_agent_rewards:
                # NEW CODE PATH: Per-agent attribution based on contribution
                box_progress = past_distance - distance
                progress_reward = self._compute_progress_attribution(
                    base_pos, box_pos, target_pos, box_progress, distance
                )
                for i in range(self.num_agents):
                    reward[:, i] += progress_reward[:, i]
                self.reward_buffer["distance_to_target_reward"] += progress_reward.sum().cpu()
            else:
                # ORIGINAL CODE PATH: Shared reward (backward compatible)
                distance_reward = self.target_reward_scale * 100 * (2 * (past_distance - distance) - 0.01 * distance)
                reward[:, :] += distance_reward.unsqueeze(1).repeat(1, self.num_agents)
                self.reward_buffer["distance_to_target_reward"] += torch.sum(distance_reward).cpu()

        # ITERATION 5: Add directional progress reward (box-to-target movement)
        if self.use_per_agent_rewards and self.prev_box_pos is not None:
            directional_progress = self._compute_directional_progress_reward(
                box_pos, target_pos, self.prev_box_pos
            )
            reward += directional_progress
            self.reward_buffer["directional_progress"] += directional_progress.sum().cpu()

        # Update previous box position for next step
        self.prev_box_pos = box_pos.clone()

        # calculate distance from each robot to box reward
        if self.approach_reward_scale != 0:
            reward_logger=[]
            for i in range(self.num_agents):
                distance = torch.norm(box_pos - base_pos[:, i, :], dim=1, keepdim=True)
                distance_reward = (-(distance+0.5)**2) * self.approach_reward_scale
                reward_logger.append(torch.sum(distance_reward).cpu())
                reward[:, i] += distance_reward.squeeze(-1)
            self.reward_buffer["approach_to_box_reward"] += np.sum(np.array(reward_logger)) 

        # calculate collision punishment
        if self.collision_punishment_scale != 0:
            punishment_logger=[]
            for i in range(self.num_agents):
                for j in range(i+1, self.num_agents):
                    distance = torch.norm(base_pos[:, i, :] - base_pos[:, j, :], dim=1, keepdim=True)
                    collsion_punishment = (1 / (0.02 + distance/3)) * self.collision_punishment_scale
                    punishment_logger.append(torch.sum(collsion_punishment).cpu())
                    reward[:, i] += collsion_punishment.squeeze(-1)
                    reward[:, j] += collsion_punishment.squeeze(-1)
            self.reward_buffer["collision_punishment"] += np.sum(np.array(punishment_logger))

        # calculate push reward for each agent
        if self.push_reward_scale != 0:
            if self.use_per_agent_rewards:
                # NEW CODE PATH: Per-agent push contribution
                box_velocity = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0, 7:9]
                push_contribution = self._compute_push_contribution(
                    base_pos, box_pos, target_pos, box_velocity
                )
                for i in range(self.num_agents):
                    reward[:, i] += push_contribution[:, i]
                self.reward_buffer["push_reward"] += push_contribution.sum().cpu()
            else:
                # ORIGINAL CODE PATH: Shared reward (backward compatible)
                push_reward = torch.zeros((self.env.num_envs,), device=self.env.device)
                push_reward[torch.norm(self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0 , 7:9],dim=1) > 0.1] = self.push_reward_scale
                reward[:, :] += push_reward.unsqueeze(1).repeat(1, self.num_agents)
                self.reward_buffer["push_reward"] += torch.sum(push_reward).cpu()
            
        # calculate OCB reward for each agent
        if self.ocb_reward_scale != 0:
            if getattr(self.cfg.rewards,"expanded_ocb_reward",False):
                original_target_direction=(target_pos[:, :2] - box_pos[:, :2])/(torch.norm((target_pos[:, :2] - box_pos[:, :2]+0.01),dim=1,keepdim=True))
                delta_yaw = target_rpy[:, 2] - box_rpy[:, 2]
                # delta_yaw -->(-pi, pi)
                delta_yaw = (delta_yaw + torch.pi) % (2 * torch.pi) - torch.pi
                # rotate target direction by delta_yaw/2
                target_direction = torch.stack([original_target_direction[:, 0] * torch.cos(-delta_yaw/2) - original_target_direction[:, 1] * torch.sin(-delta_yaw/2),
                                                original_target_direction[:, 0] * torch.sin(-delta_yaw/2) + original_target_direction[:, 1] * torch.cos(-delta_yaw/2)], dim=1)
                pass
            else:
                target_direction = (target_pos[:, :2] - box_pos[:, :2])/(torch.norm((target_pos[:, :2] - box_pos[:, :2]),dim=1,keepdim=True))

            if self.use_per_agent_rewards:
                # NEW CODE PATH: Improved positioning with engagement check
                positioning_reward = self._compute_positioning_reward(
                    base_pos, box_pos, target_pos, box_rpy, target_direction
                )
                for i in range(self.num_agents):
                    reward[:, i] += positioning_reward[:, i]
                self.reward_buffer["ocb_reward"] += positioning_reward.sum().cpu()
            else:
                # ORIGINAL CODE PATH: Standard OCB reward (backward compatible)
                vertex_list=self.cfg.asset.vertex_list
                reward_logger=[]
                for i in range(self.num_agents):
                    gf_pos=base_pos[:, i, :2] - box_pos[:,:2]
                    rotation_matrix=rotation_matrix_2D( - box_rpy[:, 2])
                    box_relative_pos=torch.bmm(rotation_matrix,gf_pos.unsqueeze(2)).squeeze(2)
                    normal_vector=self.calc_normal_vector_for_obc_reward(vertex_list,box_relative_pos)
                    rotation_matrix=rotation_matrix_2D( box_rpy[:, 2])
                    normal_vector=torch.bmm(rotation_matrix,normal_vector.to(rotation_matrix.device).unsqueeze(2)).squeeze(2)
                    ocb_reward = torch.sum( target_direction * normal_vector, dim=1) * self.ocb_reward_scale
                    reward[:, i] += ocb_reward
                    reward_logger.append(torch.sum(ocb_reward).cpu())
                self.reward_buffer["ocb_reward"] += np.sum(np.array(reward_logger))

        # ITERATION 2 & 3: Add engagement and cooperation bonuses (only for per-agent mode)
        if self.use_per_agent_rewards:
            engagement_bonus = self._compute_engagement_bonus(base_pos, box_pos)
            cooperation_bonus = self._compute_cooperation_bonus(base_pos, box_pos)

            for i in range(self.num_agents):
                reward[:, i] += engagement_bonus[:, i]
                reward[:, i] += cooperation_bonus[:, i]

            self.reward_buffer["engagement_bonus"] = self.reward_buffer.get("engagement_bonus", 0) + engagement_bonus.sum().cpu()
            self.reward_buffer["cooperation_bonus"] = self.reward_buffer.get("cooperation_bonus", 0) + cooperation_bonus.sum().cpu()

            # ITERATION 9: Add blocking penalty and same-side bonus
            blocking_penalty = self._compute_blocking_penalty(base_pos, box_pos, target_pos)
            same_side_bonus = self._compute_same_side_bonus(base_pos, box_pos, target_pos)

            for i in range(self.num_agents):
                reward[:, i] += blocking_penalty[:, i]
                reward[:, i] += same_side_bonus[:, i]

            self.reward_buffer["blocking_penalty"] += blocking_penalty.sum().cpu()
            self.reward_buffer["same_side_bonus"] += same_side_bonus.sum().cpu()

            # ITERATION 9: Debug logging (print every 100 steps for env 0 only)
            if hasattr(self.env, 'episode_length_buf') and self.env.episode_length_buf[0] % 100 == 0 and self.env.episode_length_buf[0] > 0:
                print(f"[Env 0, Step {self.env.episode_length_buf[0].item():4d}] "
                      f"Engage: {engagement_bonus[0].mean().item():.4f}, "
                      f"Coop: {cooperation_bonus[0].mean().item():.4f}, "
                      f"Block: {blocking_penalty[0].mean().item():.4f}, "
                      f"SameSide: {same_side_bonus[0].mean().item():.4f}")

        self.last_box_state = deepcopy(box_state)

        # ==================== MAPush Metrics Tracking for TensorBoard ====================
        # Compute metrics that will be logged to TensorBoard via HARL
        # Only compute if we have valid data (not during initialization)
        try:
            # 1. Success Rate - percentage of episodes that reached the target
            if self.use_per_agent_rewards:
                # ITERATION 10 FIX: Use CUMULATIVE success rate for HARL
                episode_count = max(self.reward_buffer["episode_count"], 1)
                success_rate = self.reward_buffer["success_count"] / episode_count
            else:
                # Original: instantaneous check (for backward compatibility)
                distance_to_target = self.env.dist_calculator.cal_dist(box_state, target_state)
                success = (distance_to_target < self.cfg.goal.THRESHOLD).float()
                success_rate = success.mean().cpu().item()

            # Compute distance for logging
            distance_to_target = self.env.dist_calculator.cal_dist(box_state, target_state)

            # 2. Average Distance to Target - how close the box is to target
            avg_distance_to_target = distance_to_target.mean().cpu().item()

            # 3. Collision Rate - percentage of environments with robots too close
            collision_count = 0
            total_pairs = 0
            collision_threshold = 0.5  # meters - robots closer than this are considered colliding
            for i in range(self.num_agents):
                for j in range(i+1, self.num_agents):
                    agent_distance = torch.norm(base_pos[:, i, :2] - base_pos[:, j, :2], dim=1)
                    collision_count += (agent_distance < collision_threshold).sum().cpu().item()
                    total_pairs += self.num_envs
            collision_rate = collision_count / max(total_pairs, 1)

            # 4. Reward Component Breakdown - average per-step reward from each component
            # Normalize by step count to get average per-step values
            step_count = max(self.reward_buffer["step_count"], 1)

            # Flag-dependent logging (backward compatible)
            if self.use_per_agent_rewards:
                # ITERATION 9: Per-agent mode logging (includes all bonuses and penalties)
                reward_components = {
                    "distance_to_target": float(self.reward_buffer["distance_to_target_reward"]) / step_count,
                    "approach_to_box": float(self.reward_buffer["approach_to_box_reward"]) / step_count,
                    "collision_punishment": float(self.reward_buffer["collision_punishment"]) / step_count,
                    "reach_target": float(self.reward_buffer["reach_target_reward"]) / step_count,
                    "push_reward": float(self.reward_buffer["push_reward"]) / step_count,
                    "ocb_reward": float(self.reward_buffer["ocb_reward"]) / step_count,
                    "exception_punishment": float(self.reward_buffer["exception_punishment"]) / step_count,
                    "engagement_bonus": float(self.reward_buffer.get("engagement_bonus", 0)) / step_count,
                    "cooperation_bonus": float(self.reward_buffer.get("cooperation_bonus", 0)) / step_count,
                    "blocking_penalty": float(self.reward_buffer.get("blocking_penalty", 0)) / step_count,  # ITERATION 9
                    "same_side_bonus": float(self.reward_buffer.get("same_side_bonus", 0)) / step_count,    # ITERATION 9
                }
            else:
                # Original shared mode logging (backward compatible)
                reward_components = {
                    "distance_to_target": float(self.reward_buffer["distance_to_target_reward"]) / step_count,
                    "approach_to_box": float(self.reward_buffer["approach_to_box_reward"]) / step_count,
                    "collision_punishment": float(self.reward_buffer["collision_punishment"]) / step_count,
                    "reach_target": float(self.reward_buffer["reach_target_reward"]) / step_count,
                    "push_reward": float(self.reward_buffer["push_reward"]) / step_count,
                    "ocb_reward": float(self.reward_buffer["ocb_reward"]) / step_count,
                    "exception_punishment": float(self.reward_buffer["exception_punishment"]) / step_count,
                }

            # Package metrics into info dict for HARL to log
            info["mapush_metrics"] = {
                "success_rate": success_rate,
                "avg_distance_to_target": avg_distance_to_target,
                "collision_rate": collision_rate,
                "reward_components": reward_components,
            }
        except Exception as e:
            # If metrics computation fails (e.g. during initialization), skip it
            print(f"Warning: Could not compute metrics: {e}")
            pass
        # ==================== End Metrics Tracking ====================

        return obs, reward, termination, info

    def _compute_progress_attribution(self, base_pos, box_pos, target_pos, box_progress, distance):
        """Attribute box progress to agents based on proximity and positioning.

        Args:
            base_pos: (num_envs, num_agents, 3) - agent positions
            box_pos: (num_envs, 3) - box position
            target_pos: (num_envs, 3) - target position
            box_progress: (num_envs,) - distance improvement (positive = closer)
            distance: (num_envs,) - current distance to target

        Returns:
            progress_reward: (num_envs, num_agents) - attributed progress per agent
        """
        contribution_radius = getattr(self.cfg.rewards.scales, 'progress_contribution_radius', 1.5)
        progress_reward = torch.zeros((self.env.num_envs, self.num_agents), device=self.env.device)

        # Compute target direction
        target_direction = target_pos[:, :2] - box_pos[:, :2]
        target_direction = target_direction / (torch.norm(target_direction, dim=1, keepdim=True) + 1e-6)

        # Compute contribution weights for each agent
        contribution_weights = torch.zeros((self.env.num_envs, self.num_agents), device=self.env.device)

        for i in range(self.num_agents):
            # Distance to box
            agent_to_box = box_pos[:, :2] - base_pos[:, i, :2]
            distance_to_box = torch.norm(agent_to_box, dim=1)

            # Only consider agents within contribution radius
            in_range = distance_to_box < contribution_radius

            # Compute positioning alignment
            agent_direction = agent_to_box / (torch.norm(agent_to_box, dim=1, keepdim=True) + 1e-6)
            alignment = torch.sum(agent_direction * target_direction, dim=1)
            alignment = torch.clamp(alignment, min=0.0)  # Only positive contributions

            # Weight by proximity (closer = more contribution)
            proximity_weight = torch.clamp(1.0 - distance_to_box / contribution_radius, min=0.0, max=1.0)

            # Combine factors
            weight = alignment * proximity_weight
            contribution_weights[:, i] = torch.where(in_range, weight, torch.zeros_like(weight))

        # Normalize weights (sum to 1 per environment)
        total_weight = contribution_weights.sum(dim=1, keepdim=True)
        total_weight = torch.where(total_weight > 0, total_weight, torch.ones_like(total_weight))
        contribution_weights = contribution_weights / total_weight

        # Distribute progress reward based on weights
        scaled_progress = self.target_reward_scale * 100 * (2 * box_progress - 0.01 * distance)
        for i in range(self.num_agents):
            progress_reward[:, i] = scaled_progress * contribution_weights[:, i]

        return progress_reward

    def _compute_push_contribution(self, base_pos, box_pos, target_pos, box_velocity):
        """Compute per-agent push contribution reward.

        ITERATION 10 FIX: Reward based on ACTUAL box velocity toward goal,
        not assumed force direction from agent position.

        Only reward agents that:
        1. Are in contact with the box (within threshold)
        2. Box is actually moving toward target

        Args:
            base_pos: (num_envs, num_agents, 3) - agent positions
            box_pos: (num_envs, 3) - box position
            target_pos: (num_envs, 3) - target position
            box_velocity: (num_envs, 2) - box velocity xy

        Returns:
            push_contribution: (num_envs, num_agents) - reward per agent
        """
        contact_threshold = getattr(self.cfg.rewards.scales, 'push_contact_threshold', 0.5)
        push_contribution = torch.zeros((self.env.num_envs, self.num_agents), device=self.env.device)

        # ITERATION 10: Compute ACTUAL box velocity alignment with target
        box_speed = torch.norm(box_velocity, dim=1)
        box_velocity_direction = box_velocity / (box_speed.unsqueeze(1) + 1e-6)

        # Target direction (from box toward target)
        target_direction = target_pos[:, :2] - box_pos[:, :2]
        target_direction = target_direction / (torch.norm(target_direction, dim=1, keepdim=True) + 1e-6)

        # How much is box ACTUALLY moving toward target? (-1 to +1)
        velocity_alignment = torch.sum(box_velocity_direction * target_direction, dim=1)

        for i in range(self.num_agents):
            # Check if agent is close enough to box
            agent_to_box = box_pos[:, :2] - base_pos[:, i, :2]
            distance_to_box = torch.norm(agent_to_box, dim=1)
            in_contact = distance_to_box < contact_threshold

            # ITERATION 10: Reward based on actual box movement toward goal
            # If box moves toward target AND agent is in contact = positive reward
            # If box moves away from target AND agent is in contact = negative penalty
            contribution = velocity_alignment * box_speed * self.push_reward_scale

            # Apply only if in contact
            push_contribution[:, i] = torch.where(in_contact, contribution, torch.zeros_like(contribution))

        return push_contribution

    def _compute_directional_progress_reward(self, box_pos, target_pos, prev_box_pos):
        """ITERATION 5: Explicit reward for box moving toward/away from target.

        Provides crystal-clear directional signal:
        - Box moved closer to target = POSITIVE reward
        - Box moved away from target = NEGATIVE penalty

        This is a SHARED reward (both agents get same) to encourage
        collaboration on the common objective of moving box toward target.

        Args:
            box_pos: (num_envs, 3) - current box position
            target_pos: (num_envs, 3) - target position
            prev_box_pos: (num_envs, 3) - previous box position

        Returns:
            directional_reward: (num_envs, num_agents) - shared reward per agent
        """
        directional_scale = getattr(self.cfg.rewards.scales, 'directional_progress_scale', 0.01)

        # Calculate distance change (2D, ignore Z)
        old_distance = torch.norm(prev_box_pos[:, :2] - target_pos[:, :2], dim=1)
        new_distance = torch.norm(box_pos[:, :2] - target_pos[:, :2], dim=1)

        # Progress: positive if closer, negative if farther
        progress = old_distance - new_distance

        # Scale and broadcast to both agents (shared signal)
        reward = progress * directional_scale
        return reward.unsqueeze(1).repeat(1, self.num_agents)

    def _compute_positioning_reward(self, base_pos, box_pos, target_pos, box_rpy, target_direction):
        """Improved positioning reward that only rewards engaged agents ON THE PUSH SIDE.

        ITERATION 9 FIX: Only give OCB reward to agents on the push side (behind box).
        Agents on the blocking side (between box and goal) get ZERO OCB reward.

        Args:
            base_pos: (num_envs, num_agents, 3) - agent positions
            box_pos: (num_envs, 3) - box position
            target_pos: (num_envs, 3) - target position
            box_rpy: (num_envs, 3) - box orientation
            target_direction: (num_envs, 2) - direction from box to target

        Returns:
            positioning_reward: (num_envs, num_agents) - reward per agent
        """
        engagement_radius = getattr(self.cfg.rewards.scales, 'positioning_engagement_radius', 2.0)
        positioning_reward = torch.zeros((self.env.num_envs, self.num_agents), device=self.env.device)

        # Use existing OCB calculation but add engagement check
        vertex_list = self.cfg.asset.vertex_list

        # ITERATION 9: Compute box-to-target direction for push-side check
        box_to_target = target_pos[:, :2] - box_pos[:, :2]
        box_to_target_norm = box_to_target / (torch.norm(box_to_target, dim=1, keepdim=True) + 1e-6)

        for i in range(self.num_agents):
            # Check if agent is engaged (near box)
            distance_to_box = torch.norm(base_pos[:, i, :2] - box_pos[:, :2], dim=1)
            is_engaged = distance_to_box < engagement_radius

            # ITERATION 9: Check if agent is on push side (not blocking)
            box_to_agent = base_pos[:, i, :2] - box_pos[:, :2]
            box_to_agent_norm = box_to_agent / (torch.norm(box_to_agent, dim=1, keepdim=True) + 1e-6)
            side_alignment = torch.sum(box_to_agent_norm * box_to_target_norm, dim=1)
            # Negative alignment = behind box (push side), positive = in front (blocking)
            is_push_side = side_alignment < 0.0

            # Compute OCB reward (existing logic)
            gf_pos = base_pos[:, i, :2] - box_pos[:, :2]
            rotation_matrix = rotation_matrix_2D(-box_rpy[:, 2])
            box_relative_pos = torch.bmm(rotation_matrix, gf_pos.unsqueeze(2)).squeeze(2)
            normal_vector = self.calc_normal_vector_for_obc_reward(vertex_list, box_relative_pos)
            rotation_matrix = rotation_matrix_2D(box_rpy[:, 2])
            normal_vector = torch.bmm(rotation_matrix, normal_vector.to(rotation_matrix.device).unsqueeze(2)).squeeze(2)
            ocb_reward = torch.sum(target_direction * normal_vector, dim=1) * self.ocb_reward_scale

            # Add proximity bonus
            proximity_bonus = torch.clamp(1.0 - distance_to_box / engagement_radius, min=0.0, max=1.0)

            # ITERATION 9: Only apply if engaged AND on push side
            # Blocking agents get ZERO OCB reward
            is_valid = is_engaged & is_push_side
            final_reward = ocb_reward * proximity_bonus
            positioning_reward[:, i] = torch.where(is_valid, final_reward, torch.zeros_like(final_reward))

        return positioning_reward

    def _compute_engagement_bonus(self, base_pos, box_pos):
        """Compute engagement bonus for being near the box.

        Provides positive incentive to approach and stay engaged with the task.
        Linear falloff with distance from box.

        Args:
            base_pos: Agent positions [num_envs, num_agents, 3]
            box_pos: Box positions [num_envs, 3]

        Returns:
            engagement_bonus: Per-agent engagement bonus [num_envs, num_agents]
        """
        engagement_radius = getattr(self.cfg.rewards.scales, 'engagement_bonus_radius', 2.0)
        engagement_scale = getattr(self.cfg.rewards.scales, 'engagement_bonus_scale', 0.005)

        engagement_bonus = torch.zeros((self.env.num_envs, self.num_agents), device=self.env.device)

        for i in range(self.num_agents):
            distance_to_box = torch.norm(base_pos[:, i, :2] - box_pos[:, :2], dim=1)

            # Linear bonus: closer = better (1.0 at box, 0.0 at radius)
            bonus = torch.clamp(1.0 - distance_to_box / engagement_radius, min=0.0, max=1.0)
            engagement_bonus[:, i] = bonus * engagement_scale

        return engagement_bonus

    def _compute_cooperation_bonus(self, base_pos, box_pos):
        """Compute cooperation bonus when BOTH agents are engaged with the box.

        Provides explicit incentive for coordination - both agents get reward
        only when BOTH are within cooperation radius of box.

        Args:
            base_pos: Agent positions [num_envs, num_agents, 3]
            box_pos: Box positions [num_envs, 3]

        Returns:
            cooperation_bonus: Per-agent cooperation bonus [num_envs, num_agents]
        """
        cooperation_radius = getattr(self.cfg.rewards.scales, 'cooperation_radius', 2.0)
        cooperation_scale = getattr(self.cfg.rewards.scales, 'cooperation_bonus_scale', 0.01)

        # Calculate distance from each agent to box
        distances = torch.norm(base_pos[:, :, :2] - box_pos[:, None, :2], dim=2)  # [num_envs, num_agents]

        # Check if ALL agents within cooperation radius
        all_engaged = (distances < cooperation_radius).all(dim=1)  # [num_envs]

        # Apply same bonus to both agents when cooperating
        cooperation_bonus = torch.zeros((self.env.num_envs, self.num_agents), device=self.env.device)
        cooperation_bonus[all_engaged] = cooperation_scale

        return cooperation_bonus

    def _compute_blocking_penalty(self, base_pos, box_pos, target_pos):
        """ITERATION 9: Penalize agents positioned between box and goal (blocking position).

        An agent is considered blocking if:
        1. It is within blocking_radius of the box
        2. It is positioned on the goal-side of the box (positive alignment)

        Args:
            base_pos: Agent positions [num_envs, num_agents, 3]
            box_pos: Box positions [num_envs, 3]
            target_pos: Target positions [num_envs, 3]

        Returns:
            blocking_penalty: Per-agent blocking penalty [num_envs, num_agents] (negative values)
        """
        blocking_radius = getattr(self.cfg.rewards.scales, 'blocking_radius', 2.0)
        blocking_scale = getattr(self.cfg.rewards.scales, 'blocking_penalty_scale', 0.05)
        alignment_threshold = getattr(self.cfg.rewards.scales, 'blocking_alignment_threshold', 0.3)

        blocking_penalty = torch.zeros((self.env.num_envs, self.num_agents), device=self.env.device)

        # Vector from box to target (normalized)
        box_to_target = target_pos[:, :2] - box_pos[:, :2]
        box_to_target_norm = box_to_target / (torch.norm(box_to_target, dim=1, keepdim=True) + 1e-6)

        for i in range(self.num_agents):
            # Vector from box to agent
            box_to_agent = base_pos[:, i, :2] - box_pos[:, :2]
            distance_to_box = torch.norm(box_to_agent, dim=1)
            box_to_agent_norm = box_to_agent / (distance_to_box.unsqueeze(1) + 1e-6)

            # Dot product: positive if agent is in front of box (toward goal side)
            # negative if agent is behind box (push side)
            alignment = torch.sum(box_to_agent_norm * box_to_target_norm, dim=1)

            # Agent is blocking if:
            # 1. alignment > threshold (in front of box, toward goal)
            # 2. within blocking radius (close enough to obstruct)
            is_blocking = (alignment > alignment_threshold) & (distance_to_box < blocking_radius)

            # Penalty proportional to:
            # - How directly in front (alignment)
            # - How close to box (proximity factor)
            proximity_factor = torch.clamp(1.0 - distance_to_box / blocking_radius, min=0.0, max=1.0)
            penalty = -blocking_scale * alignment * proximity_factor

            blocking_penalty[:, i] = torch.where(is_blocking, penalty, torch.zeros_like(penalty))

        return blocking_penalty

    def _compute_same_side_bonus(self, base_pos, box_pos, target_pos):
        """ITERATION 9: Bonus when both agents are on the push side (behind box relative to goal).

        Both agents must be on the correct side (negative alignment with box-to-target)
        to receive this bonus. This encourages coordinated pushing from the same side.

        Args:
            base_pos: Agent positions [num_envs, num_agents, 3]
            box_pos: Box positions [num_envs, 3]
            target_pos: Target positions [num_envs, 3]

        Returns:
            same_side_bonus: Per-agent same-side bonus [num_envs, num_agents]
        """
        same_side_scale = getattr(self.cfg.rewards.scales, 'same_side_bonus_scale', 0.02)
        alignment_threshold = getattr(self.cfg.rewards.scales, 'same_side_alignment_threshold', -0.3)
        engagement_radius = getattr(self.cfg.rewards.scales, 'engagement_bonus_radius', 1.5)

        # Vector from box to target (normalized)
        box_to_target = target_pos[:, :2] - box_pos[:, :2]
        box_to_target_norm = box_to_target / (torch.norm(box_to_target, dim=1, keepdim=True) + 1e-6)

        # Check each agent's side relative to box-target line
        agent_on_push_side = []
        agent_engaged = []

        for i in range(self.num_agents):
            # Vector from box to agent
            box_to_agent = base_pos[:, i, :2] - box_pos[:, :2]
            distance_to_box = torch.norm(box_to_agent, dim=1)
            box_to_agent_norm = box_to_agent / (distance_to_box.unsqueeze(1) + 1e-6)

            # Alignment: negative = behind box (push side), positive = in front (blocking)
            alignment = torch.sum(box_to_agent_norm * box_to_target_norm, dim=1)

            # Agent is on push side if alignment < threshold (behind box)
            is_push_side = alignment < alignment_threshold
            is_engaged = distance_to_box < engagement_radius

            agent_on_push_side.append(is_push_side)
            agent_engaged.append(is_engaged)

        # Both agents must be on push side AND engaged to get bonus
        both_on_push_side = agent_on_push_side[0] & agent_on_push_side[1]
        both_engaged = agent_engaged[0] & agent_engaged[1]
        both_good = both_on_push_side & both_engaged

        # Apply same bonus to both agents when both are correctly positioned
        same_side_bonus = torch.zeros((self.env.num_envs, self.num_agents), device=self.env.device)
        same_side_bonus[both_good] = same_side_scale

        return same_side_bonus
