from active_adaptation.envs.mdp.base import Reward
import isaaclab.utils.math as math_utils

import torch
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.utils.string import resolve_matching_names

from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from isaaclab.assets.articulation import Articulation
    from isaaclab.sensors import ContactSensor
    
class survival(Reward):
    def compute(self):
        return torch.ones(self.num_envs, 1, device=self.device)

class linvel_z_l2(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]

    def compute(self) -> torch.Tensor:
        linvel_z = self.asset.data.root_lin_vel_b[:, 2].unsqueeze(1)
        return -linvel_z.square()

class angvel_xy_l2(Reward):
    def __init__(self, env, weight: float, enabled: bool = True, body_names: str = None):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        if body_names is not None:
            self.body_ids, self.body_names = self.asset.find_bodies(body_names)
            self.body_ids = torch.tensor(self.body_ids, device=self.device)
        else:
            self.body_ids = None

    def update(self):
        if self.body_ids is not None:
            angvel = self.asset.data.body_ang_vel_w[:, self.body_ids]
        else:
            angvel = self.asset.data.root_ang_vel_w.unsqueeze(1)
        self.angvel_w = angvel

    def compute(self) -> torch.Tensor:
        r = -self.angvel_w[:, :, :2].square().sum(-1).mean(1)
        return r.reshape(self.num_envs, 1).clamp_min(-1.0)

class body_upright(Reward):
    """
    Reward for keeping the specified body upright.
    """
    def __init__(self, env, body_name: str, weight, enabled = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_id, body_name = self.asset.find_bodies(body_name)
        self.down = torch.tensor([[0., 0., -1.]], device=self.device).expand(self.num_envs, len(self.body_id), 3)
    
    def compute(self) -> torch.Tensor:
        g = quat_apply_inverse(
            self.asset.data.body_quat_w[:, self.body_id],
            self.down
        )
        rew = 1. - g[:, :, :2].square().sum(-1)
        return rew.mean(1, True)

class joint_pos_limits(Reward):
    def __init__(self, env, weight: float, joint_names: str | List[str] =".*", soft_factor: float=0.9, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = resolve_matching_names(joint_names, self.asset.joint_names)
        jpos_limits = self.asset.data.joint_pos_limits[:, self.joint_ids]
        jpos_mean = (jpos_limits[..., 0] + jpos_limits[..., 1]) / 2
        jpos_range = jpos_limits[..., 1] - jpos_limits[..., 0]
        self.soft_limits = torch.zeros_like(jpos_limits)
        self.soft_limits[..., 0] = jpos_mean - 0.5 * jpos_range * soft_factor
        self.soft_limits[..., 1] = jpos_mean + 0.5 * jpos_range * soft_factor

    def compute(self) -> torch.Tensor:
        jpos = self.asset.data.joint_pos[:, self.joint_ids]
        violation_min = (self.soft_limits[..., 0] - jpos).clamp_min(0.0)
        violation_max = (jpos - self.soft_limits[..., 1]).clamp_min(0.0)
        return -(violation_min + violation_max).sum(1, keepdim=True)

class joint_torque_limits(Reward):
    def __init__(self, env, weight: float, joint_names: str | List[str] =".*", soft_factor: float=0.9, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = resolve_matching_names(joint_names, self.asset.joint_names)
        self.soft_limits = self.asset.data.joint_effort_limits[:, self.joint_ids] * soft_factor
    
    def compute(self) -> torch.Tensor:
        applied_torque = self.asset.data.applied_torque[:, self.joint_ids]
        violation_high = (applied_torque / self.soft_limits - 1.0).clamp_min(0.0)
        violation_low = (-applied_torque / self.soft_limits - 1.0).clamp_min(0.0)
        return - (violation_high + violation_low).sum(dim=1, keepdim=True)

class action_rate_l2(Reward):
    """Penalize the rate of change of the action"""
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.action_manager = self.env.action_manager
    
    def compute(self) -> torch.Tensor:
        action_buf = self.action_manager.action_buf
        action_diff = action_buf[:, :, 0] - action_buf[:, :, 1]
        rew = - action_diff.square().sum(dim=-1, keepdim=True)
        return rew

class action_rate2_l2(Reward):
    """Penalize the second order rate of change of the action"""
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.action_manager = self.env.action_manager
    
    def compute(self) -> torch.Tensor:
        action_buf = self.action_manager.action_buf
        action_diff = (
            action_buf[:, :, 0] - 2 * action_buf[:, :, 1] + action_buf[:, :, 2]
        )
        rew = - action_diff.square().sum(dim=-1, keepdim=True)
        return rew

class joint_vel_l2(Reward):
    def __init__(self, env, joint_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, _ = self.asset.find_joints(joint_names)
        self.joint_vel = torch.zeros(
            self.num_envs, 2, len(self.joint_ids), device=self.device
        )

    def post_step(self, substep):
        self.joint_vel[:, substep % 2] = self.asset.data.joint_vel[:, self.joint_ids]

    def compute(self) -> torch.Tensor:
        joint_vel = self.joint_vel.mean(1)
        return -joint_vel.square().clamp_max(5.0).sum(1, True)


class undesired_contact(Reward):
    def __init__(self, body_names: str | List[str], thres: float=1.0, **kwargs):
        super().__init__(**kwargs)
        self.contact_forces: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids = self.contact_forces.find_bodies(body_names)[0]
        self.thres = thres
    
    def compute(self):
        
        # extract the used quantities (to enable type-hinting)
        
        # check if contact force is above threshold
        net_contact_forces = self.contact_forces.data.net_forces_w_history
        is_contact = torch.max(torch.norm(net_contact_forces[:, :, self.body_ids], dim=-1), dim=1)[0] > self.thres
        # sum over contacts for each environment
        return -torch.sum(is_contact, dim=1).unsqueeze(1)


    
class lin_vel_z_l2(Reward):
    def __init__(self, env,  weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]

    def compute(self) -> torch.Tensor:
        return -torch.square(self.asset.data.root_lin_vel_b[:, 2]).unsqueeze(1)
    
class flat_orientation_l2(Reward):
    def __init__(self, env, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
    
    def compute(self) -> torch.Tensor:
        
        return -torch.sum(torch.square(self.asset.data.projected_gravity_b[:, :2]), dim=1).unsqueeze(1)

class body_orientation_l2(Reward):
    def __init__(self, env, body_names, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self._body_ids = self.asset.find_bodies(body_names)[0]
    
    def compute(self) -> torch.Tensor:
        
        body_orientation = math_utils.quat_apply_inverse(
        self.asset.data.body_quat_w[:, self._body_ids[0], :], self.asset.data.GRAVITY_VEC_W
    )

        return -torch.sum(torch.square(body_orientation[:, :2]), dim=1).unsqueeze(1)

    
class feet_y_distance_l2(Reward):
    """Penalty = ( mean_pair_y_distance - target_y_distance )^2"""

    def __init__(self, env, left_right_pairs, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]

        body_names_idx = {n: i for i, n in enumerate(self.asset.data.body_names)}
        self._pair_indices = []
        for left_name, right_name in left_right_pairs:
            if left_name not in body_names_idx:
                raise ValueError(f"[feet_y_distance_l2] body '{left_name}' not found.")
            if right_name not in body_names_idx:
                raise ValueError(f"[feet_y_distance_l2] body '{right_name}' not found.")
            self._pair_indices.append((body_names_idx[left_name], body_names_idx[right_name]))

        self._left_ids = torch.tensor(
            [p[0] for p in self._pair_indices], device=self.device, dtype=torch.long
        )
        self._right_ids = torch.tensor(
            [p[1] for p in self._pair_indices], device=self.device, dtype=torch.long
        )

        global_body_pos = self.asset.data.body_pos_w
        root_pos_w = self.asset.data.root_state_w[:, :3]
        root_quat_w = self.asset.data.root_state_w[:, 3:7]

        num_bodies = global_body_pos.shape[1]
        root_quat_w_expanded = root_quat_w[:, None, :].expand(-1, num_bodies, -1)
        body_pos_rel = global_body_pos - root_pos_w[:, None, :]

        local_body_pos = math_utils.quat_apply_inverse(
            root_quat_w_expanded,
            body_pos_rel,
        )

        feet_y_distance = torch.abs(
            local_body_pos[:, self._left_ids, 1] - local_body_pos[:, self._right_ids, 1]
        ).mean(dim=1)

        self.feet_y_distance_target = feet_y_distance.mean().item()

    def compute(self) -> torch.Tensor:
    
        global_body_pos = self.asset.data.body_pos_w
        root_pos_w = self.asset.data.root_state_w[:, :3]
        root_quat_w = self.asset.data.root_state_w[:, 3:7]

        num_bodies = global_body_pos.shape[1]
        root_quat_w_expanded = root_quat_w[:, None, :].expand(-1, num_bodies, -1)
        body_pos_rel = global_body_pos - root_pos_w[:, None, :]

        local_body_pos = math_utils.quat_apply_inverse(
            root_quat_w_expanded,
            body_pos_rel,
        )

        feet_y_distance = torch.abs(
            local_body_pos[:, self._left_ids, 1] - local_body_pos[:, self._right_ids, 1]
        ).mean(dim=1)

        return -torch.square(feet_y_distance - self.feet_y_distance_target).unsqueeze(1)
    
class symmetry_air(Reward):
    """
    Reward for symmetric air phase.

    For each left/right body pair:
        reward += 1 if both left and right bodies are in air
        reward += 0 otherwise

    contact_time > 0.0 means the body is currently in contact.
    """

    def __init__(self, env, left_right_pairs, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)

        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self._pair_indices = []

        for left_names, right_names in left_right_pairs:
            left_ids = self.contact_sensor.find_bodies(
                left_names, preserve_order=True
            )[0]
            right_ids = self.contact_sensor.find_bodies(
                right_names, preserve_order=True
            )[0]

            if len(left_ids) == 0:
                raise ValueError(f"[symmetry_air] left body '{left_names}' not found.")
            if len(right_ids) == 0:
                raise ValueError(f"[symmetry_air] right body '{right_names}' not found.")

            if len(left_ids) != len(right_ids):
                raise ValueError(
                    f"[symmetry_air] Left/right pair must have same number of bodies, "
                    f"got {len(left_ids)} and {len(right_ids)}."
                )

            left_ids = torch.tensor(left_ids, device=self.device, dtype=torch.long)
            right_ids = torch.tensor(right_ids, device=self.device, dtype=torch.long)

            self._pair_indices.append((left_ids, right_ids))

    def compute(self) -> torch.Tensor:
        contact_time = self.contact_sensor.data.current_contact_time
        # shape: [num_envs, num_bodies]

        reward = torch.zeros(
            contact_time.shape[0],
            device=contact_time.device,
            dtype=contact_time.dtype,
        )

        for left_ids, right_ids in self._pair_indices:
            left_contact = contact_time[:, left_ids] > 0.0
            right_contact = contact_time[:, right_ids] > 0.0

            # True when both left and right are in air
            both_air = (~left_contact) & (~right_contact)

            # 여러 body가 매칭된 경우 평균
            reward += both_air.float().mean(dim=1)

        return -reward.unsqueeze(1)
    
class feet_too_near_humanoid(Reward):
    def __init__(self, env, body_names, threshold, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self._body_ids = self.asset.find_bodies(body_names)[0]
        self.threshold = threshold
        
    def compute(self) -> torch.Tensor:
        feet_pos = self.asset.data.body_pos_w[:, self._body_ids, :]
        distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
        return -(self.threshold - distance).clamp(min=0).unsqueeze(1)
