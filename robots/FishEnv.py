# pyright: reportInvalidTypeForm=false
import os
import numpy as np
import torch
import warp as wp

from robots.CartPoleEnv import DifferentiableCartPoleEnv
from fluid_encoder.encoder_utils import get_local_fluid_grid_around_rigid
from rigid_fluid_coupling import MaterialType
from utils import export_backward_data, save_fluid_velocity_field


@wp.kernel
def add_control_delta_omega_kernel(
    delta_omega: wp.array(dtype=wp.vec3),
    rigid_id: int,
    rigid_omega: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if tid == 0:
        rigid_omega[rigid_id] = delta_omega[0]


class DifferentiableFishEnv(DifferentiableCartPoleEnv):
    def __init__(self, sim, control_interval, backward_interval, control_start_step=2, include_time=True, norm_grad=False, total_sim_steps=None):
        super().__init__(sim, control_interval, backward_interval, control_start_step, include_time, norm_grad, total_sim_steps)

        # Prefer task-provided ids (fish_task sets these), otherwise try to resolve from cfg
        self.body_rigid_id = int(getattr(sim.task, "body_rigid_id", -1)) if hasattr(sim, "task") else -1
        self.tail_rigid_id = int(getattr(sim.task, "tail_rigid_id", -1)) if hasattr(sim, "task") else -1

        if self.body_rigid_id < 0 or self.tail_rigid_id < 0:
            # fallback: inspect cfg rigid bodies like fish_task
            rigid_bodies = sim.cfg.get_rigid_bodies() if hasattr(sim, "cfg") and sim.cfg is not None else []
            body_id = None
            tail_id = None
            for i, rb in enumerate(rigid_bodies):
                obj_id = int(rb.get("objectId", i + 1))
                comment = str(rb.get("comment", "")).lower()
                if body_id is None and ("fish_body" in comment or "body" in comment or "fish" in comment):
                    body_id = obj_id
                if tail_id is None and ("tail" in comment):
                    tail_id = obj_id

            if body_id is None and len(rigid_bodies) > 0:
                body_id = int(rigid_bodies[0].get("objectId", 1))
            if tail_id is None:
                if len(rigid_bodies) > 1:
                    tail_id = int(rigid_bodies[1].get("objectId", 2))
                else:
                    tail_id = body_id

            self.body_rigid_id = int(body_id)
            self.tail_rigid_id = int(tail_id)

        # keep compatibility with parent fields
        self.rod_rigid_id = self.tail_rigid_id
        self.car_rigid_id = self.body_rigid_id

    def get_state(self, step):
        rigid_x = self.sim.rigid_x_arrays[step].numpy()
        rigid_q = self.sim.rigid_quaternion_arrays[step].numpy()
        rigid_v = self.sim.rigid_v_arrays[step].numpy()
        rigid_omega = self.sim.rigid_omega_arrays[step].numpy()

        body_ids = [self.body_rigid_id, self.tail_rigid_id]
        state_parts = []

        for rid in body_ids:
            x = rigid_x[rid]
            q = rigid_q[rid]
            v = rigid_v[rid]
            omega = rigid_omega[rid]

            state_parts.extend([
                x[0], x[1], x[2],
                q[0], q[1], q[2], q[3],
                v[0], v[1], v[2],
                omega[0], omega[1], omega[2],
            ])

        if self.include_time:
            t = float(step) * float(self.sim.frame_dt)
        else:
            t = 0.0
        state_parts.append(t)

        return np.asarray(state_parts, dtype=np.float32)

    def apply_action(self, step, action_wp, tape=None):
        # For fish we treat action as a delta angular velocity applied to the fish body
        if tape is None:
            wp.launch(
                kernel=add_control_delta_omega_kernel,
                dim=1,
                inputs=[action_wp, self.body_rigid_id],
                outputs=[self.sim.rigid_omega_arrays[step]],
            )
            return

        with tape:
            wp.launch(
                kernel=add_control_delta_omega_kernel,
                dim=1,
                inputs=[action_wp, self.body_rigid_id],
                outputs=[self.sim.rigid_omega_arrays[step]],
            )

    # keep CartPoleEnv's _action_to_wp_vec3 signature
    def _action_to_wp_vec3(self, action_xyz):
        target_device = torch.device(str(self.sim.rigid_omega_arrays[0].device))
        action_torch = action_xyz.reshape(1, 3).to(device=target_device, dtype=torch.float32)
        return wp.from_torch(action_torch, dtype=wp.vec3, requires_grad=True)
