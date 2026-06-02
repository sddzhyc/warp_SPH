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
    def __init__(self, sim, control_interval, backward_interval, control_start_step=2, include_time=True, norm_grad=False, total_sim_steps=None, random_start=False, use_relative_state=False):
        super().__init__(sim, control_interval, backward_interval, control_start_step, include_time, norm_grad, total_sim_steps, random_start=random_start)
        self.use_relative_state = bool(use_relative_state)

        # Prefer task-provided ids (fish_task sets these), otherwise try to resolve from cfg
        self.body_rigid_id = int(getattr(sim.task, "body_rigid_id", -1)) if hasattr(sim, "task") else -1
        self.tail_rigid_id = int(getattr(sim.task, "tail_rigid_id", -1)) if hasattr(sim, "task") else -1
        self.state_rigid_ids = list(getattr(sim.task, "state_rigid_ids", [])) if hasattr(sim, "task") else []

        if not self.state_rigid_ids:
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
            self.state_rigid_ids = [self.body_rigid_id] if self.body_rigid_id == self.tail_rigid_id else [self.body_rigid_id, self.tail_rigid_id]

        # keep compatibility with parent fields
        self.rod_rigid_id = self.tail_rigid_id
        self.car_rigid_id = self.body_rigid_id

    def get_state(self, step):
        if self.use_relative_state:
            return self.get_state_relative(step)

        rigid_x = self.sim.rigid_x_arrays[step].numpy()
        rigid_q = self.sim.rigid_quaternion_arrays[step].numpy()
        rigid_v = self.sim.rigid_v_arrays[step].numpy()
        rigid_omega = self.sim.rigid_omega_arrays[step].numpy()

        body_ids = self.state_rigid_ids if self.state_rigid_ids else [self.body_rigid_id, self.tail_rigid_id]
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

    def get_state_relative(self, step):
        """
        Return a state vector where:
        - position is replaced by displacement from previous step (x_t - x_{t-1})
        - velocity is expressed in the fish body centroid reference frame
        - time feature is omitted
        """
        rigid_x = self.sim.rigid_x_arrays[step].numpy()
        rigid_q = self.sim.rigid_quaternion_arrays[step].numpy()
        rigid_v = self.sim.rigid_v_arrays[step].numpy()
        rigid_omega = self.sim.rigid_omega_arrays[step].numpy()

        # previous positions (use current if step==0)
        if step > 0:
            rigid_x_prev = self.sim.rigid_x_arrays[step - 1].numpy()
        else:
            rigid_x_prev = rigid_x

        body_ids = self.state_rigid_ids if self.state_rigid_ids else [self.body_rigid_id, self.tail_rigid_id]
        state_parts = []

        # choose fish body id to form local frame
        fish_body_id = body_ids[0] if body_ids else self.body_rigid_id
        fish_q = rigid_q[fish_body_id]

        def quat_mult(q1, q2):
            # q assumed in (w, x, y, z) order
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
            ], dtype=np.float32)

        def rotate_to_local_frame(v, q):
            # rotate vector v from global to local using body quaternion q
            q_conj = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)
            v_quat = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
            temp = quat_mult(q_conj, v_quat)
            res = quat_mult(temp, q)
            return res[1:4]

        for rid in body_ids:
            x = rigid_x[rid]
            x_prev = rigid_x_prev[rid]
            displacement = x - x_prev

            q = rigid_q[rid]
            v = rigid_v[rid]
            omega = rigid_omega[rid]

            # express velocity in fish body local frame
            v_local = rotate_to_local_frame(v, fish_q)

            state_parts.extend([
                float(displacement[0]), float(displacement[1]), float(displacement[2]),
                float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                float(v_local[0]), float(v_local[1]), float(v_local[2]),
                float(omega[0]), float(omega[1]), float(omega[2]),
            ])

        # no time feature appended for relative state
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
