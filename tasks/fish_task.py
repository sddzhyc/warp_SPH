# pyright: reportInvalidTypeForm=false
import numpy as np
import torch
import warp as wp

from .base_task import Task


@wp.kernel
def assign_fish_init_y_angular_velocity_kernel(
    opt_omega: wp.array(dtype=wp.vec3),
    body_rigid_id: int,
    rigid_omega: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if tid == 0:
        rigid_omega[body_rigid_id] = opt_omega[0]


@wp.kernel
def compute_fish_upright_loss_kernel(
    rigid_q: wp.array(dtype=wp.quat),
    rigid_x: wp.array(dtype=wp.vec3),
    body_rigid_id: int,
    up_vec: wp.vec3,
    target_pos: wp.vec3,
    loss: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid == 0:
        q = rigid_q[body_rigid_id]
        R = wp.quat_to_matrix(q)
        # local long axis (x axis) in world space
        local_x = wp.vec3(1.0, 0.0, 0.0)
        world_dir = R @ local_x
        d = world_dir - up_vec
        x = rigid_x[body_rigid_id]
        dp = x - target_pos
        l = 0.5 * wp.dot(d, d) + 0.5 * wp.dot(dp, dp)
        wp.atomic_add(loss, 0, l)


class FishTask(Task):
    def __init__(self, sim):
        super().__init__(sim)
        self.body_rigid_id = None
        self.tail_rigid_id = None

        self.target_up = None
        self.target_pos = None

        self.opt_omega = None
        self.optimizer = None

        self._resolve_object_ids()
        self.init_targets()
        self.init_optimizer()

    def _resolve_object_ids(self):
        rigid_bodies = self.sim.cfg.get_rigid_bodies() if hasattr(self.sim, "cfg") and self.sim.cfg is not None else []

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

        self.body_rigid_id = body_id
        self.tail_rigid_id = tail_id

    def init_targets(self):
        self.target_up = wp.vec3(0.0, 1.0, 0.0)
        self.target_pos = None
        up_np = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        target_pos_np = None
        # read from config if provided in rigid body cfg
        rigid_bodies = self.sim.cfg.get_rigid_bodies() if hasattr(self.sim, "cfg") and self.sim.cfg is not None else []
        for i, rb in enumerate(rigid_bodies):
            obj_id = int(rb.get("objectId", i + 1))
            if obj_id == self.body_rigid_id:
                if "targetUp" in rb:
                    up_np = np.array(rb.get("targetUp"), dtype=np.float32)
                    self.target_up = wp.vec3(float(up_np[0]), float(up_np[1]), float(up_np[2]))
                if "targetPosition" in rb:
                    target_pos_np = np.array(rb.get("targetPosition"), dtype=np.float32)
                elif "targetPos" in rb:
                    target_pos_np = np.array(rb.get("targetPos"), dtype=np.float32)
                elif "targetX" in rb:
                    target_pos_np = np.array(rb.get("targetX"), dtype=np.float32)
                break

        if target_pos_np is None:
            if self.sim.num_objects > 0 and self.body_rigid_id is not None:
                target_pos_np = self.sim.rigid_x_arrays[0].numpy()[self.body_rigid_id].astype(np.float32)
            else:
                target_pos_np = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.target_pos = wp.vec3(float(target_pos_np[0]), float(target_pos_np[1]), float(target_pos_np[2]))

    def init_optimizer(self):
        # initialize a full vec3 angular velocity optimization variable
        initial = (0.0, 0.0, 0.0)
        if self.sim.num_objects > 0:
            initial_omega = self.sim.rigid_omega_arrays[0].numpy()[self.body_rigid_id]
            initial = (float(initial_omega[0]), float(initial_omega[1]), float(initial_omega[2]))
        device = self.sim.rigid_omega_arrays[0].device if self.sim.num_objects > 0 else None
        torch_device = torch.device(str(device)) if device is not None else torch.device("cpu")

        init_tensor = torch.tensor([[initial[0], initial[1], initial[2]]], dtype=torch.float32, device=torch_device)
        self.opt_omega = wp.from_torch(init_tensor, dtype=wp.vec3, requires_grad=True)
        self.optimizer = wp.optim.Adam([self.opt_omega], lr=self.sim.train_rate)
        self.sim.opt_var = [self.opt_omega]

    def compute_loss(self, end_step=None):
        if self.sim.num_objects <= 0:
            return
        if end_step is None:
            end_step = self.sim.sim_steps

        wp.launch(
            kernel=compute_fish_upright_loss_kernel,
            dim=1,
            inputs=[
                self.sim.rigid_quaternion_arrays[end_step],
                self.sim.rigid_x_arrays[end_step],
                self.body_rigid_id,
                self.target_up,
                self.target_pos,
            ],
            outputs=[self.sim.loss],
        )

    def init_simulation_state(self):
        if self.sim.num_objects <= 0:
            return

        current_tape = self.sim.tapes[-1] if hasattr(self.sim, "tapes") and len(self.sim.tapes) > 0 else None
        if current_tape is not None:
            with current_tape:
                wp.launch(
                    kernel=assign_fish_init_y_angular_velocity_kernel,
                    dim=1,
                    inputs=[
                        self.opt_omega,
                        self.body_rigid_id,
                    ],
                    outputs=[self.sim.rigid_omega_arrays[0]],
                )
        else:
            wp.launch(
                kernel=assign_fish_init_y_angular_velocity_kernel,
                dim=1,
                inputs=[
                    self.opt_omega,
                    self.body_rigid_id,
                ],
                outputs=[self.sim.rigid_omega_arrays[0]],
            )

        self.sim.rbs.rigid_omega = self.sim.rigid_omega_arrays[0]

    def clear_grad(self):
        if self.opt_omega is not None and self.opt_omega.grad:
            self.opt_omega.grad.zero_()
