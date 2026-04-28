import numpy as np
import warp as wp

from .base_task import Task


@wp.kernel
def assign_rigid_init_state_kernel(
    opt_rigid_v: wp.array(dtype=wp.vec3),
    opt_rigid_omega: wp.array(dtype=wp.vec3),
    rigid_id: int,
    rigid_v: wp.array(dtype=wp.vec3),
    rigid_omega: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if tid == 0:
        rigid_v[rigid_id] = opt_rigid_v[0]
        rigid_omega[rigid_id] = opt_rigid_omega[0]


@wp.kernel
def compute_target_rigid_loss_kernel(
    rigid_x: wp.array(dtype=wp.vec3),
    target_rigid_x: wp.array(dtype=wp.vec3),
    rigid_q: wp.array(dtype=wp.quat),
    target_rigid_q: wp.array(dtype=wp.quat),
    rigid_id: int,
    position_weight: float,
    rotation_weight: float,
    loss: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid != 0:
        return

    diff_pos = rigid_x[rigid_id] - target_rigid_x[rigid_id]
    loss_pos = 0.5 * position_weight * wp.dot(diff_pos, diff_pos)

    q = rigid_q[rigid_id]
    tq = target_rigid_q[rigid_id]
    diff_rot_1 = q - tq
    diff_rot_2 = q + tq
    loss_rot_1 = 0.5 * rotation_weight * wp.dot(diff_rot_1, diff_rot_1)
    loss_rot_2 = 0.5 * rotation_weight * wp.dot(diff_rot_2, diff_rot_2)
    loss_rot = wp.min(loss_rot_1, loss_rot_2)

    wp.atomic_add(loss, 0, loss_pos + loss_rot)


class WaterRaftingTask(Task):
    def __init__(self, sim):
        super().__init__(sim)
        self.target_rigid_id = self.get_target_rigid_id()
        self.target_rigid_x = None
        self.target_rigid_q = None
        self.opt_rigid_v = None
        self.opt_rigid_omega = None
        self.optimizer = None

        self.init_targets()
        self.init_optimizer()

    def _get_rigid_body_cfgs(self):
        if hasattr(self.sim, "cfg") and self.sim.cfg is not None:
            rigid_bodies = self.sim.cfg.get_rigid_bodies()
            if rigid_bodies:
                return rigid_bodies

            raw_cfg = getattr(self.sim.cfg, "config", None)
            if isinstance(raw_cfg, dict):
                rigid_bodies_ = raw_cfg.get("RigidBodies", [])
                if rigid_bodies_:
                    return rigid_bodies_
        return []

    def get_target_rigid_id(self):
        rigid_bodies = self.sim.cfg.get_rigid_bodies() if hasattr(self.sim, "cfg") and self.sim.cfg is not None else []
        if len(rigid_bodies) > 0 and "objectId" in rigid_bodies[0]:
            return int(rigid_bodies[0]["objectId"])
        return 1

    @staticmethod
    def _euler_xyz_deg_to_quat_xyzw(euler_deg):
        ex = float(wp.radians(float(euler_deg[0])))
        ey = float(wp.radians(float(euler_deg[1])))
        ez = float(wp.radians(float(euler_deg[2])))
        q_wp = wp.quat_rpy(ex, ey, ez)
        q_wp = wp.normalize(q_wp)
        return np.array([q_wp[0], q_wp[1], q_wp[2], q_wp[3]], dtype=np.float32)

    def _get_target_rigid_cfg(self):
        rigid_bodies = self._get_rigid_body_cfgs()
        for idx, rigid_body in enumerate(rigid_bodies):
            rigid_id = int(rigid_body.get("objectId", rigid_body.get("id", idx + 1)))
            if rigid_id == self.target_rigid_id:
                return rigid_body
        return None

    def init_targets(self):
        self.target_rigid_x = wp.zeros_like(self.sim.rbs.rigid_x)
        self.target_rigid_q = wp.zeros_like(self.sim.rbs.rigid_quaternion)

        target_x_np = self.sim.rbs.rigid_x.numpy().copy()
        target_q_np = self.sim.rbs.rigid_quaternion.numpy().copy()

        target_cfg = self._get_target_rigid_cfg()
        if target_cfg is not None and self.target_rigid_id < len(target_x_np):
            if "targetX" in target_cfg:
                target_x_np[self.target_rigid_id] = np.array(target_cfg["targetX"], dtype=np.float32)
            if "targetAngleInDegree" in target_cfg:
                target_q_np[self.target_rigid_id] = self._euler_xyz_deg_to_quat_xyzw(target_cfg["targetAngleInDegree"])

        wp.copy(self.target_rigid_x, wp.array(target_x_np, dtype=wp.vec3))
        wp.copy(self.target_rigid_q, wp.array(target_q_np, dtype=wp.quat))

    def init_optimizer(self):
        initial_v = self.sim.rigid_v_arrays[0].numpy()[self.target_rigid_id]
        initial_omega = self.sim.rigid_omega_arrays[0].numpy()[self.target_rigid_id]
        device = self.sim.rigid_v_arrays[0].device

        self.opt_rigid_v = wp.array([initial_v], dtype=wp.vec3, device=device, requires_grad=True)
        self.opt_rigid_omega = wp.array([initial_omega], dtype=wp.vec3, device=device, requires_grad=True)
        self.optimizer = wp.optim.Adam([self.opt_rigid_v, self.opt_rigid_omega], lr=self.sim.train_rate)

        self.sim.opt_rigid_v = self.opt_rigid_v
        self.sim.opt_rigid_omega = self.opt_rigid_omega
        self.sim.opt_var = [self.opt_rigid_v, self.opt_rigid_omega]

    def compute_loss(self, end_step=None):
        if self.sim.num_objects <= 0:
            return

        if end_step is None:
            end_step = self.sim.sim_steps

        target_cfg = self._get_target_rigid_cfg()
        position_weight = float(target_cfg.get("positionWeight", 1.0)) if target_cfg is not None else 1.0
        rotation_weight = float(target_cfg.get("rotationWeight", 1.0)) if target_cfg is not None else 1.0

        wp.launch(
            kernel=compute_target_rigid_loss_kernel,
            dim=1,
            inputs=[
                self.sim.rigid_x_arrays[end_step],
                self.target_rigid_x,
                self.sim.rigid_quaternion_arrays[end_step],
                self.target_rigid_q,
                self.target_rigid_id,
                position_weight,
                rotation_weight,
            ],
            outputs=[self.sim.loss],
        )

    def init_simulation_state(self):
        if self.sim.num_objects <= 0:
            return

        with self.sim.tapes[-1]:
            wp.launch(
                kernel=assign_rigid_init_state_kernel,
                dim=1,
                inputs=[
                    self.opt_rigid_v,
                    self.opt_rigid_omega,
                    self.target_rigid_id,
                ],
                outputs=[
                    self.sim.rigid_v_arrays[0],
                    self.sim.rigid_omega_arrays[0],
                ],
            )

        self.sim.rbs.rigid_v = self.sim.rigid_v_arrays[0]
        self.sim.rbs.rigid_omega = self.sim.rigid_omega_arrays[0]

    def clear_grad(self):
        if self.opt_rigid_v is not None and self.opt_rigid_v.grad:
            self.opt_rigid_v.grad.zero_()
        if self.opt_rigid_omega is not None and self.opt_rigid_omega.grad:
            self.opt_rigid_omega.grad.zero_()

    def get_loss_state_info(self):
        info = {}
        if self.sim.num_objects <= 0 or self.target_rigid_id >= self.sim.num_objects:
            return info

        final_x = self.sim.rigid_x_arrays[self.sim.sim_steps].numpy()[self.target_rigid_id]
        target_x = self.target_rigid_x.numpy()[self.target_rigid_id]
        final_q = self.sim.rigid_quaternion_arrays[self.sim.sim_steps].numpy()[self.target_rigid_id]
        target_q = self.target_rigid_q.numpy()[self.target_rigid_id]
        init_v = self.sim.rigid_v_arrays[0].numpy()[self.target_rigid_id]
        init_omega = self.sim.rigid_omega_arrays[0].numpy()[self.target_rigid_id]

        info["target_final_pos"] = final_x
        info["target_target_pos"] = target_x
        info["target_final_q"] = final_q
        info["target_target_q"] = target_q
        info["target_init_v"] = init_v
        info["target_init_omega"] = init_omega
        return info

    def norm_final_grad(self):
        if self.opt_rigid_v is not None and self.opt_rigid_v.grad:
            grad_np = self.opt_rigid_v.grad.numpy()
            grad_norm = np.linalg.norm(grad_np)
            if grad_norm > 1e-10:
                self.opt_rigid_v.grad = wp.array(grad_np / grad_norm, dtype=wp.vec3, device=self.opt_rigid_v.device)

        if self.opt_rigid_omega is not None and self.opt_rigid_omega.grad:
            grad_np = self.opt_rigid_omega.grad.numpy()
            grad_norm = np.linalg.norm(grad_np)
            if grad_norm > 1e-10:
                self.opt_rigid_omega.grad = wp.array(grad_np / grad_norm, dtype=wp.vec3, device=self.opt_rigid_omega.device)