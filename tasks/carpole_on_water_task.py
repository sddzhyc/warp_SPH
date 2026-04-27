import numpy as np
import warp as wp

from .base_task import Task


@wp.kernel
def assign_car_init_velocity_kernel(
    opt_car_v: wp.array(dtype=wp.vec3),
    car_rigid_id: int,
    rigid_v: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if tid == 0:
        rigid_v[car_rigid_id] = opt_car_v[0]


@wp.kernel
def compute_rod_rotation_loss_kernel(
    rigid_q: wp.array(dtype=wp.quat),
    target_rigid_q: wp.array(dtype=wp.quat),
    rod_rigid_id: int,
    loss: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid == 0:
        q = rigid_q[rod_rigid_id]
        tq = target_rigid_q[rod_rigid_id]
        dq = q - tq
        l_rot = 0.5 * wp.dot(dq, dq)
        wp.atomic_add(loss, 0, l_rot)


class CarPoleOnWaterTask(Task):
    def __init__(self, sim):
        super().__init__(sim)
        self.rod_rigid_id = None
        self.car_rigid_id = None

        self.target_rigid_q = None
        self.target_rigid_x = None

        self.opt_rigid_v = None
        self.optimizer = None

        self._resolve_object_ids()
        self.init_targets()
        self.init_optimizer()

    @staticmethod
    def _euler_xyz_deg_to_quat_xyzw(euler_deg):
        # Convert xyz Euler angles in degrees to quaternion (x, y, z, w).
        ex, ey, ez = np.deg2rad(np.array(euler_deg, dtype=np.float32))
        q_wp = wp.quat_rpy(float(ex), float(ey), float(ez))
        q = np.array([q_wp[0], q_wp[1], q_wp[2], q_wp[3]], dtype=np.float32)
        n = np.linalg.norm(q)
        if n > 1e-8:
            q /= n
        return q

    def _resolve_object_ids(self):
        rigid_bodies = self.sim.cfg.get_rigid_bodies() if hasattr(self.sim, "cfg") and self.sim.cfg is not None else []

        rod_id = None
        car_id = None

        for i, rb in enumerate(rigid_bodies):
            obj_id = int(rb.get("objectId", i + 1))
            comment = str(rb.get("comment", "")).lower()
            if rod_id is None and ("pole" in comment or "rod" in comment):
                rod_id = obj_id
            if car_id is None and "car" in comment:
                car_id = obj_id

        if rod_id is None and len(rigid_bodies) > 0:
            rod_id = int(rigid_bodies[0].get("objectId", 1))
        if car_id is None:
            if len(rigid_bodies) > 1:
                car_id = int(rigid_bodies[1].get("objectId", 2))
            elif len(rigid_bodies) > 0:
                car_id = int(rigid_bodies[0].get("objectId", 1))
            else:
                car_id = 1

        self.rod_rigid_id = rod_id
        self.car_rigid_id = car_id

    def _get_rod_cfg(self):
        rigid_bodies = self.sim.cfg.get_rigid_bodies() if hasattr(self.sim, "cfg") and self.sim.cfg is not None else []
        for i, rb in enumerate(rigid_bodies):
            obj_id = int(rb.get("objectId", i + 1))
            if obj_id == self.rod_rigid_id:
                return rb
        return None

    def init_targets(self):
        self.target_rigid_q = wp.zeros_like(self.sim.rbs.rigid_quaternion)
        self.target_rigid_x = wp.zeros_like(self.sim.rbs.rigid_x)

        target_q_np = self.sim.rbs.rigid_quaternion.numpy().copy()
        target_x_np = self.sim.rbs.rigid_x.numpy().copy()

        rod_cfg = self._get_rod_cfg()
        if rod_cfg is not None:
            if "targetAngleInDegree" in rod_cfg and self.rod_rigid_id < len(target_q_np):
                target_q_np[self.rod_rigid_id] = self._euler_xyz_deg_to_quat_xyzw(rod_cfg["targetAngleInDegree"])
            else:
                target_q_np[self.rod_rigid_id] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

            if "targetX" in rod_cfg and self.rod_rigid_id < len(target_x_np):
                target_x_np[self.rod_rigid_id] = np.array(rod_cfg["targetX"], dtype=np.float32)

        wp.copy(self.target_rigid_q, wp.array(target_q_np, dtype=wp.quat))
        wp.copy(self.target_rigid_x, wp.array(target_x_np, dtype=wp.vec3))

    def init_optimizer(self):
        initial_v = self.sim.rigid_v_arrays[0].numpy()[self.car_rigid_id]
        device = self.sim.rigid_v_arrays[0].device

        self.opt_rigid_v = wp.array([initial_v], dtype=wp.vec3, device=device, requires_grad=True)
        self.optimizer = wp.optim.Adam([self.opt_rigid_v], lr=self.sim.train_rate)

        # Keep list form for compatibility with existing training logger/override paths.
        self.sim.opt_var = [self.opt_rigid_v]

    def compute_loss(self, end_step=None):
        if self.sim.num_objects <= 0:
            return

        if end_step is None:
            end_step = self.sim.sim_steps

        wp.launch(
            kernel=compute_rod_rotation_loss_kernel,
            dim=1,
            inputs=[
                self.sim.rigid_quaternion_arrays[end_step],
                self.target_rigid_q,
                self.rod_rigid_id,
            ],
            outputs=[self.sim.loss],
        )

    def init_simulation_state(self):
        if self.sim.num_objects <= 0:
            return

        with self.sim.tapes[-1]:
            wp.launch(
                kernel=assign_car_init_velocity_kernel,
                dim=1,
                inputs=[
                    self.opt_rigid_v,
                    self.car_rigid_id,
                ],
                outputs=[self.sim.rigid_v_arrays[0]],
            )

        self.sim.rbs.rigid_v = self.sim.rigid_v_arrays[0]

    def clear_grad(self):
        if self.opt_rigid_v is not None and self.opt_rigid_v.grad:
            self.opt_rigid_v.grad.zero_()

    def get_loss_state_info(self):
        info = {}
        if self.sim.num_objects <= 0:
            return info

        final_q = self.sim.rigid_quaternion_arrays[self.sim.sim_steps].numpy()[self.rod_rigid_id]
        target_q = self.target_rigid_q.numpy()[self.rod_rigid_id]
        final_x = self.sim.rigid_x_arrays[self.sim.sim_steps].numpy()[self.rod_rigid_id]
        target_x = self.target_rigid_x.numpy()[self.rod_rigid_id]
        car_v0 = self.sim.rigid_v_arrays[0].numpy()[self.car_rigid_id]

        info["rod_final_q"] = final_q
        info["rod_target_q"] = target_q
        info["rod_final_x"] = final_x
        info["rod_target_x"] = target_x
        info["car_init_v"] = car_v0
        return info

    def norm_final_grad(self):
        if self.opt_rigid_v is None or not self.opt_rigid_v.grad:
            return

        grad_np = self.opt_rigid_v.grad.numpy()
        n = np.linalg.norm(grad_np)
        if n > 1e-10:
            self.opt_rigid_v.grad = wp.array(grad_np / n, dtype=wp.vec3, device=self.opt_rigid_v.device)
