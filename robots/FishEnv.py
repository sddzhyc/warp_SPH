# pyright: reportInvalidTypeForm=false
import os
import numpy as np
import torch
import warp as wp

from robots.CartPoleEnv import DifferentiableCartPoleEnv
from robots.pd_controller import add_joint_pd_torque_kernel
from fluid_encoder.encoder_utils import get_local_fluid_grid_around_rigid
from rigid_fluid_coupling import MaterialType
from utils import export_backward_data, save_fluid_velocity_field


@wp.kernel
def add_control_delta_omega_kernel(
    delta_omega: wp.array(dtype=wp.float32),
    rigid_id: int,
    rigid_omega: wp.array(dtype=wp.vec3),
):
    """单刚体模式：将 body 的角速度设为 (0, action, 0)"""
    tid = wp.tid()
    if tid == 0:
        rigid_omega[rigid_id] = wp.vec3(0.0, delta_omega[0], 0.0)


@wp.kernel
def add_joint_torque_omega_kernel(
    action: wp.array(dtype=wp.float32),
    body_id: int,
    tail_id: int,
    rigid_omega: wp.array(dtype=wp.vec3),
):
    """关节力矩模式：对 body 和 tail 施加大小相等、方向相反的 Y 轴角速度增量
    模拟在 revolute 关节处施加的力矩：body 获得 +action[0]，tail 获得 -action[0]"""
    tid = wp.tid()
    if tid == 0:
        delta = action[0]
        # 在现有角速度上叠加（保留物理模拟的 omega）
        ob = rigid_omega[body_id]
        ot = rigid_omega[tail_id]
        rigid_omega[body_id] = wp.vec3(ob[0], ob[1] + delta, ob[2])
        rigid_omega[tail_id] = wp.vec3(ot[0], ot[1] - delta, ot[2])


@wp.kernel
def add_body_x_torque_kernel(
    action: wp.array(dtype=wp.float32),
    body_id: int,
    rigid_omega: wp.array(dtype=wp.vec3),
):
    """对鱼身直接施加 X 轴角速度增量（action[1]），不作用于 tail"""
    tid = wp.tid()
    if tid == 0:
        ob = rigid_omega[body_id]
        rigid_omega[body_id] = wp.vec3(ob[0] + action[1], ob[1], ob[2])


@wp.kernel
def add_joint_omega_action_kernel(
    action: wp.array(dtype=wp.float32),
    body_id: int,
    tail_id: int,
    rigid_omega: wp.array(dtype=wp.vec3),
):
    """直接关节角速度模式：action[0] = 目标关节相对角速度 (rad/s)

    将 body 和 tail 的 Y 轴角速度设置为使相对角速度等于 action[0]：
        joint_omega = tail_omega_y - body_omega_y = action[0]

    修正量均分到 body 和 tail（大小相等、方向相反）：
        delta = (action[0] - (tail_omega_y - body_omega_y)) / 2
        body  -= delta
        tail  += delta
    """
    tid = wp.tid()
    if tid == 0:
        ob = rigid_omega[body_id]
        ot = rigid_omega[tail_id]
        current_rel = ot[1] - ob[1]
        delta = (action[0] - current_rel) * 0.5
        rigid_omega[body_id] = wp.vec3(ob[0], ob[1] - delta, ob[2])
        rigid_omega[tail_id] = wp.vec3(ot[0], ot[1] + delta, ot[2])


@wp.kernel
def apply_swim_bladder_volume_delta_kernel(
    action: wp.array(dtype=wp.float32),
    action_idx: int,
    swim_bladder_volume: wp.array(dtype=wp.float32),
    min_volume: float,
    max_volume: float,
    dt: float,
):
    """将控制器的 Δv 累加到鱼鳔体积，并钳制到 [min_volume, max_volume]"""
    tid = wp.tid()
    if tid == 0:
        delta_v = action[action_idx]
        new_v = swim_bladder_volume[0] + delta_v * dt
        new_v = wp.clamp(new_v, min_volume, max_volume)
        swim_bladder_volume[0] = new_v


class DifferentiableFishEnv(DifferentiableCartPoleEnv):
    def __init__(self, sim, control_interval, backward_interval, control_start_step=2, include_time=True, norm_grad=False, total_sim_steps=None, random_start=False, use_relative_state=False, sine_action=False, omega_action=False):
        super().__init__(sim, control_interval, backward_interval, control_start_step, include_time, norm_grad, total_sim_steps, random_start=random_start)
        self.use_relative_state = bool(use_relative_state)
        self.sine_action = bool(sine_action)
        self.omega_action = bool(omega_action)

        # 从场景配置读取挡板运动频率，用于时间编码
        self._wave_freq = float(sim.wave_boundary_frequency)
        # PD 控制器增益（用于 sine_action 模式下的关节角度跟踪）
        self.pd_kp = float(getattr(sim, 'pd_kp', 100.0))
        self.pd_kd = float(getattr(sim, 'pd_kd', 2.0))
        print(f"[Robot Env] Initialized DifferentiableFishEnv with control_interval={control_interval}, backward_interval={backward_interval}, use_relative_state={self.use_relative_state}, sine_action={self.sine_action}, omega_action={self.omega_action}, wave_freq={self._wave_freq}, pd_kp={self.pd_kp}, pd_kd={self.pd_kd}")

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

        # 鱼鳔浮力：默认启用（由 JSON 配置 swimBladderEnable 控制），自动使用鱼身刚体 ID
        self.swim_bladder_body_id = self.body_rigid_id
        self.swim_bladder_enable = bool(getattr(sim, "swim_bladder_enable", False))
        sim.swim_bladder_body_id = self.body_rigid_id

    def _get_bladder_info(self):
        """覆写父类：返回 (bladder_volume, buoyancy_force)。"""
        if not self.swim_bladder_enable or self.swim_bladder_body_id < 0:
            return None, None
        V = float(self.sim.swim_bladder_volume.numpy()[0])
        # buoyancy force = fluid_density * |g| * V
        F = float(self.sim.base_density) * abs(float(self.sim.gravity)) * V
        return V, F

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

        # 鱼鳔体积
        if self.swim_bladder_enable:
            bladder_v = self.sim.swim_bladder_volume.numpy()
            state_parts.append(float(bladder_v[0]))

        if self.include_time:
            t = float(step) * float(self.sim.frame_dt)
            # 使用场景配置中的挡板运动频率进行 sin/cos 时间编码
            omega = 2.0 * np.pi * self._wave_freq
            state_parts.append(float(np.sin(omega * t)))
            state_parts.append(float(np.cos(omega * t)))
        else:
            # Append zeros for consistency when time is not included
            state_parts.append(0.0)
            state_parts.append(0.0)

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

        # 鱼鳔体积
        if self.swim_bladder_enable:
            bladder_v = self.sim.swim_bladder_volume.numpy()
            state_parts.append(float(bladder_v[0]))

        # no time feature appended for relative state
        return np.asarray(state_parts, dtype=np.float32)

    def _get_bladder_action_idx(self, action_dim):
        """返回鱼鳔体积 delta 在 action 向量中的索引。

        约定: action 的最后一维始终为鱼鳔体积变化 Δv。
        - action_dim=2: [Δω_y, Δv]（两段鱼关节力矩 + 鱼鳔）
        - action_dim=3: [Δω_y, Δω_x, Δv]（关节力矩 + X轴力矩 + 鱼鳔）
        """
        if not self.swim_bladder_enable:
            return None
        if self.body_rigid_id == self.tail_rigid_id:
            # 单段鱼
            return 0 if action_dim <= 1 else action_dim - 1
        else:
            # 两段鱼：最后一维是 Δv
            return action_dim - 1 if action_dim >= 2 else None

    def _apply_bladder_volume_delta(self, step, action_wp, action_dim, tape):
        bladder_idx = self._get_bladder_action_idx(action_dim)
        if bladder_idx is None:
            return
        # 注意：调用方如果已在 tape 上下文中，直接 launch 即可，不要再嵌套 with tape:
        wp.launch(
            kernel=apply_swim_bladder_volume_delta_kernel,
            dim=1,
            inputs=[
                action_wp,
                bladder_idx,
                self.sim.swim_bladder_volume,
                self.sim.swim_bladder_min_volume,
                self.sim.swim_bladder_max_volume,
                self.sim.dt,
            ],
        )

    def apply_action(self, step, action_wp, tape=None):
        action_dim = action_wp.shape[0] if hasattr(action_wp, 'shape') else 1
        # 累计时间偏移保证正弦波跨 segment 连续（offset 由 CartPoleEnv 管理）
        sim_time = getattr(self, '_sine_time_offset', 0.0) + float(step) * float(self.sim.frame_dt)

        # 两段鱼（body != tail）→ 关节力矩模式：等大反向作用于 body 和 tail
        if self.body_rigid_id != self.tail_rigid_id:
            # PD 控制器模式: action[0]=频率(Hz), action[1]=最大摆动角度(rad)
            # PD 控制器根据目标角度轨迹和当前关节状态计算力矩
            if self.sine_action:
                phase_offset = float(getattr(self.sim, 'sine_phase_offset', 0.0))
                if tape is None:
                    wp.launch(
                        kernel=add_joint_pd_torque_kernel,
                        dim=1,
                        inputs=[action_wp, self.body_rigid_id, self.tail_rigid_id,
                                sim_time, phase_offset, self.pd_kp, self.pd_kd,
                                self.sim.rigid_quaternion_arrays[step],
                                self.sim.rigid_omega_arrays[step]],
                        outputs=[self.sim.rigid_torque_arrays[step]],
                    )
                    # 鱼鳔体积变化: action 最后一维
                    self._apply_bladder_volume_delta(step, action_wp, action_dim, tape)
                    return
                with tape:
                    wp.launch(
                        kernel=add_joint_pd_torque_kernel,
                        dim=1,
                        inputs=[action_wp, self.body_rigid_id, self.tail_rigid_id,
                                sim_time, phase_offset, self.pd_kp, self.pd_kd,
                                self.sim.rigid_quaternion_arrays[step],
                                self.sim.rigid_omega_arrays[step]],
                        outputs=[self.sim.rigid_torque_arrays[step]],
                    )
                    self._apply_bladder_volume_delta(step, action_wp, action_dim, tape)
                return

            # 直接关节角速度模式: action[0] = 目标关节相对角速度 (rad/s)
            if self.omega_action:
                if tape is None:
                    wp.launch(
                        kernel=add_joint_omega_action_kernel,
                        dim=1,
                        inputs=[action_wp, self.body_rigid_id, self.tail_rigid_id],
                        outputs=[self.sim.rigid_omega_arrays[step]],
                    )
                    self._apply_bladder_volume_delta(step, action_wp, action_dim, tape)
                    return
                with tape:
                    wp.launch(
                        kernel=add_joint_omega_action_kernel,
                        dim=1,
                        inputs=[action_wp, self.body_rigid_id, self.tail_rigid_id],
                        outputs=[self.sim.rigid_omega_arrays[step]],
                    )
                    self._apply_bladder_volume_delta(step, action_wp, action_dim, tape)
                return

            if tape is None:
                wp.launch(
                    kernel=add_joint_torque_omega_kernel,
                    dim=1,
                    inputs=[action_wp, self.body_rigid_id, self.tail_rigid_id],
                    outputs=[self.sim.rigid_omega_arrays[step]],
                )
                # X 轴力矩仅作用于鱼身（action_dim >= 3 时 action[1] 是 X 轴力矩）
                if action_dim >= 3:
                    wp.launch(
                        kernel=add_body_x_torque_kernel,
                        dim=1,
                        inputs=[action_wp, self.body_rigid_id],
                        outputs=[self.sim.rigid_omega_arrays[step]],
                    )
                # 鱼鳔体积变化
                self._apply_bladder_volume_delta(step, action_wp, action_dim, tape)
                return
            with tape:
                wp.launch(
                    kernel=add_joint_torque_omega_kernel,
                    dim=1,
                    inputs=[action_wp, self.body_rigid_id, self.tail_rigid_id],
                    outputs=[self.sim.rigid_omega_arrays[step]],
                )
                # X 轴力矩仅作用于鱼身
                if action_dim >= 3:
                    wp.launch(
                        kernel=add_body_x_torque_kernel,
                        dim=1,
                        inputs=[action_wp, self.body_rigid_id],
                        outputs=[self.sim.rigid_omega_arrays[step]],
                    )
                # 鱼鳔体积变化
                self._apply_bladder_volume_delta(step, action_wp, action_dim, tape)
            return

        # 单段鱼 → 原行为：只控制 body（Y 轴角速度）
        if tape is None:
            wp.launch(
                kernel=add_control_delta_omega_kernel,
                dim=1,
                inputs=[action_wp, self.body_rigid_id],
                outputs=[self.sim.rigid_omega_arrays[step]],
            )
            # 鱼鳔体积变化
            self._apply_bladder_volume_delta(step, action_wp, action_dim, tape)
            return

        with tape:
            wp.launch(
                kernel=add_control_delta_omega_kernel,
                dim=1,
                inputs=[action_wp, self.body_rigid_id],
                outputs=[self.sim.rigid_omega_arrays[step]],
            )
            # 鱼鳔体积变化
            self._apply_bladder_volume_delta(step, action_wp, action_dim, tape)

    def _action_to_wp_vec3(self, action_tensor):
        target_device = torch.device(str(self.sim.rigid_omega_arrays[0].device))
        action_torch = action_tensor.to(device=target_device, dtype=torch.float32)
        return wp.from_torch(action_torch, dtype=wp.float32, requires_grad=True)