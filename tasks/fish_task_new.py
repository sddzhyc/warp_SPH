# pyright: reportInvalidTypeForm=false
import numpy as np
import warp as wp

from .fish_task import FishTask


@wp.kernel
def compute_fish_body_tail_loss_kernel(
    rigid_q: wp.array(dtype=wp.quat),
    target_rigid_q: wp.array(dtype=wp.quat),
    body_rigid_id: int,
    tail_rigid_id: int,
    loss: wp.array(dtype=float),
):
    """计算鱼身 + 鱼尾的四元数姿态 loss 之和"""
    tid = wp.tid()
    if tid == 0:
        # body loss
        qb = rigid_q[body_rigid_id]
        tqb = target_rigid_q[body_rigid_id]
        dqb = qb - tqb
        lb = 0.5 * wp.dot(dqb, dqb)

        # tail loss
        qt = rigid_q[tail_rigid_id]
        tqt = target_rigid_q[tail_rigid_id]
        dqt = qt - tqt
        lt = 0.5 * wp.dot(dqt, dqt)

        wp.atomic_add(loss, 0, lb + lt)


@wp.kernel
def compute_fish_position_loss_kernel(
    rigid_x: wp.array(dtype=wp.vec3),
    body_rigid_id: int,
    target_body_x: wp.vec3,
    position_weight: float,
    loss: wp.array(dtype=float),
):
    """计算鱼身当前位置与初始位置的距离平方 loss，激励鱼保持在原位附近"""
    tid = wp.tid()
    if tid == 0:
        current_x = rigid_x[body_rigid_id]
        d = current_x - target_body_x
        wp.atomic_add(loss, 0, position_weight * wp.dot(d, d))


@wp.kernel
def compute_fish_forward_velocity_loss_kernel(
    rigid_q: wp.array(dtype=wp.quat),
    rigid_v: wp.array(dtype=wp.vec3),
    body_rigid_id: int,
    forward_velocity_weight: float,
    target_speed: float,
    loss: wp.array(dtype=float),
):
    """计算鱼身线速度沿朝向（局部 X 轴）分量的目标速度 loss，激励鱼以 target_speed 向前游动"""
    tid = wp.tid()
    if tid == 0:
        q = rigid_q[body_rigid_id]
        v = rigid_v[body_rigid_id]
        # 局部 X 轴（鱼头方向）变换到世界空间
        forward = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
        forward_vel = wp.dot(forward, v)
        # (target_speed - forward_vel)^2，始终非负，鼓励鱼达到目标速度
        diff = target_speed - forward_vel
        wp.atomic_add(loss, 0, forward_velocity_weight * diff * diff)


@wp.kernel
def compute_tail_periodic_loss_kernel(
    rigid_q_arrays: wp.array(dtype=wp.quat, ndim=2),
    body_rigid_id: int,
    tail_rigid_id: int,
    end_step: int,
    frame_dt: float,
    control_interval: int,
    control_start_step: int,
    ref_freq: float,
    ref_amp: float,
    phase_offset: float,
    weight: float,
    loss: wp.array(dtype=float),
):
    """计算尾部周期性摆动 loss（仅控制步）:

        L_tail = (1/N) * Σ_{t ∈ control_steps} (q_t - q_ref(t))²

    其中:
        q_t        = body→tail 绕 Y 轴的当前关节角 (rad)
        q_ref(t)   = ref_amp * sin(2π * ref_freq * t * dt + phase_offset)
        N          = 控制步总数
    """
    tid = wp.tid()
    if tid == 0:
        accum = float(0.0)
        count = int(0)
        for step in range(control_start_step, end_step + 1):
            # 仅处理控制步
            if (step - control_start_step) % control_interval != 0:
                continue
            count += 1
            t = float(step) * frame_dt

            # 从 body/tail 四元数计算当前关节角（与 PD 内核公式一致）
            qb = rigid_q_arrays[step][body_rigid_id]
            qt = rigid_q_arrays[step][tail_rigid_id]

            qb_x = qb[0]; qb_y = qb[1]; qb_z = qb[2]; qb_w = qb[3]
            qt_x = qt[0]; qt_y = qt[1]; qt_z = qt[2]; qt_w = qt[3]

            # q_rel = qb^{-1} * qt
            qr_w = qb_w * qt_w + qb_x * qt_x + qb_y * qt_y + qb_z * qt_z
            qr_y = qb_w * qt_y + qb_x * qt_z - qb_y * qt_w - qb_z * qt_x

            current_angle = 2.0 * wp.atan2(qr_y, qr_w)

            # 参考正弦轨迹
            phase = 1.0 * wp.pi * ref_freq * t + phase_offset
            ref_angle = ref_amp * wp.sin(phase)

            diff = current_angle - ref_angle
            accum += weight * diff * diff

        if count > 0:
            accum = accum / float(count)
        wp.atomic_add(loss, 0, accum)


@wp.kernel
def sum_loss_components_kernel(
    loss_pose: wp.array(dtype=float),
    loss_position: wp.array(dtype=float),
    loss_velocity: wp.array(dtype=float),
    loss_tail_periodic: wp.array(dtype=float),
    loss_total: wp.array(dtype=float),
):
    """将四个组件的 loss 累加到 total loss 数组中"""
    tid = wp.tid()
    if tid == 0:
        loss_total[0] = loss_pose[0] + loss_position[0] + loss_velocity[0] + loss_tail_periodic[0]


class FishTaskNew(FishTask):
    """专门用于 fish_on_water_new 场景（body + tail 两段关节鱼）

    与 FishTask 的区别:
    - loss = body 姿态 loss + tail 姿态 loss + 前向线速度 loss
    - 由 FishEnv 的 action 模式自动切换为关节力矩（施加在 body 和 tail 上大小相等方向相反的 Y 轴角速度）
    """

    def __init__(self, sim, use_v_loss=False, no_pose_loss=False, use_x_loss=False,
                 use_tail_periodic_loss=False, tail_periodic_freq=2.0, tail_periodic_amp=0.5,
                 tail_periodic_weight=1.0, control_interval=1260, control_start_step=0):
        self.use_v_loss = use_v_loss
        self.no_pose_loss = no_pose_loss
        self.use_x_loss = use_x_loss
        self.use_tail_periodic_loss = use_tail_periodic_loss
        self.tail_periodic_freq = float(tail_periodic_freq)
        self.tail_periodic_amp = float(tail_periodic_amp)
        self.tail_periodic_weight = float(tail_periodic_weight)
        self.control_interval = int(control_interval)
        self.control_start_step = int(control_start_step)
        super().__init__(sim)
        self.forward_velocity_weight = 1.0  # 前向线速度 loss 权重
        self.target_speed = 0.2  # 目标前向速度 (m/s)，超过此速度不再奖励
        self.position_weight = 10.0  # 位置 loss 权重

    def init_targets(self):
        """为 body 和 tail 都设置目标四元数，并记录鱼身初始位置"""
        self.target_rigid_q = wp.zeros_like(self.sim.rbs.rigid_quaternion)
        target_q_np = self.sim.rbs.rigid_quaternion.numpy().copy()

        rigid_bodies = self.sim.cfg.get_rigid_bodies() if hasattr(self.sim, "cfg") and self.sim.cfg is not None else []

        import scipy.spatial.transform as st

        for i, rb in enumerate(rigid_bodies):
            obj_id = int(rb.get("objectId", i + 1))
            if obj_id not in (self.body_rigid_id, self.tail_rigid_id):
                continue

            # 从 JSON 读取目标四元数（优先 targetQuaternion，fallback targetAngleInDegree）
            if "targetQuaternion" in rb:
                q = np.array(rb["targetQuaternion"], dtype=np.float32)
            elif "targetQ" in rb:
                q = np.array(rb["targetQ"], dtype=np.float32)
            elif "targetAngleInDegree" in rb:
                euler = rb["targetAngleInDegree"]
                q = st.Rotation.from_euler('xyz', euler, degrees=True).as_quat().astype(np.float32)
            else:
                continue
            target_q_np[obj_id] = q

        wp.copy(self.target_rigid_q, wp.array(target_q_np, dtype=wp.quat))

        # 记录鱼身初始位置（用于 use_x_loss）
        if self.use_x_loss:
            body_x = self.sim.rbs.rigid_x.numpy()[self.body_rigid_id].copy()
            self.target_body_x = wp.vec3(float(body_x[0]), float(body_x[1]), float(body_x[2]))
            print(f"[Task] target body position: {self.target_body_x}")   

    def compute_loss(self, end_step=None):
        """body + tail 四元数姿态 loss + 前向线速度 loss + 位置 loss + 尾部周期性摆动 loss

        每个 loss 组件写入独立的 Warp 数组（loss_pose / loss_position / loss_velocity / loss_tail_periodic），
        最后汇总到 self.sim.loss 供 backward 使用。这样可以在训练日志中分别打印各项 loss 的值。
        """
        if self.sim.num_objects <= 0:
            return
        if end_step is None:
            end_step = self.sim.sim_steps

        if not self.no_pose_loss:
            wp.launch(
                kernel=compute_fish_body_tail_loss_kernel,
                dim=1,
                inputs=[
                    self.sim.rigid_quaternion_arrays[end_step],
                    self.target_rigid_q,
                    self.body_rigid_id,
                    self.tail_rigid_id,
                ],
                outputs=[self.sim.loss_pose],
            )

        # 位置 loss：鼓励鱼身保持在初始位置附近
        if self.use_x_loss:
            wp.launch(
                kernel=compute_fish_position_loss_kernel,
                dim=1,
                inputs=[
                    self.sim.rigid_x_arrays[end_step],
                    self.body_rigid_id,
                    self.target_body_x,
                    self.position_weight,
                ],
                outputs=[self.sim.loss_position],
            )

        # 前向线速度 loss：激励鱼沿朝向（局部 X 轴 → 世界 X 轴方向）达到 target_speed
        if self.use_v_loss:
            wp.launch(
                kernel=compute_fish_forward_velocity_loss_kernel,
                dim=1,
                inputs=[
                    self.sim.rigid_quaternion_arrays[end_step],
                    self.sim.rigid_v_arrays[end_step],
                    self.body_rigid_id,
                    self.forward_velocity_weight,
                    self.target_speed,
                ],
                outputs=[self.sim.loss_velocity],
            )

        # 尾部周期性摆动 loss：激励实际关节角跟踪正弦参考轨迹 q_ref(t) = A*sin(2π*freq*t)
        if self.use_tail_periodic_loss:
            # 将 Python list 中的逐步四元数数组拷贝到一个 2D Warp 数组中，
            # 以便内核在单次 launch 中遍历所有时间步并累积 loss。
            num_steps = int(end_step) + 1
            num_objs = int(self.sim.num_objects)
            device = self.sim.rigid_quaternion_arrays[0].device
            quat_2d = wp.empty((num_steps, num_objs), dtype=wp.quat, device=device, requires_grad=True)
            for step in range(num_steps):
                wp.copy(quat_2d[step], self.sim.rigid_quaternion_arrays[step])

            phase_offset = float(getattr(self.sim, 'sine_phase_offset', 0.0))
            wp.launch(
                kernel=compute_tail_periodic_loss_kernel,
                dim=1,
                inputs=[
                    quat_2d,
                    self.body_rigid_id,
                    self.tail_rigid_id,
                    end_step,
                    self.sim.frame_dt,
                    self.control_interval,
                    self.control_start_step,
                    self.tail_periodic_freq,
                    self.tail_periodic_amp,
                    phase_offset,
                    self.tail_periodic_weight,
                ],
                outputs=[self.sim.loss_tail_periodic],
            )
            # 打印周期性 loss 诊断信息
            if end_step > 0:
                q_arr = self.sim.rigid_quaternion_arrays[end_step].numpy()
                qb = q_arr[self.body_rigid_id]
                qt = q_arr[self.tail_rigid_id]
                qr_w = qb[3] * qt[3] + qb[0] * qt[0] + qb[1] * qt[1] + qb[2] * qt[2]
                qr_y = qb[3] * qt[1] + qb[0] * qt[2] - qb[1] * qt[3] - qb[2] * qt[0]
                ja = float(2.0 * np.arctan2(qr_y, qr_w))
                t_val = float(end_step) * float(self.sim.frame_dt)
                phase = 2.0 * np.pi * self.tail_periodic_freq * t_val + phase_offset
                ref_angle = self.tail_periodic_amp * np.sin(phase)
                print(
                    f"[tail_periodic_loss] end_step={end_step} t={t_val:.4f} "
                    f"joint_angle={ja:.4f} ref_angle={ref_angle:.4f} "
                    f"freq={self.tail_periodic_freq:.2f} amp={self.tail_periodic_amp:.2f} "
                    f"weight={self.tail_periodic_weight:.2f}"
                )

        # 将四个组件 loss 汇总到 self.sim.loss（用于 backward）
        wp.launch(
            kernel=sum_loss_components_kernel,
            dim=1,
            inputs=[
                self.sim.loss_pose,
                self.sim.loss_position,
                self.sim.loss_velocity,
                self.sim.loss_tail_periodic,
            ],
            outputs=[self.sim.loss],
        )

    def get_loss_physics_info(self, end_step):
        """返回 loss 计算使用的物理量字典，用于训练日志打印。
        根据启用的 loss 组件，提取对应物理量在当前 step 的值。
        """
        info = {}
        if self.sim.num_objects <= 0:
            return info

        # 姿态 loss 使用的物理量：body 和 tail 的四元数
        if not self.no_pose_loss:
            q_arr = self.sim.rigid_quaternion_arrays[end_step].numpy()
            tq_arr = self.target_rigid_q.numpy()
            body_q = q_arr[self.body_rigid_id].copy()
            tail_q = q_arr[self.tail_rigid_id].copy()
            target_body_q = tq_arr[self.body_rigid_id].copy()
            target_tail_q = tq_arr[self.tail_rigid_id].copy()
            # 四元数差值范数（角度误差的 2 倍，越小越好）
            body_angle_err = float(np.linalg.norm(body_q - target_body_q))
            tail_angle_err = float(np.linalg.norm(tail_q - target_tail_q))
            info["body_angle_err"] = body_angle_err
            info["tail_angle_err"] = tail_angle_err

        # 位置 loss 使用的物理量：body 当前位置
        if self.use_x_loss:
            x_arr = self.sim.rigid_x_arrays[end_step].numpy()
            body_x = x_arr[self.body_rigid_id].copy()
            target_x = np.array([self.target_body_x[0], self.target_body_x[1], self.target_body_x[2]], dtype=np.float32)
            pos_err = float(np.linalg.norm(body_x - target_x))
            info["body_pos"] = body_x
            info["pos_err"] = pos_err

        # 速度 loss 使用的物理量：body 线速度 和 前向速度分量
        if self.use_v_loss:
            v_arr = self.sim.rigid_v_arrays[end_step].numpy()
            body_v = v_arr[self.body_rigid_id].copy()
            # 计算前向速度（同 loss kernel 的算法）
            q_arr = self.sim.rigid_quaternion_arrays[end_step].numpy()
            q = q_arr[self.body_rigid_id]
            # 局部 X 轴变换到世界空间
            import scipy.spatial.transform as st
            forward = st.Rotation.from_quat([q[0], q[1], q[2], q[3]]).apply([1.0, 0.0, 0.0])
            forward_vel = float(np.dot(forward, body_v))
            info["body_vel"] = body_v
            info["forward_vel"] = forward_vel

        # 尾部周期性摆动 loss 使用的物理量：末端关节角与参考角度
        if self.use_tail_periodic_loss and end_step > 0:
            q_arr = self.sim.rigid_quaternion_arrays[end_step].numpy()
            qb = q_arr[self.body_rigid_id].copy()
            qt = q_arr[self.tail_rigid_id].copy()
            qr_w = qb[3] * qt[3] + qb[0] * qt[0] + qb[1] * qt[1] + qb[2] * qt[2]
            qr_y = qb[3] * qt[1] + qb[0] * qt[2] - qb[1] * qt[3] - qb[2] * qt[0]
            ja = float(2.0 * np.arctan2(qr_y, qr_w))
            t_val = float(end_step) * float(self.sim.frame_dt)
            phase_offset = float(getattr(self.sim, 'sine_phase_offset', 0.0))
            phase = 2.0 * np.pi * self.tail_periodic_freq * t_val + phase_offset
            ref_angle = float(self.tail_periodic_amp * np.sin(phase))
            info["tail_periodic_angle_err"] = float(ja - ref_angle)
            info["joint_angle"] = ja
            info["ref_angle"] = ref_angle

        # 读取各 loss 组件的当前值（已由 compute_loss 写入对应数组）
        info["loss_pose"] = float(self.sim.loss_pose.numpy()[0])
        info["loss_position"] = float(self.sim.loss_position.numpy()[0])
        info["loss_velocity"] = float(self.sim.loss_velocity.numpy()[0])
        info["loss_tail_periodic"] = float(self.sim.loss_tail_periodic.numpy()[0])

        return info
