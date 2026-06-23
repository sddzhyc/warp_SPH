# pyright: reportInvalidTypeForm=false
"""PD 控制器模块：根据目标关节角度轨迹，使用 PD 控制器计算应施加的关节力矩。

与 open-loop 正弦波力矩不同，PD 控制器会读取当前关节角（从 body/tail 四元数计算）
和当前关节角速度（从角速度差计算），然后计算跟踪误差并产生修正力矩。

    torque = Kp * (target_angle - current_angle) + Kd * (target_omega - current_omega)

目标轨迹:
    target_angle  = amplitude * sin(2π * frequency * time + phase_offset)
    target_omega  = amplitude * 2π * frequency * cos(2π * frequency * time + phase_offset)

其中 amplitude 为最大摆动角度（弧度），不再表示力矩幅度。
"""

import warp as wp


@wp.kernel
def add_joint_pd_torque_kernel(
    action: wp.array(dtype=wp.float32),
    body_id: int,
    tail_id: int,
    sim_time: float,
    phase_offset: float,
    Kp: float,
    Kd: float,
    rigid_q: wp.array(dtype=wp.quat),
    rigid_omega: wp.array(dtype=wp.vec3),
    rigid_torque: wp.array(dtype=wp.vec3),
):
    """PD 控制器关节力矩: action[0]=频率(Hz), action[1]=最大摆动角度(rad)

    根据目标正弦摆动轨迹，使用 PD 控制器计算力矩。
    等大反向作用于 body 和 tail。

    参数:
        action:       [频率(Hz), 最大摆动角度(rad)]
        body_id:      鱼身刚体索引
        tail_id:      鱼尾刚体索引
        sim_time:     当前模拟时间 (s)
        phase_offset: 初始相位偏移 (rad)
        Kp:           比例增益 (N·m/rad)
        Kd:           微分增益 (N·m·s/rad)
        rigid_q:      当前步的刚体四元数数组 (Warp quat: x,y,z,w)
        rigid_omega:  当前步的刚体角速度数组
        rigid_torque: 输出力矩数组（在原地叠加）
    """
    tid = wp.tid()
    if tid == 0:
        freq = action[0]
        amp = action[1]  # 最大摆动角度 (rad)

        # ---- 目标轨迹 ----
        phase = 2.0 * wp.pi * freq * sim_time + phase_offset
        target_angle = amp * wp.sin(phase)
        target_omega = amp * 2.0 * wp.pi * freq * wp.cos(phase)

        # ---- 当前关节角：通过 body→tail 相对四元数计算 ----
        # Warp quat 布局: (x, y, z, w), 即 q[0]=ix, q[1]=iy, q[2]=iz, q[3]=r
        # 不使用 wp.quat_inverse / quat_mult，而是手动展开四元数运算以确保
        # 自动微分能正确传播梯度
        qb_x = rigid_q[body_id][0]
        qb_y = rigid_q[body_id][1]
        qb_z = rigid_q[body_id][2]
        qb_w = rigid_q[body_id][3]
        qt_x = rigid_q[tail_id][0]
        qt_y = rigid_q[tail_id][1]
        qt_z = rigid_q[tail_id][2]
        qt_w = rigid_q[tail_id][3]

        # q_rel = qb^{-1} * qt, 其中 qb^{-1} = (-qb_x, -qb_y, -qb_z, qb_w)
        # Hamilton 积 (w, x, y, z 约定):
        #   w = a_w*b_w - a_x*b_x - a_y*b_y - a_z*b_z
        #   x = a_w*b_x + a_x*b_w + a_y*b_z - a_z*b_y
        #   y = a_w*b_y - a_x*b_z + a_y*b_w + a_z*b_x
        #   z = a_w*b_z + a_x*b_y - a_y*b_x + a_z*b_w
        # a = qb^{-1} = (qb_w, -qb_x, -qb_y, -qb_z) in (w,x,y,z)
        qr_w = qb_w * qt_w + qb_x * qt_x + qb_y * qt_y + qb_z * qt_z
        qr_y = qb_w * qt_y + qb_x * qt_z - qb_y * qt_w - qb_z * qt_x

        # revolute 关节绕 Y 轴: 角度 = 2 * atan2(q_rel.y, q_rel.w)
        current_angle = 2.0 * wp.atan2(qr_y, qr_w)

        # ---- 当前关节角速度：Y 轴相对角速度差 ----
        ob = rigid_omega[body_id]
        ot = rigid_omega[tail_id]
        current_omega = ot[1] - ob[1]

        # ---- PD 控制 ----
        angle_error = target_angle - current_angle
        omega_error = target_omega - current_omega
        torque = Kp * angle_error + Kd * omega_error

        # ---- 等大反向作用于 body 和 tail（绕 Y 轴）----
        # 正力矩 → 尾相对于身向 +Y 方向转动（joint_angle 增大）
        # body 获得 -torque（反作用），tail 获得 +torque
        tb = rigid_torque[body_id]
        tt = rigid_torque[tail_id]
        rigid_torque[body_id] = wp.vec3(tb[0], tb[1] - torque, tb[2])
        rigid_torque[tail_id] = wp.vec3(tt[0], tt[1] + torque, tt[2])
