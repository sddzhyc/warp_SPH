# pyright: reportInvalidTypeForm=false
import os
import numpy as np
import torch
import warp as wp
from fluid_encoder.encoder_utils import get_local_fluid_grid_around_rigid
from rigid_fluid_coupling import MaterialType
from utils import export_backward_data, save_fluid_velocity_field

@wp.kernel
def add_control_delta_v_kernel(
    delta_v: wp.array(dtype=wp.vec3),
    rigid_id: int,
    rigid_v: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if tid == 0:
        rigid_v[rigid_id] = delta_v[0]
        # rigid_v[rigid_id] = rigid_v[rigid_id] + delta_v[0] TODO: 临时直接设置速度增量，后续改为叠加

class DifferentiableCartPoleEnv:
    def __init__(self, sim, control_interval, backward_interval, control_start_step=2, include_time=True, norm_grad=False, total_sim_steps=None, random_start=False, use_relative_state=False):
        self.sim = sim
        self.control_interval = int(control_interval)
        self.backward_interval = int(backward_interval)
        self.control_start_step = int(control_start_step)
        self.include_time = bool(include_time)
        self.norm_grad = bool(norm_grad)
        self.random_start = bool(random_start)
        self.use_relative_state = bool(use_relative_state)
        # phase offset within a control interval for the first control application
        self.control_phase_offset = 0

        if self.control_interval <= 0:
            raise ValueError("control_interval must be > 0")
        if self.backward_interval <= 0:
            raise ValueError("backward_interval must be > 0")
        if self.control_start_step < 0:
            raise ValueError("control_start_step must be >= 0")

        self.rod_rigid_id = int(getattr(sim.task, "rod_rigid_id", 1))
        self.car_rigid_id = int(getattr(sim.task, "car_rigid_id", 2))

        self.action_records = []
        self.total_sim_steps = int(total_sim_steps) if total_sim_steps is not None else 0
        self.export_before_diff_reset = False
        self.export_output_interval = 10
        self.export_series_prefix = None
        self.export_ply_counter = 0
        self.export_fluid_field = False
        self.export_fluid_field_dir = None
        self.export_fluid_field_launch_time = None
        self.export_fluid_field_counter = 0
        self.export_fluid_field_grid_shape = (32, 32, 32)
        self.export_fluid_field_grid_spacing = 0.02
        self.export_fluid_field_influence_radius = None
        self.last_segment_steps = 0
        self._stored_sine_action_wp = None  # 正弦波模式：存储上一次控制输出的 action
        self._sine_time_offset = 0.0         # 正弦波模式：跨 segment 累计时间偏移

    def reset_episode(self):
        self.sim.clear_grad()
        self.sim.reset()

        self.action_records = []
        self._stored_sine_action_wp = None
        self._sine_time_offset = 0.0
        self._stored_sine_action_wp = None
        # Do NOT reset export_ply_counter here to avoid overwriting previously exported
        # PLY files across episodes/segments. The counter is initialized in the ctor
        # and may be explicitly set by caller when a fresh test run is desired.
        self.export_fluid_field_counter = 0
        # choose random phase offset for first control application within one control interval
        if self.random_start and self.control_interval > 0:
            try:
                self.control_phase_offset = int(np.random.randint(0, self.control_interval))
            except Exception:
                self.control_phase_offset = 0
        else:
            self.control_phase_offset = 0

    def set_total_sim_steps(self, total_sim_steps):
        total_sim_steps = int(total_sim_steps)
        if total_sim_steps <= 0:
            raise ValueError("total_sim_steps must be > 0")
        self.total_sim_steps = total_sim_steps

    def should_control(self, step):
        # First control step is control_start_step + control_phase_offset, then every control_interval
        first_step = int(self.control_start_step) + int(self.control_phase_offset)
        if step < first_step:
            return False
        return (step - first_step) % self.control_interval == 0

    def get_state(self, step):
        if self.use_relative_state:
            return self.get_state_relative(step)

        rigid_x = self.sim.rigid_x_arrays[step].numpy()
        rigid_q = self.sim.rigid_quaternion_arrays[step].numpy()
        rigid_v = self.sim.rigid_v_arrays[step].numpy()
        rigid_omega = self.sim.rigid_omega_arrays[step].numpy()

        body_ids = [self.car_rigid_id, self.rod_rigid_id]
        state_parts = []

        for rid in body_ids:
            x = rigid_x[rid]
            q = rigid_q[rid]
            v = rigid_v[rid]
            omega = rigid_omega[rid]

            # Per body: position(3) + quaternion(4) + velocity(3) + angular velocity(3)
            state_parts.extend([
                x[0], x[1], x[2],
                q[0], q[1], q[2], q[3],
                v[0], v[1], v[2],
                omega[0], omega[1], omega[2],
            ])

        if self.include_time:
            t = float(step) * float(self.sim.frame_dt)
            # 使用场景配置中的挡板运动频率进行 sin/cos 时间编码
            wave_freq = float(getattr(self.sim, 'wave_boundary_frequency', 1.0) or 1.0)
            omega = 2.0 * np.pi * wave_freq
            state_parts.append(float(np.sin(omega * t)))
            state_parts.append(float(np.cos(omega * t)))
        else:
            # Append zeros for consistency when time is not included
            state_parts.append(0.0)
            state_parts.append(0.0)

        return np.asarray(state_parts, dtype=np.float32)

    def get_state_relative(self, step):
        rigid_x = self.sim.rigid_x_arrays[step].numpy()
        rigid_q = self.sim.rigid_quaternion_arrays[step].numpy()
        rigid_v = self.sim.rigid_v_arrays[step].numpy()
        rigid_omega = self.sim.rigid_omega_arrays[step].numpy()

        if step > 0:
            rigid_x_prev = self.sim.rigid_x_arrays[step - 1].numpy()
        else:
            rigid_x_prev = rigid_x

        body_ids = [self.car_rigid_id, self.rod_rigid_id]
        state_parts = []

        car_q = rigid_q[self.car_rigid_id]

        def quat_mult(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ], dtype=np.float32)

        def rotate_to_local_frame(v, q):
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

            v_local = rotate_to_local_frame(v, car_q)

            state_parts.extend([
                float(displacement[0]), float(displacement[1]), float(displacement[2]),
                float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                float(v_local[0]), float(v_local[1]), float(v_local[2]),
                float(omega[0]), float(omega[1]), float(omega[2]),
            ])

        return np.asarray(state_parts, dtype=np.float32)

    def _get_fluid_particles(self, step):
        pos_all = self.sim.x_arrays[step].numpy()
        vel_all = self.sim.v_arrays[step].numpy()

        if hasattr(self.sim, "materialMarks") and hasattr(self.sim.materialMarks, "material") and self.sim.materialMarks.material is not None:
            mat = self.sim.materialMarks.material.numpy()
            mask = (mat == int(MaterialType.FLUID))
            if np.any(mask):
                return pos_all[mask], vel_all[mask]

        fluid_num = int(getattr(self.sim, "fluid_particle_num", 0))
        if fluid_num > 0:
            return pos_all[:fluid_num], vel_all[:fluid_num]

        return pos_all, vel_all

    def get_fluid_velocity_field(self, step, grid_shape=None, grid_spacing=None, influence_radius=None):
        if grid_shape is None:
            grid_shape = self.export_fluid_field_grid_shape
        if grid_spacing is None:
            grid_spacing = self.export_fluid_field_grid_spacing
        if influence_radius is None:
            influence_radius = self.export_fluid_field_influence_radius

        rigid_center = self.sim.rigid_x_arrays[step].numpy()[self.car_rigid_id].astype(np.float32)
        particle_pos, particle_vel = self._get_fluid_particles(step)
        velocity_field, weight = get_local_fluid_grid_around_rigid(
            particle_pos=particle_pos,
            particle_vel=particle_vel,
            rigid_center=rigid_center,
            grid_shape=tuple(grid_shape),
            grid_spacing=float(grid_spacing),
            influence_radius=influence_radius,
        )
        metadata = {
            "step": int(step),
            "time": float(step) * float(self.sim.frame_dt),
            "rigid_center": rigid_center.astype(np.float32),
            "grid_shape": np.asarray(tuple(grid_shape), dtype=np.int32),
            "grid_spacing": np.asarray(float(grid_spacing), dtype=np.float32),
        }
        if influence_radius is not None:
            metadata["influence_radius"] = np.asarray(float(influence_radius), dtype=np.float32)
        return velocity_field, weight, metadata

    def _action_to_wp_vec3(self, action_xyz):
        target_device = torch.device(str(self.sim.rigid_v_arrays[0].device))
        action_torch = action_xyz.reshape(1, 3).to(device=target_device, dtype=torch.float32)
        return wp.from_torch(action_torch, dtype=wp.vec3, requires_grad=True)

    def _get_bladder_info(self):
        """返回 (bladder_volume, buoyancy_force) 或 (None, None)（若鱼鳔未启用）。
        由 FishEnv 覆写以提供鱼鳔状态。"""
        return None, None

    def _get_current_joint_angle(self, step):
        """计算 body→tail 绕 Y 轴的当前关节角 (rad)。
        使用与 PD 内核相同的四元数相对旋转公式。"""
        body_id = getattr(self, 'body_rigid_id', None)
        tail_id = getattr(self, 'tail_rigid_id', None)
        if body_id is None or tail_id is None or body_id == tail_id:
            return None
        qb = self.sim.rigid_quaternion_arrays[step].numpy()[body_id].copy()
        qt = self.sim.rigid_quaternion_arrays[step].numpy()[tail_id].copy()
        # Warp quat: (x, y, z, w) — 与 PD 内核中手动展开的公式一致
        qb_x, qb_y, qb_z, qb_w = qb[0], qb[1], qb[2], qb[3]
        qt_x, qt_y, qt_z, qt_w = qt[0], qt[1], qt[2], qt[3]
        # q_rel = qb^{-1} * qt
        qr_w = qb_w * qt_w + qb_x * qt_x + qb_y * qt_y + qb_z * qt_z
        qr_y = qb_w * qt_y + qb_x * qt_z - qb_y * qt_w - qb_z * qt_x
        return float(2.0 * np.arctan2(qr_y, qr_w))

    def _get_pd_target_angle(self, action, sim_time):
        """计算 PD 控制器的目标摆动角度 (rad)。
        action[0]=频率(Hz), action[1]=最大摆动角度(rad)"""
        freq = float(action[0])
        amp = float(action[1])
        phase_offset = float(getattr(self.sim, 'sine_phase_offset', 0.0))
        phase = 2.0 * np.pi * freq * sim_time + phase_offset
        return float(amp * np.sin(phase))

    def _get_position_error(self, step):
        """返回鱼身当前位置与目标位置之差 (dx, dy, dz)，用于 use_x_loss 诊断。
        仅在 FishTaskNew 设置了 target_body_x 时有效。"""
        task = getattr(self.sim, 'task', None)
        target = getattr(task, 'target_body_x', None)
        if target is None:
            return None
        body_id = getattr(self, 'body_rigid_id', None)
        if body_id is None:
            return None
        current_x = self.sim.rigid_x_arrays[step].numpy()[body_id]
        return np.array([float(current_x[0] - target[0]),
                         float(current_x[1] - target[1]),
                         float(current_x[2] - target[2])], dtype=np.float32)

    def _get_pd_torque_info(self, action, sim_time, step):
        """返回 PD 控制器的 (torque, target_omega, current_omega, Kp, Kd)。
        仅在 sine_action 模式下有意义。"""
        joint_angle = self._get_current_joint_angle(step)
        target_angle = self._get_pd_target_angle(action, sim_time) if len(action) >= 2 else None
        if target_angle is None or joint_angle is None:
            return None

        freq = float(action[0])
        amp = float(action[1])
        phase_offset = float(getattr(self.sim, 'sine_phase_offset', 0.0))
        phase = 2.0 * np.pi * freq * sim_time + phase_offset
        target_omega = float(amp * 2.0 * np.pi * freq * np.cos(phase))

        body_id = getattr(self, 'body_rigid_id', -1)
        tail_id = getattr(self, 'tail_rigid_id', -1)
        ob = self.sim.rigid_omega_arrays[step].numpy()[body_id]
        ot = self.sim.rigid_omega_arrays[step].numpy()[tail_id]
        current_omega = float(ot[1] - ob[1])

        Kp = float(getattr(self.sim, 'pd_kp', 100.0))
        Kd = float(getattr(self.sim, 'pd_kd', 2.0))
        angle_error = target_angle - joint_angle
        omega_error = target_omega - current_omega
        torque = Kp * angle_error + Kd * omega_error

        return {
            "torque": torque,
            "target_omega": target_omega,
            "current_omega": current_omega,
            "Kp": Kp,
            "Kd": Kd,
        }

    def apply_action(self, step, action_wp, tape=None):
        if tape is None:
            wp.launch(
                kernel=add_control_delta_v_kernel,
                dim=1,
                inputs=[action_wp, self.car_rigid_id],
                outputs=[self.sim.rigid_v_arrays[step]],
            )
            return

        with tape:
            wp.launch(
                kernel=add_control_delta_v_kernel,
                dim=1,
                inputs=[action_wp, self.car_rigid_id],
                outputs=[self.sim.rigid_v_arrays[step]],
            )

    @staticmethod
    def _normalize_wp_array_grad(wp_array):
        if wp_array is None or wp_array.grad is None:
            return 0.0

        grad_np = wp_array.grad.numpy()
        grad_norm = float(np.linalg.norm(grad_np))
        if grad_norm > 1e-10:
            wp_array.grad.assign(grad_np / grad_norm)
        return grad_norm

    def _normalize_step_state_grads(self, step):
        if not self.norm_grad:
            return

        next_step = step + 1
        if next_step < len(self.sim.x_arrays):
            self._normalize_wp_array_grad(self.sim.x_arrays[next_step])
        if next_step < len(self.sim.v_arrays):
            self._normalize_wp_array_grad(self.sim.v_arrays[next_step])

        if self.sim.num_rigid_bodies > 0 and next_step < len(self.sim.rigid_x_arrays):
            self._normalize_wp_array_grad(self.sim.rigid_x_arrays[next_step])
            self._normalize_wp_array_grad(self.sim.rigid_v_arrays[next_step])
            self._normalize_wp_array_grad(self.sim.rigid_omega_arrays[next_step])
            self._normalize_wp_array_grad(self.sim.rigid_quaternion_arrays[next_step])

    def _compute_loss_with_optional_end_step(self, tape, end_step):
        with tape:
            try:
                self.sim.task.compute_loss(end_step=end_step)
            except TypeError:
                self.sim.task.compute_loss()

    def _extract_physics_info(self, end_step):
        """从 task 中提取 loss 计算使用的物理量信息。"""
        info = {}
        task = getattr(self.sim, "task", None)
        if task is not None and hasattr(task, "get_loss_physics_info"):
            info = task.get_loss_physics_info(end_step)
        return info

    def forward_backward(
        self,
        controller,
        global_step_offset,
        backward_steps,
        loss_threshold=None,
        collect_replay=False,
        replay_buffer=None,
        grad_callback=None,
    ):
        step_tapes = []
        step_action_entries = {}
        replay_transitions = [] if collect_replay else None

        for local_t in range(backward_steps):
            current_tape = wp.Tape()
            step_tapes.append(current_tape)
            self.sim.tapes = step_tapes

            if global_step_offset == 0 and local_t == 0:
                self.sim.task.init_simulation_state()

            global_t = global_step_offset + local_t
            if self.should_control(global_t):
                state_np = self.get_state(local_t)
                action_torch = controller(state_np, local_t).reshape(-1)
                action_wp = self._action_to_wp_vec3(action_torch)

                rod_quat = self.sim.rigid_quaternion_arrays[local_t].numpy()[self.rod_rigid_id].copy()
                current_time = float(global_t) * float(self.sim.frame_dt)

                # 读取鱼身前向线速度（仅 fish task 且启用 use_v_loss 时）
                use_v_loss = getattr(getattr(self.sim, 'task', None), 'use_v_loss', False)
                body_v = None
                forward_vel = None
                body_rigid_id = getattr(self, 'body_rigid_id', None)
                if body_rigid_id is not None and use_v_loss:
                    from scipy.spatial.transform import Rotation as _R
                    bv = self.sim.rigid_v_arrays[local_t].numpy()[body_rigid_id].copy()
                    bq = self.sim.rigid_quaternion_arrays[local_t].numpy()[body_rigid_id].copy()
                    fwd = _R.from_quat([bq[0], bq[1], bq[2], bq[3]]).apply([1.0, 0.0, 0.0])
                    fv = float(np.dot(fwd, bv))
                    body_v = bv
                    forward_vel = fv

                # 正弦波模式：只存储 action 供后续每步连续施加；直接力矩模式：立即施加
                if getattr(self, 'sine_action', False):
                    self._stored_sine_action_wp = action_wp
                else:
                    self.apply_action(local_t, action_wp, current_tape)

                bladder_v, bladder_f = self._get_bladder_info()
                joint_angle = self._get_current_joint_angle(local_t)
                action_np = action_torch.detach().cpu().numpy().astype(np.float32)
                target_angle = None
                torque_info = None
                if getattr(self, 'sine_action', False) and len(action_np) >= 2:
                    target_angle = self._get_pd_target_angle(action_np, current_time)
                    torque_info = self._get_pd_torque_info(action_np, current_time, local_t)
                action_record = {
                    "step": global_t,
                    "time": current_time,
                    "rod_quat": rod_quat,
                    "body_v": body_v,
                    "forward_vel": forward_vel,
                    "action": action_np,
                    "state": state_np,
                    "bladder_volume": bladder_v,
                    "buoyancy_force": bladder_f,
                    "joint_angle": joint_angle,
                    "target_angle": target_angle,
                    "torque_info": torque_info,
                }
                self.action_records.append(action_record)
                step_action_entries[local_t] = {
                    "state": state_np,
                    "action_torch": action_torch,
                    "action_wp": action_wp,
                    "record": action_record,
                }

            # 正弦波模式：每步施加存储的 [freq, amp]，实现连续摆动
            if getattr(self, 'sine_action', False) and \
               getattr(self, '_stored_sine_action_wp', None) is not None:
                self.apply_action(local_t, self._stored_sine_action_wp, current_tape)

            self.sim.step(local_t)

        loss_tape = wp.Tape()
        self.sim.loss_tape = loss_tape
        self.sim.tapes = step_tapes + [loss_tape]

        self._compute_loss_with_optional_end_step(loss_tape, backward_steps)

        with wp.ScopedTimer("Backward Propagation"):
            loss_tape.backward(self.sim.loss)

            action_grad_norms = []
            batched_action_torch_vars = []
            batched_grad_tensors = []

            for local_t in reversed(range(backward_steps)):
                self._normalize_step_state_grads(local_t)
                step_tapes[local_t].backward()

                action_entry = step_action_entries.get(local_t)
                if action_entry is None:
                    continue

                action_wp = action_entry["action_wp"]
                action_torch = action_entry["action_torch"]
                state_np = action_entry["state"]
                record = action_entry["record"]

                if action_wp.grad is not None:
                    grad_tensor = wp.to_torch(action_wp.grad).reshape(-1).to(
                        device=action_torch.device,
                        dtype=action_torch.dtype,
                    )
                else:
                    grad_tensor = torch.zeros_like(action_torch)

                grad_vec = grad_tensor.detach().cpu().numpy().astype(np.float32)
                action_grad_norms.append(float(torch.linalg.norm(grad_tensor).item()))
                record["grad"] = grad_vec.copy()

                fv_str = f" fwd_vel={record['forward_vel']:.4f}" if record.get('forward_vel') is not None else ""
                bv_str = f" body_v=({record['body_v'][0]:.3f},{record['body_v'][1]:.3f},{record['body_v'][2]:.3f})" if record.get('body_v') is not None else ""
                bl_str = f" bladder_v={record['bladder_volume']:.3e} buoy_f={record['buoyancy_force']:.4f}" if record.get('bladder_volume') is not None else ""
                ja_str = f" joint_angle={record['joint_angle']:.4f}" if record.get('joint_angle') is not None else ""
                ta_str = f" target_angle={record['target_angle']:.4f}" if record.get('target_angle') is not None else ""
                ti = record.get('torque_info') or {}
                tq_str = f" torque={ti['torque']:.4f}" if ti.get('torque') is not None else ""
                pe = record.get('pos_error')
                pe_str = f" pos_err=({pe[0]:.3f},{pe[1]:.3f},{pe[2]:.3f})" if pe is not None else ""
                print(
                    f"[action step] step={record['step']} time={record['time']} "
                    f"rod_quat={record['rod_quat']} action={record['action']}"
                    f"{fv_str}{bv_str}{bl_str}{ja_str}{ta_str}{tq_str}{pe_str}"
                    f" grad={record['grad']}"
                )

                if collect_replay:
                    if replay_buffer is None:
                        raise ValueError("replay_buffer must be provided when collect_replay is True")
                    transition = {
                        "state": np.asarray(state_np, dtype=np.float32).copy(),
                        "action_grad": np.asarray(grad_vec, dtype=np.float32).copy(),
                        "action": np.asarray(record["action"], dtype=np.float32).copy(),
                        "step": int(record["step"]),
                        "time": float(record["time"]),
                        "loss": None,
                    }
                    # buffer transitions locally and only flush after loss is computed
                    replay_transitions.append(transition)
                    continue

                if grad_callback is not None:
                    grad_callback(action_torch, grad_tensor, record)

                batched_action_torch_vars.append(action_torch)
                batched_grad_tensors.append(grad_tensor)

        loss_val = float(np.asarray(self.sim.loss.numpy()).reshape(-1)[0])
        # If we collected replay transitions, fill their loss and append to replay_buffer
        if collect_replay and replay_transitions:
            for transition in replay_transitions:
                transition["loss"] = np.asarray(loss_val, dtype=np.float32).copy()
                if hasattr(replay_buffer, "append"):
                    replay_buffer.append(transition)
                else:
                    replay_buffer.add(**transition)
        avg_action_grad_norm = float(np.mean(np.asarray(action_grad_norms, dtype=np.float32))) if action_grad_norms else 0.0
        controls = int(len(step_action_entries))

        exceed_threshold = loss_threshold is not None and loss_val > loss_threshold

        physics_info = self._extract_physics_info(backward_steps)

        return {
            "loss": loss_val,
            "action_grad_norm": avg_action_grad_norm,
            "controls": controls,
            "early_terminated": bool(exceed_threshold),
            "action_torch_vars": batched_action_torch_vars,
            "action_grad_tensors": batched_grad_tensors,
            "physics_info": physics_info,
        }

    def forward_backward_with_controller(
        self,
        controller,
        optimizer,
        global_step_offset,
        backward_steps,
        loss_threshold=None,
        grad_clip_norm=None,
        grad_callback=None,
        batch_buffer=None,
        collect_batch=False,
    ):
        # If collect_batch is True, optimizer may be None and parameter
        # gradients will be collected into `batch_buffer` instead of
        # applying an immediate optimizer.step().
        if not collect_batch and optimizer is None:
            raise ValueError("optimizer must be provided for backward_with_controller when not collecting batch grads")

        step_tapes = []
        step_action_entries = {}

        for local_t in range(backward_steps):
            current_tape = wp.Tape()
            step_tapes.append(current_tape)
            self.sim.tapes = step_tapes

            if global_step_offset == 0 and local_t == 0:
                self.sim.task.init_simulation_state()

            global_t = global_step_offset + local_t
            if self.should_control(global_t):
                state_np = self.get_state(local_t)
                action_torch = controller(state_np, local_t).reshape(-1)
                action_wp = self._action_to_wp_vec3(action_torch)

                rod_quat = self.sim.rigid_quaternion_arrays[local_t].numpy()[self.rod_rigid_id].copy()
                current_time = float(global_t) * float(self.sim.frame_dt)

                # 读取鱼身前向线速度（仅 fish task 且启用 use_v_loss 时）
                use_v_loss = getattr(getattr(self.sim, 'task', None), 'use_v_loss', False)
                body_v = None
                forward_vel = None
                body_rigid_id = getattr(self, 'body_rigid_id', None)
                if body_rigid_id is not None and use_v_loss:
                    from scipy.spatial.transform import Rotation as _R
                    bv = self.sim.rigid_v_arrays[local_t].numpy()[body_rigid_id].copy()
                    bq = self.sim.rigid_quaternion_arrays[local_t].numpy()[body_rigid_id].copy()
                    fwd = _R.from_quat([bq[0], bq[1], bq[2], bq[3]]).apply([1.0, 0.0, 0.0])
                    fv = float(np.dot(fwd, bv))
                    body_v = bv
                    forward_vel = fv

                # 正弦波模式：只存储 action 供后续每步连续施加；直接力矩模式：立即施加
                if getattr(self, 'sine_action', False):
                    self._stored_sine_action_wp = action_wp
                else:
                    self.apply_action(local_t, action_wp, current_tape)

                bladder_v, bladder_f = self._get_bladder_info()
                joint_angle = self._get_current_joint_angle(local_t)
                action_np = action_torch.detach().cpu().numpy().astype(np.float32)
                target_angle = None
                torque_info = None
                pos_error = None
                if getattr(self, 'sine_action', False) and len(action_np) >= 2:
                    target_angle = self._get_pd_target_angle(action_np, current_time)
                    torque_info = self._get_pd_torque_info(action_np, current_time, local_t)
                pos_error = self._get_position_error(local_t)
                action_record = {
                    "step": global_t,
                    "time": current_time,
                    "rod_quat": rod_quat,
                    "body_v": body_v,
                    "forward_vel": forward_vel,
                    "action": action_np,
                    "state": state_np,
                    "bladder_volume": bladder_v,
                    "buoyancy_force": bladder_f,
                    "joint_angle": joint_angle,
                    "target_angle": target_angle,
                    "torque_info": torque_info,
                    "pos_error": pos_error,
                }
                self.action_records.append(action_record)
                step_action_entries[local_t] = {
                    "state": state_np,
                    "action_torch": action_torch,
                    "action_wp": action_wp,
                    "record": action_record,
                }

            # 正弦波模式：每步施加存储的 [freq, amp]，实现连续摆动
            if getattr(self, 'sine_action', False) and \
               getattr(self, '_stored_sine_action_wp', None) is not None:
                self.apply_action(local_t, self._stored_sine_action_wp, current_tape)

            self.sim.step(local_t)

        loss_tape = wp.Tape()
        self.sim.loss_tape = loss_tape
        self.sim.tapes = step_tapes + [loss_tape]

        self._compute_loss_with_optional_end_step(loss_tape, backward_steps)

        if optimizer is not None:
            optimizer.zero_grad()

        with wp.ScopedTimer("Backward Propagation"):
            loss_tape.backward(self.sim.loss)

            action_grad_norms = []
            records = []
            for local_t in reversed(range(backward_steps)):
                self._normalize_step_state_grads(local_t)
                step_tapes[local_t].backward()

                action_entry = step_action_entries.get(local_t)
                if action_entry is None:
                    continue

                action_wp = action_entry["action_wp"]
                action_torch = action_entry["action_torch"]
                record = action_entry["record"]

                if action_wp.grad is not None:
                    grad_tensor = wp.to_torch(action_wp.grad).reshape(-1).to(
                        device=action_torch.device,
                        dtype=action_torch.dtype,
                    )
                else:
                    grad_tensor = torch.zeros_like(action_torch)

                grad_vec = grad_tensor.detach().cpu().numpy().astype(np.float32)
                action_grad_norms.append(float(torch.linalg.norm(grad_tensor).item()))
                record["grad"] = grad_vec.copy()

                records.append(record)

                if grad_callback is not None:
                    grad_callback(action_torch, grad_tensor, record)

                # Backpropagate into controller parameters for this action
                torch.autograd.backward([action_torch], grad_tensors=[grad_tensor])

            for i in reversed(records):
                fv_str = f" fwd_vel={i['forward_vel']:.4f}" if i.get('forward_vel') is not None else ""
                bv_str = f" body_v=({i['body_v'][0]:.3f},{i['body_v'][1]:.3f},{i['body_v'][2]:.3f})" if i.get('body_v') is not None else ""
                bl_str = f" bladder_v={i['bladder_volume']:.3e} buoy_f={i['buoyancy_force']:.4f}" if i.get('bladder_volume') is not None else ""
                ja_str = f" joint_angle={i['joint_angle']:.4f}" if i.get('joint_angle') is not None else ""
                ta_str = f" target_angle={i['target_angle']:.4f}" if i.get('target_angle') is not None else ""
                ti = i.get('torque_info') or {}
                tq_str = f" torque={ti['torque']:.4f}" if ti.get('torque') is not None else ""
                pe = i.get('pos_error')
                pe_str = f" pos_err=({pe[0]:.3f},{pe[1]:.3f},{pe[2]:.3f})" if pe is not None else ""
                print(
                    f"[action step] step={i['step']} time={i['time']} "
                    f"rod_quat={i['rod_quat']} action={i['action']}"
                    f"{fv_str}{bv_str}{bl_str}{ja_str}{ta_str}{tq_str}{pe_str}"
                    f" grad={i['grad']}"
                )

        loss_val = float(np.asarray(self.sim.loss.numpy()).reshape(-1)[0])
        avg_action_grad_norm = float(np.mean(np.asarray(action_grad_norms, dtype=np.float32))) if action_grad_norms else 0.0
        controls = int(len(step_action_entries))
        physics_info = self._extract_physics_info(backward_steps)

        exceed_threshold = loss_threshold is not None and loss_val > loss_threshold
        if exceed_threshold:
            print(
                f"[segment] skip optimizer step: loss {loss_val:.6e} exceeds threshold "
                f"{loss_threshold:.2e}."
            )
            if optimizer is not None:
                optimizer.zero_grad()
            return {
                "loss": loss_val,
                "action_grad_norm": avg_action_grad_norm,
                "controls": controls,
                "early_terminated": True,
                "physics_info": physics_info,
            }

        # If collecting batch gradients, capture per-parameter gradients into batch_buffer
        if collect_batch and batch_buffer is not None:
            # Flatten parameter gradients into a single 1D numpy array
            grad_list = []
            for p in controller.parameters():
                if p.grad is not None:
                    g = p.grad.detach().cpu().reshape(-1).numpy().astype(np.float32)
                else:
                    g = np.zeros(int(p.numel()), dtype=np.float32)
                grad_list.append(g)
            flat_grad = np.concatenate(grad_list, axis=0) if grad_list else np.array([], dtype=np.float32)

            # Store minimal batch element: param gradients + loss + controls info
            batch_item = {
                "param_grad": flat_grad,
                "loss": np.asarray(loss_val, dtype=np.float32).copy(),
                "controls": controls,
            }
            batch_buffer.append(batch_item)

            # Clear gradients on controller to avoid accumulation across segments
            for p in controller.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

            return {
                "loss": loss_val,
                "action_grad_norm": avg_action_grad_norm,
                "controls": controls,
                "early_terminated": False,
                "physics_info": physics_info,
            }

        # Otherwise perform immediate optimizer step
        if grad_clip_norm is not None and grad_clip_norm > 0:
            params = [param for group in optimizer.param_groups for param in group["params"]]
            torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
        optimizer.step()

        return {
            "loss": loss_val,
            "action_grad_norm": avg_action_grad_norm,
            "controls": controls,
            "early_terminated": False,
            "physics_info": physics_info,
        }

    def _backward_one_segment(
        self,
        tape,
        end_step,
        action_torch_vars,
        action_wp_vars,
        action_records,
        loss_threshold=None,
    ):
        self._compute_loss_with_optional_end_step(tape, end_step)
        tape.backward(self.sim.loss)

        loss_val = float(np.asarray(self.sim.loss.numpy()).reshape(-1)[0])
        physics_info = self._extract_physics_info(end_step)
        # Build physics info string for print
        physics_str = ""
        if physics_info:
            parts = []
            if "body_angle_err" in physics_info:
                parts.append(f"body_angle_err={physics_info['body_angle_err']:.4f}")
            if "tail_angle_err" in physics_info:
                parts.append(f"tail_angle_err={physics_info['tail_angle_err']:.4f}")
            if "pos_err" in physics_info:
                parts.append(f"pos_err={physics_info['pos_err']:.4f}")
            if "forward_vel" in physics_info:
                parts.append(f"forward_vel={physics_info['forward_vel']:.4f}")
            if "tail_periodic_angle_err" in physics_info:
                parts.append(f"tail_periodic_angle_err={physics_info['tail_periodic_angle_err']:.4f}")
            if "loss_pose" in physics_info:
                parts.append(f"L_pose={physics_info['loss_pose']:.4e}")
            if "loss_position" in physics_info:
                parts.append(f"L_pos={physics_info['loss_position']:.4e}")
            if "loss_velocity" in physics_info:
                parts.append(f"L_vel={physics_info['loss_velocity']:.4e}")
            if "loss_tail_periodic" in physics_info:
                parts.append(f"L_tail={physics_info['loss_tail_periodic']:.4e}")
            if parts:
                physics_str = " | " + " ".join(parts)
        print(f"[segment] end_step={end_step} loss = {loss_val:.6e}{physics_str}")

        action_grads = []
        for idx, action_wp in enumerate(action_wp_vars):
            record = action_records[idx]

            grad_np = action_wp.grad.numpy()[0] if action_wp.grad is not None else np.zeros(3, dtype=np.float32)
            grad_vec = np.array([grad_np[0], grad_np[1], grad_np[2]], dtype=np.float32)
            action_grads.append(grad_vec)

            bl_str = f" bladder_v={record['bladder_volume']:.3e} buoy_f={record['buoyancy_force']:.4f}" if record.get('bladder_volume') is not None else ""
            print(
                f"[action {idx}] step={record['step']} time={record['time']} "
                f"rod_quat={record['rod_quat']} action={record['action']}"
                f"{bl_str}"
                f" grad={grad_vec}"
            )

        action_grad_norm = 0.0
        if action_grads:
            action_grad_norm = float(np.linalg.norm(np.stack(action_grads, axis=0)))

        exceed_threshold = loss_threshold is not None and loss_val > loss_threshold
        if exceed_threshold:
            print(
                f"[segment] skip optimizer step: loss {loss_val:.6e} exceeds threshold "
                f"{loss_threshold:.2e}."
            )
            return loss_val, action_grad_norm, len(action_torch_vars), True

        return loss_val, action_grad_norm, len(action_torch_vars), False

    def rollout(
        self,
        controller,
        optimizer=None,
        grad_clip_norm=None,
        loss_threshold=None,
        collect_replay=False,
        replay_buffer=None,
        grad_callback=None,
    ):
        self.reset_episode()
        total_sim_steps = int(self.total_sim_steps)
        if total_sim_steps <= 0:
            raise ValueError("total_sim_steps must be set on env before rollout")

        diff_horizon = self.backward_interval * self.control_interval
        if diff_horizon <= 0:
            raise ValueError("backward_interval must be > 0")

        global_step = 0
        processed_steps = 0
        segment_count = 0
        total_controls = 0
        segment_losses = []
        segment_action_grad_norms = []
        early_terminated = False
        rollout_action_torch = []
        rollout_action_grad_tensors = []
        last_physics_info = {}

        while global_step < total_sim_steps:
            backward_steps = min(diff_horizon, total_sim_steps - global_step)
            self.sim.sim_steps = backward_steps

            if collect_replay:
                seg_stats = self.forward_backward(
                    controller=controller,
                    backward_steps=backward_steps,
                    global_step_offset=global_step,
                    loss_threshold=loss_threshold,
                    collect_replay=collect_replay,
                    replay_buffer=replay_buffer,
                    grad_callback=grad_callback,
                )
            else:
                seg_stats = self.forward_backward_with_controller(
                    controller=controller,
                    optimizer=optimizer,
                    backward_steps=backward_steps,
                    global_step_offset=global_step,
                    loss_threshold=loss_threshold,
                    grad_clip_norm=grad_clip_norm,
                    grad_callback=grad_callback,
                )

            segment_count += 1
            total_controls += int(seg_stats["controls"])
            segment_losses.append(float(seg_stats["loss"]))
            segment_action_grad_norms.append(float(seg_stats["action_grad_norm"]))
            processed_steps += backward_steps
            last_physics_info = seg_stats.get("physics_info", {})

            if seg_stats["early_terminated"]:
                early_terminated = True
                print(
                    f"[rollout] segment {segment_count} early stop: loss {seg_stats['loss']:.6e} exceeds threshold {loss_threshold:.2e}."
                )
                break

            global_step += backward_steps
            # 累加正弦波时间偏移，保证跨 segment 相位连续
            if getattr(self, 'sine_action', False):
                self._sine_time_offset += backward_steps * self.sim.frame_dt
            if global_step >= total_sim_steps:
                break

            if self.export_before_diff_reset and self.export_series_prefix:
                seg_output_interval = max(1, int(self.export_output_interval))
                seg_series_prefix = self.export_series_prefix

                seg_dir = os.path.dirname(seg_series_prefix)
                if seg_dir:
                    os.makedirs(seg_dir, exist_ok=True)

                print(
                    f"[rollout] export diff_segment {segment_count}: steps={backward_steps}, "
                    f"output_interval={seg_output_interval}, prefix={seg_series_prefix}"
                )
                self.export_ply_counter = export_backward_data(
                    self.sim,
                    backward_steps,
                    seg_output_interval,
                    seg_series_prefix,
                    start_cnt=self.export_ply_counter,
                )

            self.sim.diff_reset(backward_steps)

        final_loss = float(segment_losses[-1]) if segment_losses else 0.0
        avg_action_grad_norm = float(np.mean(np.asarray(segment_action_grad_norms, dtype=np.float32))) if segment_action_grad_norms else 0.0

        return {
            "loss": final_loss,
            "action_grad_norm": avg_action_grad_norm,
            "controls": int(total_controls),
            "segments": int(segment_count),
            "total_steps": int(processed_steps if early_terminated else total_sim_steps),
            "early_terminated": bool(early_terminated),
            "action_torch_vars": rollout_action_torch,
            "action_grad_tensors": rollout_action_grad_tensors,
            "physics_info": last_physics_info,
        }

    def rollout_forward_only(self, controller, total_sim_steps=None):
        self.reset_episode()

        control_count = 0
        action_records = []
        if total_sim_steps is None:
            total_sim_steps = int(self.total_sim_steps)
        else:
            total_sim_steps = int(total_sim_steps)
        diff_horizon = self.backward_interval * self.control_interval
        if diff_horizon <= 0:
            raise ValueError("backward_interval must be > 0")

        if total_sim_steps <= 0:
            raise ValueError("total_sim_steps must be > 0")

        self.last_segment_steps = 0

        controller.eval()
        global_step = 0

        with torch.no_grad():
            while global_step < total_sim_steps:
                backward_steps = min(diff_horizon, total_sim_steps - global_step)
                self.sim.sim_steps = backward_steps
                self.last_segment_steps = backward_steps

                if global_step == 0:
                    self.sim.task.init_simulation_state()

                for local_t in range(backward_steps):
                    global_t = global_step + local_t
                    if self.should_control(global_t):
                        state_np = self.get_state(local_t)
                        action_torch = controller(state_np, local_t).reshape(-1)
                        action_wp = self._action_to_wp_vec3(action_torch)

                        if self.export_fluid_field:
                            velocity_field, weight, metadata = self.get_fluid_velocity_field(local_t)
                            launch_time_tag = self.export_fluid_field_launch_time or "launch_unknown"
                            current_time_tag = f"t{float(metadata['time']):.6f}".replace("-", "m")
                            fluid_field_name = f"{launch_time_tag}_{current_time_tag}.npz"
                            fluid_field_dir = self.export_fluid_field_dir
                            fluid_field_path = os.path.join(fluid_field_dir, fluid_field_name) if fluid_field_dir else fluid_field_name
                            save_fluid_velocity_field(
                                fluid_field_path,
                                velocity_field,
                                weight=weight,
                                metadata=metadata,
                            )

                        rod_quat = self.sim.rigid_quaternion_arrays[local_t].numpy()[self.rod_rigid_id].copy()
                        current_time = float(global_t) * float(self.sim.frame_dt)

                        # 正弦波模式：只存储 action 供后续每步连续施加
                        if getattr(self, 'sine_action', False):
                            self._stored_sine_action_wp = action_wp
                        else:
                            self.apply_action(local_t, action_wp, tape=None)

                        bladder_v, bladder_f = self._get_bladder_info()
                        joint_angle = self._get_current_joint_angle(local_t)
                        action_np = action_torch.detach().cpu().numpy().astype(np.float32)
                        target_angle = None
                        torque_info = None
                        pos_error = None
                        if getattr(self, 'sine_action', False) and len(action_np) >= 2:
                            target_angle = self._get_pd_target_angle(action_np, current_time)
                            torque_info = self._get_pd_torque_info(action_np, current_time, local_t)
                        pos_error = self._get_position_error(local_t)
                        action_records.append(
                            {
                                "step": global_t,
                                "time": current_time,
                                "rod_quat": rod_quat,
                                "action": action_np,
                                "bladder_volume": bladder_v,
                                "buoyancy_force": bladder_f,
                                "joint_angle": joint_angle,
                                "target_angle": target_angle,
                                "torque_info": torque_info,
                                "pos_error": pos_error,
                            }
                        )
                        control_count += 1

                    # 正弦波模式：每步施加存储的 [freq, amp]，实现连续摆动
                    if getattr(self, 'sine_action', False) and \
                       getattr(self, '_stored_sine_action_wp', None) is not None:
                        self.apply_action(local_t, self._stored_sine_action_wp, tape=None)

                    self.sim.step(local_t)

                global_step += backward_steps
                # 累加正弦波时间偏移，保证跨 segment 相位连续
                if getattr(self, 'sine_action', False):
                    self._sine_time_offset += backward_steps * self.sim.frame_dt
                if global_step < total_sim_steps:
                    if self.export_before_diff_reset and self.export_series_prefix:
                        seg_output_interval = max(1, int(self.export_output_interval))
                        seg_series_prefix = self.export_series_prefix

                        seg_dir = os.path.dirname(seg_series_prefix)
                        if seg_dir:
                            os.makedirs(seg_dir, exist_ok=True)

                        print(
                            f"[rollout] export diff_segment (test): steps={backward_steps}, "
                            f"output_interval={seg_output_interval}, prefix={seg_series_prefix}"
                        )
                        self.export_ply_counter = export_backward_data(
                            self.sim,
                            backward_steps,
                            seg_output_interval,
                            seg_series_prefix,
                            start_cnt=self.export_ply_counter,
                        )
                    self.sim.diff_reset(backward_steps)

        try:
            # compute_loss expects an index within the simulator's per-segment buffers.
            # Use the last simulated segment length (self.last_segment_steps) to avoid
            # indexing beyond the allocated arrays when the full trajectory is larger
            # than the per-segment buffers.
            self.sim.task.compute_loss(end_step=self.last_segment_steps)
        except TypeError:
            self.sim.task.compute_loss()

        loss_val = float(np.asarray(self.sim.loss.numpy()).reshape(-1)[0])
        physics_info = self._extract_physics_info(self.last_segment_steps)
        physics_str = ""
        if physics_info:
            parts = []
            if "body_angle_err" in physics_info:
                parts.append(f"body_angle_err={physics_info['body_angle_err']:.4f}")
            if "tail_angle_err" in physics_info:
                parts.append(f"tail_angle_err={physics_info['tail_angle_err']:.4f}")
            if "pos_err" in physics_info:
                parts.append(f"pos_err={physics_info['pos_err']:.4f}")
            if "forward_vel" in physics_info:
                parts.append(f"forward_vel={physics_info['forward_vel']:.4f}")
            if "tail_periodic_angle_err" in physics_info:
                parts.append(f"tail_periodic_angle_err={physics_info['tail_periodic_angle_err']:.4f}")
            if "loss_pose" in physics_info:
                parts.append(f"L_pose={physics_info['loss_pose']:.4e}")
            if "loss_position" in physics_info:
                parts.append(f"L_pos={physics_info['loss_position']:.4e}")
            if "loss_velocity" in physics_info:
                parts.append(f"L_vel={physics_info['loss_velocity']:.4e}")
            if "loss_tail_periodic" in physics_info:
                parts.append(f"L_tail={physics_info['loss_tail_periodic']:.4e}")
            if parts:
                physics_str = " | " + " ".join(parts)
        print(f"[test] loss = {loss_val:.6e} | controls={control_count}{physics_str}")
        for idx, record in enumerate(action_records):
            bl_str = f" bladder_v={record['bladder_volume']:.3e} buoy_f={record['buoyancy_force']:.4f}" if record.get('bladder_volume') is not None else ""
            ja_str = f" joint_angle={record['joint_angle']:.4f}" if record.get('joint_angle') is not None else ""
            ta_str = f" target_angle={record['target_angle']:.4f}" if record.get('target_angle') is not None else ""
            ti = record.get('torque_info') or {}
            tq_str = f" torque={ti['torque']:.4f}" if ti.get('torque') is not None else ""
            pe = record.get('pos_error')
            pe_str = f" pos_err=({pe[0]:.3f},{pe[1]:.3f},{pe[2]:.3f})" if pe is not None else ""
            print(
                f"[test action {idx}] step={record['step']} time={record['time']} "
                f"rod_quat={record['rod_quat']} action={record['action']}"
                f"{bl_str}{ja_str}{ta_str}{tq_str}{pe_str}"
            )

        return loss_val, control_count