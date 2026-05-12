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
    def __init__(self, sim, control_interval, backward_interval, control_start_step=2, include_time=True, norm_grad=False, total_sim_steps=None):
        self.sim = sim
        self.control_interval = int(control_interval)
        self.backward_interval = int(backward_interval)
        self.control_start_step = int(control_start_step)
        self.include_time = bool(include_time)
        self.norm_grad = bool(norm_grad)

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

    def reset_episode(self):
        self.sim.clear_grad()
        self.sim.reset()

        self.action_records = []
        self.export_ply_counter = 0
        self.export_fluid_field_counter = 0

    def set_total_sim_steps(self, total_sim_steps):
        total_sim_steps = int(total_sim_steps)
        if total_sim_steps <= 0:
            raise ValueError("total_sim_steps must be > 0")
        self.total_sim_steps = total_sim_steps

    def should_control(self, step):
        if step < self.control_start_step:
            return False
        return (step - self.control_start_step) % self.control_interval == 0

    def get_state(self, step):
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
        else:
            t = 0.0
        state_parts.append(t)

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

        for local_t in range(backward_steps):
            current_tape = wp.Tape()
            step_tapes.append(current_tape)
            self.sim.tapes = step_tapes

            if global_step_offset == 0 and local_t == 0:
                self.sim.task.init_simulation_state()

            global_t = global_step_offset + local_t
            if self.should_control(global_t):
                state_np = self.get_state(local_t)
                action_torch = controller(state_np).reshape(-1)
                action_wp = self._action_to_wp_vec3(action_torch)

                rod_quat = self.sim.rigid_quaternion_arrays[local_t].numpy()[self.rod_rigid_id].copy()
                current_time = float(global_t) * float(self.sim.frame_dt)

                self.apply_action(local_t, action_wp, current_tape)

                action_record = {
                    "step": global_t,
                    "time": current_time,
                    "rod_quat": rod_quat,
                    "action": action_torch.detach().cpu().numpy().astype(np.float32),
                    "state": state_np,
                }
                self.action_records.append(action_record)
                step_action_entries[local_t] = {
                    "state": state_np,
                    "action_torch": action_torch,
                    "action_wp": action_wp,
                    "record": action_record,
                }

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

                print(
                    f"[action step] step={record['step']} time={record['time']} "
                    f"rod_quat={record['rod_quat']} action={record['action']} "
                    f"grad={record['grad']}"
                )

                if collect_replay:
                    if replay_buffer is None:
                        raise ValueError("replay_buffer must be provided when collect_replay is True")
                    replay_buffer.add(
                        state=state_np,
                        action_grad=grad_vec,
                        action=record["action"],
                        step=record["step"],
                        time=record["time"],
                        loss=None,
                    )
                    continue

                if grad_callback is not None:
                    grad_callback(action_torch, grad_tensor, record)

                batched_action_torch_vars.append(action_torch)
                batched_grad_tensors.append(grad_tensor)

        loss_val = float(np.asarray(self.sim.loss.numpy()).reshape(-1)[0])
        avg_action_grad_norm = float(np.mean(np.asarray(action_grad_norms, dtype=np.float32))) if action_grad_norms else 0.0
        controls = int(len(step_action_entries))

        exceed_threshold = loss_threshold is not None and loss_val > loss_threshold

        return {
            "loss": loss_val,
            "action_grad_norm": avg_action_grad_norm,
            "controls": controls,
            "early_terminated": bool(exceed_threshold),
            "action_torch_vars": batched_action_torch_vars,
            "action_grad_tensors": batched_grad_tensors,
        }

    def backward_with_controller(
        self,
        controller,
        optimizer,
        global_step_offset,
        backward_steps,
        loss_threshold=None,
        grad_clip_norm=None,
        grad_callback=None,
    ):
        if optimizer is None:
            raise ValueError("optimizer must be provided for backward_with_controller")

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
                action_torch = controller(state_np).reshape(-1)
                action_wp = self._action_to_wp_vec3(action_torch)

                rod_quat = self.sim.rigid_quaternion_arrays[local_t].numpy()[self.rod_rigid_id].copy()
                current_time = float(global_t) * float(self.sim.frame_dt)

                self.apply_action(local_t, action_wp, current_tape)

                action_record = {
                    "step": global_t,
                    "time": current_time,
                    "rod_quat": rod_quat,
                    "action": action_torch.detach().cpu().numpy().astype(np.float32),
                    "state": state_np,
                }
                self.action_records.append(action_record)
                step_action_entries[local_t] = {
                    "state": state_np,
                    "action_torch": action_torch,
                    "action_wp": action_wp,
                    "record": action_record,
                }

            self.sim.step(local_t)

        loss_tape = wp.Tape()
        self.sim.loss_tape = loss_tape
        self.sim.tapes = step_tapes + [loss_tape]

        self._compute_loss_with_optional_end_step(loss_tape, backward_steps)

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

                torch.autograd.backward([action_torch], grad_tensors=[grad_tensor])

            for i in reversed(records):
                print(
                    f"[action step] step={i['step']} time={i['time']} "
                    f"rod_quat={i['rod_quat']} action={i['action']} "
                    f"grad={i['grad']}"
                )

        loss_val = float(np.asarray(self.sim.loss.numpy()).reshape(-1)[0])
        avg_action_grad_norm = float(np.mean(np.asarray(action_grad_norms, dtype=np.float32))) if action_grad_norms else 0.0
        controls = int(len(step_action_entries))

        exceed_threshold = loss_threshold is not None and loss_val > loss_threshold
        if exceed_threshold:
            print(
                f"[segment] skip optimizer step: loss {loss_val:.6e} exceeds threshold "
                f"{loss_threshold:.2e}."
            )
            optimizer.zero_grad()
            return {
                "loss": loss_val,
                "action_grad_norm": avg_action_grad_norm,
                "controls": controls,
                "early_terminated": True,
            }

        if grad_clip_norm is not None and grad_clip_norm > 0:
            params = [param for group in optimizer.param_groups for param in group["params"]]
            torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
        optimizer.step()

        return {
            "loss": loss_val,
            "action_grad_norm": avg_action_grad_norm,
            "controls": controls,
            "early_terminated": False,
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
        print(f"[segment] end_step={end_step} loss = {loss_val:.6e}")

        action_grads = []
        for idx, action_wp in enumerate(action_wp_vars):
            record = action_records[idx]

            grad_np = action_wp.grad.numpy()[0] if action_wp.grad is not None else np.zeros(3, dtype=np.float32)
            grad_vec = np.array([grad_np[0], grad_np[1], grad_np[2]], dtype=np.float32)
            action_grads.append(grad_vec)

            print(
                f"[action {idx}] step={record['step']} time={record['time']} "
                f"rod_quat={record['rod_quat']} action={record['action']} "
                f"grad={grad_vec}"
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
                seg_stats = self.backward_with_controller(
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

            if seg_stats["early_terminated"]:
                early_terminated = True
                print(
                    f"[rollout] segment {segment_count} early stop: loss {seg_stats['loss']:.6e} exceeds threshold {loss_threshold:.2e}."
                )
                break

            global_step += backward_steps
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
                        action_torch = controller(state_np).reshape(-1)
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

                        self.apply_action(local_t, action_wp, tape=None)

                        action_records.append(
                            {
                                "step": global_t,
                                "time": current_time,
                                "rod_quat": rod_quat,
                                "action": action_torch.detach().cpu().numpy().astype(np.float32),
                            }
                        )
                        control_count += 1

                    self.sim.step(local_t)

                global_step += backward_steps
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
        print(f"[test] loss = {loss_val:.6e} | controls={control_count}")
        for idx, record in enumerate(action_records):
            print(
                f"[test action {idx}] step={record['step']} time={record['time']} "
                f"rod_quat={record['rod_quat']} action={record['action']}"
            )

        return loss_val, control_count