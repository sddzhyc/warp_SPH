# pyright: reportInvalidTypeForm=false
import argparse
import datetime
import os

import numpy as np
import torch
import math
import tensorboardX
import warp as wp

from robots.CartPoleEnv import DifferentiableCartPoleEnv
from robots.FishEnv import DifferentiableFishEnv
from config_builder import SimConfig
from SimSPH_diff import SimSPH_diff
from SimDFSPH_diff import SimDFSPH_diff
from SimDFSPH_diff_no_fluid_tape import SimDFSPH_diff as SimDFSPH_diff_no_fluid_tape
from rigid_fluid_coupling import MaterialType
from robots.controller import Controller
from robots.controller_with_fluid import ControllerWithFluid
from fluid_encoder.ae_CNN_3d_pytorch import AutoEncoder3D
from fluid_encoder.encoder_utils import get_local_fluid_grid_around_rigid, normalize_fluid_field
from utils import export_backward_data, build_export_settings, build_default_log_dir, build_checkpoint_dir, build_controller_run_tag, save_fluid_velocity_field

GRID_SHAPE = (64, 32, 32)
LOCAL_FLUID_GRID_CELL_SIZE = 0.03

is_fish_task = False

def _resolve_local_fluid_grid_shape(config, cell_size=None):
    if cell_size is None:
        cell_size = float(LOCAL_FLUID_GRID_CELL_SIZE)
    _ = config
    grid_shape = np.asarray(GRID_SHAPE, dtype=np.int32)
    local_extent = grid_shape.astype(np.float32) * float(cell_size)
    return tuple(int(v) for v in grid_shape.tolist()), local_extent, float(cell_size)



def derive_sim_steps(cfg, sim_time=None):
    end_time = sim_time if sim_time is not None else cfg.get_cfg("end_time")
    if end_time is None:
        raise ValueError("Simulation time must be provided by --t or scene config end_time.")

    time_step_size = cfg.get_cfg("timeStepSize")
    if time_step_size is None:
        raise ValueError("Scene config must define timeStepSize.")

    return max(1, int(np.ceil(float(end_time) / float(time_step_size))))


def _resolve_rigid_log_targets(sim):
    task = getattr(sim, "task", None)
    if task is not None and task.__class__.__name__ == "CarPoleOnWaterTask":
        rod_id = int(getattr(task, "rod_rigid_id", -1))
        car_id = int(getattr(task, "car_rigid_id", -1))
        targets = []
        if car_id >= 0:
            targets.append(("Car", car_id))
        if rod_id >= 0:
            targets.append(("Rod", rod_id))
        if targets:
            return targets

    return [("Body1", 1)]


def _get_sim_state(sim, step, include_time):
    rod_rigid_id = int(getattr(sim.task, "rod_rigid_id", 1))
    car_rigid_id = int(getattr(sim.task, "car_rigid_id", 2))
    state_rigid_ids = list(getattr(sim.task, "state_rigid_ids", [])) if getattr(sim, "task", None) is not None else []

    rigid_x = sim.rigid_x_arrays[step].numpy()
    rigid_q = sim.rigid_quaternion_arrays[step].numpy()
    rigid_v = sim.rigid_v_arrays[step].numpy()
    rigid_omega = sim.rigid_omega_arrays[step].numpy()

    state_parts = []
    body_ids = state_rigid_ids if state_rigid_ids else [car_rigid_id, rod_rigid_id]
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

    t = float(step) * float(sim.frame_dt) if include_time else 0.0
    state_parts.append(t)
    return np.asarray(state_parts, dtype=np.float32)


def _get_fluid_particles_from_step(sim, step):
    pos_all = sim.x_arrays[step].numpy()
    vel_all = sim.v_arrays[step].numpy()

    # Prefer explicit material tags when available; fallback to front fluid range.
    if hasattr(sim, "materialMarks") and hasattr(sim.materialMarks, "material") and sim.materialMarks.material is not None:
        mat = sim.materialMarks.material.numpy()
        mask = (mat == int(MaterialType.FLUID))
        if np.any(mask):
            return pos_all[mask], vel_all[mask]

    fluid_num = int(getattr(sim, "fluid_particle_num", 0))
    if fluid_num > 0:
        return pos_all[:fluid_num], vel_all[:fluid_num]

    return pos_all, vel_all


def _save_particles_ply(file_path, positions, velocities):
    # utility removed: no-op placeholder to preserve API compatibility
    return


def _build_local_fluid_grid(sim, step, grid_shape, grid_spacing, influence_radius):
    task = getattr(sim, "task", None)
    state_rigid_ids = list(getattr(task, "state_rigid_ids", [])) if task is not None else []
    if state_rigid_ids:
        car_rigid_id = int(state_rigid_ids[0])
    else:
        car_rigid_id = int(getattr(task, "car_rigid_id", 1))

    rigid_pos = sim.rigid_x_arrays[step].numpy()
    car_rigid_id = max(0, min(car_rigid_id, rigid_pos.shape[0] - 1))
    rigid_center = rigid_pos[car_rigid_id]
    particle_pos, particle_vel = _get_fluid_particles_from_step(sim, step)
    vel_grid, _ = get_local_fluid_grid_around_rigid(
        particle_pos=particle_pos,
        particle_vel=particle_vel,
        rigid_center=rigid_center,
        grid_shape=grid_shape,
        grid_spacing=float(grid_spacing),
        influence_radius=influence_radius,
    )
    return vel_grid



def _encode_local_fluid_grid(ae_model, vel_grid, device):
    # Input shape expected by 3D AE: (B, C, D, H, W)
    grid_norm = normalize_fluid_field(vel_grid[None, ...])[0]
    ae_in = torch.as_tensor(grid_norm[None, ...], dtype=torch.float32, device=device)
    with torch.no_grad():
        feat = ae_model.encode(ae_in).reshape(-1)
    return feat


class FluidAEPolicyAdapter(torch.nn.Module):
    """Wraps ControllerWithFluid to keep Env API controller(state)->action unchanged."""

    def __init__(self, sim, base_controller, ae_model, grid_shape, grid_spacing, influence_radius):
        super().__init__()
        self.sim = sim
        self.base_controller = base_controller
        self.ae_model = ae_model
        self.grid_shape = tuple(grid_shape)
        self.grid_spacing = float(grid_spacing)
        self.influence_radius = influence_radius

    def _controller_device(self):
        try:
            return next(self.base_controller.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def forward(self, state, step=None):
        s = torch.as_tensor(state, dtype=torch.float32, device=self._controller_device())
        if s.dim() != 1:
            raise ValueError("FluidAEPolicyAdapter.forward expects a 1D state vector")

        if step is None:
            step = 0
        step = int(max(0, min(int(step), len(self.sim.x_arrays) - 1)))
        vel_grid = _build_local_fluid_grid(
            sim=self.sim,
            step=step,
            grid_shape=self.grid_shape,
            grid_spacing=self.grid_spacing,
            influence_radius=self.influence_radius,
        )
        ae_feat = _encode_local_fluid_grid(self.ae_model, vel_grid, self._controller_device())
        return self.base_controller(s, ae_feat, step=step)


def _resolve_action_bounds(sim, action_dim=3):
    global is_fish_task
    if is_fish_task:
        print("[task] Resolving action bounds for FishTask (z-axis angular velocity fixed to 0)")
        action_min = [-1.0] * int(action_dim)
        action_max = [1.0] * int(action_dim)
        if int(action_dim) >= 3:
            action_min[2] = 0.0
            action_max[2] = 0.0
    else:
        if int(action_dim) == 2:
            action_min = [-0.5, -0.5]
            action_max = [0.5, 0.0]
        else:
            action_min = [-0.5] * int(action_dim)
            action_max = [0.5] * int(action_dim)
            if int(action_dim) >= 2:
                action_max[1] = 0.0

    return action_min, action_max


def _build_policy_controller(sim, args):
    if not args.use_fluid:
        action_min, action_max = _resolve_action_bounds(sim, action_dim=3)
        return Controller(action_min=action_min, action_max=action_max)

    if not args.encoder_path:
        raise ValueError("use_fluid requires --encoder_path")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load checkpoint first and infer latent channel count from its encoder conv weights.
    ae_state = torch.load(args.encoder_path, map_location=device)
    ae_model = AutoEncoder3D(in_channels=3,).to(device)
    try:
        ae_model.load_state_dict(ae_state)
    except Exception:
        # fall back to non-strict load to allow minor key mismatches
        ae_model.load_state_dict(ae_state, strict=False)
    ae_model.eval()
    for p in ae_model.parameters():
        p.requires_grad_(False)

    grid_shape = tuple(getattr(args, "fluid_grid_shape", (32, 32, 32)))
    grid_spacing = float(getattr(args, "fluid_grid_spacing", LOCAL_FLUID_GRID_CELL_SIZE))

    sample_grid = _build_local_fluid_grid(
        sim=sim,
        step=0,
        grid_shape=grid_shape,
        grid_spacing=grid_spacing,
        influence_radius=args.fluid_influence_radius,
    )
    sample_feat = _encode_local_fluid_grid(ae_model, sample_grid, device)
    # _get_sim_state always appends a trailing time slot; relative state does not include it.
    include_time_for_state = bool(args.time_in_state and not getattr(args, "use_relative_state", False))
    state_dim = int(_get_sim_state(sim, step=0, include_time=include_time_for_state).shape[0])
    if getattr(args, "use_relative_state", False):
        state_dim -= 1
    action_dim = 3

    base_controller = ControllerWithFluid(
        state_dim=state_dim,
        ae_feat_dim=int(sample_feat.numel()),
        action_dim=action_dim,
        action_min=_resolve_action_bounds(sim, action_dim=action_dim)[0],
        action_max=_resolve_action_bounds(sim, action_dim=action_dim)[1],
    ).to(device)
    wrapped = FluidAEPolicyAdapter(
        sim=sim,
        base_controller=base_controller,
        ae_model=ae_model,
        grid_shape=grid_shape,
        grid_spacing=grid_spacing,
        influence_radius=args.fluid_influence_radius,
    )
    return wrapped


def _make_env_for_sim(sim, args, sim_steps):
    """Return appropriate Differentiable env for the given simulator task."""
    # choose FishEnv when task is FishTask
    global is_fish_task
    is_fish_task = False
    task = getattr(sim, "task", None)
    if task is not None:
        name = task.__class__.__name__
        if name == "FishTask" or "fish" in name.lower():
            is_fish_task = True
            print(f"[task] Detected task '{name}', using DifferentiableFishEnv with angular velocity control and symmetric action bounds.")
    EnvCls = DifferentiableFishEnv if is_fish_task else DifferentiableCartPoleEnv
    env = EnvCls(
        sim=sim,
        control_interval=args.control_interval,
        backward_interval=args.backward_interval,
        control_start_step=args.control_start_step,
        include_time=args.time_in_state,
        norm_grad=args.norm_grad,
        total_sim_steps=sim_steps,
        random_start=getattr(args, "random_start", False),
        use_relative_state=getattr(args, "use_relative_state", False),
    )
    return env


def _log_first_segment_step_grads(writer, sim, env):
    if writer is None:
        return

    for target_name, target_id in _resolve_rigid_log_targets(sim):
        if target_id < 0 or target_id >= sim.num_objects:
            continue

        for time_step in range(len(sim.rigid_x_arrays)):
            for name, array_list in [
                ("Pos", sim.rigid_x_arrays),
                ("Vel", sim.rigid_v_arrays),
                ("Omega", sim.rigid_omega_arrays),
                ("VelForJoint", getattr(sim, "rigid_v_for_joint_arrays", [])),
                ("OmegaForJoint", getattr(sim, "rigid_omega_for_joint_arrays", [])),
                ("DeltaVel", getattr(sim, "rigid_delta_v_arrays", [])),
                ("DeltaOmega", getattr(sim, "rigid_delta_omega_arrays", [])),
                ("Quat", sim.rigid_quaternion_arrays),
                ("Force", sim.rigid_force_arrays),
                ("Torque", sim.rigid_torque_arrays),
            ]:
                if time_step >= len(array_list):
                    continue
                if array_list[time_step].grad is None:
                    continue

                grad_data = array_list[time_step].grad.numpy()
                if target_id >= len(grad_data):
                    continue

                grad_val = grad_data[target_id]
                tag_prefix = f"Rigid_Step_Grad-{target_name}_id{target_id}/{name}"
                if len(grad_val) >= 3:
                    writer.add_scalar(f"{tag_prefix}/x", float(grad_val[0]), time_step)
                    writer.add_scalar(f"{tag_prefix}/y", float(grad_val[1]), time_step)
                    writer.add_scalar(f"{tag_prefix}/z", float(grad_val[2]), time_step)
                if len(grad_val) == 4:
                    writer.add_scalar(f"{tag_prefix}/w", float(grad_val[3]), time_step)

    for idx, record in enumerate(getattr(env, "action_records", [])):
        grad_vec = record.get("grad", None)
        if grad_vec is None:
            continue
        step = int(record.get("step", idx))
        writer.add_scalar("Action_Step_Grad/x", float(grad_vec[0]), step)
        writer.add_scalar("Action_Step_Grad/y", float(grad_vec[1]), step)
        writer.add_scalar("Action_Step_Grad/z", float(grad_vec[2]), step)
        writer.add_scalar("Action_Step_Grad/norm", float(np.linalg.norm(grad_vec)), step)


def train_controller(sim, args, sim_steps, export_settings=None, ckpt_dir=None):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = _make_env_for_sim(sim, args, sim_steps)

    if export_settings is not None:
        output_ply, _, output_interval, series_prefix, _, _ = export_settings
        env.export_before_diff_reset = False
        env.export_output_interval = int(output_interval)
        env.export_series_prefix = series_prefix

    controller = _build_policy_controller(sim, args)
    load_controller_checkpoint(controller, args.model_path)
    # optimizer = torch.optim.Adam(controller.parameters(), lr=args.lr)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=args.lr)
    scheduler = build_lr_scheduler(optimizer, args) if getattr(args, "lr_scheduler", False) else None

    writer = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = tensorboardX.SummaryWriter(log_dir=args.log_dir)

    if ckpt_dir is None:
        ckpt_dir = os.path.join(build_checkpoint_dir(args), build_controller_run_tag(args))

    os.makedirs(ckpt_dir, exist_ok=True)

    best_loss = float("inf")
    best_ckpt_path = None

    batch_size = 1 if args.batch is None else int(args.batch)
    if batch_size <= 0:
        raise ValueError("batch must be > 0")
    batch_buffer = []

    total_sim_steps = int(env.total_sim_steps)
    if total_sim_steps <= 0:
        raise ValueError("total_sim_steps must be set on env before training")
    diff_horizon = int(env.backward_interval) * int(env.control_interval)
    if diff_horizon <= 0:
        raise ValueError("backward_interval and control_interval must produce diff_horizon > 0")

    for epoch in range(args.iters):
        controller.train()
        env.export_before_diff_reset = bool(
            export_settings is not None
            and output_ply
            and series_prefix
            and (epoch == args.iters - 1)
        )
        env.reset_episode()

        global_step = 0
        processed_steps = 0
        segment_count = 0
        control_count = 0
        segment_losses = []
        segment_action_grad_norms = []
        early_terminated = False
        batch_update_losses = []
        batch_sample_losses = []
        batch_updates = 0

        # Continue collecting segments until we have at least `batch_size` samples.
        # If we reach the end of the episode (global_step >= total_sim_steps), reset
        # the environment and continue collecting across episodes.
        while len(batch_buffer) < batch_size:
            if global_step >= total_sim_steps:
                env.reset_episode()
                global_step = 0

            backward_steps = min(diff_horizon, total_sim_steps - global_step)
            sim.sim_steps = backward_steps

            # Run differentiable segment and collect per-segment parameter gradients
            seg_stats = env.forward_backward_with_controller(
                controller=controller,
                optimizer=None,
                global_step_offset=global_step,
                backward_steps=backward_steps,
                loss_threshold=args.loss_threshold,
                collect_batch=True,
                batch_buffer=batch_buffer,
            )

            segment_count += 1
            control_count += int(seg_stats["controls"])
            segment_losses.append(float(seg_stats["loss"]))
            segment_action_grad_norms.append(float(seg_stats["action_grad_norm"]))
            processed_steps += backward_steps

            if seg_stats["early_terminated"]:
                early_terminated = True
                print(
                    f"[train_controller] segment {segment_count} early stop: loss {seg_stats['loss']:.6e} exceeds "
                    f"threshold {args.loss_threshold:.2e}."
                )
                # end current episode early and start next one
                env.reset_episode()
                global_step = 0
                continue

            # Export segment if requested
            if env.export_before_diff_reset and env.export_series_prefix:
                seg_output_interval = max(1, int(env.export_output_interval))
                seg_series_prefix = env.export_series_prefix
                seg_dir = os.path.dirname(seg_series_prefix)
                if seg_dir:
                    os.makedirs(seg_dir, exist_ok=True)
                print(
                    f"[rollout] export diff_segment {segment_count}: steps={backward_steps}, "
                    f"output_interval={seg_output_interval}, prefix={seg_series_prefix}"
                )
                env.export_ply_counter = export_backward_data(
                    sim,
                    backward_steps,
                    seg_output_interval,
                    seg_series_prefix,
                    start_cnt=env.export_ply_counter,
                )

            sim.diff_reset(backward_steps)
            global_step += backward_steps

        update_loss, batch_sample_loss, updates_done = buffer_update_controller(
            controller=controller,
            optimizer=optimizer,
            batch_buffer=batch_buffer,
            updates_per_iter=1,
            grad_clip_norm=args.grad_clip_norm,
            scheduler=scheduler,
        )
        if updates_done > 0:
            batch_update_losses.append(float(update_loss))
            batch_sample_losses.append(float(batch_sample_loss))
            batch_updates += int(updates_done)
            # batch updated for this epoch

        loss_val = float(segment_losses[-1]) if segment_losses else 0.0
        action_grad_norm = float(np.mean(np.asarray(segment_action_grad_norms, dtype=np.float32))) if segment_action_grad_norms else 0.0
        batch_loss_val = float(np.mean(np.asarray(batch_update_losses, dtype=np.float32))) if batch_update_losses else float("nan")
        batch_sample_loss_val = float(np.mean(np.asarray(batch_sample_losses, dtype=np.float32))) if batch_sample_losses else float("nan")

        current_lr = _get_current_lr(optimizer)
        print(f"[LR] Current LR: {current_lr:.6e}")


        print(
            f"Epoch {epoch:04d} | avg_loss={loss_val:.6e} | controls={control_count} | "
            f"segments={segment_count} | avg_action_grad_norm={action_grad_norm:.6e} | "
            f"batch_updates={batch_updates} | batch_loss={batch_loss_val:.6e} | "
            f"batch_sample_loss={batch_sample_loss_val:.6e}"
        )

        if writer is not None:
            writer.add_scalar("Loss/train", loss_val, epoch)
            writer.add_scalar("Loss/batch", batch_loss_val, epoch)
            writer.add_scalar("Loss/batch_samples", batch_sample_loss_val, epoch)
            writer.add_scalar("LR/current", _get_current_lr(optimizer), epoch)
            writer.add_scalar("Grad/action_norm", action_grad_norm, epoch)
            writer.add_scalar("Control/count", float(control_count), epoch)
            writer.add_scalar("Backward/segments", float(segment_count), epoch)
            writer.add_scalar("Batch/updates", float(batch_updates), epoch)
            if args.iters == 1 and epoch == 0:
                _log_first_segment_step_grads(writer, sim, env)

        if loss_val < best_loss:
            best_loss = loss_val
            loss_tag = "loss_unknown" if loss_val is None or not np.isfinite(loss_val) else f"loss_{loss_val:.6e}"
            best_ckpt_path = os.path.join(ckpt_dir, f"model_best_{loss_tag}.pth")
            torch.save(controller.state_dict(), best_ckpt_path)
            print(f"Saved best model to: {best_ckpt_path}")

    if export_settings is not None and output_ply and series_prefix:
        export_steps = min(int(sim.sim_steps), len(sim.x_arrays))
        seg_dir = os.path.dirname(series_prefix)
        if seg_dir:
            os.makedirs(seg_dir, exist_ok=True)
        print(
            f"[rollout] export final diff_segment: steps={export_steps}, "
            f"output_interval={int(output_interval)}, prefix={series_prefix}"
        )
        export_backward_data(
            sim,
            export_steps,
            output_interval,
            series_prefix,
            include_last_step=True,
        )

    if writer is not None:
        writer.close()


def train_controller_online(sim, args, sim_steps, export_settings=None, ckpt_dir=None):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = _make_env_for_sim(sim, args, sim_steps)

    if export_settings is not None:
        output_ply, _, output_interval, series_prefix, _, _ = export_settings
        env.export_before_diff_reset = False
        env.export_output_interval = int(output_interval)
        env.export_series_prefix = series_prefix
    else:
        output_ply, series_prefix = False, None

    controller = _build_policy_controller(sim, args)
    load_controller_checkpoint(controller, args.model_path)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=args.lr)
    scheduler = build_lr_scheduler(optimizer, args) if getattr(args, "lr_scheduler", False) else None

    writer = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = tensorboardX.SummaryWriter(log_dir=args.log_dir)

    if ckpt_dir is None:
        ckpt_dir = os.path.join(build_checkpoint_dir(args), build_controller_run_tag(args))

    os.makedirs(ckpt_dir, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(args.iters):
        controller.train()
        env.export_before_diff_reset = bool(
            export_settings is not None
            and output_ply
            and series_prefix
            and (epoch == args.iters - 1)
        )

        rollout_stats = env.rollout(
            controller=controller,
            optimizer=optimizer,
            grad_clip_norm=args.grad_clip_norm,
            loss_threshold=args.loss_threshold,
        )

        # advance scheduler once per epoch and log current LR
        try:
            scheduler.step()
        except Exception:
            pass
        current_lr = None
        try:
            current_lr = scheduler.get_last_lr()[0]
        except Exception:
            current_lr = _get_current_lr(optimizer)
        print(f"[LR] Current LR: {current_lr:.6e}")

        loss_val = rollout_stats["loss"]
        action_grad_norm = rollout_stats["action_grad_norm"]
        control_count = rollout_stats["controls"]
        segment_count = rollout_stats["segments"]

        print(
            f"Epoch {epoch:04d} | avg_loss={loss_val:.6e} | controls={control_count} | "
            f"segments={segment_count} | avg_action_grad_norm={action_grad_norm:.6e}"
        )

        if writer is not None:
            writer.add_scalar("Loss/train", loss_val, epoch)
            writer.add_scalar("LR/current", _get_current_lr(optimizer), epoch)
            writer.add_scalar("Grad/action_norm", action_grad_norm, epoch)
            writer.add_scalar("Control/count", float(control_count), epoch)
            writer.add_scalar("Backward/segments", float(segment_count), epoch)
            if args.iters == 1 and epoch == 0:
                _log_first_segment_step_grads(writer, sim, env)

        if loss_val < best_loss:
            best_loss = loss_val
            loss_tag = "loss_unknown" if loss_val is None or not np.isfinite(loss_val) else f"loss_{loss_val:.6e}"
            best_ckpt_path = os.path.join(ckpt_dir, f"model_best_{loss_tag}.pth")
            torch.save(controller.state_dict(), best_ckpt_path)
            print(f"Saved best model to: {best_ckpt_path}")

    if export_settings is not None and output_ply and series_prefix:
        export_steps = min(int(sim.sim_steps), len(sim.x_arrays))
        seg_dir = os.path.dirname(series_prefix)
        if seg_dir:
            os.makedirs(seg_dir, exist_ok=True)
        print(
            f"[rollout] export final diff_segment: steps={export_steps}, "
            f"output_interval={int(output_interval)}, prefix={series_prefix}"
        )
        export_backward_data(
            sim,
            export_steps,
            output_interval,
            series_prefix,
            include_last_step=True,
        )

    if writer is not None:
        writer.close()


def _get_controller_device(controller):
    try:
        return next(controller.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def _get_current_lr(optimizer):
    if not optimizer.param_groups:
        return float("nan")
    return float(optimizer.param_groups[0].get("lr", float("nan")))


def _batch_buffer_clear(batch_buffer):
    if hasattr(batch_buffer, "clear"):
        batch_buffer.clear()
    else:
        del batch_buffer[:]

def build_lr_scheduler(optimizer, args):
    total_steps = max(1, int(args.iters) * 1)
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def buffer_update_controller(controller, optimizer, batch_buffer, updates_per_iter, grad_clip_norm, scheduler=None):
    if len(batch_buffer) == 0:
        return 0.0, 0.0, 0

    device = _get_controller_device(controller)
    update_losses = []
    batch_sample_losses = []
    updates_done = 0

    if hasattr(batch_buffer, "_storage"):
        transitions = list(batch_buffer._storage)
    else:
        transitions = list(batch_buffer)

    if len(transitions) == 0:
        return 0.0, 0.0, 0

    # Use only precomputed parameter gradients from batch items.
    if "param_grad" not in transitions[0]:
        raise ValueError("batch items must contain 'param_grad'; per_sample_losses path has been removed")

    grads_np = np.stack([item["param_grad"] for item in transitions], axis=0)
    losses_np = np.asarray([item.get("loss", 0.0) for item in transitions], dtype=np.float32)

    for _ in range(int(updates_per_iter)):
        avg_grad = np.mean(grads_np, axis=0)
        avg_grad_t = torch.as_tensor(avg_grad, dtype=torch.float32, device=device)

        optimizer.zero_grad()

        # Scatter averaged flat gradient back into each parameter tensor.
        offset = 0
        for p in controller.parameters():
            n = int(p.numel())
            if n > 0:
                g_slice = avg_grad_t[offset: offset + n].view_as(p).to(device)
                if p.grad is None:
                    p.grad = g_slice.clone()
                else:
                    p.grad.data.copy_(g_slice)
                offset += n

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass

        update_losses.append(float(np.mean(losses_np)))
        batch_sample_losses.append(float(np.mean(losses_np)))
        updates_done += 1

    # Keep buffer samples aligned with current controller parameters.
    _batch_buffer_clear(batch_buffer)

    return (
        float(np.mean(np.asarray(update_losses, dtype=np.float32))),
        float(np.mean(np.asarray(batch_sample_losses, dtype=np.float32))),
        updates_done,
    )


def train_controller_with_sim_backward(sim, args, sim_steps, export_settings=None, ckpt_dir=None):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.use_fluid:
        raise ValueError("use_fluid is not supported in sim_backward_controller mode. Use --train_ar instead.")

    action_min, action_max = _resolve_action_bounds(sim, action_dim=3)
    controller = Controller(action_min=action_min, action_max=action_max)
    load_controller_checkpoint(controller, args.model_path)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=args.lr)
    scheduler = build_lr_scheduler(optimizer, args) if getattr(args, "lr_scheduler", False) else None

    if not hasattr(sim, "task") or not hasattr(sim.task, "opt_rigid_v"):
        raise RuntimeError("Current task does not expose opt_rigid_v for SimSPH_diff.backward controller training.")

    writer = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = tensorboardX.SummaryWriter(log_dir=args.log_dir)

    if ckpt_dir is None:
        ckpt_dir = os.path.join(build_checkpoint_dir(args), build_controller_run_tag(args))
    os.makedirs(ckpt_dir, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(args.iters):
        controller.train()

        # One backward pass -> one action gradient.
        sim.clear_grad()
        sim.reset()
        state_np = _get_sim_state(sim, step=0, include_time=args.time_in_state)
        action_torch = controller(state_np).reshape(-1)

        target_device = torch.device(str(sim.task.opt_rigid_v.device))
        action_torch_for_wp = action_torch.reshape(1, 3).to(device=target_device, dtype=torch.float32)
        sim.task.opt_rigid_v.assign(wp.from_torch(action_torch_for_wp, dtype=wp.vec3))

        sim.backward()
        if args.norm_grad:
            sim.norm_final_grad()

        loss_val = float(np.asarray(sim.loss.numpy()).reshape(-1)[0])

        if sim.task.opt_rigid_v.grad is not None:
            grad_tensor = wp.to_torch(sim.task.opt_rigid_v.grad).reshape(-1).to(
                device=action_torch.device,
                dtype=action_torch.dtype,
            )
        else:
            grad_tensor = torch.zeros_like(action_torch)
        grad_vec = grad_tensor.detach().cpu().numpy().astype(np.float32)
        action_grad_norm = float(torch.linalg.norm(grad_tensor).item())

        action_np = action_torch.detach().cpu().numpy().astype(np.float32)

        optimizer.zero_grad()
        torch.autograd.backward([action_torch], grad_tensors=[grad_tensor])

        if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], args.grad_clip_norm)
        optimizer.step()
        try:
            scheduler.step()
        except Exception:
            pass

        print(
            f"Epoch {epoch:04d} | loss={loss_val:.6e} | one_action_grad_norm={action_grad_norm:.6e} "
            f"| action={action_np} | grad={grad_vec}"
        )

        if writer is not None:
            writer.add_scalar("Loss/train", loss_val, epoch)
            writer.add_scalar("LR/current", _get_current_lr(optimizer), epoch)
            writer.add_scalar("Grad/action_norm", action_grad_norm, epoch)
            writer.add_scalar("Action/x", float(action_np[0]), epoch)
            writer.add_scalar("Action/y", float(action_np[1]), epoch)
            writer.add_scalar("Action/z", float(action_np[2]), epoch)
            writer.add_scalar("ActionGrad/x", float(grad_vec[0]), epoch)
            writer.add_scalar("ActionGrad/y", float(grad_vec[1]), epoch)
            writer.add_scalar("ActionGrad/z", float(grad_vec[2]), epoch)
            if args.iters == 1 and epoch == 0:
                _log_first_segment_step_grads(writer, sim, None)

        if loss_val < best_loss:
            best_loss = loss_val
            loss_tag = "loss_unknown" if loss_val is None or not np.isfinite(loss_val) else f"loss_{loss_val:.6e}"
            best_ckpt_path = os.path.join(ckpt_dir, f"model_best_{loss_tag}.pth")
            torch.save(controller.state_dict(), best_ckpt_path)
            print(f"Saved best model to: {best_ckpt_path}")

    if export_settings is not None:
        output_ply, _, output_interval, series_prefix, _, _ = export_settings
        if output_ply:
            export_steps = min(int(sim.sim_steps), len(sim.x_arrays))
            seg_dir = os.path.dirname(series_prefix)
            if seg_dir:
                os.makedirs(seg_dir, exist_ok=True)
            print(
                f"[rollout] export final diff_segment: steps={export_steps}, "
                f"output_interval={int(output_interval)}, prefix={series_prefix}"
            )
            export_backward_data(
                sim,
                export_steps,
                output_interval,
                series_prefix,
                include_last_step=True,
            )

    if writer is not None:
        writer.close()

def load_controller_checkpoint(controller, ckpt_path):
    if not ckpt_path:
        return False

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    controller.load_state_dict(state_dict)
    print(f"Loaded controller checkpoint from: {ckpt_path}")
    return True


def test_controller(sim, args, sim_steps, export_settings=None):
    controller = _build_policy_controller(sim, args)
    load_controller_checkpoint(controller, args.model_path)

    env = _make_env_for_sim(sim, args, sim_steps)

    if args.save_fluid_field:
        os.makedirs(args.fluid_field_out_dir, exist_ok=True)
        env.export_fluid_field = True
        env.export_fluid_field_dir = args.fluid_field_out_dir
        env.export_fluid_field_launch_time = getattr(args, "command_start_time_tag", None)
        env.export_fluid_field_grid_shape = tuple(getattr(args, "fluid_grid_shape", (32, 32, 32)))
        env.export_fluid_field_grid_spacing = float(getattr(args, "fluid_grid_spacing", LOCAL_FLUID_GRID_CELL_SIZE))
        env.export_fluid_field_influence_radius = args.fluid_influence_radius

    if export_settings is not None:
        output_ply, _, output_interval, series_prefix, _, _ = export_settings
        if output_ply:
            env.export_before_diff_reset = True
            env.export_output_interval = int(output_interval)
            env.export_series_prefix = series_prefix
            env.export_ply_counter = 0

    loss_val, control_count = env.rollout_forward_only(controller, total_sim_steps=sim_steps)

    print(f"Test complete | loss={loss_val:.6e} | controls={control_count}")


def build_simulator(args):
    config = SimConfig(scene_file_path=args.scene_file)

    sim_steps = args.num_timesteps
    if sim_steps is None:
        sim_steps = derive_sim_steps(config, sim_time=args.t)

    print(f"Using sim_steps: {sim_steps}")
    segment_steps = args.backward_interval * args.control_interval + 1
    print(f"segment_steps: {segment_steps}")

    with wp.ScopedDevice(args.device):
        if args.method == 1:
            dfsph_cls = SimDFSPH_diff if args.dfsph_impl == "default" else SimDFSPH_diff_no_fluid_tape
            print(f"Using DFSPH implementation: {args.dfsph_impl}")
            sim = dfsph_cls(
                config,
                stage_path=args.stage_path,
                container=None,
                sim_steps=segment_steps,
                ply_path=args.ply_path,
                lr=args.lr,
                h_scale=args.h_scale,
                use_custom_grad=args.custom_grad,
                use_norm_grad=args.norm_grad,
                use_diffFR=args.difffr,
                verbose=args.verbose,
            )
        else:
            sim = SimSPH_diff(
                config,
                stage_path=args.stage_path,
                container=None,
                sim_steps=segment_steps,
                ply_path=args.ply_path,
                lr=args.lr,
                h_scale=args.h_scale,
                use_custom_grad=args.custom_grad,
                use_norm_grad=args.norm_grad,
                use_diffFR=args.difffr,
                verbose=args.verbose,
            )

    return sim, config, sim_steps


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--scene_file", type=str, required=True, help="Scene file path.")
    parser.add_argument("--device", type=str, default=None, help="Warp device override.")
    parser.add_argument("--method", type=int, default=0, help="0 for WCSPH, 1 for DFSPH.")
    parser.add_argument(
        "--dfsph_impl",
        type=str,
        default="no_fluid_tape",
        choices=["default", "no_fluid_tape"],
        help="Choose DFSPH differentiable implementation when --method=1.",
    )
    parser.add_argument("--num_timesteps", type=int, default=None, help="Override simulation steps.")
    parser.add_argument("--t", type=float, default=None, help="Total simulation time used to derive sim_steps and path tags.")

    parser.add_argument("--iters", type=int, default=50, help="Training iterations.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Controller optimizer learning rate.")
    parser.add_argument("--seed", type=int, default=12, help="Random seed.")
    parser.add_argument("--grad_clip_norm", type=float, default=None, help="Optional grad clip norm.")
    parser.add_argument("--batch", type=int, default=None, help="Online update batch size for action-gradient application; default 1 when omitted.")
    parser.add_argument("--use_sim-backward", dest="use_sim_backward", action="store_true", help="Train controller using SimSPH_diff.backward with one action gradient per backward call.")
    parser.add_argument("--replay_mode", action="store_true", help="Train controller from a replay buffer of collected action gradients.")
    parser.add_argument("--replay_capacity", type=int, default=200, help="Maximum number of replay samples to keep.")
    parser.add_argument("--replay_batch_size", type=int, default=64, help="Mini-batch size for replay updates.")
    parser.add_argument("--replay_updates_per_iter", type=int, default=1, help="Number of replay optimizer steps after each rollout.")
    parser.add_argument("--replay_warmup", type=int, default=64, help="Minimum replay buffer size before replay updates start.")
    parser.add_argument("--backward_interval", "--bi",dest="backward_interval", type=int, default=1,help="Number of control intervals covered by one backward pass.",)
    parser.add_argument("--control_interval", type=int, default=1260, help="Control interval in simulation steps.")
    parser.add_argument("--control_start_step", type=int, default=0, help="First step where control is applied.")
    parser.add_argument("--time_in_state", action="store_true", help="Include time feature in state.")
    parser.add_argument("--random_start", action="store_true", help="Randomize first control time within one control interval on episode reset.")
    parser.add_argument("--use_relative_state", action="store_true", help="Use relative state representation for robot envs.")
    parser.add_argument("--scheduler_steps_per_epoch", type=int, default=1, help="Estimated optimizer steps per epoch used by the LR scheduler.")
    parser.add_argument("--lr_scheduler", action="store_true", help="Enable learning-rate scheduler (off by default).")

    # AE encoded policy options.
    parser.add_argument("--use_fluid", action="store_true", help="Use local fluid 3D grid encoded by encoder as additional controller input.")
    parser.add_argument("--encoder_path", type=str, 
                        default="fluid_encoder/models/20260505_163437/autoencoder_best_test_loss_9.931702e-03.pth", help="Path to AutoEncoder3D checkpoint (.pth/.pkl).")
    # latent channel count will be inferred from the AE checkpoint when provided
    # Fluid grid
    parser.add_argument("--fluid_grid_spacing", type=float, default=LOCAL_FLUID_GRID_CELL_SIZE, help="Uniform world-space cell size of the local fluid grid.")
    parser.add_argument("--fluid_influence_radius", type=float, default=None, help="Optional particle filter radius around car COM; defaults to grid half-extent norm.")
    parser.add_argument("--save_fluid_field", action="store_true", help="Save local fluid velocity fields during forward-only rollout.")
    parser.add_argument("--fluid_field_out_dir", type=str, default="datasets/fluid_grid_dataset", help="Output directory for saved fluid fields.")
    
    parser.add_argument("--disable_ply_export", action="store_true", help="Disable PLY export for test or collection runs.")
    parser.add_argument("--train_ar", action="store_true", help="Train controller with fluid AE features (AR mode).")
    parser.add_argument("--test_ar", action="store_true", help="Test controller with fluid AE features (AR mode).")

    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard log directory.")
    parser.add_argument("--model_path", type=str, default=None, help="Model path used to initialize controller or run test.")
    parser.add_argument("--test", action="store_true", help="Run non-differentiable controller test only.")
    parser.add_argument("--loss_threshold", type=float, default=100, help="Max loss threshold; training stops early if exceeded.")
    parser.add_argument("--stage_path", type=str, default="example_sph.usd", help="USD stage output path.")
    parser.add_argument("--ply_path", type=str, default=None, help="Optional PLY initialization path.")
    parser.add_argument("--h_scale", type=float, default=1.0, help="Smoothing-length scale.")
    parser.add_argument("--custom_grad", action="store_true", help="Enable custom gradient mode.")
    parser.add_argument("--norm_grad", action="store_true", help="Enable state-gradient normalization in simulator.")
    parser.add_argument("--difffr", action="store_true", help="Enable diffFR mode.")
    parser.add_argument("--verbose", action="store_true", help="Verbose simulator logs.")

    return parser


def main():
    args = get_parser().parse_args()
    args.command_start_time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.train_ar:
        args.use_fluid = True
    if args.test_ar:
        args.use_fluid = True
        args.test = True

    sim_time = args.t
    sim, config, sim_steps = build_simulator(args)
    if sim_time is None:
        sim_time = config.get_cfg("end_time")

    args.fluid_grid_shape, args.fluid_grid_extent, args.fluid_grid_spacing = _resolve_local_fluid_grid_shape(
        config,
    )

    args.backward_interval = int(args.backward_interval)
    if args.backward_interval <= 0:
        raise ValueError("backward_interval must be > 0")
    backward_interval_tag = f"interval_{args.backward_interval}"
    run_tag = build_controller_run_tag(args)

    export_settings = build_export_settings(args, config, end_time=sim_time)
    if args.disable_ply_export and export_settings is not None:
        output_ply, output_obj, output_interval, series_prefix, frame_time, scene_name_with_grad_mode = export_settings
        export_settings = (False, output_obj, output_interval, series_prefix, frame_time, scene_name_with_grad_mode)

    ckpt_dir = os.path.join(build_checkpoint_dir(args, end_time=sim_time), backward_interval_tag, run_tag)

    batch_tag = None if args.batch is None else f"batch_{int(args.batch)}"
    if args.log_dir is None:
        args.log_dir = os.path.join(build_default_log_dir(args, end_time=sim_time), backward_interval_tag, run_tag)
    else:
        if batch_tag is not None:
            args.log_dir = os.path.join(args.log_dir, batch_tag)
        args.log_dir = os.path.join(args.log_dir, backward_interval_tag, run_tag)
    print(f"TensorBoard logging to {args.log_dir}")
    print(f"Checkpoint directory: {ckpt_dir}")
    print(f"Using backwardInterval (signal spans): {args.backward_interval}")

    if args.test:
        test_controller(sim, args, sim_steps, export_settings=export_settings)
    elif args.use_sim_backward:
        train_controller_with_sim_backward(sim, args, sim_steps, export_settings=export_settings, ckpt_dir=ckpt_dir)
    elif args.batch is not None:
        train_controller(sim, args, sim_steps, export_settings=export_settings, ckpt_dir=ckpt_dir)
    else:
        train_controller_online(sim, args, sim_steps, export_settings=export_settings, ckpt_dir=ckpt_dir)

    if export_settings is not None:
        output_ply, _, _, series_prefix, _, _ = export_settings
        if output_ply:
            print(f"PLY export prefix: {series_prefix}")
            print(f"PLY export directory: {os.path.dirname(series_prefix)}")
        else:
            print("PLY export disabled by scene config.")


if __name__ == "__main__":
    main()
