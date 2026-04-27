# pyright: reportInvalidTypeForm=false
import argparse
import os

import numpy as np
import torch
import tensorboardX
import warp as wp

from robots.CartPoleEnv import DifferentiableCartPoleEnv
from config_builder import SimConfig
from SimSPH_diff import SimSPH_diff
from SimDFSPH_diff import SimDFSPH_diff
from SimDFSPH_diff_no_fluid_tape import SimDFSPH_diff as SimDFSPH_diff_no_fluid_tape
from robots.grad_replay_buffer import GradReplayBuffer
from robots.controller import Controller
from utils import export_backward_data, build_export_settings, build_default_log_dir, build_checkpoint_dir, build_controller_run_tag



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

    rigid_x = sim.rigid_x_arrays[step].numpy()
    rigid_q = sim.rigid_quaternion_arrays[step].numpy()
    rigid_v = sim.rigid_v_arrays[step].numpy()
    rigid_omega = sim.rigid_omega_arrays[step].numpy()

    state_parts = []
    for rid in [car_rigid_id, rod_rigid_id]:
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

    env = DifferentiableCartPoleEnv(
        sim=sim,
        control_interval=args.control_interval,
        backward_interval=args.backward_interval,
        control_start_step=args.control_start_step,
        include_time=args.time_in_state,
        norm_grad=args.norm_grad,
    )

    if export_settings is not None:
        output_ply, _, output_interval, series_prefix, _, _ = export_settings
        env.export_before_diff_reset = False
        env.export_output_interval = int(output_interval)
        env.export_series_prefix = series_prefix

    controller = Controller()
    load_controller_checkpoint(controller, args.model_path)
    optimizer = torch.optim.Adam(controller.parameters(), lr=args.lr)

    writer = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = tensorboardX.SummaryWriter(log_dir=args.log_dir)

    if ckpt_dir is None:
        ckpt_dir = os.path.join(build_checkpoint_dir(args), build_controller_run_tag(args))

    os.makedirs(ckpt_dir, exist_ok=True)

    best_loss = float("inf")
    best_ckpt_path = None
    last_loss_val = None

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

        loss_val = rollout_stats["loss"]
        last_loss_val = loss_val
        action_grad_norm = rollout_stats["action_grad_norm"]
        control_count = rollout_stats["controls"]
        segment_count = rollout_stats["segments"]
        early_terminated = bool(rollout_stats.get("early_terminated", False))


        print(
            f"Epoch {epoch:04d} | avg_loss={loss_val:.6e} | controls={control_count} | "
            f"segments={segment_count} | avg_action_grad_norm={action_grad_norm:.6e}"
        )

        if writer is not None:
            writer.add_scalar("Loss/train", loss_val, epoch)
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

    last_loss_tag = "loss_unknown" if last_loss_val is None or not np.isfinite(last_loss_val) else f"loss_{last_loss_val:.6e}"
    last_ckpt_path = os.path.join(ckpt_dir, f"model_last_{last_loss_tag}.pth")
    torch.save(controller.state_dict(), last_ckpt_path)

    print(f"Saved last model to: {last_ckpt_path}")

    if writer is not None:
        writer.close()


def train_controller_online(sim, args, sim_steps, export_settings=None, ckpt_dir=None):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = DifferentiableCartPoleEnv(
        sim=sim,
        control_interval=args.control_interval,
        backward_interval=args.backward_interval,
        control_start_step=args.control_start_step,
        include_time=args.time_in_state,
        norm_grad=args.norm_grad,
    )

    if export_settings is not None:
        output_ply, _, output_interval, series_prefix, _, _ = export_settings
        env.export_before_diff_reset = False
        env.export_output_interval = int(output_interval)
        env.export_series_prefix = series_prefix
    else:
        output_ply, series_prefix = False, None

    controller = Controller()
    load_controller_checkpoint(controller, args.model_path)
    optimizer = torch.optim.Adam(controller.parameters(), lr=args.lr)

    writer = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = tensorboardX.SummaryWriter(log_dir=args.log_dir)

    if ckpt_dir is None:
        ckpt_dir = os.path.join(build_checkpoint_dir(args), build_controller_run_tag(args))

    os.makedirs(ckpt_dir, exist_ok=True)

    best_loss = float("inf")
    last_loss_val = None

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

        loss_val = rollout_stats["loss"]
        last_loss_val = loss_val
        action_grad_norm = rollout_stats["action_grad_norm"]
        control_count = rollout_stats["controls"]
        segment_count = rollout_stats["segments"]

        print(
            f"Epoch {epoch:04d} | avg_loss={loss_val:.6e} | controls={control_count} | "
            f"segments={segment_count} | avg_action_grad_norm={action_grad_norm:.6e}"
        )

        if writer is not None:
            writer.add_scalar("Loss/train", loss_val, epoch)
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

    last_loss_tag = "loss_unknown" if last_loss_val is None or not np.isfinite(last_loss_val) else f"loss_{last_loss_val:.6e}"
    last_ckpt_path = os.path.join(ckpt_dir, f"model_last_{last_loss_tag}.pth")
    torch.save(controller.state_dict(), last_ckpt_path)

    print(f"Saved last model to: {last_ckpt_path}")

    if writer is not None:
        writer.close()


def _get_controller_device(controller):
    try:
        return next(controller.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _replay_update_controller(controller, optimizer, replay_buffer, batch_size, updates_per_iter, grad_clip_norm):
    if len(replay_buffer) == 0:
        return 0.0, 0

    device = _get_controller_device(controller)
    update_losses = []
    updates_done = 0

    for _ in range(int(updates_per_iter)):
        batch = replay_buffer.sample(batch_size)

        states = torch.as_tensor(batch["state"], dtype=torch.float32, device=device)
        action_grads = torch.as_tensor(batch["action_grad"], dtype=torch.float32, device=device)

        optimizer.zero_grad()
        pred_actions = controller(states)
        surrogate_loss = (pred_actions * action_grads).sum(dim=-1).mean()
        surrogate_loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], grad_clip_norm)
        optimizer.step()

        update_losses.append(float(surrogate_loss.detach().cpu().item()))
        updates_done += 1

    return float(np.mean(np.asarray(update_losses, dtype=np.float32))), updates_done


def train_controller_with_sim_backward(sim, args, sim_steps, export_settings=None, ckpt_dir=None):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    controller = Controller()
    load_controller_checkpoint(controller, args.model_path)
    optimizer = torch.optim.Adam(controller.parameters(), lr=args.lr)

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
    last_loss_val = None

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
        last_loss_val = loss_val

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

        print(
            f"Epoch {epoch:04d} | loss={loss_val:.6e} | one_action_grad_norm={action_grad_norm:.6e} "
            f"| action={action_np} | grad={grad_vec}"
        )

        if writer is not None:
            writer.add_scalar("Loss/train", loss_val, epoch)
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
            print(f"exporting simulation data in backward to: {series_prefix}")
            export_backward_data(sim, sim_steps, output_interval, series_prefix)

    last_loss_tag = "loss_unknown" if last_loss_val is None or not np.isfinite(last_loss_val) else f"loss_{last_loss_val:.6e}"
    last_ckpt_path = os.path.join(ckpt_dir, f"model_last_{last_loss_tag}.pth")
    torch.save(controller.state_dict(), last_ckpt_path)
    print(f"Saved last model to: {last_ckpt_path}")

    if writer is not None:
        writer.close()


def train_controller_with_replay(sim, args, sim_steps, export_settings=None, ckpt_dir=None):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = DifferentiableCartPoleEnv(
        sim=sim,
        control_interval=args.control_interval,
        backward_interval=args.backward_interval,
        control_start_step=args.control_start_step,
        include_time=args.time_in_state,
        norm_grad=args.norm_grad,
    )

    if export_settings is not None:
        output_ply, _, output_interval, series_prefix, _, _ = export_settings
        env.export_before_diff_reset = False
        env.export_output_interval = int(output_interval)
        env.export_series_prefix = series_prefix

    controller = Controller()
    load_controller_checkpoint(controller, args.model_path)
    optimizer = torch.optim.Adam(controller.parameters(), lr=args.lr)

    replay_buffer = GradReplayBuffer(args.replay_capacity)
    state_dim = int(env.get_state(0).shape[0])
    print(f"Replay mode enabled | state_dim={state_dim} | buffer_capacity={args.replay_capacity}")

    writer = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = tensorboardX.SummaryWriter(log_dir=args.log_dir)

    if ckpt_dir is None:
        ckpt_dir = os.path.join(build_checkpoint_dir(args), build_controller_run_tag(args))

    os.makedirs(ckpt_dir, exist_ok=True)

    best_loss = float("inf")
    last_loss_val = None

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
            grad_clip_norm=args.grad_clip_norm,
            loss_threshold=args.loss_threshold,
            collect_replay=True,
            replay_buffer=replay_buffer,
        )

        rollout_loss = rollout_stats["loss"]
        last_loss_val = rollout_loss
        action_grad_norm = rollout_stats["action_grad_norm"]
        control_count = rollout_stats["controls"]
        buffer_size = len(replay_buffer)

        replay_loss = float("nan")
        replay_updates = 0
        if buffer_size >= args.replay_warmup:
            controller.train()
            replay_loss, replay_updates = _replay_update_controller(
                controller=controller,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                batch_size=args.replay_batch_size,
                updates_per_iter=args.replay_updates_per_iter,
                grad_clip_norm=args.grad_clip_norm,
            )

        print(
            f"Epoch {epoch:04d} | rollout_loss={rollout_loss:.6e} | replay_loss={replay_loss:.6e} | "
            f"controls={control_count} | buffer={buffer_size} | replay_updates={replay_updates} | "
            f"avg_action_grad_norm={action_grad_norm:.6e}"
        )

        if writer is not None:
            writer.add_scalar("Loss/rollout", rollout_loss, epoch)
            if replay_updates > 0:
                writer.add_scalar("Loss/replay", replay_loss, epoch)
            writer.add_scalar("Grad/action_norm", action_grad_norm, epoch)
            writer.add_scalar("Control/count", float(control_count), epoch)
            writer.add_scalar("Replay/buffer_size", float(buffer_size), epoch)
            writer.add_scalar("Replay/updates", float(replay_updates), epoch)
            if args.iters == 1 and epoch == 0:
                _log_first_segment_step_grads(writer, sim, env)

        if rollout_loss < best_loss:
            best_loss = rollout_loss
            loss_tag = "loss_unknown" if rollout_loss is None or not np.isfinite(rollout_loss) else f"loss_{rollout_loss:.6e}"
            best_ckpt_path = os.path.join(ckpt_dir, f"model_best_{loss_tag}.pth")
            torch.save(controller.state_dict(), best_ckpt_path)
            print(f"Saved best model to: {best_ckpt_path}")

    last_loss_tag = "loss_unknown" if last_loss_val is None or not np.isfinite(last_loss_val) else f"loss_{last_loss_val:.6e}"
    last_ckpt_path = os.path.join(ckpt_dir, f"model_last_{last_loss_tag}.pth")
    torch.save(controller.state_dict(), last_ckpt_path)

    print(f"Saved last model to: {last_ckpt_path}")

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
    controller = Controller()
    load_controller_checkpoint(controller, args.model_path)

    env = DifferentiableCartPoleEnv(
        sim=sim,
        control_interval=args.control_interval,
        backward_interval=args.backward_interval,
        control_start_step=args.control_start_step,
        include_time=args.time_in_state,
        norm_grad=args.norm_grad,
    )

    loss_val, control_count = env.rollout_forward_only(controller)

    if export_settings is not None:
        output_ply, _, output_interval, series_prefix, _, _ = export_settings
        if output_ply:
            print(f"exporting simulation data for test to: {series_prefix}")
            export_backward_data(sim, sim_steps, output_interval, series_prefix)

    print(f"Test complete | loss={loss_val:.6e} | controls={control_count}")


def build_simulator(args):
    config = SimConfig(scene_file_path=args.scene_file)

    sim_steps = args.num_timesteps
    if sim_steps is None:
        sim_steps = derive_sim_steps(config, sim_time=args.t)

    print(f"Using sim_steps: {sim_steps}")

    with wp.ScopedDevice(args.device):
        if args.method == 1:
            dfsph_cls = SimDFSPH_diff if args.dfsph_impl == "default" else SimDFSPH_diff_no_fluid_tape
            print(f"Using DFSPH implementation: {args.dfsph_impl}")
            sim = dfsph_cls(
                config,
                stage_path=args.stage_path,
                container=None,
                sim_steps=sim_steps,
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
                sim_steps=sim_steps,
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
    parser.add_argument("--sim_backward_controller", action="store_true", help="Train controller using SimSPH_diff.backward with one action gradient per backward call.")
    parser.add_argument("--replay_mode", action="store_true", help="Train controller from a replay buffer of collected action gradients.")
    parser.add_argument("--replay_capacity", type=int, default=200, help="Maximum number of replay samples to keep.")
    parser.add_argument("--replay_batch_size", type=int, default=64, help="Mini-batch size for replay updates.")
    parser.add_argument("--replay_updates_per_iter", type=int, default=1, help="Number of replay optimizer steps after each rollout.")
    parser.add_argument("--replay_warmup", type=int, default=64, help="Minimum replay buffer size before replay updates start.")
    parser.add_argument("--backward_interval", "--bi",dest="backward_interval", type=int, default=1,help="Number of control intervals covered by one backward pass.",)
    parser.add_argument("--control_interval", type=int, default=1260, help="Control interval in simulation steps.")
    parser.add_argument("--control_start_step", type=int, default=0, help="First step where control is applied.")
    parser.add_argument("--time_in_state", action="store_true", help="Include time feature in state.")

    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard log directory.")
    parser.add_argument("--model_path", type=str, default=None, help="Model path used to initialize controller or run test.")
    parser.add_argument("--test_only", action="store_true", help="Run non-differentiable controller test only.")
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

    sim_time = args.t
    sim, config, sim_steps = build_simulator(args)
    if sim_time is None:
        sim_time = config.get_cfg("end_time")

    args.backward_interval = int(args.backward_interval)
    if args.backward_interval <= 0:
        raise ValueError("backward_interval must be > 0")
    backward_interval_tag = f"interval_{args.backward_interval}"
    run_tag = build_controller_run_tag(args)

    export_settings = build_export_settings(args, config, end_time=sim_time)

    ckpt_dir = os.path.join(build_checkpoint_dir(args, end_time=sim_time), backward_interval_tag, run_tag)

    if args.log_dir is None:
        args.log_dir = os.path.join(build_default_log_dir(args, end_time=sim_time), backward_interval_tag, run_tag)
    else:
        args.log_dir = os.path.join(args.log_dir, backward_interval_tag, run_tag)
    print(f"TensorBoard logging to {args.log_dir}")
    print(f"Checkpoint directory: {ckpt_dir}")
    print(f"Using backwardInterval (signal spans): {args.backward_interval}")

    if args.test_only:
        test_controller(sim, args, sim_steps, export_settings=export_settings)
    elif args.sim_backward_controller:
        train_controller_with_sim_backward(sim, args, sim_steps, export_settings=export_settings, ckpt_dir=ckpt_dir)
    elif args.replay_mode:
        train_controller_with_replay(sim, args, sim_steps, export_settings=export_settings, ckpt_dir=ckpt_dir)
    else:
        train_controller_online(sim, args, sim_steps, export_settings=export_settings, ckpt_dir=ckpt_dir)


if __name__ == "__main__":
    main()
