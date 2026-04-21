# pyright: reportInvalidTypeForm=false
import argparse
import os

import numpy as np
import torch
import tensorboardX
import warp as wp

from config_builder import SimConfig
from SimSPH_diff import SimSPH_diff
from SimDFSPH_diff import SimDFSPH_diff
from SimDFSPH_diff_no_fluid_tape import SimDFSPH_diff as SimDFSPH_diff_no_fluid_tape
from robots.controller import Controller
from utils import export_backward_data, build_export_settings, build_default_log_dir, build_checkpoint_dir


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


def derive_sim_steps(cfg, sim_time=None):
    end_time = sim_time if sim_time is not None else cfg.get_cfg("end_time")
    if end_time is None:
        raise ValueError("Simulation time must be provided by --t or scene config end_time.")

    time_step_size = cfg.get_cfg("timeStepSize")
    if time_step_size is None:
        raise ValueError("Scene config must define timeStepSize.")

    return max(1, int(np.ceil(float(end_time) / float(time_step_size))))


class DifferentiableCartPoleEnv:
    def __init__(self, sim, control_interval, backward_interval, control_start_step=2, include_time=True):
        self.sim = sim
        self.control_interval = int(control_interval)
        self.backward_interval = int(backward_interval)
        self.control_start_step = int(control_start_step)
        self.include_time = bool(include_time)

        if self.control_interval <= 0:
            raise ValueError("control_interval must be > 0")
        if self.backward_interval <= 0:
            raise ValueError("backward_interval must be > 0")
        if self.control_start_step < 0:
            raise ValueError("control_start_step must be >= 0")

        self.rod_rigid_id = int(getattr(sim.task, "rod_rigid_id", 1))
        self.car_rigid_id = int(getattr(sim.task, "car_rigid_id", 2))

        self.action_wp_vars = []
        self.action_torch_vars = []
        self.action_steps = []
        self.action_records = []

    def reset_episode(self):
        self.sim.clear_grad()
        self.sim.reset()

        self.action_wp_vars = []
        self.action_torch_vars = []
        self.action_steps = []
        self.action_records = []

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

    def _action_to_wp_vec3(self, action_xyz):
        action_np = action_xyz.detach().cpu().numpy().astype(np.float32)
        delta_v_np = np.array([[action_np[0], action_np[1], action_np[2]]], dtype=np.float32)
        return wp.array(
            delta_v_np,
            dtype=wp.vec3,
            device=self.sim.rigid_v_arrays[0].device,
            requires_grad=True,
        )

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

    def _compute_loss_with_optional_end_step(self, tape, end_step):
        with tape:
            try:
                self.sim.task.compute_loss(end_step=end_step)
            except TypeError:
                self.sim.task.compute_loss()

    def _backward_one_segment(self, tape, end_step, action_torch_vars, action_wp_vars, action_records, optimizer, grad_clip_norm):
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

        optimizer.zero_grad()
        if action_torch_vars:
            grad_tensors = [
                torch.tensor(g, dtype=action_torch_vars[0].dtype, device=action_torch_vars[0].device)
                for g in action_grads
            ]
            torch.autograd.backward(action_torch_vars, grad_tensors=grad_tensors)
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], grad_clip_norm)
            optimizer.step()

        return loss_val, action_grad_norm, len(action_torch_vars)

    def rollout(self, controller, optimizer, grad_clip_norm=None):
        self.reset_episode()

        tape = wp.Tape()
        self.sim.tapes = [tape]

        segment_span_steps = self.backward_interval * self.control_interval

        segment_action_wp_vars = []
        segment_action_torch_vars = []
        segment_action_records = []
        segment_start_step = None

        segment_loss_vals = []
        segment_grad_norms = []
        total_controls = 0

        for t in range(self.sim.sim_steps):
            if self.should_control(t):
                if segment_start_step is None:
                    segment_start_step = t

                state_np = self.get_state(t)
                action_torch = controller(state_np).reshape(-1)
                action_wp = self._action_to_wp_vec3(action_torch)

                rod_quat = self.sim.rigid_quaternion_arrays[t].numpy()[self.rod_rigid_id].copy()
                current_time = float(t) * float(self.sim.frame_dt)

                self.apply_action(t, action_wp, tape)

                self.action_steps.append(t)
                self.action_torch_vars.append(action_torch)
                self.action_wp_vars.append(action_wp)
                segment_action_torch_vars.append(action_torch)
                segment_action_wp_vars.append(action_wp)
                self.action_records.append(
                    {
                        "step": t,
                        "time": current_time,
                        "rod_quat": rod_quat,
                        "action": action_torch.detach().cpu().numpy().astype(np.float32),
                    }
                )
                segment_action_records.append(self.action_records[-1])

            self.sim.step(t)

            is_last_step = (t == self.sim.sim_steps - 1)
            reach_segment_span = (
                segment_start_step is not None
                and (t + 1) >= (segment_start_step + segment_span_steps)
            )
            should_backward_now = bool(segment_action_wp_vars) and (reach_segment_span or is_last_step)

            if should_backward_now:
                end_step = t + 1
                loss_val, action_grad_norm, controls_used = self._backward_one_segment(
                    tape=tape,
                    end_step=end_step,
                    action_torch_vars=segment_action_torch_vars,
                    action_wp_vars=segment_action_wp_vars,
                    action_records=segment_action_records,
                    optimizer=optimizer,
                    grad_clip_norm=grad_clip_norm,
                )

                segment_loss_vals.append(loss_val)
                segment_grad_norms.append(action_grad_norm)
                total_controls += controls_used

                # Clear grads and tape graph before continuing differential rollout.
                self.sim.clear_grad()
                tape = wp.Tape()
                self.sim.tapes = [tape]

                segment_action_wp_vars = []
                segment_action_torch_vars = []
                segment_action_records = []
                segment_start_step = None

        if not segment_loss_vals:
            return {
                "loss": float("nan"),
                "action_grad_norm": 0.0,
                "controls": 0,
                "segments": 0,
            }

        return {
            "loss": float(np.mean(np.asarray(segment_loss_vals, dtype=np.float32))),
            "action_grad_norm": float(np.mean(np.asarray(segment_grad_norms, dtype=np.float32))) if segment_grad_norms else 0.0,
            "controls": int(total_controls),
            "segments": len(segment_loss_vals),
        }

    def rollout_forward_only(self, controller):
        self.reset_episode()

        control_count = 0
        action_records = []

        controller.eval()
        with torch.no_grad():
            for t in range(self.sim.sim_steps):
                if self.should_control(t):
                    state_np = self.get_state(t)
                    action_torch = controller(state_np).reshape(-1)
                    action_wp = self._action_to_wp_vec3(action_torch)

                    rod_quat = self.sim.rigid_quaternion_arrays[t].numpy()[self.rod_rigid_id].copy()
                    current_time = float(t) * float(self.sim.frame_dt)

                    self.apply_action(t, action_wp, tape=None)

                    action_records.append(
                        {
                            "step": t,
                            "time": current_time,
                            "rod_quat": rod_quat,
                            "action": action_torch.detach().cpu().numpy().astype(np.float32),
                        }
                    )
                    control_count += 1

                self.sim.step(t)

        try:
            self.sim.task.compute_loss(end_step=self.sim.sim_steps)
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


def train_controller(sim, args, sim_steps, export_settings=None, ckpt_dir=None):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = DifferentiableCartPoleEnv(
        sim=sim,
        control_interval=args.control_interval,
        backward_interval=args.backward_interval,
        control_start_step=args.control_start_step,
        include_time=(not args.disable_time_in_state),
    )

    controller = Controller()
    load_controller_checkpoint(controller, args.model_path)
    optimizer = torch.optim.Adam(controller.parameters(), lr=args.lr)

    writer = None
    if args.log_dir and args.iters >= 2:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = tensorboardX.SummaryWriter(log_dir=args.log_dir)

    if ckpt_dir is None:
        ckpt_dir = build_checkpoint_dir(args)

    os.makedirs(ckpt_dir, exist_ok=True)

    best_loss = float("inf")
    best_ckpt_path = None
    last_loss_val = None

    for epoch in range(args.iters):
        controller.train()
        rollout_stats = env.rollout(
            controller=controller,
            optimizer=optimizer,
            grad_clip_norm=args.grad_clip_norm,
        )

        loss_val = rollout_stats["loss"]
        last_loss_val = loss_val
        action_grad_norm = rollout_stats["action_grad_norm"]
        control_count = rollout_stats["controls"]
        segment_count = rollout_stats["segments"]

        if np.isnan(loss_val):
            print(f"[Train] Early stopping: loss is NaN at epoch {epoch}.")
            break

        print(
            f"Epoch {epoch:04d} | avg_loss={loss_val:.6e} | controls={control_count} | "
            f"segments={segment_count} | avg_action_grad_norm={action_grad_norm:.6e}"
        )

        if writer is not None:
            writer.add_scalar("Loss/train", loss_val, epoch)
            writer.add_scalar("Grad/action_norm", action_grad_norm, epoch)
            writer.add_scalar("Control/count", float(control_count), epoch)
            writer.add_scalar("Backward/segments", float(segment_count), epoch)

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
        include_time=(not args.disable_time_in_state),
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
    parser.add_argument(
        "--backward_interval",
        "--backwardInterval",
        dest="backward_interval",
        type=int,
        default=5,
        help="Number of control intervals covered by one backward pass.",
    )

    parser.add_argument("--control_interval", type=int, default=100, help="Control interval in simulation steps.")
    parser.add_argument("--control_start_step", type=int, default=2, help="First step where control is applied.")
    parser.add_argument("--disable_time_in_state", action="store_true", help="Disable time feature in state.")

    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard log directory.")
    parser.add_argument("--model_path", type=str, default=None, help="Model path used to initialize controller or run test.")
    parser.add_argument("--test_only", action="store_true", help="Run non-differentiable controller test only.")
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

    export_settings = build_export_settings(args, config, end_time=sim_time)

    ckpt_dir = os.path.join(build_checkpoint_dir(args, end_time=sim_time), backward_interval_tag)

    if args.log_dir is None:
        args.log_dir = os.path.join(build_default_log_dir(args, end_time=sim_time), backward_interval_tag)
    print(f"TensorBoard logging to {args.log_dir}")
    print(f"Checkpoint directory: {ckpt_dir}")
    print(f"Using backwardInterval (signal spans): {args.backward_interval}")

    if args.test_only:
        test_controller(sim, args, sim_steps, export_settings=export_settings)
    else:
        train_controller(sim, args, sim_steps, export_settings=export_settings, ckpt_dir=ckpt_dir)


if __name__ == "__main__":
    main()
