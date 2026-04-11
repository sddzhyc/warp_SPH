import os
import datetime
import argparse
import warp as wp
import warp.optim
import numpy as np
import tensorboardX
from SimSPH_diff import SimSPH_diff
from SimDFSPH_diff import SimDFSPH_diff
from SimSPH import HASH_GRID_SIZE_SCALE
from config_builder import SimConfig
from plot_utils import plot_grid_search_results, save_grid_search_to_csv
from kernel_func import custom_grad_scale, custom_h_scale, use_replay_kernel, replay_scale

# wp.config.verify_autograd_array_access = True
wp.config.verbose = False
warp.config.verbose_warnings = False
warp.config.quiet = False   

def export_backward_data(sim, sim_steps, output_interval, series_prefix):
    cnt_ply = 0
    for time_step in range(sim_steps):
        if time_step % output_interval == 0:
            sim.export_ply_from_diff(f'{series_prefix}', time_step, cnt_ply )
            cnt_ply += 1

def compute_temporal_avg_grad(grad_buffer, current_grad, window_size):
    grad_buffer.append(current_grad)
    if len(grad_buffer) > window_size:
        grad_buffer.pop(0)
    return np.mean(np.array(grad_buffer), axis=0)

def run_training(sim, args, log_dir=None, enable_writer=True):
    writer = None
    if enable_writer and log_dir is not None:
        writer = tensorboardX.SummaryWriter(log_dir=log_dir)
        if args.iters == 1:
            print(f"TensorBoard step logging to {log_dir}")
        else:
            print(f"TensorBoard logging to {log_dir}")

    sim.optimizer.lr = args.lr

    lr_scheduler = None
    torch_lr_optimizer = None
    if args.reduce_lr:
        import torch
        lr_anchor = torch.nn.Parameter(torch.tensor(0.0))
        torch_lr_optimizer = torch.optim.Adam([lr_anchor], lr=args.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            torch_lr_optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=1e-4,
            min_lr=args.min_lr,
        )
        sim.optimizer.lr = float(torch_lr_optimizer.param_groups[0]["lr"])
        print(f"ReduceLROnPlateau enabled: initial lr = {sim.optimizer.lr}")

    grad_buffer = []
    last_loss_scalar = None

    for i in range(args.iters):
        print(f"------------Starting training for {i}/{args.iters} iterations------------")
        sim.backward()

        loss_val = sim.loss.numpy()
        last_loss_scalar = float(np.asarray(loss_val).reshape(-1)[0])
        print(f"Iteration {i}: Loss = {loss_val}")

        # 检查loss为nan时提前终止
        if np.isnan(last_loss_scalar):
            print(f"[Train] Early stopping: Loss is NaN at iter {i}, aborting training.")
            sim.clear_grad()
            sim.reset()
            break

        state_info = sim.task.get_loss_state_info()
        if state_info:
            print(f"Loss State Info: {state_info}")

        if sim.num_objects > 0:
            if isinstance(sim.opt_var, list):
                print(f"[grad] opt_var grad before norm:\n", [v.grad.numpy() for v in sim.opt_var])
            else:
                print(f"[grad] opt_var grad before norm:\n", sim.opt_var.grad.numpy())

            if args.norm_grad:
                sim.norm_final_grad()
                if isinstance(sim.opt_var, list):
                    print("[grad] opt_var grad after norm:\n", [v.grad.numpy() for v in sim.opt_var])
                else:
                    print("[grad] opt_var grad after norm:\n", sim.opt_var.grad.numpy())

            if isinstance(sim.opt_var, list):
                if args.avg_grad:
                    print("avg_grad not supported for list opt_var")
                sim.optimizer.step([v.grad for v in sim.opt_var])
                print("opt_var after optimization:", [v.numpy() for v in sim.opt_var])
            else:
                current_grad = sim.opt_var.grad.numpy().copy()
                print("[grad] opt_var grad: ", current_grad)
                if args.avg_grad:
                    avg_grad = compute_temporal_avg_grad(grad_buffer, current_grad, args.grad_win)
                    print(f"[grad] opt_var grad after temporal averaging:\n", avg_grad)
                    avg_grad_wp = wp.array(avg_grad, dtype=sim.opt_var.dtype, device=args.device)
                    sim.optimizer.step([avg_grad_wp])
                else:
                    sim.optimizer.step([sim.opt_var.grad])
                print("fluid opt_v_fluid after optimization:", sim.opt_var.numpy())

            if lr_scheduler is not None and torch_lr_optimizer is not None:
                prev_lr = float(torch_lr_optimizer.param_groups[0]["lr"])
                lr_scheduler.step(last_loss_scalar)
                new_lr = float(torch_lr_optimizer.param_groups[0]["lr"])
                sim.optimizer.lr = new_lr
                if new_lr != prev_lr:
                    print(f"[lr scheduler] ReduceLROnPlateau: {prev_lr:.6e} -> {new_lr:.6e}")
                if new_lr <= args.min_lr:
                    print(f"[Train] Early stopping triggered at iter {i}: lr {new_lr:.6e} <= min_lr {args.min_lr:.6e}")
                    break

            if writer is not None and args.iters == 1:
                target_id = 1
                if target_id < sim.num_objects:
                    for time_step in range(len(sim.rigid_x_arrays)):
                        for name, array_list in [
                            ("Pos", sim.rigid_x_arrays),
                            ("Vel", sim.rigid_v_arrays),
                            ("Omega", sim.rigid_omega_arrays),
                            ("Quat", sim.rigid_quaternion_arrays),
                            ("Force", sim.rigid_force_arrays),
                            ("Torque", sim.rigid_torque_arrays)
                        ]:
                            if time_step < len(array_list) and array_list[time_step].grad is not None:
                                grad_data = array_list[time_step].grad.numpy()
                                if target_id < len(grad_data):
                                    grad_val = grad_data[target_id]
                                    if len(grad_val) >= 3:
                                        writer.add_scalar(f'Rigid_Step_Grad/{name}/x', float(grad_val[0]), time_step)
                                        writer.add_scalar(f'Rigid_Step_Grad/{name}/y', float(grad_val[1]), time_step)
                                        writer.add_scalar(f'Rigid_Step_Grad/{name}/z', float(grad_val[2]), time_step)
                                    if len(grad_val) == 4:
                                        writer.add_scalar(f'Rigid_Step_Grad/{name}/w', float(grad_val[3]), time_step)

            if writer is not None and args.iters > 1:
                writer.add_scalar('Loss/train', loss_val, i)
                writer.add_scalar('LR/train', sim.optimizer.lr, i)

                opt_vars = sim.opt_var if isinstance(sim.opt_var, list) else [sim.opt_var]
                for var_idx, var in enumerate(opt_vars):
                    var_name = f"Var_{var_idx}"
                    if var.grad:
                        grad = var.grad.numpy()
                        if len(grad.shape) > 1 or (len(grad.shape) == 1 and grad.shape[0] > 1):
                            grad_norm = np.linalg.norm(grad)
                            writer.add_scalar(f'Grad/{var_name}_norm', grad_norm, i)
                            flat_grad = grad.flatten()
                            for comp_idx in range(min(3, len(flat_grad))):
                                writer.add_scalar(f'Grad/{var_name}_comp_{comp_idx}', flat_grad[comp_idx], i)
                        else:
                            writer.add_scalar(f'Grad/{var_name}', grad, i)

                    var_value = var.numpy()
                    flat_value = np.asarray(var_value).reshape(-1)
                    if flat_value.size == 1:
                        writer.add_scalar(f'Var/{var_name}', float(flat_value[0]), i)
                    else:
                        for comp_idx, comp_val in enumerate(flat_value):
                            writer.add_scalar(f'Var/{var_name}_comp_{comp_idx}', float(comp_val), i)

        if args.early_stop_loss is not None and last_loss_scalar < args.early_stop_loss:
            print(f"Early stopping triggered at iter {i}: loss {last_loss_scalar:.6e} < threshold {args.early_stop_loss:.6e}")
            break

    if writer is not None:
        writer.close()

    return last_loss_scalar

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scene_file',
                        default='',
                        help='scene file')
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_sph.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_timesteps", type=int, default=None, help="Total number of frames. If omitted, read from scene config.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")
    parser.add_argument("--test_gradient", action="store_true", help="Run gradient computation test.")
    parser.add_argument("--train", action="store_true", help="Run optimization training loop.")
    parser.add_argument("--iters", type=int, default=10, help="Number of training iterations.")
    parser.add_argument("--early_stop_loss", type=float, default=None, help="Stop training early when loss is below this threshold.")
    # parser.add_argument("--sim_steps", type=int, default=320, help="Number of simulation steps for gradient computation.")
    parser.add_argument("--ply_path", type=str, default=None, help="Path to PLY file for initialization.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for optimizer.")
    parser.add_argument("--reduce_lr", action="store_true", help="Enable ReduceLROnPlateau-based automatic LR decay (PyTorch scheduler).")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum LR threshold for scheduler and LR-based early stop.")
    # Grid Search Arguments
    parser.add_argument("--vx", type=float, default=None, help="Initial vx override.")
    parser.add_argument("--vy", type=float, default=None, help="Initial vy override.")
    parser.add_argument("--grid_search", action="store_true", help="Run 2D grid search for vx and vy together.")
    parser.add_argument("--grid_search_vy", action="store_true", help="Run grid search for vy.")
    parser.add_argument("--grid_search_vx", action="store_true", help="Run grid search for vx.")
    parser.add_argument("--vy_min", type=float, default=-30, help="Min vy for grid search.")
    parser.add_argument("--vy_max", type=float, default=0, help="Max vy for grid search.")
    parser.add_argument("--vy_samples", type=int, default=4, help="Number of samples for vy.")
    parser.add_argument("--vx_min", type=float, default=0, help="Min vx for grid search.")
    parser.add_argument("--vx_max", type=float, default=30, help="Max vx for grid search.")
    parser.add_argument("--vx_samples", type=int, default=4, help="Number of samples for vx.")
    parser.add_argument("--grad_win", type=int, default=10, help="Window size for gradient averaging.")
    parser.add_argument("--avg_grad", action="store_true", help="Use temporal averaged gradient.")
    parser.add_argument("--norm_grad", action="store_true", help="Normalize gradients before optimization.")
    parser.add_argument("--difffr", action="store_true", help="Enable diffFR mode (sparser cross-step gradient flow).")
    parser.add_argument("--method", type=int, default=0, help="Simulation method: 0 for WCSPH, 1 for DFSPH.")
    parser.add_argument("--custom_grad", action="store_true", help="Use custom gradient implementation.")
    parser.add_argument("--init_id", type=int, default=None, help="Select initial rigid-body values from scene JSON keys like velocity_i / angularVelocity_i.")
    h_scale = 1

    parser.add_argument("export_all", action="store_true", help="Export all simulation data.")
    args = parser.parse_args()
    grid_search_mode_count = int(args.grid_search) + int(args.grid_search_vy) + int(args.grid_search_vx)
    if grid_search_mode_count > 1:
        raise ValueError("Please set only one of --grid_search, --grid_search_vy, or --grid_search_vx.")
    use_custom_grad = args.custom_grad
    use_difffr = args.difffr
    scene_path = args.scene_file
    config = SimConfig(scene_file_path=scene_path)

    if args.init_id is not None:
        if args.init_id < 0:
            raise ValueError("--init_id must be >= 0")
        velocity_key = f"velocity_{args.init_id}"
        angular_velocity_key = f"angularVelocity_{args.init_id}"
        rigid_bodies = config.get_rigid_bodies()
        for rigid in rigid_bodies:
            if velocity_key in rigid:
                rigid["velocity"] = rigid[velocity_key]
            if angular_velocity_key in rigid:
                rigid["angularVelocity"] = rigid[angular_velocity_key]
        print(f"Using initial rigid-body keys: {velocity_key}, {angular_velocity_key}")

    end_time = config.get_cfg("end_time")
    if end_time is None:
        raise ValueError("Scene config must define end_time.")
    time_step_size = config.get_cfg("timeStepSize")
    if time_step_size is None:
        raise ValueError("Scene config must define timeStepSize to derive simulation steps from end_time.")
    sim_steps = max(1, int(np.ceil(float(end_time) / float(time_step_size))))
    print(f"Using sim_steps derived from end_time={end_time} and timeStepSize={time_step_size}: {sim_steps}")
    # Robust scene name extraction for Windows/Unix paths
    scene_name = f'{os.path.splitext(os.path.basename(scene_path))[0]}/h{h_scale}/'
    if args.init_id is not None:
        scene_name += f"init{args.init_id}/"
    scene_name += f"hg{HASH_GRID_SIZE_SCALE}/"
    if use_custom_grad:
        scene_name += f"g2_custom_h{custom_h_scale}_g{custom_grad_scale}/"
        if use_replay_kernel:
            scene_name += f"replay_{replay_scale}/"
    elif use_difffr:
        scene_name += "g2_difffr/"
    else:
        scene_name += "g2/"
    if args.avg_grad:
        scene_name += "grad_win{}".format(args.grad_win)
    if args.norm_grad:
        scene_name += "_normed"
    # if args.method == 1:
    #     scene_name += "_dfsph"

    # export settings
    output_frames = config.get_cfg("exportFrame")
    fps = config.get_cfg("fps")
    if fps == None:
        fps = 60
    frame_time = 1.0 / fps

    output_interval = 10  # int(frame_time / config.get_cfg("timeStepSize"))
    total_time = config.get_cfg("totalTime")
    if total_time == None:
        total_time = 10.0

    total_rounds = int(total_time / config.get_cfg("timeStepSize"))
    
    # if config.get_cfg("outputInterval"):
    #     output_interval = config.get_cfg("outputInterval")
    # output_interval = 10
    print(f"Output interval (in steps): {output_interval}")
    output_ply = config.get_cfg("exportPly")
    output_obj = config.get_cfg("exportObj")
    # Use zero-padded frame index in filename
    series_prefix = f"outputs/{scene_name}_diff_output/particle_object_{{:06d}}.ply"
    if output_frames:
        os.makedirs(f"outputs/{scene_name}_output_img", exist_ok=True)
    if output_ply:
        os.makedirs(f"outputs/{scene_name}_diff_output", exist_ok=True)

    # os.makedirs(f"{scene_name}_output", exist_ok=True)
    simulation_method = config.get_cfg("simulationMethod")

    # Reuse the already-resolved args from parse_args() above.

    with wp.ScopedDevice(args.device):
        # skip Taichi container initialization
        container = None
        if args.method == 1:
            sim = SimDFSPH_diff(config, stage_path=args.stage_path, container = container, sim_steps=sim_steps, ply_path=args.ply_path, lr = args.lr, 
            h_scale=h_scale, use_custom_grad=use_custom_grad, use_norm_grad=args.norm_grad, use_diffFR=use_difffr, verbose=True)
        else:
            sim = SimSPH_diff(config, stage_path=args.stage_path, container = container, sim_steps=sim_steps, ply_path=args.ply_path, lr = args.lr, 
            h_scale=h_scale, use_custom_grad=use_custom_grad, use_norm_grad=args.norm_grad, use_diffFR=use_difffr)

        if args.vx is not None or args.vy is not None:
            if isinstance(sim.opt_var, list):
                target_var = sim.opt_var[0]
            else:
                target_var = sim.opt_var
            
            initial_val_np = target_var.numpy().copy()
            if args.vx is not None:
                initial_val_np[0][0] = args.vx
            if args.vy is not None:
                initial_val_np[0][1] = args.vy
                
            new_var = wp.array(initial_val_np, dtype=wp.vec3, device=args.device, requires_grad=True)
            if isinstance(sim.opt_var, list):
                sim.opt_var[0] = new_var
                if hasattr(sim.task, 'opt_rigid_v'):
                    sim.task.opt_rigid_v = new_var
            else:
                sim.opt_var = new_var
                if hasattr(sim.task, 'opt_v_fluid'):
                    sim.task.opt_v_fluid = new_var
            print(f"Overridden initial opt_var to: {sim.task.opt_rigid_v.numpy()}")
        if args.grid_search or args.grid_search_vy or args.grid_search_vx:
            import csv
            
            time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            if args.grid_search:
                log_dir = f"runs/grid_search/{scene_name}/vx{args.vx_min}-{args.vx_max}_vy{args.vy_min}-{args.vy_max}/{time_str}"
            elif args.grid_search_vy:
                log_dir = f"runs/grid_search/{scene_name}/vy{args.vy_min}-{args.vy_max}/{time_str}"
            elif args.grid_search_vx:
                log_dir = f"runs/grid_search/{scene_name}/vx{args.vx_min}-{args.vx_max}/{time_str}"
            writer = tensorboardX.SummaryWriter(log_dir=log_dir)
            print(f"Grid search logging to {log_dir}")
            
            # Handle both list of variables (SkipStoneTask) and single variable (BallDemoTask)
            if isinstance(sim.opt_var, list):
                # Assuming the first variable in the list is the one we want to grid search on (e.g., rigid_v)
                target_var = sim.opt_var[0]
            else:
                target_var = sim.opt_var
                
            initial_val_np = target_var.numpy().copy()

            is_2d_grid_search = args.grid_search
            if is_2d_grid_search:
                vx_values = np.linspace(args.vx_min, args.vx_max, args.vx_samples)
                vy_values = np.linspace(args.vy_min, args.vy_max, args.vy_samples)
                search_axis = None
            elif args.grid_search_vx:
                vx_values = np.linspace(args.vx_min, args.vx_max, args.vx_samples)
                vy_values = np.array([initial_val_np[0][1]])
                search_axis = "vx"
            else:
                vx_values = np.array([initial_val_np[0][0]])
                vy_values = np.linspace(args.vy_min, args.vy_max, args.vy_samples)
                search_axis = "vy"

            total_samples = len(vx_values) * len(vy_values)
            records = []
            loss_list = []
            grad_list = []
            search_values = []

            for vx_idx, vx_value in enumerate(vx_values):
                for vy_idx, vy_value in enumerate(vy_values):
                    sample_idx = vx_idx * len(vy_values) + vy_idx

                    del sim # Ensure we free memory from previous sim before creating a new one for the next sample
                    # Re-instantiate sim for each sample
                    if args.method == 1:
                        sim = SimDFSPH_diff(config, stage_path=args.stage_path, container=container, sim_steps=sim_steps, ply_path=args.ply_path, lr=args.lr, h_scale=h_scale, use_custom_grad=use_custom_grad, use_norm_grad=args.norm_grad, use_diffFR=use_difffr, verbose=True)
                    else:
                        sim = SimSPH_diff(config, stage_path=args.stage_path, container=container, sim_steps=sim_steps, ply_path=args.ply_path, lr=args.lr, h_scale=h_scale, use_custom_grad=use_custom_grad, use_norm_grad=args.norm_grad, use_diffFR=use_difffr)

                    # Set opt_var
                    current_val = initial_val_np.copy()
                    current_val[0][0] = vx_value
                    current_val[0][1] = vy_value

                    new_var = wp.array(current_val, dtype=wp.vec3, device=args.device, requires_grad=True)

                    if isinstance(sim.opt_var, list):
                        sim.opt_var[0] = new_var
                        if hasattr(sim.task, 'opt_rigid_v'):
                            sim.task.opt_rigid_v = new_var
                    else:
                        sim.opt_var = new_var
                        if hasattr(sim.task, 'opt_v_fluid'):
                            sim.task.opt_v_fluid = new_var

                    if is_2d_grid_search:
                        print(f"------------ Grid Search {sample_idx + 1}/{total_samples}: vx = {vx_value:.4f}, vy = {vy_value:.4f} ------------")
                    else:
                        axis_value = vx_value if search_axis == "vx" else vy_value
                        print(f"------------ Grid Search {sample_idx + 1}/{total_samples}: {search_axis} = {axis_value:.4f} ------------")

                    if args.train:
                        sample_train_log_dir = f"runs/{scene_name}/vx_{vx_value:.2f}_vy_{vy_value:.2f}/{time_str}"
                    
                        loss_val = run_training(sim, args, log_dir=sample_train_log_dir, enable_writer=True)
                        if loss_val is None:
                            loss_val = sim.loss.numpy()[0]
                    else:
                        sim.backward()
                        loss_val = sim.loss.numpy()[0]

                    if np.isnan(loss_val):
                        print("Loss is NaN!")
                        loss_val = 1e9

                    current_target_var = sim.opt_var[0] if isinstance(sim.opt_var, list) else sim.opt_var
                    print(f"[grad] opt_var grad before norm:\n", current_target_var.grad.numpy())
                    if args.norm_grad:
                        sim.norm_final_grad()
                        print("[grad] opt_var grad after norm:\n", current_target_var.grad.numpy())

                    grad_val = current_target_var.grad.numpy()[0]
                    grad_vx = grad_val[0]
                    grad_vy = grad_val[1]

                    print(f"Loss: {loss_val:.6f}, Grad_vx: {grad_vx:.6f}, Grad_vy: {grad_vy:.6f}")

                    records.append((vx_value, vy_value, loss_val, grad_vx, grad_vy))

                    writer.add_scalar('GridSearch/Loss', loss_val, sample_idx)
                    writer.add_scalar('GridSearch/Grad_vx', grad_vx, sample_idx)
                    writer.add_scalar('GridSearch/Grad_vy', grad_vy, sample_idx)
                    writer.add_scalar('GridSearch/VX', vx_value, sample_idx)
                    writer.add_scalar('GridSearch/VY', vy_value, sample_idx)
                    if args.train:
                        writer.add_scalar('GridSearch/TrainIters', args.iters, sample_idx)
                    if not is_2d_grid_search:
                        axis_value = vx_value if search_axis == "vx" else vy_value
                        grad_axis = grad_vx if search_axis == "vx" else grad_vy
                        search_values.append(axis_value)
                        loss_list.append(loss_val)
                        grad_list.append(grad_axis)
                        writer.add_scalar(f'GridSearch/Grad_{search_axis}', grad_axis, sample_idx)
                        writer.add_scalar(f'GridSearch/{search_axis.upper()}', axis_value, sample_idx)

            if is_2d_grid_search:
                csv_path = os.path.join(log_dir, "grid_search_data_2d.csv")
                with open(csv_path, mode="w", newline="") as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(["vx", "vy", "loss", "grad_vx", "grad_vy"])
                    csv_writer.writerows(records)
            else:
                axis_min = args.vx_min if search_axis == "vx" else args.vy_min
                axis_max = args.vx_max if search_axis == "vx" else args.vy_max
                plot_grid_search_results(search_values, loss_list, grad_list, axis_min, axis_max, writer)
                csv_path = os.path.join(log_dir, "grid_search_data.csv")
                save_grid_search_to_csv(search_values, loss_list, grad_list, csv_path)

            if records:
                best_record = min(records, key=lambda item: item[2])
                print(f"Best sample: vx={best_record[0]:.6f}, vy={best_record[1]:.6f}, loss={best_record[2]:.6f}")

            writer.close()
            print("Grid search completed.")

        elif args.train:

            time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_vx = sim.task.opt_rigid_v.numpy()[0][0] if hasattr(sim.task, 'opt_rigid_v') else None
            log_vy = sim.task.opt_rigid_v.numpy()[0][1] if hasattr(sim.task, 'opt_rigid_v') else None
            log_dir = f"runs/{scene_name}/vx_{log_vx}_vy_{log_vy}/{time_str}_lr_{args.lr}"

            sim.reset()  # Ensure we start from the initial state before training
            last_loss = run_training(sim, args, log_dir=log_dir, enable_writer=True)
            if last_loss is not None:
                try:
                    print(f"FINAL_LOSS: {float(last_loss):.6f}")
                except Exception:
                    print(f"FINAL_LOSS: {last_loss}")

            print(f"exporting simulation data in backward to: {series_prefix}")
            export_backward_data(sim, sim_steps, output_interval, series_prefix)
            # sim.print_all_rigid_grads()
        else:
            cnt_ply = 0
            for time_step in range(sim_steps):
                # example.render()
                if time_step % output_interval == 0:
                    if output_ply:
                        sim.export_ply_from_diff(series_prefix, time_step, cnt_ply)
                    if output_obj:
                        for r_body_id in container.object_id_rigid_body:
                            with open(f"{scene_name}_output/obj_{r_body_id}_{time_step:06}.obj", "w") as f:
                                e = container.object_collection[r_body_id]["mesh"].export(file_type='obj')
                                f.write(e)
                    cnt_ply += 1

                sim.step(time_step)
            # example.partio_export()
            #if output_frames:
                # if cnt % output_interval == 0:
                #     window.write_image(f"{scene_name}_output_img/{cnt:06}.png")
        # if example.renderer:
        #     example.renderer.save()
    movement_speed = 0.02
    background_color = (0, 0, 0)  # 0xFFFFFF
    particle_color = (1, 1, 1)

    # Invisible objects
    invisible_objects = config.get_cfg("invisibleObjects")
    if not invisible_objects:
        invisible_objects = []
