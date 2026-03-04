import os
import argparse
import argparse
import warp as wp
import warp.optim
import numpy as np
import tensorboardX
from SimSPH_diff import SimSPH_diff
from SimDFSPH_diff import SimDFSPH_diff
from config_builder import SimConfig
from plot_utils import plot_grid_search_results, save_grid_search_to_csv

# wp.config.verify_autograd_array_access = True
wp.config.verbose = False
warp.config.verbose_warnings = False
warp.config.quiet = False   

def export_backward_data(sim, num_timesteps, output_interval, series_prefix):
    cnt_ply = 0
    for time_step in range(num_timesteps):
        if time_step % output_interval == 0:
            sim.export_ply_from_diff(f'{series_prefix}', time_step, cnt_ply )
            cnt_ply += 1

def compute_temporal_avg_grad(grad_buffer, current_grad, window_size):
    grad_buffer.append(current_grad)
    if len(grad_buffer) > window_size:
        grad_buffer.pop(0)
    return np.mean(np.array(grad_buffer), axis=0)

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
    parser.add_argument("--num_timesteps", type=int, default=1400, help="Total number of frames.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")
    parser.add_argument("--test_gradient", action="store_true", help="Run gradient computation test.")
    parser.add_argument("--train", action="store_true", help="Run optimization training loop.")
    parser.add_argument("--iters", type=int, default=10, help="Number of training iterations.")
    # parser.add_argument("--sim_steps", type=int, default=320, help="Number of simulation steps for gradient computation.")
    parser.add_argument("--ply_path", type=str, default=None, help="Path to PLY file for initialization.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for optimizer.")
    # Grid Search Arguments
    parser.add_argument("--grid_search_vy", action="store_true", help="Run grid search for vy.")
    parser.add_argument("--grid_search_vx", action="store_true", help="Run grid search for vx.")
    parser.add_argument("--vy_min", type=float, default=-10, help="Min vy for grid search.")
    parser.add_argument("--vy_max", type=float, default=-3, help="Max vy for grid search.")
    parser.add_argument("--vy_samples", type=int, default=350, help="Number of samples for vy.")
    parser.add_argument("--vx_min", type=float, default=8, help="Min vx for grid search.")
    parser.add_argument("--vx_max", type=float, default=13, help="Max vx for grid search.")
    parser.add_argument("--vx_samples", type=int, default=250, help="Number of samples for vx.")
    parser.add_argument("--grad_win", type=int, default=10, help="Window size for gradient averaging.")
    parser.add_argument("--avg_grad", action="store_true", help="Use temporal averaged gradient.")
    parser.add_argument("--norm_grad", action="store_true", help="Normalize gradients before optimization.")
    parser.add_argument("--method", type=int, default=0, help="Simulation method: 0 for WCSPH, 1 for DFSPH.")
    parser.add_argument("--custom_grad", action="store_true", help="Use custom gradient implementation.")
    h_scale = 1

    parser.add_argument("export_all", action="store_true", help="Export all simulation data.")
    args = parser.parse_args()
    if args.grid_search_vy and args.grid_search_vx:
        raise ValueError("Please set only one of --grid_search_vy or --grid_search_vx.")
    use_custom_grad = args.custom_grad
    scene_path = args.scene_file
    config = SimConfig(scene_file_path=scene_path)
    # Robust scene name extraction for Windows/Unix paths
    scene_name = f'{os.path.splitext(os.path.basename(scene_path))[0]}/h{h_scale}/'
    if use_custom_grad:
        scene_name += "g2_custom_h4/"
    else:
        scene_name += "g2/"
    if args.grid_search_vy:
        scene_name = "grid_search/" + scene_name + "_vy_{}-{}".format(args.vy_min, args.vy_max)
    elif args.grid_search_vx:
        scene_name = "grid_search/" + scene_name + "_vx_{}-{}".format(args.vx_min, args.vx_max)
    if args.avg_grad:
        scene_name += "grad_win{}".format(args.grad_win)
    if args.norm_grad:
        scene_name += "_normed"
    if args.method == 1:
        scene_name += "_dfsph"

    # export settings
    output_frames = config.get_cfg("exportFrame")
    fps = config.get_cfg("fps")
    if fps == None:
        fps = 60
    frame_time = 1.0 / fps

    output_interval = int(frame_time / config.get_cfg("timeStepSize"))
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

    # warp_example code
    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        # skip Taichi container initialization
        container = None
        # If running visualization loop (not training/testing), we need enough steps allocated
        # sim_steps = args.sim_steps 
        sim_steps = args.num_timesteps

        if args.method == 1:
            sim = SimDFSPH_diff(config, stage_path=args.stage_path, container = container, sim_steps=sim_steps, ply_path=args.ply_path, lr = args.lr, 
            h_scale=h_scale, use_custom_grad=use_custom_grad, use_norm_grad=args.norm_grad)
        else:
            sim = SimSPH_diff(config, stage_path=args.stage_path, container = container, sim_steps=sim_steps, ply_path=args.ply_path, lr = args.lr, 
            h_scale=h_scale, use_custom_grad=use_custom_grad, use_norm_grad=args.norm_grad)

        if args.grid_search_vy or args.grid_search_vx:
            import datetime
            
            time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = f"runs/{scene_name}_grid_search/{time_str}"
            writer = tensorboardX.SummaryWriter(log_dir=log_dir)
            print(f"Grid search logging to {log_dir}")
            
            if args.grid_search_vy:
                search_axis = "vy"
                axis_index = 1
                axis_min = args.vy_min
                axis_max = args.vy_max
                axis_samples = args.vy_samples
            else:
                search_axis = "vx"
                axis_index = 0
                axis_min = args.vx_min
                axis_max = args.vx_max
                axis_samples = args.vx_samples

            search_values = np.linspace(axis_min, axis_max, axis_samples)
            
            # Handle both list of variables (SkipStoneTask) and single variable (BallDemoTask)
            if isinstance(sim.opt_var, list):
                # Assuming the first variable in the list is the one we want to grid search on (e.g., rigid_v)
                target_var = sim.opt_var[0]
            else:
                target_var = sim.opt_var
                
            initial_val_np = target_var.numpy().copy()
            
            loss_list = []
            grad_list = []
            
            for i, axis_value in enumerate(search_values):
                # Update target axis component
                current_val = initial_val_np.copy()
                current_val[0][axis_index] = axis_value
                
                # Re-create the array to ensure clean state
                new_var = wp.array(current_val, dtype=wp.vec3, device=args.device, requires_grad=True)
                
                # Update the task's optimization variable
                if isinstance(sim.opt_var, list):
                    sim.opt_var[0] = new_var
                    # Also need to update the specific task attribute if it exists
                    if hasattr(sim.task, 'opt_rigid_v'):
                        sim.task.opt_rigid_v = new_var
                else:
                    sim.opt_var = new_var
                    if hasattr(sim.task, 'opt_v_fluid'):
                        sim.task.opt_v_fluid = new_var
                
                print(f"--- Grid Search {i+1}/{axis_samples}: {search_axis} = {axis_value:.4f} ---")
                
                # Run simulation and backward pass
                sim.backward()
                
                loss_val = sim.loss.numpy()[0]
                # Check for nan
                if np.isnan(loss_val):
                    print("Loss is NaN!")
                    loss_val = 1e9 # sentinel
                    
                if args.norm_grad:
                    sim.norm_final_grad()
                    
                # Get gradient from the updated variable
                if isinstance(sim.opt_var, list):
                    current_target_var = sim.opt_var[0]
                else:
                    current_target_var = sim.opt_var
                    
                if args.norm_grad:
                    print("opt_var grad after norm:\n", current_target_var.grad.numpy())
                    
                grad_val = current_target_var.grad.numpy()[0] # [gx, gy, gz]
                grad_axis = grad_val[axis_index]
                
                print(f"Loss: {loss_val:.6f}, Grad_{search_axis}: {grad_axis:.6f}")
                
                loss_list.append(loss_val)
                grad_list.append(grad_axis)
                
                # Write to TensorBoard
                # Use i as step. Log vy as a metric.
                # TensorBoard global_step 必须是整数，为了保留小数精度，乘以 100 作为横坐标
                # step_val = int(vy * 100)
                writer.add_scalar('GridSearch/Loss', loss_val, i)
                writer.add_scalar(f'GridSearch/Grad_{search_axis}', grad_axis, i)
                writer.add_scalar(f'GridSearch/{search_axis.upper()}', axis_value, i)

            # Generate Summary Plot
            plot_grid_search_results(search_values, loss_list, grad_list, axis_min, axis_max, writer)
            
            # Save data to CSV
            csv_path = os.path.join(log_dir, "grid_search_data.csv")
            save_grid_search_to_csv(search_values, loss_list, grad_list, csv_path)

            writer.close()
            print("Grid search completed.")

        elif args.train:
            # Initialize TensorBoard writer
            import datetime
            time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = f"runs/{scene_name}/{time_str}_lr_{args.lr}"
            writer = None

            best_param = None

            loss_ema = None
            initial_loss = None
            plateau_counter = 0
            
            grad_buffer = []

            for i in range(args.iters):
                print(f"------------Starting training for {i}/{args.iters} iterations------------")
                # print(f"grad x_arrays[{0}]:\n", sim.x_arrays[0].grad.numpy())
                sim.backward()
                
                loss_val = sim.loss.numpy()
                print(f"Iteration {i}: Loss = {loss_val}")
                
                # Print loss state info
                state_info = sim.task.get_loss_state_info()
                if state_info:
                    print(f"Loss State Info: {state_info}")

                if sim.num_objects > 0:
                    # grad check
                    # print("grad rigid_v0:\n", sim.rigid_v_arrays[0].grad.numpy())
                    # for j in range(sim_steps):
                    #     sim.rigid_grad_print(1, j)
                    
                    if args.norm_grad:
                        sim.norm_final_grad()
                        if isinstance(sim.opt_var, list):
                            print("opt_var grad after norm:\n", [v.grad.numpy() for v in sim.opt_var])
                        else:
                            print("opt_var grad after norm:\n", sim.opt_var.grad.numpy())

                    if isinstance(sim.opt_var, list):
                        current_grad = [v.grad.numpy().copy() for v in sim.opt_var]
                        print("opt_var grad: ", current_grad)
                        if args.avg_grad:
                            # avg_grad not supported for list opt_var
                            print("avg_grad not supported for list opt_var")
                            sim.optimizer.step([v.grad for v in sim.opt_var])
                        else:
                            sim.optimizer.step([v.grad for v in sim.opt_var])
                        grad_opt = [v.grad.numpy() for v in sim.opt_var]
                        print("opt_var after optimization:", [v.numpy() for v in sim.opt_var])
                    else:
                        current_grad = sim.opt_var.grad.numpy().copy()
                        print("opt_var grad: ", current_grad)
                        if args.avg_grad:
                            # Use helper function for temporal averaging
                            avg_grad = compute_temporal_avg_grad(grad_buffer, current_grad, args.grad_win)
                            
                            print("fluid avg grad:\n", avg_grad)
                            
                            # Create warp array for averaged gradient
                            avg_grad_wp = wp.array(avg_grad, dtype=sim.opt_var.dtype, device=args.device)
                            sim.optimizer.step([avg_grad_wp])
                            grad_fluid = avg_grad[0]
                        else:
                            sim.optimizer.step([sim.opt_var.grad])
                            grad_fluid = sim.opt_var.grad.numpy()[0]
                        print("fluid opt_v_fluid after optimization:", sim.opt_var.numpy())

                    if args.iters > 1:
                        if writer is None: # create writer on first use
                            writer = tensorboardX.SummaryWriter(log_dir=log_dir)
                            print(f"TensorBoard logging to {log_dir}")

                        writer.add_scalar('Loss/train', loss_val, i)
                        writer.add_scalar('LR/train', sim.optimizer.lr, i)
                        
                        # Log gradients for all optimized variables from sim.opt_var
                        if isinstance(sim.opt_var, list):
                             opt_vars = sim.opt_var
                        else:
                             opt_vars = [sim.opt_var]

                        for var_idx, var in enumerate(opt_vars):
                            if var.grad:
                                grad = var.grad.numpy()
                                var_name = f"Var_{var_idx}"
                                
                                # If it's a vector/array, log norm and components of first element
                                if len(grad.shape) > 1 or (len(grad.shape) == 1 and grad.shape[0] > 1):
                                    grad_norm = np.linalg.norm(grad)
                                    writer.add_scalar(f'Grad/{var_name}_norm', grad_norm, i)
                                    
                                    # Log first few components if available
                                    flat_grad = grad.flatten()
                                    for comp_idx in range(min(3, len(flat_grad))):
                                        writer.add_scalar(f'Grad/{var_name}_comp_{comp_idx}', flat_grad[comp_idx], i)
                                else:
                                    # Scalar
                                    writer.add_scalar(f'Grad/{var_name}', grad, i)

                # print("rigid_v after optimization:", sim.rbs.rigid_v.numpy())
                # if sim.num_objects > 0:
                #     v_opt = sim.rigid_v_arrays[0].numpy()
                #     print("Optimized rigid initial linear velocities:", v_opt)

                # if loss_val < min_loss or args.export_all:
                #     min_loss = loss_val
                #     iter_dir = f"{scene_name}_diff_output/iter_{i:03d}"
                #     os.makedirs(iter_dir, exist_ok=True)
                #     iter_series_prefix = f"{iter_dir}/particle_object_{{:06d}}.ply"
                #     print(f"New loss {min_loss} at iteration {i}, exporting simulation data...")
                #     export_backward_data(sim, args.num_timesteps, output_interval, iter_series_prefix)
            
            print("Training finished. Running final simulation with optimized parameters...")

            print("exporting simulation data in backward")
            print(f"Exporting data to: {series_prefix}")
            export_backward_data(sim, args.num_timesteps, output_interval, series_prefix)
            # sim.print_all_rigid_grads()
        else:
            cnt_ply = 0
            for time_step in range(args.num_timesteps):
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
