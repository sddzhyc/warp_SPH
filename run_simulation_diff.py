import os
import argparse
import warp as wp
import warp.optim
import numpy as np
import tensorboardX
import taichi as ti 

# from particle_system_np import ParticleSystem
from SimSPH_diff import SimSPH_diff
from particle_system import ParticleSystem
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
    parser.add_argument("--num_timesteps", type=int, default=160, help="Total number of frames.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")
    parser.add_argument("--test_gradient", action="store_true", help="Run gradient computation test.")
    parser.add_argument("--train", action="store_true", help="Run optimization training loop.")
    parser.add_argument("--iters", type=int, default=10, help="Number of training iterations.")
    # parser.add_argument("--sim_steps", type=int, default=320, help="Number of simulation steps for gradient computation.")
    parser.add_argument("--ply_path", type=str, default=None, help="Path to PLY file for initialization.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for optimizer.")
    # Grid Search Arguments
    parser.add_argument("--grid_search_vy", action="store_true", help="Run grid search for vy.")
    parser.add_argument("--vy_min", type=float, default=3, help="Min vy for grid search.")
    parser.add_argument("--vy_max", type=float, default=15, help="Max vy for grid search.")
    parser.add_argument("--vy_samples", type=int, default=1200, help="Number of samples for vy.")
    parser.add_argument("--grad_win", type=int, default=10, help="Window size for gradient averaging.")
    parser.add_argument("--avg_grad", action="store_true", help="Use temporal averaged gradient.")
    parser.add_argument("--norm_grad", action="store_true", help="Normalize gradients before optimization.")
    parser.add_argument("--method", type=int, default=0, help="Simulation method: 0 for WCSPH, 1 for DFSPH.")

    parser.add_argument("export_all", action="store_true", help="Export all simulation data.")
    args = parser.parse_args()

    scene_path = args.scene_file
    config = SimConfig(scene_file_path=scene_path)
    # Robust scene name extraction for Windows/Unix paths
    scene_name = os.path.splitext(os.path.basename(scene_path))[0]+'/h1/'
    if args.grid_search_vy:
        scene_name = "grid_search/" + scene_name + "_vy_{}-{}".format(args.vy_min, args.vy_max)
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
    series_prefix = f"{scene_name}_diff_output/particle_object_{{:06d}}.ply"
    if output_frames:
        os.makedirs(f"{scene_name}_output_img", exist_ok=True)
    if output_ply:
        os.makedirs(f"{scene_name}_diff_output", exist_ok=True)

    # os.makedirs(f"{scene_name}_output", exist_ok=True)
    simulation_method = config.get_cfg("simulationMethod")

    # warp_example code
    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        container = ParticleSystem(config, GGUI=True)
        # prepare the container before creating the simulation so SimSPH

        # If running visualization loop (not training/testing), we need enough steps allocated
        # sim_steps = args.sim_steps
        sim_steps = args.num_timesteps

        sim = SimSPH_diff(config, stage_path=args.stage_path, container = container, sim_steps=sim_steps, ply_path=args.ply_path, lr = args.lr)
        # set target x/rotation for loss computation
        wp.copy(sim.target_x, sim.x)
        if sim.num_objects > 0:
            wp.copy(sim.target_rigid_x, sim.rbs.rigid_x)
            target_q_np = np.zeros((sim.num_objects, 4), dtype=np.float32)
            target_q_np[:, 3] = 1.0
            target_q_np[1,:] = np.array([0.0, 0.7071, 0.0, 0.7071], dtype=np.float32)  # 90 degrees around Y axis
            sim.target_rigid_q = wp.array(target_q_np, dtype=wp.quat, device=args.device)
            print("Target rigid quaternions:\n", sim.target_rigid_q.numpy())

        if args.grid_search_vy:
            import datetime
            
            time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = f"runs/{scene_name}_grid_search/{time_str}"
            writer = tensorboardX.SummaryWriter(log_dir=log_dir)
            print(f"Grid search logging to {log_dir}")
            
            vy_values = np.linspace(args.vy_min, args.vy_max, args.vy_samples)
            initial_val_np = sim.opt_v_fluid.numpy().copy()
            
            loss_list = []
            grad_list = []
            
            for i, vy in enumerate(vy_values):
                # Update vy
                current_val = initial_val_np.copy()
                current_val[0][1] = vy # Update y component
                
                # Re-create the array to ensure clean state (or copy_from)
                # Important: requires_grad=True must be set
                sim.opt_v_fluid = wp.array(current_val, dtype=wp.vec3, device=args.device, requires_grad=True)
                # Also update sim.opt_var reference if it's used elsewhere, though backward uses opt_v_fluid
                sim.opt_var = sim.opt_v_fluid 
                
                print(f"--- Grid Search {i+1}/{args.vy_samples}: vy = {vy:.4f} ---")
                
                # Run simulation and backward pass
                sim.backward()
                
                loss_val = sim.loss.numpy()[0]
                # Check for nan
                if np.isnan(loss_val):
                    print("Loss is NaN!")
                    loss_val = 1e9 # sentinel
                    
                if args.norm_grad:
                    sim.norm_final_grad()
                    print("fluid opt_v_fluid grad after norm:\n", sim.opt_v_fluid.grad.numpy())
                grad_val = sim.opt_v_fluid.grad.numpy()[0] # [gx, gy, gz]
                grad_y = grad_val[1]
                
                print(f"Loss: {loss_val:.6f}, Grad_y: {grad_y:.6f}")
                
                loss_list.append(loss_val)
                grad_list.append(grad_y)
                
                # Write to TensorBoard
                # Use i as step. Log vy as a metric.
                # TensorBoard global_step 必须是整数，为了保留小数精度，乘以 100 作为横坐标
                # step_val = int(vy * 100)
                writer.add_scalar('GridSearch/Loss', loss_val, i)
                writer.add_scalar('GridSearch/Grad_y', grad_y, i)
                writer.add_scalar('GridSearch/Vy', vy, i)

            # Generate Summary Plot
            plot_grid_search_results(vy_values, loss_list, grad_list, args.vy_min, args.vy_max, writer)
            
            # Save data to CSV
            csv_path = os.path.join(log_dir, "grid_search_data.csv")
            save_grid_search_to_csv(vy_values, loss_list, grad_list, csv_path)

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

                if sim.num_objects > 0:
                    # grad check
                    # print("grad rigid_v0:\n", sim.rigid_v_arrays[0].grad.numpy())
                    # for j in range(sim_steps):
                    #     sim.rigid_grad_print(1, j)
                    
                    if args.norm_grad:
                        sim.norm_final_grad()
                        print("fluid opt_v_fluid grad after norm:\n", sim.opt_var.grad.numpy())

                    current_grad = sim.opt_var.grad.numpy().copy()
                    print("fluid opt_v_fluid grad:\n", current_grad)
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

                    if args.iters > 1:
                        if writer is None: # create writer on first use
                            writer = tensorboardX.SummaryWriter(log_dir=log_dir)
                            print(f"TensorBoard logging to {log_dir}")

                        writer.add_scalar('Loss/train', loss_val, i)
                        writer.add_scalar('LR/train', sim.optimizer.lr, i)
                        writer.add_scalar('Grad/opt_v_fluid_norm', np.linalg.norm(grad_fluid), i)
                        writer.add_scalar('Grad/opt_v_fluid_x', grad_fluid[0], i)
                        writer.add_scalar('Grad/opt_v_fluid_y', grad_fluid[1], i)
                        writer.add_scalar('Grad/opt_v_fluid_z', grad_fluid[2], i)
                        writer.add_scalar('Grad/current_grad_fluid_y', current_grad[0][1], i)
                
                print("fluid opt_v_fluid after optimization:", sim.opt_var.numpy())
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
