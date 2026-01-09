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

ti.init(arch=ti.gpu, device_memory_fraction=0.5)

# wp.config.verify_autograd_array_access = True

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
    parser.add_argument("--num_timesteps", type=int, default=320, help="Total number of frames.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")
    parser.add_argument("--test_gradient", action="store_true", help="Run gradient computation test.")
    parser.add_argument("--train", action="store_true", help="Run optimization training loop.")
    parser.add_argument("--iters", type=int, default=10, help="Number of training iterations.")
    # parser.add_argument("--sim_steps", type=int, default=320, help="Number of simulation steps for gradient computation.")
    parser.add_argument("--ply_path", type=str, default=None, help="Path to PLY file for initialization.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for optimizer.")
    args = parser.parse_args()

    scene_path = args.scene_file
    config = SimConfig(scene_file_path=scene_path)
    # Robust scene name extraction for Windows/Unix paths
    scene_name = os.path.splitext(os.path.basename(scene_path))[0]

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
        os.makedirs(f"{scene_name}_output", exist_ok=True)

    os.makedirs(f"{scene_name}_output", exist_ok=True)
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

        if args.train:
            # Initialize TensorBoard writer
            import datetime
            time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = f"runs/{scene_name}_{time_str}_lr_{args.lr}"
            writer = tensorboardX.SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logging to {log_dir}")

            for i in range(args.iters):
                print(f"------------Starting training for {i}/{args.iters} iterations------------")
                # print(f"grad x_arrays[{0}]:\n", sim.x_arrays[0].grad.numpy())
                sim.backward()
                
                loss_val = sim.loss.numpy()
                print(f"Iteration {i}: Loss = {loss_val}")
                writer.add_scalar('Loss/train', loss_val, i)

                if sim.num_objects > 0:
                    # grad check
                    # print("grad rigid_v0:\n", sim.rigid_v_arrays[0].grad.numpy())
                    # for j in range(sim_steps):
                    #     sim.rigid_grad_print(1, j)
                    print("fluid opt_v_fluid grad:\n", sim.opt_var.grad.numpy())
                    # sim.optimizer.step([sim.rigid_v_arrays[0].grad])
                    sim.optimizer.step([sim.opt_var.grad])
                    grad_fluid = sim.opt_var.grad.numpy()[0]
                    writer.add_scalar('Grad/opt_v_fluid_norm', np.linalg.norm(grad_fluid), i)
                    writer.add_scalar('Grad/opt_v_fluid_x', grad_fluid[0], i)
                    writer.add_scalar('Grad/opt_v_fluid_y', grad_fluid[1], i)
                    writer.add_scalar('Grad/opt_v_fluid_z', grad_fluid[2], i)

                print("fluid opt_v_fluid after optimization:", sim.opt_var.numpy())
                # print("rigid_v after optimization:", sim.rbs.rigid_v.numpy())
                # if sim.num_objects > 0:
                #     v_opt = sim.rigid_v_arrays[0].numpy()
                #     print("Optimized rigid initial linear velocities:", v_opt)

            
            print("Training finished. Running final simulation with optimized parameters...")
            # # Copy optimized initial state to simulation state
            # wp.copy(sim.x, sim.x_arrays[0])
            # wp.copy(sim.v, sim.v_arrays[0])
            # if sim.num_objects > 0:
            #     wp.copy(sim.rbs.rigid_x, sim.rigid_x_arrays[0])
            #     wp.copy(sim.rbs.rigid_v, sim.rigid_v_arrays[0])
            #     wp.copy(sim.rbs.rigid_omega, sim.rigid_omega_arrays[0])
            #     wp.copy(sim.rbs.rigid_quaternion, sim.rigid_quaternion_arrays[0])
            print("exporting simulation data in backward")
            cnt_ply = 0
            for time_step in range(args.num_timesteps):
                if time_step % output_interval == 0:
                    if output_ply:
                        sim.export_ply_from_diff(f'{series_prefix}', time_step, cnt_ply )
                        cnt_ply += 1
                    if output_obj:
                        for r_body_id in container.object_id_rigid_body:
                            with open(f"{scene_name}_output/obj_{r_body_id}_{time_step:06}.obj", "w") as f:
                                e = container.object_collection[r_body_id]["mesh"].export(file_type='obj')
                                f.write(e)
                    time_step += 1

                # sim.step(time_step)

        elif args.test_gradient:
            # Set target to be the initial position (trying to keep particles stationary)
            # Since initial velocity is 0, and gravity exists, they will fall.
            # The optimizer should try to give them upward velocity to counteract gravity.
            # Forward pass
            # print(f"Running forward pass for {args.sim_steps} steps...")
            # sim.forward()
            # print(f"Loss: {sim.loss.numpy()[0]}")

            # Backward pass
            print("Running backward pass...")
            sim.backward()

            # Output gradients
            # We are optimizing initial velocity v_arrays[0]
            grad_v = sim.v_arrays[0].grad.numpy()
            print("Gradient of initial velocity (first 10 particles):")
            print(grad_v[:10])
            
            # Check for non-zero gradients
            print(f"Max gradient magnitude: {np.max(np.abs(grad_v))}")
            print(f"Mean gradient magnitude: {np.mean(np.abs(grad_v))}")

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
