import os
import argparse
import warp as wp
import numpy as np

import taichi as ti 

# from particle_system_np import ParticleSystem
from SimSPH_diff import SimSPH_diff
from particle_system import ParticleSystem
from config_builder import SimConfig

ti.init(arch=ti.gpu, device_memory_fraction=0.5)

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
    parser.add_argument("--num_frames", type=int, default=80000, help="Total number of frames.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")
    parser.add_argument("--test_gradient", action="store_true", help="Run gradient computation test.")
    parser.add_argument("--sim_steps", type=int, default=1, help="Number of simulation steps for gradient computation.")
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

    # output_interval = int(frame_time / config.get_cfg("timeStepSize"))
    total_time = config.get_cfg("totalTime")
    if total_time == None:
        total_time = 10.0

    total_rounds = int(total_time / config.get_cfg("timeStepSize"))
    
    # if config.get_cfg("outputInterval"):
    #     output_interval = config.get_cfg("outputInterval")
    output_interval = int(0.016 / config.get_cfg("timeStepSize"))
    output_ply = config.get_cfg("exportPly")
    output_obj = config.get_cfg("exportObj")
    # Use zero-padded frame index in filename
    series_prefix = f"{scene_name}_output/particle_object_{{:06d}}.ply"
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

        example = SimSPH_diff(config, stage_path=args.stage_path, container = container, sim_steps=args.sim_steps)
        
        if args.test_gradient:
            # Set target to be the initial position (trying to keep particles stationary)
            # Since initial velocity is 0, and gravity exists, they will fall.
            # The optimizer should try to give them upward velocity to counteract gravity.
            wp.copy(example.target_x, example.x)
            if example.num_objects > 0:
                wp.copy(example.target_rigid_x, example.rbs.rigid_x)
                wp.copy(example.target_rigid_q, example.rbs.rigid_quaternion)
            
            # Forward pass
            print(f"Running forward pass for {args.sim_steps} steps...")
            example.forward()
            print(f"Loss: {example.loss.numpy()[0]}")

            # Backward pass
            print("Running backward pass...")
            example.backward()

            # Output gradients
            # We are optimizing initial velocity v_arrays[0]
            grad_v = example.v_arrays[0].grad.numpy()
            print("Gradient of initial velocity (first 10 particles):")
            print(grad_v[:10])
            
            # Check for non-zero gradients
            print(f"Max gradient magnitude: {np.max(np.abs(grad_v))}")
            print(f"Mean gradient magnitude: {np.mean(np.abs(grad_v))}")

        else:
            cnt = 0
            cnt_ply = 0
            for time_step in range(args.num_frames):
                # example.render()
                if cnt % output_interval == 0:
                    if output_ply:
                        print(f"Exporting frame {cnt_ply} to PLY on time step {cnt}.")
                        example.export_ply(series_prefix, cnt_ply)
                    if output_obj:
                        for r_body_id in container.object_id_rigid_body:
                            with open(f"{scene_name}_output/obj_{r_body_id}_{cnt_ply:06}.obj", "w") as f:
                                e = container.object_collection[r_body_id]["mesh"].export(file_type='obj')
                                f.write(e)
                    cnt_ply += 1

                example.step(time_step)
                cnt += 1
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
