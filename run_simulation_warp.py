import os
import argparse
import warp as wp

import taichi as ti 

from SimSPH import SimSPH
# from particle_system_np import ParticleSystem
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
    parser.add_argument("--num_frames", type=int, default=400, help="Total number of frames.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")
    args = parser.parse_args()

    scene_path = args.scene_file
    config = SimConfig(scene_file_path=scene_path)
    scene_name = scene_path.split("/")[-1].split(".")[0]

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
    
    if config.get_cfg("outputInterval"):
        output_interval = config.get_cfg("outputInterval")

    output_ply = config.get_cfg("exportPly")
    output_obj = config.get_cfg("exportObj")
    series_prefix = "{}_output/particle_object_{}.ply".format(scene_name, "{}")
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

        example = SimSPH(config, stage_path=args.stage_path, container = container)
        cnt = 0
        cnt_ply = 0
        for time_step in range(args.num_frames):
            # example.render()
            if cnt % output_interval == 0:
                if output_ply:
                    obj_id = 0
                    # Save particle positions to PLY for the specific object
                    #obj_data = container.dump(obj_id=obj_id)
                    np_pos = example.x.numpy()
                    # print(container.object_collection)
                    writer = ti.tools.PLYWriter(num_vertices=np_pos.shape[0])
                    writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
                    writer.export_frame_ascii(cnt_ply, series_prefix.format(0))
                if output_obj:
                    for r_body_id in container.object_id_rigid_body:
                        with open(f"{scene_name}_output/obj_{r_body_id}_{cnt_ply:06}.obj", "w") as f:
                            e = container.object_collection[r_body_id]["mesh"].export(file_type='obj')
                            f.write(e)
                cnt_ply += 1

            example.step(time_step)
            # example.partio_export()
            #if output_frames:
                # if cnt % output_interval == 0:
                #     window.write_image(f"{scene_name}_output_img/{cnt:06}.png")


        # if example.renderer:
        #     example.renderer.save()
        radius = 0.002
    movement_speed = 0.02
    background_color = (0, 0, 0)  # 0xFFFFFF
    particle_color = (1, 1, 1)

    # Invisible objects
    invisible_objects = config.get_cfg("invisibleObjects")
    if not invisible_objects:
        invisible_objects = []
