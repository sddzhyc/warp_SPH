import datetime
import os

from SimSPH import HASH_GRID_SIZE_SCALE
from kernel_func import custom_grad_scale, custom_h_scale, use_0_order_grad


def export_backward_data(sim, sim_steps, output_interval, series_prefix):
    cnt_ply = 0
    for time_step in range(sim_steps):
        if time_step % output_interval == 0:
            sim.export_ply_from_diff(f"{series_prefix}", time_step, cnt_ply)
            cnt_ply += 1


def build_grad_mode_info(args):
    grad_mode_info = ""
    if args.difffr:
        grad_mode_info += "difffr/"
    if args.custom_grad:
        if use_0_order_grad:
            grad_mode_info += f"g2_custom_const{custom_grad_scale}/"
        else:
            grad_mode_info += f"g2_custom_h{custom_h_scale}_g{custom_grad_scale}/"
    else:
        grad_mode_info += "g2/"
    if args.norm_grad:
        grad_mode_info += "_normed"
    return grad_mode_info


def build_end_time_tag(end_time):
    if end_time is None:
        return "et_unknown"
    return f"et{end_time}"


def build_scene_name_with_grad_mode(args, end_time=None):
    scene_name = f"{os.path.splitext(os.path.basename(args.scene_file))[0]}/h{args.h_scale}/"
    scene_name += f"hg{HASH_GRID_SIZE_SCALE}/"
    scene_name += f"{build_end_time_tag(end_time)}/"
    return f"{scene_name}controller/{build_grad_mode_info(args)}"


def build_export_settings(args, config, end_time=None):
    output_ply = config.get_cfg("exportPly")
    output_obj = config.get_cfg("exportObj")
    fps = config.get_cfg("fps")
    if fps is None:
        fps = 60
    frame_time = 1.0 / fps

    output_interval = 10
    if end_time is None:
        end_time = config.get_cfg("end_time")
    scene_name_with_grad_mode = build_scene_name_with_grad_mode(args, end_time=end_time)
    backward_interval = getattr(args, "backward_interval", None)
    if backward_interval is not None:
        scene_name_with_grad_mode = f"{scene_name_with_grad_mode.rstrip('/')}/interval_{int(backward_interval)}/"
    series_prefix = f"outputs/{scene_name_with_grad_mode}_diff_output/particle_object_{{:06d}}.ply"

    if config.get_cfg("exportFrame"):
        os.makedirs(f"outputs/{scene_name_with_grad_mode}_output_img", exist_ok=True)
    if output_ply:
        os.makedirs(f"outputs/{scene_name_with_grad_mode}_diff_output", exist_ok=True)

    return output_ply, output_obj, output_interval, series_prefix, frame_time, scene_name_with_grad_mode


def build_default_log_dir(args, end_time=None):
    scene_name = build_scene_name_with_grad_mode(args, end_time=end_time)
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"runs/{scene_name}/{time_str}_lr_{args.lr}"


def build_checkpoint_dir(args, end_time=None):
    # Keep checkpoint paths based only on grad mode information.
    grad_mode_info = build_grad_mode_info(args)
    end_time_tag = build_end_time_tag(end_time)
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join("checkpoints", end_time_tag, f"{grad_mode_info}/{time_str}_lr_{args.lr}")
