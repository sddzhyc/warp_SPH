# pyright: reportInvalidTypeForm=false
import argparse
import datetime
import json
from pathlib import Path
import os
import sys
import time
import types

import numpy as np
import torch
import math
import warp as wp

import newton
import newton.examples


def _compat_export_backward_data(sim, sim_steps, output_interval, series_prefix, start_cnt=0, include_last_step=False):
    cnt_ply: int = int(start_cnt)
    output_interval = max(1, int(output_interval))
    for time_step in range(int(sim_steps)):
        if time_step % output_interval == 0:
            sim.export_ply_from_diff(f"{series_prefix}", time_step, cnt_ply)
            cnt_ply += 1
    if include_last_step and int(sim_steps) > 0:
        last_step: int = int(sim_steps) - 1
        if last_step % output_interval != 0:
            sim.export_ply_from_diff(f"{series_prefix}", last_step, cnt_ply)
            cnt_ply += 1
    return cnt_ply


def _compat_save_fluid_velocity_field(file_path, velocity_field, weight=None, metadata=None):
    file_dir: str = os.path.dirname(file_path)
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)
    payload: dict = {"velocity_field": np.asarray(velocity_field, dtype=np.float32)}
    if weight is not None:
        payload["weight"] = np.asarray(weight, dtype=np.float32)
    if metadata:
        for key, value in metadata.items():
            if value is not None:
                payload[key] = np.asarray(value)
    np.savez_compressed(file_path, **payload)


_robots_utils_module = types.ModuleType("utils")
_robots_utils_module.export_backward_data = _compat_export_backward_data
_robots_utils_module.save_fluid_velocity_field = _compat_save_fluid_velocity_field
sys.modules.setdefault("robots.utils", _robots_utils_module)
sys.modules.setdefault("wanphys.examples.robots.utils", _robots_utils_module)

from wanphys._src.fluid.fluid_particle.SPH_diff import (
    DFSPHDiffDomain,
    DiffCompositeSimulation,
    MaterialMarks,
    MaterialType,
    SPHDiffDomain,
    SPHDiffModel,
)
from wanphys._src.fluid.fluid_particle.SPH_diff.sim_utils import export_ply_points
from wanphys._src.fluid.fluid_particle.SPH_diff.tasks import FishTask
from wanphys.utils import load_mesh_file, load_point_cloud
from wanphys.examples.fluid_dfsph_diff_skip_stone import _sample_rigid_body_surface
try:
    from robots.CartPoleEnv import DifferentiableCartPoleEnv
    from robots.FishEnv import DifferentiableFishEnv
    from robots.controller import Controller
    from robots.controller_with_fluid import ControllerWithFluid
    from robots.fluid_encoder.ae_CNN_3d_pytorch import AutoEncoder3D
    from robots.fluid_encoder.encoder_utils import get_local_fluid_grid_around_rigid, normalize_fluid_field
except ImportError:
    from .robots.CartPoleEnv import DifferentiableCartPoleEnv
    from .robots.FishEnv import DifferentiableFishEnv
    from .robots.controller import Controller
    from .robots.controller_with_fluid import ControllerWithFluid
    from .robots.fluid_encoder.ae_CNN_3d_pytorch import AutoEncoder3D
    from .robots.fluid_encoder.encoder_utils import get_local_fluid_grid_around_rigid, normalize_fluid_field

try:
    import tensorboardX
except ImportError:
    class _NoOpSummaryWriter:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

        def add_scalar(self, *args, **kwargs):
            _ = args, kwargs

        def close(self):
            return

    class _NoOpTensorboardX:
        SummaryWriter = _NoOpSummaryWriter

    tensorboardX = _NoOpTensorboardX()

export_backward_data = _compat_export_backward_data
save_fluid_velocity_field = _compat_save_fluid_velocity_field

GRID_SHAPE = (64, 32, 32)
LOCAL_FLUID_GRID_CELL_SIZE = 0.03

is_fish_task = False


DEFAULT_FISH_SCENE_PATH = (
    Path(__file__).resolve().parent.parent
    / "_src"
    / "fluid"
    / "fluid_particle"
    / "SPH_diff"
    / "tasks"
    / "fish_on_water.json"
)


def build_grad_mode_info(args):
    grad_mode_info: str = ""
    if args.difffr:
        grad_mode_info += "difffr/"
    grad_mode_info += "custom_grad/" if args.custom_grad else "g2/"
    if args.norm_grad:
        grad_mode_info += "_normed"
    return grad_mode_info


def build_end_time_tag(end_time):
    if end_time is None:
        return "et_unknown"
    return f"et{end_time}"


def build_scene_name_with_grad_mode(args, end_time=None):
    scene_name: str = f"{os.path.splitext(os.path.basename(args.scene_file))[0]}/h{args.h_scale}/"
    scene_name += f"{build_end_time_tag(end_time)}/"
    replay_mode_enabled: bool = bool(getattr(args, "replay_mode", False))
    mode_tag: str = "replay_mode" if replay_mode_enabled else "direct_mode"
    return f"{scene_name}controller/{mode_tag}/{build_grad_mode_info(args)}"


def build_export_settings(args, config, end_time=None):
    output_ply: bool = bool(config.get_cfg("exportPly"))
    output_obj: bool = bool(config.get_cfg("exportObj"))
    fps: float = float(config.get_cfg("fps", 60.0))
    frame_time: float = 1.0 / fps
    output_interval: int = 10
    if end_time is None:
        end_time = config.get_cfg("end_time")
    scene_name_with_grad_mode: str = build_scene_name_with_grad_mode(args, end_time=end_time)
    backward_interval = getattr(args, "backward_interval", None)
    if backward_interval is not None:
        scene_name_with_grad_mode = f"{scene_name_with_grad_mode.rstrip('/')}/interval_{int(backward_interval)}/"
    series_prefix: str = f"outputs/{scene_name_with_grad_mode}_diff_output/particle_object_{{:06d}}.ply"
    if config.get_cfg("exportFrame"):
        os.makedirs(f"outputs/{scene_name_with_grad_mode}_output_img", exist_ok=True)
    if output_ply:
        os.makedirs(f"outputs/{scene_name_with_grad_mode}_diff_output", exist_ok=True)
    return output_ply, output_obj, output_interval, series_prefix, frame_time, scene_name_with_grad_mode


def build_default_log_dir(args, end_time=None):
    scene_name: str = build_scene_name_with_grad_mode(args, end_time=end_time)
    if getattr(args, "batch", None) is not None:
        scene_name = f"{scene_name.rstrip('/')}/batch_{int(args.batch)}/"
    time_str: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"runs/{scene_name}/{time_str}_lr_{args.lr}"


def build_controller_run_tag(args):
    time_tag: str = "time_in_state" if bool(args.time_in_state) else "no_t"
    loss_threshold_tag: str = f"max_loss_{float(args.loss_threshold):g}"
    return f"{time_tag}_{loss_threshold_tag}"


def build_checkpoint_dir(args, end_time=None):
    grad_mode_info: str = build_grad_mode_info(args)
    end_time_tag: str = build_end_time_tag(end_time)
    time_str: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"checkpoints/{end_time_tag}/{grad_mode_info}/{time_str}_lr_{args.lr}"


class FishSceneConfig:
    """Small compatibility wrapper for JSON scene configs."""

    def __init__(self, scene_file_path):
        self.scene_path: Path = _resolve_scene_path(scene_file_path)
        with open(self.scene_path, "r", encoding="utf-8") as scene_file:
            self.scene_cfg: dict = json.load(scene_file)
        self.configuration: dict = self.scene_cfg.get("Configuration", {})

    def get_cfg(self, key, default=None):
        return self.configuration.get(key, default)

    def get_rigid_bodies(self):
        return self.scene_cfg.get("RigidBodies", [])

    def get_fluid_blocks(self):
        return self.scene_cfg.get("FluidBlocks", [])


def _resolve_scene_path(scene_file_path):
    default_scene_path: Path = DEFAULT_FISH_SCENE_PATH.expanduser().resolve()
    if scene_file_path is None:
        return default_scene_path

    scene_candidate: Path = Path(scene_file_path).expanduser()
    if scene_candidate.is_absolute():
        return scene_candidate.resolve()

    scene_from_cwd: Path = (Path.cwd() / scene_candidate).resolve()
    if scene_from_cwd.exists():
        return scene_from_cwd

    scene_from_tasks: Path = (default_scene_path.parent / scene_candidate.name).resolve()
    if scene_from_tasks.exists():
        return scene_from_tasks

    return scene_from_cwd


def _resolve_asset_path(scene_path, asset_path):
    asset_candidate: Path = Path(str(asset_path)).expanduser()
    if asset_candidate.is_absolute():
        return asset_candidate.resolve()

    candidates: list[Path] = [
        (scene_path.parent / asset_candidate).resolve(),
        (Path.cwd() / asset_candidate).resolve(),
        (Path(__file__).resolve().parent.parent / "assets" / asset_candidate.name).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _parse_vec3(value):
    if value is None:
        return 0.0, 0.0, 0.0
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError("Need 3 values for vec3")
        return float(value[0]), float(value[1]), float(value[2])
    if isinstance(value, str):
        values: list[float] = [float(x) for x in value.replace(",", " ").split()]
        if len(values) != 3:
            raise ValueError("Need 3 values for vec3")
        return float(values[0]), float(values[1]), float(values[2])
    raise ValueError(f"Expected vec3-like value, got {value!r}")


def _rotation_from_axis_angle(axis, angle_deg):
    axis_np: np.ndarray = np.asarray(axis, dtype=np.float32)
    norm: float = float(np.linalg.norm(axis_np))
    if norm < 1.0e-8 or abs(float(angle_deg)) < 1.0e-8:
        return wp.quat_identity()

    axis_np = axis_np / norm
    return wp.quat_from_axis_angle(
        wp.vec3(float(axis_np[0]), float(axis_np[1]), float(axis_np[2])),
        math.radians(float(angle_deg)),
    )


def _sample_fallback_ellipsoid_surface(translation, scale, particle_radius):
    translation_np: np.ndarray = np.asarray(translation, dtype=np.float32)
    radii_np: np.ndarray = np.maximum(np.asarray(scale, dtype=np.float32), float(particle_radius))
    diameter: float = max(float(particle_radius) * 2.0, 1.0e-6)
    counts_np: np.ndarray = np.maximum(np.ceil((2.0 * radii_np) / diameter).astype(np.int32), 4)
    points: list[np.ndarray] = []
    for ix in range(int(counts_np[0]) + 1):
        theta: float = math.pi * float(ix) / float(max(1, counts_np[0]))
        for iy in range(int(counts_np[1]) * 2):
            phi: float = 2.0 * math.pi * float(iy) / float(max(1, counts_np[1] * 2))
            unit_np: np.ndarray = np.array(
                [math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi)],
                dtype=np.float32,
            )
            points.append(translation_np + unit_np * radii_np)
    return np.asarray(points, dtype=np.float32)


def _emit_fluid_block(builder, config, block, spacing, particle_radius):
    scene_path: Path = config.scene_path
    density: float = float(block.get("density", config.get_cfg("density0", 1000.0)))
    m_v0: float = float(0.8 * ((float(particle_radius) * 2.0) ** 3))
    mass: float = density * m_v0
    vel: tuple[float, float, float] = _parse_vec3(block.get("velocity", config.get_cfg("initialVelocity", [0.0, 0.0, 0.0])))
    translation_np: np.ndarray = np.asarray(_parse_vec3(block.get("translation", [0.0, 0.0, 0.0])), dtype=np.float32)

    ply_file: str | None = block.get("plyFile")
    if ply_file:
        ply_path: Path = _resolve_asset_path(scene_path, ply_file)
        if ply_path.exists():
            points_np: np.ndarray = load_point_cloud(ply_path, scale=1.0).astype(np.float32) + translation_np
            print(f"[scene] Loaded {len(points_np)} fluid particles from {ply_path}")
            for point_np in points_np:
                builder.add_particle(
                    pos=wp.vec3(float(point_np[0]), float(point_np[1]), float(point_np[2])),
                    vel=wp.vec3(*vel),
                    mass=mass,
                    radius=float(particle_radius),
                )
            return int(len(points_np))
        print(f"[scene] Fluid PLY not found, falling back to block emission: {ply_path}")

    start_np: np.ndarray = np.asarray(_parse_vec3(block["start"]), dtype=np.float32) + translation_np
    end_np: np.ndarray = np.asarray(_parse_vec3(block["end"]), dtype=np.float32) + translation_np
    bbox_size_np: np.ndarray = np.maximum(end_np - start_np, float(spacing))
    counts_np: np.ndarray = np.maximum(np.floor(bbox_size_np / float(spacing)).astype(np.int32), 1)
    emitted: int = 0
    for ix in range(int(counts_np[0])):
        for iy in range(int(counts_np[1])):
            for iz in range(int(counts_np[2])):
                point_np: np.ndarray = start_np + float(spacing) * np.asarray([ix, iy, iz], dtype=np.float32)
                builder.add_particle(
                    pos=wp.vec3(float(point_np[0]), float(point_np[1]), float(point_np[2])),
                    vel=wp.vec3(*vel),
                    mass=mass,
                    radius=float(particle_radius),
                )
                emitted += 1
    print(f"[scene] Emitted {emitted} fluid block particles")
    return emitted


def _emit_rigid_body_boundary(builder, config, rigid_cfg, particle_radius):
    geom_file: str | None = rigid_cfg.get("geometryFile")
    translation: tuple[float, float, float] = _parse_vec3(rigid_cfg.get("translation", [0.0, 0.0, 0.0]))
    rotation_axis = rigid_cfg.get("rotationAxis", [0.0, 0.0, 1.0])
    rotation_angle: float = float(rigid_cfg.get("rotationAngle", 0.0))
    scale: tuple[float, float, float] = _parse_vec3(rigid_cfg.get("scale", [1.0, 1.0, 1.0]))
    rotation = _rotation_from_axis_angle(rotation_axis, rotation_angle)
    builder.add_body(xform=wp.transform(wp.vec3(*translation), rotation))

    geom_path: Path | None = _resolve_asset_path(config.scene_path, geom_file) if geom_file else None
    if geom_path is not None and geom_path.exists():
        # Validate that the mesh is loadable through the same utility used by skip_stone.
        load_mesh_file(geom_path, scale=1.0, compute_normals=True)
        points_np: np.ndarray = _sample_rigid_body_surface(
            geom_path,
            translation=translation,
            scale=scale,
            rotation_axis=rotation_axis,
            rotation_angle=rotation_angle,
            particle_radius=particle_radius,
        )
        print(f"[scene] Sampled {len(points_np)} rigid boundary particles from {geom_path}")
    else:
        print(f"[scene] Rigid geometry not found, using ellipsoid fallback: {geom_path}")
        points_np = _sample_fallback_ellipsoid_surface(translation, scale, particle_radius)

    vel: tuple[float, float, float] = _parse_vec3(rigid_cfg.get("velocity", [0.0, 0.0, 0.0]))
    for point_np in points_np:
        builder.add_particle(
            pos=wp.vec3(float(point_np[0]), float(point_np[1]), float(point_np[2])),
            vel=wp.vec3(*vel),
            mass=0.0,
            radius=float(particle_radius),
        )
    return int(len(points_np))


class ControllerTrainingViewer:
    """Interactive Newton GL viewer for controller training (skip_stone-style)."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.enabled = self._is_interactive_viewer(args)
        self.should_stop = False
        self.sim_time = 0.0
        self.radii = None
        self.colors = None

    @staticmethod
    def _is_interactive_viewer(args):
        viewer_type = str(getattr(args, "viewer", "gl")).lower()
        if viewer_type != "gl":
            return False
        return not bool(getattr(args, "headless", False))

    def setup(self, sim):
        if not self.enabled:
            return

        newton_model = getattr(sim, "newton_model", None)
        if newton_model is None:
            self.enabled = False
            return

        spacing = float(getattr(sim, "particle_spacing", getattr(sim, "particle_radius", 0.026) * 2.0))
        self.viewer.set_model(newton_model)
        self.viewer.show_particles = False
        self.viewer.show_collision = False
        self.viewer.show_contacts = False
        self.viewer.show_visual = True

        if hasattr(self.viewer, "camera"):
            self.viewer.camera_control_enable = False
            self.viewer.camera.pos = wp.vec3(-0.50, -7.80, 3.00)
            self.viewer.camera.pitch = -15
            self.viewer.camera.yaw = 90
            self.viewer.camera.fov = 30

        state = sim.composite.get_state("sphdiff")
        n = int(newton_model.particle_count)
        self.radii = wp.full(n, 0.5 * spacing, dtype=wp.float32, device=newton_model.device)
        self.colors = self._build_particle_colors(state, n, newton_model.device)
        print("[viewer] Interactive GL viewer enabled.")

    @staticmethod
    def _build_particle_colors(state, particle_count, device):
        colors_np = np.tile(np.array([0.10, 0.40, 0.90], dtype=np.float32), (particle_count, 1))

        boundary_indices = getattr(state, "boundary_indices", None)
        if boundary_indices is not None and hasattr(boundary_indices, "numpy"):
            rigid_idx = boundary_indices.numpy().astype(np.int32)
            if rigid_idx.size > 0:
                colors_np[rigid_idx] = np.array([0.95, 0.45, 0.20], dtype=np.float32)
                return wp.array(colors_np, dtype=wp.vec3, device=device)

        object_id = getattr(state, "object_id", None)
        if object_id is not None and hasattr(object_id, "numpy"):
            obj_np = object_id.numpy().astype(np.int32)
            rigid_mask = obj_np > 0
            colors_np[rigid_mask] = np.array([0.95, 0.45, 0.20], dtype=np.float32)
            return wp.array(colors_np, dtype=wp.vec3, device=device)

        if getattr(state, "material_marks", None) is not None:
            material = state.material_marks.material
            if material is not None:
                material_np = material.numpy().astype(np.int32)
                rigid_mask = material_np == int(MaterialType.SOLID)
                colors_np[rigid_mask] = np.array([0.95, 0.45, 0.20], dtype=np.float32)

        return wp.array(colors_np, dtype=wp.vec3, device=device)

    def reset_time(self):
        self.sim_time = 0.0

    def is_running(self):
        if not self.enabled:
            return True
        if hasattr(self.viewer, "is_running"):
            return bool(self.viewer.is_running())
        return True

    def wait_if_paused(self):
        if not self.enabled:
            return
        while bool(getattr(self.viewer, "_paused", False)):
            if not self.is_running():
                self.should_stop = True
                return
            self.render_frame(None)
            time.sleep(0.01)

    def render_frame(self, sim):
        if not self.enabled:
            return
        if sim is None:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.end_frame()
            return

        state = sim.composite.get_state("sphdiff")
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(state.newton_state)
        self.viewer.log_points(
            "sphdiff_water",
            points=state.particle_q,
            radii=self.radii,
            colors=self.colors,
            hidden=False,
        )
        self.viewer.end_frame()

    def on_sim_step(self, sim, frame_dt):
        if not self.enabled:
            return
        self.wait_if_paused()
        if self.should_stop or not self.is_running():
            self.should_stop = True
            return
        self.sim_time += float(frame_dt)
        self.render_frame(sim)


def _safe_close_viewer(viewer):
    if viewer is None or not hasattr(viewer, "close"):
        return
    try:
        viewer.close()
    except Exception as exc:
        print(f"[viewer] close failed: {exc}")


class SPHDiffControllerSimAdapter:
    """Compatibility layer used by the existing controller training envs."""

    def __init__(self, composite, domain, config, sim_steps, frame_dt, train_rate):
        self.composite: DiffCompositeSimulation = composite
        self.sphdiff_domain: SPHDiffDomain = domain
        self.cfg: FishSceneConfig = config
        self.sim_steps: int = int(sim_steps)
        self.frame_dt: float = float(frame_dt)
        self.train_rate: float = float(train_rate)
        self.loss: wp.array = wp.zeros((1,), dtype=float, requires_grad=True, device=domain.model._device)
        self.loss_tape = None
        self.tapes: list[wp.Tape] = []
        self.task = None
        self.opt_var: list[wp.array] = []
        self.fluid_particle_num: int = 0
        self.num_objects: int = 0
        self.num_rigid_bodies: int = 0
        self.materialMarks: MaterialMarks | None = None
        self.x_arrays: list[wp.array] = []
        self.v_arrays: list[wp.array] = []
        self.rigid_x_arrays: list[wp.array] = []
        self.rigid_quaternion_arrays: list[wp.array] = []
        self.rigid_v_arrays: list[wp.array] = []
        self.rigid_omega_arrays: list[wp.array] = []
        self.rigid_v_for_joint_arrays: list[wp.array] = []
        self.rigid_omega_for_joint_arrays: list[wp.array] = []
        self.rigid_delta_v_arrays: list[wp.array] = []
        self.rigid_delta_omega_arrays: list[wp.array] = []
        self.rigid_force_arrays: list[wp.array] = []
        self.rigid_torque_arrays: list[wp.array] = []
        self.rbs = None
        self.newton_model = None
        self.particle_radius = 0.026
        self.particle_spacing = 0.052
        self.viewer_session = None
        self._sync_initial_state()

    def _current_state(self):
        return self.composite.get_state("sphdiff")

    def _reset_history(self):
        self.x_arrays = []
        self.v_arrays = []
        self.rho_arrays = []
        self.pressure_arrays = []
        self.viscous_forces_arrays = []
        self.a_arrays = []
        self.debug_val_arrays = []
        self.density_change_arrays = []
        self.density_adv_arrays = []
        self.dfsph_factor_arrays = []
        self.rigid_x_arrays = []
        self.rigid_quaternion_arrays = []
        self.rigid_v_arrays = []
        self.rigid_omega_arrays = []
        self.rigid_v_for_joint_arrays = []
        self.rigid_omega_for_joint_arrays = []
        self.rigid_delta_v_arrays = []
        self.rigid_delta_omega_arrays = []
        self.rigid_force_arrays = []
        self.rigid_torque_arrays = []

    def _sync_initial_state(self):
        state = self._current_state()
        if hasattr(state, "initialize_rigid_bodies"):
            state.initialize_rigid_bodies()
        self._reset_history()
        self._append_state_arrays(state)

    def _append_state_arrays(self, state):
        self.rbs = state.rbs
        self.materialMarks = getattr(state, "material_marks", None)
        self.num_objects = int(getattr(state, "num_objects", 0))
        self.num_rigid_bodies = self.num_objects
        self.particle_max_num = int(getattr(state, "particle_max_num", state.particle_q.shape[0]))
        self.m_V = getattr(state, "m_V", None)
        self.object_id = getattr(state, "object_id", None)
        if self.materialMarks is not None and self.materialMarks.material is not None:
            material_np: np.ndarray = self.materialMarks.material.numpy()
            self.fluid_particle_num = int(np.count_nonzero(material_np == int(MaterialType.FLUID)))
        else:
            mass_np: np.ndarray = self.sphdiff_domain.model_adapter.particle_mass.numpy()
            self.fluid_particle_num = int(np.count_nonzero(np.abs(mass_np) != 0.0))

        self.x_arrays.append(state.particle_q)
        self.v_arrays.append(state.particle_qd)
        if hasattr(state, "rho") and state.rho is not None:
            self.rho_arrays.append(state.rho)
        if hasattr(state, "pressure") and state.pressure is not None:
            self.pressure_arrays.append(state.pressure)
        if hasattr(state, "viscous_forces") and state.viscous_forces is not None:
            self.viscous_forces_arrays.append(state.viscous_forces)
        if hasattr(state, "a") and state.a is not None:
            self.a_arrays.append(state.a)
        if hasattr(state, "debug_val") and state.debug_val is not None:
            self.debug_val_arrays.append(state.debug_val)
        if hasattr(state, "density_change") and state.density_change is not None:
            self.density_change_arrays.append(state.density_change)
        if hasattr(state, "density_adv") and state.density_adv is not None:
            self.density_adv_arrays.append(state.density_adv)
        if hasattr(state, "dfsph_factor") and state.dfsph_factor is not None:
            self.dfsph_factor_arrays.append(state.dfsph_factor)
        self.rigid_x_arrays.append(state.rbs.rigid_x)
        self.rigid_quaternion_arrays.append(state.rbs.rigid_quaternion)
        self.rigid_v_arrays.append(state.rbs.rigid_v)
        self.rigid_omega_arrays.append(state.rbs.rigid_omega)
        self.rigid_v_for_joint_arrays.append(state.rbs.rigid_v)
        self.rigid_omega_for_joint_arrays.append(state.rbs.rigid_omega)
        self.rigid_delta_v_arrays.append(state.rbs.rigid_v)
        self.rigid_delta_omega_arrays.append(state.rbs.rigid_omega)
        self.rigid_force_arrays.append(state.rbs.rigid_force)
        self.rigid_torque_arrays.append(state.rbs.rigid_torque)

    def clear_grad(self):
        self.loss = wp.zeros((1,), dtype=float, requires_grad=True, device=self.sphdiff_domain.model._device)
        for state in (getattr(self.sphdiff_domain, "_state_in", None), getattr(self.sphdiff_domain, "_state_out", None)):
            if state is not None and hasattr(state, "clear_grad"):
                state.clear_grad()
        if self.task is not None and hasattr(self.task, "clear_grad"):
            self.task.clear_grad()

    def reset(self):
        session = getattr(self, "viewer_session", None)
        if session is not None:
            session.reset_time()
        self.loss = wp.zeros((1,), dtype=float, requires_grad=True, device=self.sphdiff_domain.model._device)
        self.loss_tape = None
        self.tapes = []
        self.sphdiff_domain.solver.tapes = []
        self.composite.state_history.clear()
        self.sphdiff_domain.create_state()
        if hasattr(self.sphdiff_domain._state_in, "initialize_rigid_bodies"):
            self.sphdiff_domain._state_in.initialize_rigid_bodies()
        if hasattr(self.sphdiff_domain._state_out, "initialize_rigid_bodies"):
            self.sphdiff_domain._state_out.initialize_rigid_bodies()
        self.composite._states_in["sphdiff"] = self.sphdiff_domain._state_in
        self.composite._states_out["sphdiff"] = self.sphdiff_domain._state_out
        self.composite._time = 0.0
        if self.task is not None:
            self.task.state = self.sphdiff_domain._state_in
            if hasattr(self.task, "init_targets"):
                self.task.init_targets()
        self._sync_initial_state()

    def diff_reset(self, backward_steps):
        """Delegate diff_reset to composite simulation."""
        self.composite.diff_reset(backward_steps)
        # Reset local history arrays
        self._reset_history()
        state = self._current_state()
        self._append_state_arrays(state)

    def step(self, local_t):
        _ = local_t
        session = getattr(self, "viewer_session", None)
        if session is not None and session.enabled and session.should_stop:
            return
        self.composite.step(self.frame_dt)
        state = self._current_state()
        self._append_state_arrays(state)
        if session is not None:
            session.on_sim_step(self, self.frame_dt)

    def norm_final_grad(self):
        if self.task is not None and hasattr(self.task, "norm_final_grad"):
            try:
                self.task.norm_final_grad(None, self.materialMarks)
            except TypeError:
                pass

    def backward(self):
        self.reset()
        target_steps: int = int(self.sim_steps)
        for _step in range(target_steps):
            self.step(_step)

    def export_ply_from_diff(self, series_prefix, time_step, cnt_ply, verbose=False):
        return
        def grad_or_zeros(wp_array, reference_np, dtype=np.float32):
            if wp_array is not None and wp_array.grad is not None:
                return wp_array.grad.numpy()
            return np.zeros_like(reference_np, dtype=dtype)

        if time_step < 0 or time_step >= len(self.x_arrays):
            return

        np_pos = self.x_arrays[time_step].numpy()
        np_v = self.v_arrays[time_step].numpy()
        np_vel = np.linalg.norm(np_v, axis=1)

        if time_step < len(self.rho_arrays):
            np_rho = self.rho_arrays[time_step].numpy()
        else:
            np_rho = np.zeros(self.particle_max_num, dtype=np.float32)

        if self.m_V is not None:
            np_mV = self.m_V.numpy()
        else:
            np_mV = np.full(self.particle_max_num, getattr(self, "m_V0", 1.0e-5), dtype=np.float32)

        if self.object_id is not None:
            np_obj_id = self.object_id.numpy()
        else:
            np_obj_id = np.zeros(self.particle_max_num, dtype=np.int32)

        if time_step < len(self.pressure_arrays):
            pf = self.pressure_arrays[time_step].numpy()
        else:
            pf = np.zeros(self.particle_max_num, dtype=np.float32)

        if time_step < len(self.viscous_forces_arrays):
            vf = self.viscous_forces_arrays[time_step].numpy()
        else:
            vf = np.zeros((self.particle_max_num, 3), dtype=np.float32)

        if time_step < len(self.a_arrays):
            np_a = self.a_arrays[time_step].numpy()
        else:
            np_a = np.zeros((self.particle_max_num, 3), dtype=np.float32)

        if time_step < len(self.debug_val_arrays):
            debug_val = self.debug_val_arrays[time_step].numpy()
        else:
            debug_val = np.zeros(self.particle_max_num, dtype=np.float32)

        density_change = np.zeros(self.particle_max_num, dtype=np.float32)
        if hasattr(self, "density_change_arrays") and time_step < len(self.density_change_arrays):
            density_change = self.density_change_arrays[time_step].numpy()

        density_adv = np.zeros(self.particle_max_num, dtype=np.float32)
        if hasattr(self, "density_adv_arrays") and time_step < len(self.density_adv_arrays):
            density_adv = self.density_adv_arrays[time_step].numpy()

        dfsph_factor = np.zeros(self.particle_max_num, dtype=np.float32)
        if hasattr(self, "dfsph_factor_arrays") and time_step < len(self.dfsph_factor_arrays):
            dfsph_factor = self.dfsph_factor_arrays[time_step].numpy()

        grad_x = grad_or_zeros(self.x_arrays[time_step], np_pos)
        grad_v = grad_or_zeros(self.v_arrays[time_step], np_v)
        grad_rho = grad_or_zeros(self.rho_arrays[time_step] if time_step < len(self.rho_arrays) else None, np_rho)
        grad_p = grad_or_zeros(self.pressure_arrays[time_step] if time_step < len(self.pressure_arrays) else None, pf)
        grad_a = grad_or_zeros(self.a_arrays[time_step] if time_step < len(self.a_arrays) else None, np_a)

        material_np = np.zeros(self.particle_max_num, dtype=np.int32)
        is_dynamic_np = np.ones(self.particle_max_num, dtype=np.int32)
        if self.materialMarks is not None:
            if self.materialMarks.material is not None:
                material_np = self.materialMarks.material.numpy().astype(np.int32)
            if self.materialMarks.is_dynamic is not None:
                is_dynamic_np = self.materialMarks.is_dynamic.numpy().astype(np.int32)

        out_path = series_prefix.format(cnt_ply)
        export_ply_points(out_path, np_pos.astype(np.float32), {
            "rho": np_rho.astype(np.float32),
            "pressure": pf.astype(np.float32),
            "mV": np_mV.astype(np.float32),
            "object_id": np_obj_id.astype(np.int32),
            "debug_val": debug_val.astype(np.float32),
            "viscous_fx": vf[:, 0].astype(np.float32),
            "viscous_fy": vf[:, 1].astype(np.float32),
            "viscous_fz": vf[:, 2].astype(np.float32),
            "ax": np_a[:, 0].astype(np.float32),
            "ay": np_a[:, 1].astype(np.float32),
            "az": np_a[:, 2].astype(np.float32),
            "vx": np_v[:, 0].astype(np.float32),
            "vy": np_v[:, 1].astype(np.float32),
            "vz": np_v[:, 2].astype(np.float32),
            "vel": np_vel.astype(np.float32),
            "material": material_np,
            "is_dynamic": is_dynamic_np,
            "density_change": density_change.astype(np.float32),
            "density_adv": density_adv.astype(np.float32),
            "dfsph_factor": dfsph_factor.astype(np.float32),
            "grad_x_x": grad_x[:, 0].astype(np.float32),
            "grad_x_y": grad_x[:, 1].astype(np.float32),
            "grad_x_z": grad_x[:, 2].astype(np.float32),
            "grad_v_x": grad_v[:, 0].astype(np.float32),
            "grad_v_y": grad_v[:, 1].astype(np.float32),
            "grad_v_z": grad_v[:, 2].astype(np.float32),
            "grad_rho": grad_rho.astype(np.float32),
            "grad_pressure": grad_p.astype(np.float32),
            "grad_a_x": grad_a[:, 0].astype(np.float32),
            "grad_a_y": grad_a[:, 1].astype(np.float32),
            "grad_a_z": grad_a[:, 2].astype(np.float32),
        })
        if verbose:
            print(f"[ply] Exported {out_path} with {len(np_pos)} particles")


class ControllerFishTask(FishTask):
    """FishTask compatibility shim for controller envs that expect task.sim."""

    def __init__(self, sim):
        self.sim = sim
        super().__init__(sim)

    def init_simulation_state(self, tape=None):
        _ = tape
        return super().init_simulation_state()


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
    if getattr(sim, "task", None) is None and hasattr(sim, "cfg") and sim.cfg is not None:
        rigid_bodies = sim.cfg.get_rigid_bodies() if hasattr(sim.cfg, "get_rigid_bodies") else []
        for rb in rigid_bodies:
            comment = str(rb.get("comment", "")).lower()
            if "fish" in comment:
                sim.task = FishTask(sim)
                break

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

    # action gradient 维度自适应: CartPole=3 (force xyz), Fish=1 (angular velocity y)
    _action_grad_labels = {0: "x", 1: "y", 2: "z", 3: "w"}
    for idx, record in enumerate(getattr(env, "action_records", [])):
        grad_vec = record.get("grad", None)
        if grad_vec is None:
            continue
        step = int(record.get("step", idx))
        for dim in range(len(grad_vec)):
            label = _action_grad_labels.get(dim, f"dim{dim}")
            writer.add_scalar(f"Action_Step_Grad/{label}", float(grad_vec[dim]), step)
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
            if env._viewer_should_stop():
                print("[train_controller] viewer closed, stop training.")
                break
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
        if env._viewer_should_stop():
            print("[train_controller_online] viewer closed, stop training.")
            break
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
    config = FishSceneConfig(scene_file_path=args.scene_file)
    if not config.scene_path.exists():
        raise FileNotFoundError(f"Scene json not found: {config.scene_path}")
    print(f"[scene] Loading fish_on_water scene: {config.scene_path}")

    sim_steps: int | None = args.num_timesteps
    if sim_steps is None:
        sim_steps = derive_sim_steps(config, sim_time=args.t)
    sim_steps = int(sim_steps)

    print(f"Using sim_steps: {sim_steps}")
    segment_steps: int = int(args.backward_interval) * int(args.control_interval) + 1
    print(f"segment_steps: {segment_steps}")

    cfg: dict = config.configuration
    particle_radius: float = float(cfg.get("particleRadius", cfg.get("radius", 0.026)))
    spacing: float = particle_radius * 2.0
    sim_dt: float = float(cfg.get("timeStepSize", 0.001))
    rest_density: float = float(cfg.get("density0", 1000.0))
    viscosity: float = float(config.get_fluid_blocks()[0].get("viscosity", cfg.get("viscosity", 0.1))) if config.get_fluid_blocks() else float(cfg.get("viscosity", 0.1))

    def _create_adapter():
        builder: newton.ModelBuilder = newton.ModelBuilder()
        fluid_particle_num: int = 0
        for block in config.get_fluid_blocks():
            fluid_particle_num += _emit_fluid_block(
                builder=builder,
                config=config,
                block=block,
                spacing=spacing,
                particle_radius=particle_radius,
            )

        rigid_particle_num: int = 0
        for rigid_cfg in config.get_rigid_bodies():
            rigid_particle_num += _emit_rigid_body_boundary(
                builder=builder,
                config=config,
                rigid_cfg=rigid_cfg,
                particle_radius=particle_radius,
            )

        newton_model = builder.finalize()
        bounds_min: tuple[float, float, float] = _parse_vec3(cfg.get("domainStart", [-2.1, 0.0, -0.83]))
        bounds_max: tuple[float, float, float] = _parse_vec3(cfg.get("domainEnd", [2.1, 4.0, 0.83]))
        gravity: tuple[float, float, float] = _parse_vec3(cfg.get("gravitation", [0.0, -9.81, 0.0]))
        sphdiff_model: SPHDiffModel = SPHDiffModel(
            h=float(particle_radius) * 4.0 * float(args.h_scale),
            rest_density=rest_density,
            viscosity=viscosity,
            c0=float(cfg.get("stiffness", 7000.0)),
            gamma=float(cfg.get("exponent", 7.0)),
            gravity=gravity,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            bounds_restitution=0.3,
            bounds_damping=0.02,
            max_dt=sim_dt,
            fixed_dt=sim_dt,
            device=newton_model.device,
        )
        sphdiff_model.cfg = config.scene_cfg

        domain_cls = DFSPHDiffDomain if int(args.method) == 1 else SPHDiffDomain
        print(f"[scene] Using SPH_diff domain: {domain_cls.__name__}")
        if int(args.method) == 1:
            print("[scene] --dfsph_impl is ignored; using SPH_diff.DFSPHDiffDomain")

        sphdiff_domain = domain_cls(
            newton_model,
            sphdiff_model,
            target_time=None,
            target_timesteps=None,
            dt=sim_dt,
        )
        composite: DiffCompositeSimulation = DiffCompositeSimulation()
        composite.add_domain(sphdiff_domain)
        adapter: SPHDiffControllerSimAdapter = SPHDiffControllerSimAdapter(
            composite=composite,
            domain=sphdiff_domain,
            config=config,
            sim_steps=segment_steps,
            frame_dt=sim_dt,
            train_rate=args.lr,
        )
        adapter.fluid_particle_num = fluid_particle_num
        adapter.newton_model = newton_model
        adapter.particle_radius = particle_radius
        adapter.particle_spacing = spacing
        task: ControllerFishTask = ControllerFishTask(adapter)
        adapter.task = task
        sphdiff_domain._task = task
        sphdiff_domain.solver._runtime_task = task
        sphdiff_domain.solver._runtime_task_initialized = False
        print(
            f"[scene] Created fish SPH_diff scene: fluid_particles={fluid_particle_num}, "
            f"rigid_boundary_particles={rigid_particle_num}, total_particles={newton_model.particle_count}"
        )
        return adapter

    if args.device is None:
        sim = _create_adapter()
    else:
        with wp.ScopedDevice(args.device):
            sim = _create_adapter()

    return sim, config, sim_steps


def get_parser():
    parser = newton.examples.create_parser()

    parser.add_argument("--scene_file", type=str, default=str(DEFAULT_FISH_SCENE_PATH), help="Scene file path.")
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
    parser.add_argument(
        "--test-controller",
        dest="test_controller",
        action="store_true",
        help="Run non-differentiable controller test only.",
    )
    parser.add_argument("--loss_threshold", type=float, default=100, help="Max loss threshold; training stops early if exceeded.")
    parser.add_argument("--stage_path", type=str, default="example_sph.usd", help="USD stage output path.")
    parser.add_argument("--ply_path", type=str, default=None, help="Optional PLY initialization path.")
    parser.add_argument("--h_scale", type=float, default=1.0, help="Smoothing-length scale.")
    parser.add_argument("--custom_grad", action="store_true", help="Enable custom gradient mode.")
    parser.add_argument("--norm_grad", action="store_true", help="Enable state-gradient normalization in simulator.")
    parser.add_argument("--difffr", action="store_true", help="Enable diffFR mode.")
    parser.add_argument("--verbose", action="store_true", help="Verbose simulator logs.")
    parser.add_argument(
        "--paused",
        action="store_true",
        help="Start viewer paused (press Space to resume).",
    )

    return parser


def main():
    wp.init()
    viewer, args = newton.examples.init(get_parser())
    if args.paused and hasattr(viewer, "_paused"):
        viewer._paused = True

    args.command_start_time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.train_ar:
        args.use_fluid = True
    if args.test_ar:
        args.use_fluid = True
        args.test_controller = True

    sim_time = args.t
    completed = False
    try:
        sim, config, sim_steps = build_simulator(args)
        viewer_session = ControllerTrainingViewer(viewer, args)
        viewer_session.setup(sim)
        sim.viewer_session = viewer_session
    except Exception:
        _safe_close_viewer(viewer)
        raise

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

    try:
        if args.test_controller:
            test_controller(sim, args, sim_steps, export_settings=export_settings)
        elif args.use_sim_backward:
            train_controller_with_sim_backward(sim, args, sim_steps, export_settings=export_settings, ckpt_dir=ckpt_dir)
        elif args.batch is not None:
            train_controller(sim, args, sim_steps, export_settings=export_settings, ckpt_dir=ckpt_dir)
        else:
            train_controller_online(sim, args, sim_steps, export_settings=export_settings, ckpt_dir=ckpt_dir)
        completed = True
    finally:
        if not completed:
            _safe_close_viewer(viewer)

    if export_settings is not None:
        output_ply, _, _, series_prefix, _, _ = export_settings
        if output_ply:
            print(f"PLY export prefix: {series_prefix}")
            print(f"PLY export directory: {os.path.dirname(series_prefix)}")
        else:
            print("PLY export disabled by scene config.")


if __name__ == "__main__":
    main()
