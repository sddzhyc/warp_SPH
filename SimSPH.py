from IISPH_kernel import compute_aii_and_density_deviation, compute_pressure_a as compute_pressure_a_iisph, predict_velocity, update_pressure_and_compute_avg_error
from kernel_utils import add_ball_kernel, add_particles_kernel

from rigid_fluid_coupling import MaterialMarks, MaterialType, RigidBodies, compute_moving_boundary_volume, compute_static_boundary_volume, solve_rigid_body, update_rigid_particle_info, compute_rigid_cm_mass_kernel, finalize_rigid_cm_kernel, compute_rigid_inertia_kernel, finalize_rigid_inertia_kernel
from sim_utils import export_ply_points, load_ply_points
from sph_kernel import compute_non_presure_forces, compute_pressure, compute_pressure_a
from sph_kernel_diff import compute_density, enforce_boundary_3D_warp, kick as kick_diff, drift as drift_diff

import os
import numpy as np
import warp as wp
import trimesh as tm
import json
from functools import reduce
class SimSPH:
    def initialize(self):
        print(f"Initialized particle volumes m_V0 = {self.m_V0}")
        # ps.initialize_particle_system()
        # TODO: 实现流固耦合：移植compute_rigid_rest_cm、compute_rigid_mass_info
        wp.launch(
            kernel=compute_static_boundary_volume,
            dim=self.particle_max_num,
            inputs=[self.grid.id, self.x, self.m_V, self.density_normalization_no_mass, self.smoothing_length,
                    self.materialMarks],
        )
        wp.launch(
            kernel=compute_moving_boundary_volume,
            dim=self.particle_max_num,
            inputs=[self.grid.id, self.x, self.m_V, self.density_normalization_no_mass, self.smoothing_length,
                    self.materialMarks],
        )
        # 打印m_V
        # m_V_np = self.m_V.numpy()
        # print(f"Computed boundary volumes, sample m_V: {m_V_np[:10]}")


    def __init__(self, config = None, container = None, method=0, stage_path="example_sph.usd", ply_path=None, h_scale=1.0):
        """
        If `container` (a `BaseContainer`) is provided, SimSPH will use the container's
        particle arrays as the source of truth. Otherwise it falls back to the original
        random-initialized behavior.
        """
        self.ps = container
        self.h_scale = float(h_scale)

        self.verbose = False
        # render params
        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_time = 0.0
        # get simulation params from config
        if (config != None):
            self.cfg = config
            self.dim = 3
            self.frame_dt = config.get_cfg("timeStepSize") # 采用config中的时间步长
            self.sim_step_to_frame_ratio = config.get_cfg("numberOfStepsPerRenderUpdate")

            d_start = config.get_cfg("domainStart")
            if d_start is None:
                self.domain_start = np.array([0.0, 0.0, 0.0])
            else:
                self.domain_start = np.array(d_start)

            d_end = config.get_cfg("domainEnd")
            if d_end is None:
                self.domian_end = np.array([1.0, 1.0, 1.0])
                self.domain_end = self.domian_end
            else:
                self.domian_end = np.array(d_end)
                self.domain_end = np.array(d_end)
                
            ds = (self.domian_end - self.domain_start).astype(np.float32)
            self.domain_size = wp.vec3(ds[0], ds[1], ds[2])

            self.particle_radius = config.get_cfg("particleRadius")
            # self.smoothing_length = self.particle_radius     # 0.8
            self.particle_diameter = 2 * self.particle_radius
            self.smoothing_length = self.particle_radius * 4.0 * self.h_scale
            # self.smoothing_length = 1.8 * self.particle_radius * 2.0 # 0.8 # 一般为排列距离的1.3到1.5倍 #taichi版本：self.support_radius = self.particle_radius * 4.0  # support radius

            self.stiffness = config.get_cfg("stiffness") # 20
            self.exponent = config.get_cfg("exponent")
            self.base_density = config.get_cfg("density0")   # 1.0
            # self.base_density = 0.015667
            # self.m_V0 = self.ps.m_V0 #  0.8 * self.particle_diameter ** self.dim
            self.m_V0 = 0.8 * self.particle_diameter ** self.dim # 修改为设定体积而非质量
            # self.particle_mass = 0.01 * self.smoothing_length**3  # 为什么原example采用0.01?
            self.particle_mass = self.m_V0 * self.base_density # 设置后粒子不稳定？ # TODO:改为每个粒子分别存储
            self.dt = config.get_cfg("timeStepSize")    # 0.01 * self.smoothing_length
            self.dynamic_visc = config.get_cfg("viscosity")
            if self.dynamic_visc is None:
                self.dynamic_visc = 0.1 # 0.025
            self.surface_tension = config.get_cfg("surfaceTension")
            if self.surface_tension is None:
                self.surface_tension = 0.0
            self.damping_coef = -0.95
            self.gravity = config.get_cfg("gravitation")[1]  # -0.1
            # 打印 m_V0、 base_density、particle_mass、smoothing_length
            print("----------------------------------------------------------------")
            print(
                f"m_V0 = {self.m_V0}, base_density = {self.base_density}, "
                f"particle_mass = {self.particle_mass}, smoothing_length = {self.smoothing_length}"
                f"stiffness = {self.stiffness}, exponent = {self.exponent}"
            )
            print("----------------------------------------------------------------")

            # Grid related properties
            self.grid_size = 10.0 * self.smoothing_length
            self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
            print("grid size: ", self.grid_num)
            self.padding = self.smoothing_length

            self.grid = wp.HashGrid(self.grid_num[0], self.grid_num[1], self.grid_num[2])
            # All objects id and its particle num
            self.object_collection = dict()
            self.object_id_rigid_body = set()

        self.time_step = 0.0
        # recompute constants
        self.density_normalization = (315.0 * self.particle_mass) / (
            64.0 * np.pi * self.smoothing_length**9
        )
        self.density_normalization_no_mass = 315.0 / (
            64.0 * np.pi * self.smoothing_length**9
        )
        self.pressure_normalization = -(45.0 * self.particle_mass) / (np.pi * self.smoothing_length**6)
        self.pressure_normalization_no_mass = -45.0 / (np.pi * self.smoothing_length**6)
        self.viscous_normalization = (45.0 * self.dynamic_visc * self.particle_mass) / (
            np.pi * self.smoothing_length**6
        )
        self.viscous_normalization_no_mass = (45.0 * self.dynamic_visc) / (
            np.pi * self.smoothing_length**6
        )
        self.sim_step_to_frame_ratio = 1
        # self.sim_step_to_frame_ratio = int(32 / self.smoothing_length)

        if ply_path:
            if os.path.isdir(ply_path) and os.path.exists(os.path.join(ply_path, "params.json")):
                 self.init_from_houdini_geo(ply_path)
            else:
                 self.init_from_ply(ply_path)
        elif self.ps is None:
            self.init_from_generated_geo(config)
        else:
            self.ti_to_warp()
        # 调试导出时使用，注意在ti_to_warp初始化n之后定义
        self.neibor_nums = wp.zeros(self.particle_max_num, dtype=wp.int32)
        self.pressure_forces = wp.zeros(self.particle_max_num, dtype=wp.vec3)
        self.viscous_forces = wp.zeros(self.particle_max_num, dtype=wp.vec3)
    
        self.USE_METHOD = method
        if self.USE_METHOD == 1:
            self.init_IISPH()
        self.initialize()
        
        # Save initial state for reset
        self.save_initial_state()

    def save_initial_state(self):
        self.x_initial = wp.zeros_like(self.x)
        self.v_initial = wp.zeros_like(self.v)
        self.rho_initial = wp.zeros_like(self.rho)
        self.object_id_initial = wp.zeros_like(self.object_id)
        self.material_initial = wp.zeros_like(self.materialMarks.material)
        self.is_dynamic_initial = wp.zeros_like(self.materialMarks.is_dynamic)

        wp.copy(self.x_initial, self.x)
        wp.copy(self.v_initial, self.v)
        wp.copy(self.rho_initial, self.rho)
        wp.copy(self.object_id_initial, self.object_id)
        wp.copy(self.material_initial, self.materialMarks.material)
        wp.copy(self.is_dynamic_initial, self.materialMarks.is_dynamic)

        if self.num_objects > 0:
            self.rigid_x_initial = wp.zeros_like(self.rbs.rigid_x)
            self.rigid_v_initial = wp.zeros_like(self.rbs.rigid_v)
            self.rigid_omega_initial = wp.zeros_like(self.rbs.rigid_omega)
            self.rigid_quaternion_initial = wp.zeros_like(self.rbs.rigid_quaternion)
            
            wp.copy(self.rigid_x_initial, self.rbs.rigid_x)
            wp.copy(self.rigid_v_initial, self.rbs.rigid_v)
            wp.copy(self.rigid_omega_initial, self.rbs.rigid_omega)
            wp.copy(self.rigid_quaternion_initial, self.rbs.rigid_quaternion)

        self.num_particles_curr_initial = self.num_particles_curr

    def reset(self):
        wp.copy(self.x, self.x_initial)
        wp.copy(self.v, self.v_initial)
        wp.copy(self.rho, self.rho_initial)
        wp.copy(self.object_id, self.object_id_initial)
        wp.copy(self.materialMarks.material, self.material_initial)
        wp.copy(self.materialMarks.is_dynamic, self.is_dynamic_initial)

        self.num_particles_curr = self.num_particles_curr_initial
        
        if self.num_objects > 0:
            wp.copy(self.rbs.rigid_x, self.rigid_x_initial)
            wp.copy(self.rbs.rigid_v, self.rigid_v_initial)
            wp.copy(self.rbs.rigid_omega, self.rigid_omega_initial)
            wp.copy(self.rbs.rigid_quaternion, self.rigid_quaternion_initial)
    def compute_cube_particle_num(self, start, end):
        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(start[i], end[i], self.particle_diameter))
        return reduce(lambda x, y: x * y, [len(n) for n in num_dim])

    def load_rigid_body(self, rigid_body):
        obj_id = rigid_body["objectId"]
        geometry_file = rigid_body["geometryFile"]
        if not os.path.isabs(geometry_file):
             # assume path is relative to this file's directory (project root)
             geometry_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), geometry_file)
        mesh = tm.load(geometry_file)
        mesh.apply_scale(rigid_body["scale"])
        offset = np.array(rigid_body["translation"])
        if "rotationAngle" in rigid_body and "rotationAxis" in rigid_body:
            angle = rigid_body["rotationAngle"] / 360 * 2 * 3.1415926
            direction = rigid_body["rotationAxis"]
            rot_matrix = tm.transformations.rotation_matrix(angle, direction, mesh.vertices.mean(axis=0))
            mesh.apply_transform(rot_matrix)
        mesh.vertices += offset
        try:
             tm.repair.fill_holes(mesh)
        except:
             pass
        voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter).fill()
        voxelized_points_np = voxelized_mesh.points
        return voxelized_points_np

    def load_points_from_ply_config(self, obj_cfg):
        ply_file = obj_cfg.get("plyFile")
        if ply_file is None:
            raise ValueError("plyFile is not provided in object config")

        if not os.path.isabs(ply_file):
            ply_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ply_file)

        pos, _ = load_ply_points(ply_file)
        points = np.array(pos, dtype=np.float32)

        # if "scale" in obj_cfg:
        #     scale = np.array(obj_cfg["scale"], dtype=np.float32)
        #     points = points * scale

        # if "rotationAngle" in obj_cfg and "rotationAxis" in obj_cfg:
        #     angle = obj_cfg["rotationAngle"] / 360.0 * 2.0 * np.pi
        #     direction = obj_cfg["rotationAxis"]
        #     center = points.mean(axis=0) if points.shape[0] > 0 else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        #     rot_matrix = tm.transformations.rotation_matrix(angle, direction, center)
        #     points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
        #     points = (points_h @ rot_matrix.T)[:, :3].astype(np.float32)

        # if "translation" in obj_cfg:
        #     translation = np.array(obj_cfg["translation"], dtype=np.float32)
        #     points = points + translation

        return points

    def add_points(self, object_id, points_np, velocity, density, is_dynamic, color, material):
        num_new_particles = int(points_np.shape[0])
        start = self.num_particles_curr
        end = start + num_new_particles
        if end > self.particle_max_num:
            raise RuntimeError(f"Particle overflow: {end} > {self.particle_max_num}")

        wp_new_positions = wp.array(np.array(points_np, dtype=np.float32), dtype=wp.vec3)
        if velocity is None:
            wp_velocity_arr = wp.zeros(shape=num_new_particles, dtype=wp.vec3)
        else:
            wp_velocity_arr = wp.full(
                shape=num_new_particles,
                value=wp.vec3(float(velocity[0]), float(velocity[1]), float(velocity[2])),
                dtype=wp.vec3,
            )

        wp_density_arr = wp.full(shape=num_new_particles, value=density, dtype=float)
        wp_material_arr = wp.full(shape=num_new_particles, value=material, dtype=int)
        wp_is_dynamic_arr = wp.full(shape=num_new_particles, value=int(is_dynamic), dtype=int)
        wp_color_arr = wp.full(
            shape=num_new_particles,
            value=wp.vec3i(int(color[0]), int(color[1]), int(color[2])),
            dtype=wp.vec3i,
        )

        wp.launch(
            kernel=add_particles_kernel,
            dim=num_new_particles,
            inputs=[
                start,
                self.x,
                self.x_0,
                self.v,
                self.rho,
                self.materialMarks.material,
                self.materialMarks.is_dynamic,
                self.object_id,
                self.m_V,
                self.color,
                wp_new_positions,
                wp_velocity_arr,
                wp_density_arr,
                wp_material_arr,
                wp_is_dynamic_arr,
                object_id,
                self.m_V0,
                wp_color_arr,
            ],
        )

        self.num_particles_curr += num_new_particles

    def add_cube(self, object_id, lower_corner, cube_size, velocity, density, is_dynamic, color, material):
        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(lower_corner[i], lower_corner[i] + cube_size[i],
                          self.particle_diameter)) #TODO: hack: reduce particle interval for WCSPH
        num_new_particles = reduce(lambda x, y: x * y, [len(n) for n in num_dim])
        
        new_positions = np.array(np.meshgrid(*num_dim, sparse=False, indexing='ij'), dtype=np.float32)
        new_positions = new_positions.reshape(-1, reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        
        wp_new_positions = wp.array(new_positions, dtype=wp.vec3)

        if velocity is None:
            wp_velocity_arr = wp.zeros(shape=num_new_particles, dtype=wp.vec3)
        else:
            wp_velocity_arr = wp.full(shape=num_new_particles, value=wp.vec3(float(velocity[0]), float(velocity[1]), float(velocity[2])), dtype=wp.vec3)
        
        # Directly create Warp arrays, avoiding numpy intermediates where simple
        wp_density_arr = wp.full(shape=num_new_particles, value=density, dtype=float)
        wp_material_arr = wp.full(shape=num_new_particles, value=material, dtype=int)
        wp_is_dynamic_arr = wp.full(shape=num_new_particles, value=is_dynamic, dtype=int)
        
        # Color needs an array of vec3i, simpler to construct via numpy first or a small kernel?
        # Actually wp.full with a vec3i value works if passed correctly
        # But color is a numpy array input [r,g,b]. 
        # wp.full(..., value=wp.vec3i(*color))
        wp_color_arr = wp.full(shape=num_new_particles, value=wp.vec3i(int(color[0]), int(color[1]), int(color[2])), dtype=wp.vec3i)
        
        start = self.num_particles_curr
        end = start + num_new_particles
        if end > self.particle_max_num:
            raise RuntimeError(f"Particle overflow: {end} > {self.particle_max_num}")
            
        # Warp kernel based initialization
        wp.launch(
            kernel=add_particles_kernel,
            dim=num_new_particles,
            inputs=[
                start,
                self.x, self.x_0, self.v, self.rho, self.materialMarks.material, self.materialMarks.is_dynamic,
                self.object_id, self.m_V, self.color,
                wp_new_positions,
                wp_velocity_arr,
                wp_density_arr,
                wp_material_arr,
                wp_is_dynamic_arr,
                object_id,
                self.m_V0,
                wp_color_arr
            ]
        )
        self.num_particles_curr += num_new_particles

    # rigid body computation functions are now kernelized and moved to rigid_fluid_coupling.py
    def compute_emitted_fluid_particle_num(self, n_pipes, quality):
        fluid_p_num = 15000 + 5000
        return int(fluid_p_num * n_pipes * quality ** 3)

    def init_from_houdini_geo(self, ply_path):
        print("Init from Houdini GEO at", ply_path)
        
        # 1. Load params.json
        params_file = os.path.join(ply_path, "params.json")
        with open(params_file, 'r') as f:
            params = json.load(f)
            
        self.particle_radius = params.get("radius", 0.025)
        self.particle_diameter = 2 * self.particle_radius
        self.smoothing_length = self.particle_radius * 4.0 * self.h_scale
        
        # Domain
        # if "bounding_box" in params:
        #     bbox = params["bounding_box"]
        #     self.domain_start = np.array(bbox[0], dtype=np.float32)
        #     self.domain_end = np.array(bbox[1], dtype=np.float32)
        #     ds = (self.domain_end - self.domain_start).astype(np.float32)
        #     self.domain_size = wp.vec3(ds[0], ds[1], ds[2])
        # else:
        #     self.domain_start = np.array([-10.0, -10.0, -10.0], dtype=np.float32)
        #     self.domain_end = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        #     ds = (self.domain_end - self.domain_start).astype(np.float32)
        #     self.domain_size = wp.vec3(ds[0], ds[1], ds[2])

        if hasattr(self, "cfg") and self.cfg is not None:
            self.domain_start = np.array(self.cfg.get_cfg("domainStart"), dtype=np.float32)
            self.domain_end = np.array(self.cfg.get_cfg("domainEnd"), dtype=np.float32)
            ds = (self.domain_end - self.domain_start).astype(np.float32)
            self.domain_size = wp.vec3(ds[0], ds[1], ds[2])
        else:
            print("Warning: Config not found, using default domain.")
            self.domain_start = np.array([-10.0, -10.0, -10.0], dtype=np.float32)
            self.domain_end = np.array([10.0, 10.0, 10.0], dtype=np.float32)
            ds = (self.domain_end - self.domain_start).astype(np.float32)
            self.domain_size = wp.vec3(ds[0], ds[1], ds[2])

        # # Physics Defaults
        # self.stiffness = 50000.0
        # self.exponent = 7.0
        # self.base_density = 1000.0
        # self.dim = 3
        # self.m_V0 = 0.8 * self.particle_diameter ** self.dim
        # self.particle_mass = self.m_V0 * self.base_density
        # self.dt = 0.0001
        # self.dynamic_visc = 0.01
        # self.gravity = -9.8
        
        print(f"Loaded params: radius={self.particle_radius}, m_V0={self.m_V0}")
        
        # Grid
        self.grid_size = 10.0 * self.smoothing_length
        grid_dims = np.ceil((self.domain_end - self.domain_start) / self.grid_size).astype(int)
        self.grid_num = grid_dims
        self.grid = wp.HashGrid(self.grid_num[0], self.grid_num[1], self.grid_num[2])
        
        print("Grid num:", self.grid_num)

        # 2. Load Particles
        fluid_pos, fluid_attrs = load_ply_points(os.path.join(ply_path, "fluid.ply"))
        ball_pos, ball_attrs = load_ply_points(os.path.join(ply_path, "ball.ply"))
        
        num_fluid = fluid_pos.shape[0]
        num_ball = ball_pos.shape[0]

        # Read rigid properties from scene config (if provided)
        rigid_cfg = None
        rigid_obj_id = 1
        rigid_density = self.base_density
        rigid_velocity = None
        rigid_angular_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        rigid_is_dynamic = 1
        if hasattr(self, "cfg") and self.cfg is not None:
            rigid_bodies_cfg = self.cfg.get_rigid_bodies()
            if len(rigid_bodies_cfg) > 0:
                rigid_cfg = rigid_bodies_cfg[0]
                rigid_obj_id = int(rigid_cfg.get("objectId", 1))
                rigid_density = float(rigid_cfg.get("density", self.base_density))
                if "velocity" in rigid_cfg:
                    rigid_velocity = np.array(rigid_cfg["velocity"], dtype=np.float32)
                if "angularVelocity" in rigid_cfg:
                    rigid_angular_velocity = np.array(rigid_cfg["angularVelocity"], dtype=np.float32)
                rigid_is_dynamic = int(rigid_cfg.get("isDynamic", True))
        
        self.particle_max_num = num_fluid + num_ball
        self.num_particles_curr = self.particle_max_num
        self.num_rigid_bodies = 1 # One ball
        self.num_objects = 2 # 1 fluid (id 0) + 1 rigid (id 1)
        print(f"Loaded {num_fluid} fluid particles and {num_ball} rigid particles from Houdini GEO.")
        # 3. Allocate All Warp Arrays
        self.object_id = wp.zeros(self.particle_max_num, dtype=wp.int32)
        self.x = wp.zeros(self.particle_max_num, dtype=wp.vec3, requires_grad=True)
        self.v = wp.zeros(self.particle_max_num, dtype=wp.vec3, requires_grad=True)
        self.m_V = wp.zeros(self.particle_max_num, dtype=wp.float32)
        self.rho = wp.zeros(self.particle_max_num, dtype=float, requires_grad=True)
        self.pressure = wp.zeros(self.particle_max_num, dtype=float, requires_grad=True)
        self.color = wp.zeros(self.particle_max_num, dtype=wp.vec3i)
        
        self.materialMarks = MaterialMarks()
        self.materialMarks.material = wp.zeros(self.particle_max_num, dtype=wp.int32)
        self.materialMarks.is_dynamic = wp.zeros(self.particle_max_num, dtype=wp.int32)
        
        self.x_0 = wp.zeros(self.particle_max_num, dtype=wp.vec3)
        self.a = wp.zeros(self.particle_max_num, dtype=wp.vec3, requires_grad=True)  # Init acceleration

        # 4. Fill Data
        
        # -- Fluid (Id 0) --
        # Position
        wp_fluid_pos = wp.array(fluid_pos, dtype=wp.vec3)
        wp.copy(self.x, wp_fluid_pos, dest_offset=0, count=num_fluid)
        
        # Velocity
        if 'v1' in fluid_attrs:
            v_np = np.stack([fluid_attrs['v1'], fluid_attrs['v2'], fluid_attrs['v3']], axis=1).astype(np.float32)
            wp_v = wp.array(v_np, dtype=wp.vec3)
            wp.copy(self.v, wp_v, dest_offset=0, count=num_fluid)

        # Mass Volume & Density
        # We can fill range
        # Use simple kernel or copy from full array
        wp_mV_fluid = wp.full(num_fluid, self.m_V0, dtype=wp.float32)
        wp.copy(self.m_V, wp_mV_fluid, dest_offset=0, count=num_fluid)
        
        wp_rho_fluid = wp.full(num_fluid, self.base_density, dtype=float)
        wp.copy(self.rho, wp_rho_fluid, dest_offset=0, count=num_fluid)

        # Marks
        wp_mat_fluid = wp.full(num_fluid, 1, dtype=int) # 1 = Fluid
        wp.copy(self.materialMarks.material, wp_mat_fluid, dest_offset=0, count=num_fluid)
        
        wp_dyn_fluid = wp.full(num_fluid, 1, dtype=int)
        wp.copy(self.materialMarks.is_dynamic, wp_dyn_fluid, dest_offset=0, count=num_fluid)
        
        # Object ID (0) initialized by zeros so no need to copy if 0.
        
        # -- Rigid Ball (Id 1) --
        start_ball = num_fluid
        
        # Position
        wp_ball_pos = wp.array(ball_pos, dtype=wp.vec3)
        wp.copy(self.x, wp_ball_pos, dest_offset=start_ball, count=num_ball)

        # Velocity
        if rigid_velocity is not None:
            wp_v_b = wp.full(num_ball, value=wp.vec3(float(rigid_velocity[0]), float(rigid_velocity[1]), float(rigid_velocity[2])), dtype=wp.vec3)
            wp.copy(self.v, wp_v_b, dest_offset=start_ball, count=num_ball)
        elif 'v1' in ball_attrs:
            v_np_b = np.stack([ball_attrs['v1'], ball_attrs['v2'], ball_attrs['v3']], axis=1).astype(np.float32)
            wp_v_b = wp.array(v_np_b, dtype=wp.vec3)
            wp.copy(self.v, wp_v_b, dest_offset=start_ball, count=num_ball)

        # Mass Volume & Density
        wp_mV_ball = wp.full(num_ball, self.m_V0, dtype=wp.float32)
        wp.copy(self.m_V, wp_mV_ball, dest_offset=start_ball, count=num_ball)
        
        wp_rho_ball = wp.full(num_ball, rigid_density, dtype=float)
        wp.copy(self.rho, wp_rho_ball, dest_offset=start_ball, count=num_ball)
        
        # Marks
        wp_mat_ball = wp.full(num_ball, 0, dtype=int) # 0 = Solid
        wp.copy(self.materialMarks.material, wp_mat_ball, dest_offset=start_ball, count=num_ball)
        
        wp_dyn_ball = wp.full(num_ball, rigid_is_dynamic, dtype=int)
        wp.copy(self.materialMarks.is_dynamic, wp_dyn_ball, dest_offset=start_ball, count=num_ball)

        # Object ID (from config if provided)
        wp_obj_ball = wp.full(num_ball, rigid_obj_id, dtype=int)
        wp.copy(self.object_id, wp_obj_ball, dest_offset=start_ball, count=num_ball)

        # -- Set x_0 for all particles --
        wp.copy(self.x_0, self.x)

        # 5. Initialize RigidBodies struct
        self.rbs = RigidBodies()
        self.rbs.rigid_x = wp.zeros(self.num_objects, dtype=wp.vec3, requires_grad=True)
        self.rbs.rigid_v = wp.zeros(self.num_objects, dtype=wp.vec3, requires_grad=True)
        self.rbs.rigid_omega = wp.zeros(self.num_objects, dtype=wp.vec3, requires_grad=True)
        self.rbs.rigid_mass = wp.zeros(self.num_objects, dtype=float)
        self.rbs.rigid_inv_mass = wp.zeros(self.num_objects, dtype=float)
        self.rbs.rigid_inertia = wp.zeros(self.num_objects, dtype=wp.mat33)
        self.rbs.rigid_inertia0 = wp.zeros(self.num_objects, dtype=wp.mat33)
        self.rbs.rigid_inv_inertia = wp.zeros(self.num_objects, dtype=wp.mat33)
        self.rbs.rigid_rest_cm = wp.zeros(self.num_objects, dtype=wp.vec3)
        self.rbs.rigid_force = wp.zeros(self.num_objects, dtype=wp.vec3, requires_grad=True)
        self.rbs.rigid_torque = wp.zeros(self.num_objects, dtype=wp.vec3, requires_grad=True)
        self.rbs.rigid_v0 = wp.zeros(self.num_objects, dtype=wp.vec3)
        self.rbs.rigid_omega0 = wp.zeros(self.num_objects, dtype=wp.vec3)
        
        # Quaternion Identity
        q_np = np.zeros((self.num_objects, 4), dtype=np.float32)
        q_np[:, 3] = 1.0
        self.rbs.rigid_quaternion = wp.array(q_np, dtype=wp.quat, requires_grad=True)

        # Initialize rigid-body initial linear/angular velocity from config
        if rigid_obj_id < self.num_objects:
            rigid_v_np = np.zeros((self.num_objects, 3), dtype=np.float32)
            rigid_omega_np = np.zeros((self.num_objects, 3), dtype=np.float32)
            if rigid_velocity is not None:
                rigid_v_np[rigid_obj_id] = rigid_velocity
            rigid_omega_np[rigid_obj_id] = rigid_angular_velocity
            wp.copy(self.rbs.rigid_v, wp.array(rigid_v_np, dtype=wp.vec3))
            wp.copy(self.rbs.rigid_v0, wp.array(rigid_v_np, dtype=wp.vec3))
            wp.copy(self.rbs.rigid_omega, wp.array(rigid_omega_np, dtype=wp.vec3))
            wp.copy(self.rbs.rigid_omega0, wp.array(rigid_omega_np, dtype=wp.vec3))

        # Compute Rigid Properties (Mass, CM, etc.) using kernels
        
        # Temporary accumulators [mass, x*m, y*m, z*m] per object? Kernel compute_rigid_cm_mass_kernel uses rigid_rest_cm + rigid_mass
        # compute_rigid_cm_mass_kernel inputs: object_id, x, m_V, rho, rigid_mass, rigid_rest_cm, m_V0
        
        wp.launch(
            kernel=compute_rigid_cm_mass_kernel,
            dim=self.particle_max_num,
            inputs=[
                self.object_id,
                self.x,
                self.m_V,
                self.rho,
                self.rbs.rigid_mass,
                self.rbs.rigid_rest_cm, 
                self.m_V0
            ]
        )
        
        wp.launch(
            kernel=finalize_rigid_cm_kernel,
            dim=self.num_objects,
            inputs=[
                self.rbs.rigid_mass,
                self.rbs.rigid_rest_cm,
                self.num_objects
            ]
        )
        
        # Set Rigid X to CM
        wp.copy(self.rbs.rigid_x, self.rbs.rigid_rest_cm)
        
        # Initialize rigid velocity from particle velocity average?
        # For now assume 0 or infer?
        # If we have particle velocities, we should probably set rigid body velocity to match average linear momentum.
        # But this part is tricky without a dedicated kernel to sum momentum.
        # Assuming initial rigid velocity is 0 unless params.json or external logic sets it.
        # params.json provided does NOT set velocity.
        
        # Compute Inertia
        rigid_inertia_accum_flat = wp.zeros(self.num_objects * 9, dtype=float)

        wp.launch(
            kernel=compute_rigid_inertia_kernel,
            dim=self.particle_max_num,
            inputs=[
                self.object_id,
                self.x,
                self.m_V,
                self.rho,
                self.rbs.rigid_rest_cm,
                rigid_inertia_accum_flat,
                self.m_V0
            ]
        )
        
        wp.launch(
            kernel=finalize_rigid_inertia_kernel,
            dim=self.num_objects,
            inputs=[
                self.rbs.rigid_mass,
                rigid_inertia_accum_flat,
                self.rbs.rigid_inertia,
                self.rbs.rigid_inv_inertia,
                self.rbs.rigid_inv_mass,
                self.rbs.rigid_inertia0,
                self.num_objects
            ]
        )
        
        # Store initial inertia
        wp.copy(self.rbs.rigid_inertia0, self.rbs.rigid_inertia)

    def init_from_generated_geo(self, cfg):
        #========== Compute number of particles ==========#
        #### Process Fluid Blocks ####
        fluid_blocks = cfg.get_fluid_blocks()
        fluid_particle_num = 0
        for fluid in fluid_blocks:
            if "plyFile" in fluid and fluid["plyFile"]:
                particle_points_np = self.load_points_from_ply_config(fluid)
                fluid["particlePoints"] = particle_points_np
                particle_num = particle_points_np.shape[0]
            else:
                particle_num = self.compute_cube_particle_num(fluid["start"], fluid["end"])
            fluid["particleNum"] = particle_num
            self.object_collection[fluid["objectId"]] = fluid
            fluid_particle_num += particle_num

        #### Process Rigid Blocks ####
        rigid_blocks = cfg.get_rigid_blocks()
        rigid_particle_num = 0
        for rigid in rigid_blocks:
            if "plyFile" in rigid and rigid["plyFile"]:
                particle_points_np = self.load_points_from_ply_config(rigid)
                rigid["particlePoints"] = particle_points_np
                particle_num = particle_points_np.shape[0]
            else:
                particle_num = self.compute_cube_particle_num(rigid["start"], rigid["end"])
            rigid["particleNum"] = particle_num
            self.object_collection[rigid["objectId"]] = rigid
            rigid_particle_num += particle_num
        
        #### Process Rigid Bodies ####
        rigid_bodies = cfg.get_rigid_bodies()
        for rigid_body in rigid_bodies:
            if "plyFile" in rigid_body and rigid_body["plyFile"]:
                particle_points_np = self.load_points_from_ply_config(rigid_body)
            else:
                particle_points_np = self.load_rigid_body(rigid_body)
            rigid_body["particleNum"] = particle_points_np.shape[0]
            rigid_body["particlePoints"] = particle_points_np
            self.object_collection[rigid_body["objectId"]] = rigid_body
            rigid_particle_num += particle_points_np.shape[0]
        
        self.fluid_particle_num = fluid_particle_num
        self.solid_particle_num = rigid_particle_num
        self.emitted_particle_num = 2000
        # self.emitted_particle_num = self.compute_emitted_fluid_particle_num(cfg.get_cfg("nPipe"), cfg.get_cfg("quality"))
        self.particle_max_num = fluid_particle_num + rigid_particle_num + self.emitted_particle_num
        self.num_rigid_bodies = len(rigid_blocks)+len(rigid_bodies)

        if len(self.object_collection) > 0:
            self.num_objects = max(self.object_collection.keys()) + 1
        else:
            self.num_objects = 0

        if len(rigid_blocks) > 0:
            print("Warning: currently rigid block functions are not completed, may lead to unexpected behaviour")
            input("Press Enter to continue")

        #### TODO: Handle the Particle Emitter ####
        # self.particle_max_num += emitted particles
        print(f"Particle max num: {self.particle_max_num}")
        self.num_particles_curr = 0

        #========== Allocate memory ==========#
        # Rigid body properties
        if self.num_rigid_bodies > 0:
            # We keep these on host for initialization convenience inside the loops
            self.np_rigid_v0 = np.zeros((self.num_objects, self.dim), dtype=np.float32)
            self.np_rigid_v = np.zeros((self.num_objects, self.dim), dtype=np.float32)
            self.np_rigid_omega = np.zeros((self.num_objects, 3), dtype=np.float32)
            self.np_rigid_omega0 = np.zeros((self.num_objects, 3), dtype=np.float32)


        # Particle related properties
        # Allocate Warp arrays directly
        self.object_id = wp.zeros(self.particle_max_num, dtype=wp.int32)
        self.x = wp.zeros(self.particle_max_num, dtype=wp.vec3, requires_grad=True)
        self.x_0 = wp.zeros(self.particle_max_num, dtype=wp.vec3)
        self.v = wp.zeros(self.particle_max_num, dtype=wp.vec3, requires_grad=True)
        # self.acceleration = wp.zeros(self.particle_max_num, dtype=wp.vec3) # Not used?
        self.m_V = wp.zeros(self.particle_max_num, dtype=wp.float32)
        # self.m = wp.zeros(self.particle_max_num, dtype=wp.float32) # Not used?
        self.rho = wp.zeros(self.particle_max_num, dtype=float, requires_grad=True)
        self.pressure = wp.zeros(self.particle_max_num, dtype=float, requires_grad=True)
        
        self.color = wp.zeros(self.particle_max_num, dtype=wp.vec3i)

        self.materialMarks = MaterialMarks()
        self.materialMarks.material = wp.zeros(self.particle_max_num, dtype=wp.int32)
        self.materialMarks.is_dynamic = wp.zeros(self.particle_max_num, dtype=wp.int32)
        
        self.a = wp.zeros(self.particle_max_num, dtype=wp.vec3, requires_grad=True)


        #========== Initialize particles ==========#

        # Fluid block
        for fluid in fluid_blocks:
            obj_id = fluid["objectId"]
            velocity = fluid["velocity"]
            density = fluid["density"]
            self.base_density = density # record for later use in emitted particles
            color = fluid.get("color", [255, 255, 255])
            if "particlePoints" in fluid:
                self.add_points(
                    object_id=obj_id,
                    points_np=fluid["particlePoints"],
                    velocity=velocity,
                    density=density,
                    is_dynamic=1,
                    color=color,
                    material=1,
                )
            else:
                offset = np.array(fluid["translation"])
                start = np.array(fluid["start"]) + offset
                end = np.array(fluid["end"]) + offset
                scale = np.array(fluid["scale"])
                self.add_cube(object_id=obj_id,
                              lower_corner=start,
                              cube_size=(end-start)*scale,
                              velocity=velocity,
                              density=density, 
                              is_dynamic=1, # enforce fluid dynamic
                              color=color,
                              material=1) # 1 indicates fluid
        
        # TODO: Handle rigid block
        # Rigid block
        for rigid in rigid_blocks:
            obj_id = rigid["objectId"]
            velocity = rigid["velocity"]
            angular_velocity = rigid["angularVelocity"]
            density = rigid["density"]
            color = rigid.get("color", [255, 255, 255])
            is_dynamic = rigid["isDynamic"]
            self.np_rigid_v[obj_id] = velocity
            self.np_rigid_v0[obj_id] = velocity
            self.np_rigid_omega[obj_id] = angular_velocity
            self.np_rigid_omega0[obj_id] = angular_velocity
            if "particlePoints" in rigid:
                self.add_points(
                    object_id=obj_id,
                    points_np=rigid["particlePoints"],
                    velocity=velocity,
                    density=density,
                    is_dynamic=is_dynamic,
                    color=color,
                    material=0,
                )
            else:
                offset = np.array(rigid["translation"])
                start = np.array(rigid["start"]) + offset
                end = np.array(rigid["end"]) + offset
                scale = np.array(rigid["scale"])
                self.add_cube(object_id=obj_id,
                              lower_corner=start,
                              cube_size=(end-start)*scale,
                              velocity=velocity,
                              density=density, 
                              is_dynamic=is_dynamic,
                              color=color,
                              material=0) # 1 indicates solid

        # Rigid bodies
        for rigid_body in rigid_bodies:
            obj_id = rigid_body["objectId"]
            self.object_id_rigid_body.add(obj_id)
            num_particles_obj = rigid_body["particleNum"]
            particle_points_np = rigid_body["particlePoints"]
            is_dynamic = rigid_body["isDynamic"]
            if is_dynamic:
                velocity = np.array(rigid_body["velocity"], dtype=np.float32)
                if "angularVelocity" in rigid_body:
                    angular_velocity = np.array(rigid_body["angularVelocity"], dtype=np.float32)
                else:
                    angular_velocity = np.array([0.0 for _ in range(self.dim)], dtype=np.float32)
            else:
                velocity = np.array([0.0 for _ in range(self.dim)], dtype=np.float32)
                angular_velocity = np.array([0.0 for _ in range(self.dim)], dtype=np.float32)
            density = rigid_body["density"]
            color = np.array(rigid_body.get("color", [255, 255, 255]), dtype=np.int32)
            self.np_rigid_v[obj_id] = velocity
            self.np_rigid_v0[obj_id] = velocity
            self.np_rigid_omega[obj_id] = angular_velocity
            self.np_rigid_omega0[obj_id] = angular_velocity
            
            start = self.num_particles_curr
            end = start + num_particles_obj
            if end > self.particle_max_num:
                raise RuntimeError(f"Particle overflow: {end} > {self.particle_max_num}")
                
            # Direct Warp array creation
            wp_density_arr = wp.full(shape=num_particles_obj, value=density, dtype=float)
            wp_material_arr = wp.full(shape=num_particles_obj, value=0, dtype=int)
            wp_is_dynamic_arr = wp.full(shape=num_particles_obj, value=int(is_dynamic), dtype=int)
            # color is np array [r,g,b]
            wp_color_arr = wp.full(shape=num_particles_obj, value=wp.vec3i(int(color[0]), int(color[1]), int(color[2])), dtype=wp.vec3i)

            wp_new_positions = wp.array(np.array(particle_points_np, dtype=np.float32), dtype=wp.vec3)
            wp_velocity_arr = wp.full(shape=num_particles_obj, value=wp.vec3(float(velocity[0]), float(velocity[1]), float(velocity[2])), dtype=wp.vec3)

            # Warp kernel based initialization
            wp.launch(
                kernel=add_particles_kernel,
                dim=num_particles_obj,
                inputs=[
                    start,
                    self.x, self.x_0, self.v, self.rho, self.materialMarks.material, self.materialMarks.is_dynamic,
                    self.object_id, self.m_V, self.color,
                    wp_new_positions,
                    wp_velocity_arr,
                    wp_density_arr,
                    wp_material_arr,
                    wp_is_dynamic_arr,
                    obj_id,
                    self.m_V0,
                    wp_color_arr
                ]
            )
            self.num_particles_curr += num_particles_obj

        # some stntance after init needed to run
        # Rigid body logic moved to kernels

        # Map to Warp
        self.rbs = RigidBodies()
        if self.num_rigid_bodies > 0:
            # Allocate arrays on device
            self.rbs.rigid_rest_cm = wp.zeros(self.num_objects, dtype=wp.vec3)
            self.rbs.rigid_x = wp.zeros(self.num_objects, dtype=wp.vec3, requires_grad=True)
            self.rbs.rigid_v0 = wp.array(self.np_rigid_v0, dtype=wp.vec3)
            self.rbs.rigid_v = wp.array(self.np_rigid_v, dtype=wp.vec3, requires_grad=True)
            self.rbs.rigid_force = wp.zeros(self.num_objects, dtype=wp.vec3, requires_grad=True)
            self.rbs.rigid_torque = wp.zeros(self.num_objects, dtype=wp.vec3, requires_grad=True)
            self.rbs.rigid_omega = wp.array(self.np_rigid_omega, dtype=wp.vec3, requires_grad=True)
            self.rbs.rigid_omega0 = wp.array(self.np_rigid_omega0, dtype=wp.vec3)
            
            # Quaternion initialization (identity = 0,0,0,1 in x,y,z,w)
            q_np = np.zeros((self.num_objects, 4), dtype=np.float32)
            q_np[:, 3] = 1.0 
            self.rbs.rigid_quaternion = wp.array(q_np, dtype=wp.quat)
            
            self.rbs.rigid_mass = wp.zeros(self.num_objects, dtype=float)
            self.rbs.rigid_inv_mass = wp.zeros(self.num_objects, dtype=float)
            self.rbs.rigid_inertia = wp.zeros(self.num_objects, dtype=wp.mat33)
            self.rbs.rigid_inertia0 = wp.zeros(self.num_objects, dtype=wp.mat33)
            self.rbs.rigid_inv_inertia = wp.zeros(self.num_objects, dtype=wp.mat33)

            # --- Compute Rigid Properties on GPU ---
            
            # Temporary accumulators
            # rigid_rest_cm_accum_flat = wp.zeros(self.num_objects * 3, dtype=float)
            rigid_inertia_accum_flat = wp.zeros(self.num_objects * 9, dtype=float)

            # 1. Compute Mass and Weighted Position Sum
            wp.launch(
                kernel=compute_rigid_cm_mass_kernel,
                dim=self.particle_max_num,
                inputs=[
                    self.object_id,
                    self.x,
                    self.m_V,
                    self.rho,
                    self.rbs.rigid_mass,
                    self.rbs.rigid_rest_cm, 
                    self.m_V0
                ]
            )

            # 2. Finalize Rest CM
            wp.launch(
                kernel=finalize_rigid_cm_kernel,
                dim=self.num_objects,
                inputs=[
                    self.rbs.rigid_mass,
                    self.rbs.rigid_rest_cm,
                    self.num_objects
                ]
            )

            # Initialize rigid_x from rigid_rest_cm
            wp.copy(self.rbs.rigid_x, self.rbs.rigid_rest_cm)

            # 3. Compute Inertia Tensor
            wp.launch(
                kernel=compute_rigid_inertia_kernel,
                dim=self.particle_max_num,
                inputs=[
                    self.object_id,
                    self.x,
                    self.m_V,
                    self.rho,
                    self.rbs.rigid_rest_cm,
                    rigid_inertia_accum_flat,
                    self.m_V0
                ]
            )

            # 4. Finalize Inertia
            wp.launch(
                kernel=finalize_rigid_inertia_kernel,
                dim=self.num_objects,
                inputs=[
                    self.rbs.rigid_mass,
                    rigid_inertia_accum_flat,
                    self.rbs.rigid_inertia,
                    self.rbs.rigid_inv_inertia,
                    self.rbs.rigid_inv_mass,
                    self.rbs.rigid_inertia0,
                    self.num_objects
                ]
            )

        # self.object_id = wp.zeros(self.particle_max_num, dtype=wp.int32)
        print(f"n: {self.particle_max_num}, x shape: {self.x.shape}")
        self.print_rigid_info()  
        # Already allocated and filled via add_particles/kernels, no need to recreate from np arrays
        # But we previously mapped from np_x, so let's clean that up.
        
        # self.x = wp.array(self.np_x, dtype=wp.vec3, requires_grad=True)
        # self.x_0 = wp.array(self.np_x_0, dtype=wp.vec3)
        # self.v = wp.array(self.np_v, dtype=wp.vec3, requires_grad=True)
        # self.rho = wp.array(self.np_rho, dtype=float, requires_grad=True)
        # self.pressure = wp.array(self.np_pressure, dtype=float, requires_grad=True)
        # self.m_V = wp.array(self.np_m_V, dtype=wp.float32)

        # self.materialMarks = MaterialMarks()
        # self.materialMarks.material = wp.array(self.np_material, dtype=wp.int32)
        # self.materialMarks.is_dynamic = wp.array(self.np_is_dynamic, dtype=wp.int32)
        
        # self.a = wp.zeros(self.particle_max_num, dtype=wp.vec3, requires_grad=True)

    def init_IISPH(self):
        self.a_ii = wp.zeros(self.particle_max_num, dtype=float)
        self.density_deviation = wp.zeros(self.particle_max_num, dtype=float)
        self.last_pressure = wp.zeros(self.particle_max_num, dtype=float)
        self.avg_density_error = wp.zeros(1, dtype=float) # Keep as array for atomic add in kernel

        self.pressure_a = wp.zeros(self.particle_max_num, dtype=wp.vec3)
    def ti_to_warp(self,):
            # use container values
            # self.n = int(self.ps.particle_num.to_numpy())
            self.fluid_particle_num = int(self.ps.fluid_particle_num)
            self.solid_particle_num = int(self.ps.solid_particle_num)
            self.particle_max_num = int(self.ps.particle_max_num)
            self.num_particles_curr = self.particle_max_num
            self.num_rigid_bodies = int(self.ps.num_rigid_bodies)
            self.num_objects = int(self.ps.num_objects)
            print(f"Current particle num: {self.particle_max_num}, Particle max num: {self.particle_max_num}")
            # map Taichi fields into Warp RigidBodies arrays
            self.rbs = RigidBodies()
            if self.num_rigid_bodies > 0:
                self.rbs.rigid_rest_cm = wp.array(self.ps.rigid_rest_cm.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.vec3)
                self.rbs.rigid_x       = wp.array(self.ps.rigid_x.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.vec3, requires_grad=True)
                self.rbs.rigid_v0      = wp.array(self.ps.rigid_v0.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.vec3)
                self.rbs.rigid_v       = wp.array(self.ps.rigid_v.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.vec3, requires_grad=True)
                self.rbs.rigid_force   = wp.array(self.ps.rigid_force.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.vec3, requires_grad=True)
                self.rbs.rigid_torque  = wp.array(self.ps.rigid_torque.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.vec3, requires_grad=True)
                # omegas (3-components)
                self.rbs.rigid_omega  = wp.array(self.ps.rigid_omega.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.vec3, requires_grad=True)
                self.rbs.rigid_omega0 = wp.array(self.ps.rigid_omega0.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.vec3)
                # quaternions (shape: n_obj x 4) -> Warp quat
                q_np = self.ps.rigid_quaternion.to_numpy()[:self.num_objects].astype(np.float32)
                # reorder from (w, x, y, z) -> (x, y, z, w) to match Warp quat layout
                if q_np.ndim == 1:
                    q_np = q_np.reshape(1, 4)
                q_np = q_np[:, [1, 2, 3, 0]].copy()
                self.rbs.rigid_quaternion = wp.array(q_np, dtype=wp.quat)
                # scalar masses
                self.rbs.rigid_mass     = wp.array(self.ps.rigid_mass.to_numpy()[:self.num_objects].astype(np.float32), dtype=float)
                self.rbs.rigid_inv_mass = wp.array(self.ps.rigid_inv_mass.to_numpy()[:self.num_objects].astype(np.float32), dtype=float)
                # inertia matrices (n_obj x 3 x 3)
                self.rbs.rigid_inertia     = wp.array(self.ps.rigid_inertia.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.mat33)
                self.rbs.rigid_inertia0    = wp.array(self.ps.rigid_inertia0.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.mat33)
                self.rbs.rigid_inv_inertia = wp.array(self.ps.rigid_inv_inertia.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.mat33)
                # self.print_rigid_info()            
            
            # allocate arrays and initialize
            self.object_id = wp.array(self.ps.object_id.to_numpy()[: self.particle_max_num].astype(np.int32), dtype=wp.int32)
            # self.x = wp.zeros(self.particle_max_num, dtype=wp.vec3, requires_grad=True)
            # self.v = wp.zeros(self.particle_max_num, dtype=wp.vec3, requires_grad=True)
            # self.rho = wp.zeros(self.n, dtype=float, requires_grad=True)
            self.a = wp.zeros(self.particle_max_num, dtype=wp.vec3, requires_grad=True)
            px = self.ps.x.to_numpy()[: self.particle_max_num].astype(np.float32)
            self.x = wp.array(px, dtype=wp.vec3)
            px_0 = self.ps.x_0.to_numpy()[: self.particle_max_num].astype(np.float32)
            self.x_0 =  wp.array(px_0, dtype=wp.vec3)
            prho = self.ps.density.to_numpy()[: self.particle_max_num].astype(np.float32)
            self.rho = wp.array(prho, dtype=float)
            pv = self.ps.v.to_numpy()[: self.particle_max_num].astype(np.float32)
            self.v = wp.array(pv, dtype=wp.vec3)
            self.pressure = wp.zeros(self.particle_max_num, dtype=float, requires_grad=True)
            print(f"n: {self.particle_max_num}, x shape: {self.x.shape}, rho[1] = {prho[1]}")

            self.materialMarks = MaterialMarks()
            self.materialMarks.material = wp.array(self.ps.material.to_numpy()[: self.particle_max_num].astype(np.int32), dtype=wp.int32)
            self.materialMarks.is_dynamic = wp.array(self.ps.is_dynamic.to_numpy()[: self.particle_max_num].astype(np.int32), dtype=wp.int32)

            self.m_V = wp.array(self.ps.m_V.to_numpy()[: self.particle_max_num].astype(np.float32), dtype=wp.float32)

    def init_from_ply(self, ply_path):
        print(f"Loading initial state from {ply_path}")
        pos, attrs = load_ply_points(ply_path)
        num_particles = pos.shape[0]
        
        print(f"Initializing particle arrays with {num_particles} particles from PLY")
        self.particle_max_num = num_particles

        if self.ps is not None:
            self.fluid_particle_num = int(self.ps.fluid_particle_num)
            self.solid_particle_num = int(self.ps.solid_particle_num)
            # self.particle_max_num = int(self.ps.particle_max_num) # Don't overwrite this from PS, use PLY count
            self.num_rigid_bodies = int(self.ps.num_rigid_bodies)
            self.num_objects = int(self.ps.num_objects)
            
            self.rbs = RigidBodies()
            if self.num_rigid_bodies > 0:
                self.rbs.rigid_rest_cm = wp.array(self.ps.rigid_rest_cm.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.vec3)
                self.rbs.rigid_x       = wp.array(self.ps.rigid_x.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.vec3, requires_grad=True)
                self.rbs.rigid_v0      = wp.array(self.ps.rigid_v0.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.vec3)
                self.rbs.rigid_v       = wp.array(self.ps.rigid_v.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.vec3, requires_grad=True)
                self.rbs.rigid_force   = wp.array(self.ps.rigid_force.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.vec3, requires_grad=True)
                self.rbs.rigid_torque  = wp.array(self.ps.rigid_torque.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.vec3, requires_grad=True)
                self.rbs.rigid_omega  = wp.array(self.ps.rigid_omega.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.vec3, requires_grad=True)
                self.rbs.rigid_omega0 = wp.array(self.ps.rigid_omega0.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.vec3)
                
                q_np = self.ps.rigid_quaternion.to_numpy()[:self.num_objects].astype(np.float32)
                if q_np.ndim == 1:
                    q_np = q_np.reshape(1, 4)
                q_np = q_np[:, [1, 2, 3, 0]].copy()
                self.rbs.rigid_quaternion = wp.array(q_np, dtype=wp.quat)
                
                self.rbs.rigid_mass     = wp.array(self.ps.rigid_mass.to_numpy()[:self.num_objects].astype(np.float32), dtype=float)
                self.rbs.rigid_inv_mass = wp.array(self.ps.rigid_inv_mass.to_numpy()[:self.num_objects].astype(np.float32), dtype=float)
                self.rbs.rigid_inertia     = wp.array(self.ps.rigid_inertia.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.mat33)
                self.rbs.rigid_inertia0    = wp.array(self.ps.rigid_inertia0.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.mat33)
                self.rbs.rigid_inv_inertia = wp.array(self.ps.rigid_inv_inertia.to_numpy()[:self.num_objects].astype(np.float32), dtype=wp.mat33)
        else:
             # If no PS, maybe initialize empty RBS?
             self.rbs = RigidBodies() # Assuming default constructor works or handles empty
             self.num_objects = 0
             self.num_rigid_bodies = 0
             # Count fluid particles if material is available, otherwise assume all are fluid if not specified
             if 'material' in attrs:
                 mat = attrs['material'].astype(np.int32)
                 self.fluid_particle_num = np.sum(mat == MaterialType.FLUID)
                 self.solid_particle_num = np.sum(mat == MaterialType.SOLID)
             else:
                 self.fluid_particle_num = self.particle_max_num
                 self.solid_particle_num = 0
        
        # Allocate basic fields
        self.x = wp.array(pos, dtype=wp.vec3, requires_grad=True)
        
        if 'vx' in attrs and 'vy' in attrs and 'vz' in attrs:
            vel = np.stack([attrs['vx'], attrs['vy'], attrs['vz']], axis=1).astype(np.float32)
            self.v = wp.array(vel, dtype=wp.vec3, requires_grad=True)
        else:
            self.v = wp.zeros(self.particle_max_num, dtype=wp.vec3, requires_grad=True)
            
        if 'rho' in attrs:
            self.rho = wp.array(attrs['rho'], dtype=float, requires_grad=True)
        else:
            self.rho = wp.zeros(self.particle_max_num, dtype=float, requires_grad=True)

        self.pressure = wp.zeros(self.particle_max_num, dtype=float, requires_grad=True)
        self.a = wp.zeros(self.particle_max_num, dtype=wp.vec3, requires_grad=True)
        
        if 'mV' in attrs:
            self.m_V = wp.array(attrs['mV'], dtype=float, requires_grad=True)
        else:
            self.m_V = wp.zeros(self.particle_max_num, dtype=float, requires_grad=True)
            
        if 'object_id' in attrs:
            self.object_id = wp.array(attrs['object_id'].astype(np.int32), dtype=wp.int32)
        else:
            self.object_id = wp.zeros(self.particle_max_num, dtype=wp.int32)
            
        # Initialize material marks
        if not hasattr(self, 'materialMarks'):
            self.materialMarks = MaterialMarks()
        
        if 'material' in attrs:
            self.materialMarks.material = wp.array(attrs['material'].astype(np.int32), dtype=wp.int32)
        else:
            self.materialMarks.material = wp.zeros(self.particle_max_num, dtype=wp.int32)
            
        if 'is_dynamic' in attrs:
            self.materialMarks.is_dynamic = wp.array(attrs['is_dynamic'].astype(np.int32), dtype=wp.int32)
        else:
            self.materialMarks.is_dynamic = wp.zeros(self.particle_max_num, dtype=wp.int32)
            
        self.x_0 = wp.array(pos, dtype=wp.vec3)

        
        print("Initialization from PLY complete.")

    def substep_WCSPH(self):
        wp.launch(
            kernel=compute_pressure,
            dim=self.particle_max_num,
            inputs=[self.rho, self.pressure, self.materialMarks,
                    self.stiffness, self.exponent, self.base_density],
        )

    def substep_IISPH(self):
        
        wp.launch(
            kernel=predict_velocity,
            dim=self.particle_max_num,
            inputs=[self.a, self.gravity, float(self.dt), self.materialMarks, self.v,]
        )

        wp.launch(
            kernel=compute_aii_and_density_deviation,
            dim=self.particle_max_num,
            inputs=[
                self.grid.id, self.x, self.v, self.rho, self.m_V,
                self.materialMarks, self.smoothing_length, self.base_density, float(self.dt),
                self.a_ii, self.density_deviation
            ]
        )
        
        # Pressure Solve
        wp.copy(self.last_pressure, self.pressure)
        self.pressure.zero_() 
        
        cnt_iter = 0
        while cnt_iter < 1000:
            self.avg_density_error.zero_()
            # self.pressure_a.zero_()
            wp.launch(
                kernel=compute_pressure_a_iisph,
                dim=self.particle_max_num,
                inputs=[
                    self.grid.id, self.x, self.rho,
                    self.pressure, 
                    self.m_V, self.materialMarks, self.smoothing_length, self.base_density,
                    self.pressure_a
                ]
            )
            wp.launch(
                kernel=update_pressure_and_compute_avg_error,
                dim=self.particle_max_num,
                inputs=[
                    self.grid.id, self.x, self.pressure_a, self.m_V,
                    self.a_ii, self.density_deviation,
                    self.pressure, 
                    self.materialMarks, self.smoothing_length, self.base_density, float(self.dt),
                    0.5, # omega
                    self.avg_density_error
                ]
            )
            cnt_iter += 1
            err = self.avg_density_error.numpy()[0]
            if err / self.fluid_particle_num < 0.001:
                break

        # Final update force for integration
        self.a.zero_()
        wp.copy(self.a, self.pressure_a)

    def sub_step(self):
        # compute density of points
        wp.launch(
            kernel=compute_density,
            dim=self.particle_max_num,
            inputs=[self.grid.id, self.x,
                # self.density_normalization_no_mass,
                    1.0, # cubic kernel don't need normalization
                    self.smoothing_length,
                self.materialMarks, self.m_V, self.base_density,
                self.rho],
        )

        wp.launch(
        kernel=compute_non_presure_forces,
        dim=self.particle_max_num,
        inputs=[
            self.grid.id,
            self.x,
            self.v,
            self.rho,
            self.dynamic_visc,
            self.smoothing_length,
            self.materialMarks,
            self.m_V,
            self.base_density,
            self.viscous_forces,
            self.object_id,
            self.rbs,
            self.surface_tension,
            self.gravity,
            self.a
        ],
    )

        if self.USE_METHOD == 1:
            self.substep_IISPH()
            self.a.zero_()

        else:
            self.substep_WCSPH()

        wp.launch(
                kernel=compute_pressure_a,
                dim=self.particle_max_num,
                inputs=[
                self.grid.id,
                self.x,
                self.v,
                self.rho,
                self.pressure,
                self.base_density,
                1.0,  # cubic kernel don't need normalization
                self.smoothing_length,
                self.materialMarks,
                self.m_V,
                self.pressure_forces,
                self.neibor_nums,
                self.object_id,
                self.rbs,
                self.a
                ]
        )
        # kick
        wp.launch(kernel=kick_diff, dim=self.particle_max_num, inputs=[self.a, self.dt, self.v, self.v])

        # drift
        wp.launch(kernel=drift_diff, dim=self.particle_max_num, inputs=[self.x, self.v, self.dt, self.x])

    def step(self, t):
        self.time_step = t
        with wp.ScopedTimer("step", active=self.verbose):
            for _ in range(self.sim_step_to_frame_ratio):
                with wp.ScopedTimer("grid build", active=self.verbose):
                    # build grid
                    #self.grid.build(self.x, self.smoothing_length)
                    self.grid.build(self.x, self.grid_size)

                with wp.ScopedTimer("forces", active=self.verbose):
                    wp.launch(
                        kernel=compute_moving_boundary_volume,
                        dim=self.particle_max_num,
                        inputs=[self.grid.id, self.x, self.m_V, self.density_normalization_no_mass, self.smoothing_length,
                                self.materialMarks],
                    )

                    wp.launch(
                        kernel=enforce_boundary_3D_warp,
                        dim=self.particle_max_num,
                        inputs=[self.x, self.v,
                                self.materialMarks,
                                wp.vec3(*self.domain_start),
                                wp.vec3(*self.domain_end),
                                self.padding,
                        ]
                    )

                    self.sub_step()
                    g = wp.vec3(0.0, self.gravity, 0.0)

                    wp.launch(
                        kernel=solve_rigid_body,
                        dim=self.num_objects,
                        inputs=[self.rbs, g, self.dt, self.rbs]
                    )
                    # wp.launch(kernel=solve_rigid_body, dim=self.num_rigid_bodies, inputs=[self.rbs, g, self.dt]) # 该实现有问题
                    wp.launch(
                        kernel=update_rigid_particle_info,
                        dim=self.particle_max_num,
                        inputs=[self.x, self.v, self.x_0,
                            self.object_id,
                            self.materialMarks,
                            self.rbs,
                        ]
                    )

            self.sim_time += self.frame_dt

        # with wp.ScopedTimer("render"):
        #     self.renderer.begin_frame(self.sim_time)
        #     self.renderer.render_points(
        #         points=self.x.numpy(), radius=self.smoothing_length, name="points", colors=(0.8, 0.3, 0.2)
        #     )
        #     self.renderer.end_frame()
        # if self.time_step >= 1500 and self.time_step % 100 == 0:
        #     self.print_rigid_info()  

    def print_rigid_info(self):
        if self.num_rigid_bodies > 0:
            masses = self.rbs.rigid_mass.numpy()
            pos = self.rbs.rigid_x.numpy()
            vel = self.rbs.rigid_v.numpy()
            omega = self.rbs.rigid_omega.numpy()
            quat = self.rbs.rigid_quaternion.numpy()
            rest_cm = self.rbs.rigid_rest_cm.numpy()

            print(f"[rbs] num={self.num_rigid_bodies}")
            for i in range(1, self.num_objects): # 跳过流体
                print(
                    f" id={i} mass={masses[i]:.6f} pos={pos[i]} rest_cm={rest_cm[i]}\n vel={vel[i]} "
                    f"omega={omega[i]} quat={quat[i]}"
                )
    
    def export_ply(self, series_prefix, cnt_ply):
        np_pos = self.x.numpy()
        np_rho = self.rho.numpy()
        # m_V: use computed per-particle if available, else fallback to constant m_V0
        np_mV = self.m_V.numpy()
        np_obj_id = self.object_id.numpy()
        # also export per-particle force diagnostics (split vec3 into scalar components)
        pf = self.pressure_forces.numpy()
        vf = self.viscous_forces.numpy()
        np_a = self.a.numpy()
        np_v = self.v.numpy()  # velocity
        out_path = series_prefix.format(cnt_ply)
        export_ply_points(out_path, np_pos.astype(np.float32), {
            'rho': np_rho.astype(np.float32),
            'pressure': self.pressure.numpy().astype(np.float32),
            'mV': np_mV.astype(np.float32),
            'object_id': np_obj_id.astype(np.int32),
            'neighbor_num': self.neibor_nums.numpy().astype(np.int32),
            'pressure_fx': pf[:,0].astype(np.float32),
            'pressure_fy': pf[:,1].astype(np.float32),
            'pressure_fz': pf[:,2].astype(np.float32),
            'viscous_fx': vf[:,0].astype(np.float32),
            'viscous_fy': vf[:,1].astype(np.float32),
            'viscous_fz': vf[:,2].astype(np.float32),
            'ax': np_a[:,0].astype(np.float32),
            'ay': np_a[:,1].astype(np.float32),
            'az': np_a[:,2].astype(np.float32),
            'vx': np_v[:,0].astype(np.float32),
            'vy': np_v[:,1].astype(np.float32),
            'vz': np_v[:,2].astype(np.float32),
            'material': self.materialMarks.material.numpy().astype(np.int32),
            'is_dynamic': self.materialMarks.is_dynamic.numpy().astype(np.int32),
        })

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_sph.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=480, help="Total number of frames.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    args = parser.parse_known_args()[0]
    series_prefix = "demo_output/particle_object_{}.ply".format( "{}")
    with wp.ScopedDevice(args.device):
        example = SimSPH(stage_path=args.stage_path)
        cnt_ply = 0
        for time_step in range(args.num_frames):
            # example.render()
            example.step(time_step)
            example.export_ply(series_prefix, cnt_ply)
            cnt_ply += 1
            # example.partio_export()