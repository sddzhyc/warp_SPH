from particle_system import ParticleSystem
from rigid_fluid_coupling import MaterialMarks, RigidBodies, compute_moving_boundary_volume, compute_static_boundary_volume, solve_rigid_body, update_rigid_particle_info
from sim_utils import export_ply_points, load_ply_points
from sph_kernel import *

import numpy as np
import warp as wp
# optional dependency for flexible PLY export with custom attributes

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
        m_V_np = self.m_V.numpy()
        print(f"Computed boundary volumes, sample m_V: {m_V_np[:10]}")


    def __init__(self,config = None, container: ParticleSystem = None, stage_path="example_sph.usd", ply_path=None):
        """
        If `container` (a `BaseContainer`) is provided, SimSPH will use the container's
        particle arrays as the source of truth. Otherwise it falls back to the original
        random-initialized behavior.
        """
        self.ps = container

        self.verbose = False
        # render params
        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_time = 0.0
        # get simulation params from config
        if (config != None):
            self.dim = 3
            self.frame_dt = config.get_cfg("timeStepSize") # 采用config中的时间步长
            self.sim_step_to_frame_ratio = config.get_cfg("numberOfStepsPerRenderUpdate")

            self.domain_start = np.array([0.0, 0.0, 0.0])
            self.domain_start = np.array(config.get_cfg("domainStart"))

            self.domain_end = np.array([1.0, 1.0, 1.0])
            self.domian_end = np.array(config.get_cfg("domainEnd"))
            ds = (self.domian_end - self.domain_start).astype(np.float32)
            self.domain_size = wp.vec3(ds[0], ds[1], ds[2])

            self.particle_radius = config.get_cfg("particleRadius")
            # self.smoothing_length = self.particle_radius     # 0.8
            self.particle_diameter = 2 * self.particle_radius
            self.support_radius = self.particle_radius * 4.0 # move from difftaichi version
            self.smoothing_length = self.support_radius
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
            self.dynamic_visc = 0.1 # 0.025
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
            self.grid_size = 2.0 * self.support_radius
            self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
            print("grid size: ", self.grid_num)
            self.padding = self.support_radius

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
        if self.ps is None:
            self.smoothing_length = 0.8
            self.width = 80.0
            self.height = 80.0
            self.length = 80.0
            self.stiffness = 20
            self.base_density = 1.0
            self.particle_mass = 0.01 * self.smoothing_length**3
            self.dt =0.01 * self.smoothing_length
            self.dynamic_visc = 0.025
            self.damping_coef = -0.95
            self.gravity = -0.1
            # original initialization
            self.particle_max_num = int(
                self.height * (self.width / 4.0) * (self.height / 4.0) / (self.smoothing_length**3)
            )
            # self.n = 10000
            # allocate arrays and initialize
            self.x = wp.empty(self.particle_max_num, dtype=wp.vec3, requires_grad=True)
            self.v = wp.zeros(self.particle_max_num, dtype=wp.vec3, requires_grad=True)
            self.rho = wp.zeros(self.particle_max_num, dtype=float, requires_grad=True)
            self.a = wp.zeros(self.particle_max_num, dtype=wp.vec3, requires_grad=True)
            print(f"Using demo init with {self.particle_max_num} particles, dh={self.smoothing_length}")
            wp.launch(
                kernel=initialize_particles,
                dim=self.particle_max_num,
                inputs=[self.x, self.smoothing_length, self.width, self.height, self.length],
            )

            grid_size = int(self.height / (self.smoothing_length))
            self.grid = wp.HashGrid(grid_size, grid_size, grid_size)
            # ensure volume array exists for export consistency
            # self.m_V0 = getattr(self, 'm_V0', 0.01 * self.smoothing_length**3)
            self.m_V = wp.array(np.full(self.particle_max_num, self.m_V0, dtype=np.float32), dtype=wp.float32)
        else:
            if ply_path:
                self.init_from_ply(ply_path)
            else:
                self.ti_to_warp()
            # 调试导出时使用，注意在ti_to_warp初始化n之后定义
            self.neibor_nums = wp.zeros(self.particle_max_num, dtype=wp.int32)
            self.pressure_forces = wp.zeros(self.particle_max_num, dtype=wp.vec3)
            self.viscous_forces = wp.zeros(self.particle_max_num, dtype=wp.vec3)

            self.initialize()
        # renderer
        # self.renderer = None
        # if stage_path:
        #     self.renderer = wp.render.UsdRenderer(stage_path)

    def ti_to_warp(self,):
            # use container values
            # self.n = int(self.ps.particle_num.to_numpy())
            self.fluid_particle_num = int(self.ps.fluid_particle_num)
            self.solid_particle_num = int(self.ps.solid_particle_num)
            self.particle_max_num = int(self.ps.particle_max_num)
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

        # Initialize Rigid Bodies from PS if available
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

    def step(self, t):
        self.time_step = t
        with wp.ScopedTimer("step"):
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
                    # compute density of points
                    wp.launch(
                        kernel=compute_density,
                        dim=self.particle_max_num,
                        inputs=[self.grid.id, self.x, self.rho, 
                                # self.density_normalization_no_mass, 
                                1.0, # cubic kernel don't need normalization
                                self.smoothing_length,
                                self.materialMarks, self.m_V, self.base_density],
                    )

                    wp.launch(
                        kernel=compute_pressure,
                        dim=self.particle_max_num,
                        inputs=[self.rho, self.pressure, self.materialMarks,
                                self.stiffness, self.exponent, self.base_density],
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
                            self.rbs
                        ],
                    )

                    # get new acceleration
                    wp.launch(
                        kernel=get_acceleration,
                        dim=self.particle_max_num,
                        inputs=[
                            self.grid.id,
                            self.x,
                            self.v,
                            self.rho,
                            self.pressure,
                            self.a,
                            self.stiffness,
                            self.exponent,
                            self.base_density,
                            self.gravity,
                            # poly6核函数相关参数
                            # self.pressure_normalization_no_mass,
                            # self.viscous_normalization_no_mass,
                            1.0,  # cubic kernel don't need normalization
                            self.dynamic_visc, # cubic kernel only use dynamic_visc
                            self.smoothing_length,
                            self.materialMarks,
                            self.m_V,
                            self.pressure_forces,
                            self.viscous_forces,
                            self.neibor_nums,
                            self.object_id,
                            self.rbs
                        ],
                    )
                    # self.print_rigid_info()
                    # apply bounds
                    # wp.launch(
                    #     kernel=apply_bounds,
                    #     dim=self.n,
                    #     inputs=[self.x, self.v, self.damping_coef, self.width, self.height, self.length,
                    #             self.materialMarks],
                    # )
                    wp.launch(
                        kernel=enforce_boundary_3D_warp,
                        dim=self.particle_max_num,
                        inputs=[self.x, self.v,
                                self.materialMarks,
                                self.domain_size,
                                self.padding,
                        ]
                    )

                    # kick
                    wp.launch(kernel=kick, dim=self.particle_max_num, inputs=[self.v, self.a, self.dt])

                    # drift
                    wp.launch(kernel=drift, dim=self.particle_max_num, inputs=[self.x, self.v, self.dt])

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
                    f"  id={i} mass={masses[i]:.6f} pos={pos[i]} vel={vel[i]} "
                    f"omega={omega[i]} quat={quat[i]} rest_cm={rest_cm[i]}"
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