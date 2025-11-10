from particle_system import ParticleSystem
from rigid_fluid_coupling import MaterialMarks, RigidBodies, compute_moving_boundary_volume, compute_static_boundary_volume, solve_rigid_body, update_rigid_particle_info
from sph_diff_warp import *

import taichi as ti 
import numpy as np
import warp as wp

class SimSPH:
    def ti_to_warp(self,):
            # use container values
            self.n = int(self.ps.particle_num.to_numpy())
            # allocate arrays and initialize
            self.x = wp.empty(self.n, dtype=wp.vec3, requires_grad=True)
            self.v = wp.zeros(self.n, dtype=wp.vec3, requires_grad=True)
            # self.rho = wp.zeros(self.n, dtype=float, requires_grad=True)
            self.a = wp.zeros(self.n, dtype=wp.vec3, requires_grad=True)
            #TODO: transfer other arrays such as material properties
            px = self.ps.x.to_numpy()[: self.n].astype(np.float32)
            # pv = self.ps.v.to_numpy()[: self.n].astype(np.float32)
            prho = self.ps.density.to_numpy()[: self.n].astype(np.float32)
            self.x = wp.array(px, dtype=wp.vec3)
            # self.v = wp.array(pv, dtype=wp.vec3)
            self.rho = wp.array(prho, dtype=float)
            print(f"n: {self.n}, x shape: {self.x.shape}, rho[1] = {prho[1]}")

            self.materialMarks = MaterialMarks()
            self.materialMarks.material = wp.array(self.ps.material.to_numpy()[: self.n].astype(np.int32), dtype=wp.int32)
            self.materialMarks.is_dynamic = wp.array(self.ps.is_dynamic.to_numpy()[: self.n].astype(np.int32), dtype=wp.int32)
            # rigid related info
            self.num_rigid_bodies = self.ps.num_rigid_bodies
            self.num_objects = self.ps.num_objects
            self.dim = self.ps.dim
            
            self.object_id = wp.array(self.ps.object_id.to_numpy()[: self.n].astype(np.int32), dtype=wp.int32) # Originally particle_max_num
            px_0 = self.ps.x_0.to_numpy()[: self.n].astype(np.float32)
            self.x_0 =  wp.array(px_0, dtype=wp.vec3)
            self.rbs = RigidBodies()
            # map Taichi fields into Warp RigidBodies arrays
            n_obj = int(self.num_objects)
            self.rbs.rigid_rest_cm = wp.array(self.ps.rigid_rest_cm.to_numpy()[:n_obj].astype(np.float32), dtype=wp.vec3)
            self.rbs.rigid_x       = wp.array(self.ps.rigid_x.to_numpy()[:n_obj].astype(np.float32), dtype=wp.vec3)
            self.rbs.rigid_v0      = wp.array(self.ps.rigid_v0.to_numpy()[:n_obj].astype(np.float32), dtype=wp.vec3)
            self.rbs.rigid_v       = wp.array(self.ps.rigid_v.to_numpy()[:n_obj].astype(np.float32), dtype=wp.vec3)
            self.rbs.rigid_force   = wp.array(self.ps.rigid_force.to_numpy()[:n_obj].astype(np.float32), dtype=wp.vec3)
            self.rbs.rigid_torque  = wp.array(self.ps.rigid_torque.to_numpy()[:n_obj].astype(np.float32), dtype=wp.vec3)
            # omegas (3-components)
            self.rbs.rigid_omega  = wp.array(self.ps.rigid_omega.to_numpy()[:n_obj].astype(np.float32), dtype=wp.vec3)
            self.rbs.rigid_omega0 = wp.array(self.ps.rigid_omega0.to_numpy()[:n_obj].astype(np.float32), dtype=wp.vec3)
            # quaternions (shape: n_obj x 4) -> Warp quat
            q_np = self.ps.rigid_quaternion.to_numpy()[:n_obj].astype(np.float32)
            self.rbs.rigid_quaternion = wp.array(q_np, dtype=wp.quat)
            # scalar masses
            self.rbs.rigid_mass     = wp.array(self.ps.rigid_mass.to_numpy()[:n_obj].astype(np.float32), dtype=float)
            self.rbs.rigid_inv_mass = wp.array(self.ps.rigid_inv_mass.to_numpy()[:n_obj].astype(np.float32), dtype=float)
            # inertia matrices (n_obj x 3 x 3)
            self.rbs.rigid_inertia     = wp.array(self.ps.rigid_inertia.to_numpy()[:n_obj].astype(np.float32), dtype=wp.mat33)
            self.rbs.rigid_inertia0    = wp.array(self.ps.rigid_inertia0.to_numpy()[:n_obj].astype(np.float32), dtype=wp.mat33)
            self.rbs.rigid_inv_inertia = wp.array(self.ps.rigid_inv_inertia.to_numpy()[:n_obj].astype(np.float32), dtype=wp.mat33)
            self.print_rigid_info()
            grid_size = int(self.ps.grid_num[0]) if hasattr(self.ps, 'grid_num') else max(1, int(self.height / (4.0 * self.smoothing_length)))
            self.grid = wp.HashGrid(grid_size, grid_size, grid_size)
    
    def initialize(self, ps):
        self.m_V = wp.array(np.full(self.n, self.m_V0, dtype=np.float32), dtype=wp.float32)
        
        # ps.initialize_particle_system()
        
        wp.launch(
            kernel=compute_static_boundary_volume,
            dim=self.n,
            inputs=[self.grid.id, self.x, self.m_V, self.density_normalization, self.smoothing_length,
                    self.materialMarks],
        )
        wp.launch(
            kernel=compute_moving_boundary_volume,
            dim=self.n,
            inputs=[self.grid.id, self.x, self.m_V, self.density_normalization, self.smoothing_length,
                    self.materialMarks],
        )


    def __init__(self,config = None, container: ParticleSystem = None, stage_path="example_sph.usd"):
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
            self.particle_radius = config.get_cfg("particleRadius")
            self.smoothing_length = 2.1 * self.particle_radius     # 0.8
            self.width = config.get_cfg("domainEnd")[1] # 80.0
            self.height = config.get_cfg("domainEnd")[2] # 80.0
            self.length = config.get_cfg("domainEnd")[0] # 80.0
            self.isotropic_exp = config.get_cfg("stiffness") # 20
            self.base_density = config.get_cfg("density0")   # 1.0
            # self.m_V0 = self.ps.m_V0 #  0.8 * self.particle_diameter ** self.dim
            self.m_V0 = 0.01 * self.smoothing_length**3 # 修改为设定体积而非质量
            # self.particle_mass = 0.01 * self.smoothing_length**3  # 为什么原example采用0.01?
            self.particle_mass = self.m_V0 * self.base_density # 设置后粒子不稳定？
            self.dt = config.get_cfg("timeStepSize")    # 0.01 * self.smoothing_length
            self.dynamic_visc = 0.025
            self.damping_coef = -0.95
            self.gravity = config.get_cfg("gravitation")[1]  # -0.1
        else:
            self.smoothing_length = 0.8
            self.width = 80.0
            self.height = 80.0
            self.length = 80.0
            self.isotropic_exp = 20
            self.base_density = 1.0
            self.particle_mass = 0.01 * self.smoothing_length**3
            self.dt =0.01 * self.smoothing_length
            self.dynamic_visc = 0.025
            self.damping_coef = -0.95
            self.gravity = -0.1

        self.time_step = 0.0
        # recompute constants
        self.density_normalization = (315.0 * self.particle_mass) / (
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
        self.sim_step_to_frame_ratio = int(32 / self.smoothing_length)
        if self.ps is None:
            # original initialization
            self.n = int(
                self.height * (self.width / 4.0) * (self.height / 4.0) / (self.smoothing_length**3)
            )
            # self.n = 10000
            # allocate arrays and initialize
            self.x = wp.empty(self.n, dtype=wp.vec3, requires_grad=True)
            self.v = wp.zeros(self.n, dtype=wp.vec3, requires_grad=True)
            self.rho = wp.zeros(self.n, dtype=float, requires_grad=True)
            self.a = wp.zeros(self.n, dtype=wp.vec3, requires_grad=True)
            print(f"Using demo init with {self.n} particles, dh={self.smoothing_length}")
            wp.launch(
                kernel=initialize_particles,
                dim=self.n,
                inputs=[self.x, self.smoothing_length, self.width, self.height, self.length],
            )

            grid_size = int(self.height / (4.0 * self.smoothing_length))
            self.grid = wp.HashGrid(grid_size, grid_size, grid_size)
        else:
            self.ti_to_warp()
            self.initialize(container)
            grid_size = int(self.height / (4.0 * self.smoothing_length))
            self.grid = wp.HashGrid(grid_size, grid_size, grid_size)

            # self.rbs = RigidBodies()
            # self.rbs.rigid_rest_cm   = wp.zeros(n_obj, dtype=wp.vec3)
            # self.rbs.rigid_x         = wp.zeros(n_obj, dtype=wp.vec3)
            # self.rbs.rigid_v0        = wp.zeros(n_obj, dtype=wp.vec3)
            # self.rbs.rigid_v         = wp.zeros(n_obj, dtype=wp.vec3)
            # self.rbs.rigid_force     = wp.zeros(n_obj, dtype=wp.vec3)
            # self.rbs.rigid_torque    = wp.zeros(n_obj, dtype=wp.vec3)

            # self.rbs.rigid_omega     = wp.zeros(n_obj, dtype=wp.vec3)
            # self.rbs.rigid_omega0    = wp.zeros(n_obj, dtype=wp.vec3)

            # self.rbs.rigid_mass      = wp.zeros(n_obj, dtype=float)
            # self.rbs.rigid_inv_mass  = wp.zeros(n_obj, dtype=float)

            # self.rbs.rigid_quaternion = wp.zeros(n_obj, dtype=wp.quat)

            # self.rbs.rigid_inertia       = wp.zeros(n_obj, dtype=wp.mat33)
            # self.rbs.rigid_inertia0      = wp.zeros(n_obj, dtype=wp.mat33)
            # self.rbs.rigid_inv_inertia   = wp.zeros(n_obj, dtype=wp.mat33)

        # renderer
        # self.renderer = None
        # if stage_path:
        #     self.renderer = wp.render.UsdRenderer(stage_path)

    """     
    def partio_export(self):
        particleSet=partio.create()
        X=particleSet.addAttribute("position",partio.VECTOR,3)
        V=particleSet.addAttribute("velocity",partio.VECTOR,3)
        RHO = particleSet.addAttribute("rho",partio.FLOAT,1)
        id = particleSet.addAttribute("id",partio.INT,1)
        particleSet.addParticles(self.n)
        # 一次性取出 numpy 数组，wp.array不允许索引元素
        xs = self.x.numpy()
        vs = self.v.numpy()
        rhos = self.rho.numpy()
        for i in range(self.n):
            particleSet.set(id, i, (int(i),))
            particleSet.set(X, i, (float(xs[i][0]), float(xs[i][1]), float(xs[i][2])))
            particleSet.set(V, i, (float(vs[i][0]), float(vs[i][1]), float(vs[i][2])))
            particleSet.set(RHO, i, (float(rhos[i]),))

        EXPORT_PATH = "pario_export"
        # TODO: 如果路径不存在则创建路径
        os.makedirs(EXPORT_PATH, exist_ok=True)
        partio.write(f"{EXPORT_PATH}/sph_{self.time_step}.bgeo",particleSet) # write uncompressed
        #partio.write("circle.bgeo",particleSet,True) # write compressed
        #partio.write("circle.bgeo.gz",particleSet) # write compressed 
    """
    def step(self, t):
        self.time_step = t
        with wp.ScopedTimer("step"):
            for _ in range(self.sim_step_to_frame_ratio):
                with wp.ScopedTimer("grid build", active=self.verbose):
                    # build grid
                    self.grid.build(self.x, self.smoothing_length)

                with wp.ScopedTimer("forces", active=self.verbose):
                    # compute density of points
                    wp.launch(
                        kernel=compute_density,
                        dim=self.n,
                        inputs=[self.grid.id, self.x, self.rho, self.density_normalization, self.smoothing_length,
                                self.materialMarks],
                    )

                    # get new acceleration
                    wp.launch(
                        kernel=get_acceleration,
                        dim=self.n,
                        inputs=[
                            self.grid.id,
                            self.x,
                            self.v,
                            self.rho,
                            self.a,
                            self.isotropic_exp,
                            self.base_density,
                            self.gravity,
                            self.pressure_normalization_no_mass,
                            self.viscous_normalization_no_mass,
                            self.smoothing_length,
                            self.materialMarks,
                            self.m_V,
                            self.object_id,
                            self.rbs
                        ],
                    )

                    # apply bounds
                    wp.launch(
                        kernel=apply_bounds,
                        dim=self.n,
                        inputs=[self.x, self.v, self.damping_coef, self.width, self.height, self.length,
                                self.materialMarks],
                    )

                    # kick
                    wp.launch(kernel=kick, dim=self.n, inputs=[self.v, self.a, self.dt])

                    # drift
                    wp.launch(kernel=drift, dim=self.n, inputs=[self.x, self.v, self.dt])

                    g = wp.vec3(0.0, self.gravity, 0.0)

                    wp.launch(kernel=solve_rigid_body, dim=self.num_objects, inputs=[self.rbs, g, self.dt])
                    # wp.launch(kernel=solve_rigid_body, dim=self.num_rigid_bodies, inputs=[self.rbs, g, self.dt]) # 该实现有问题
                    self.print_rigid_info()
                    wp.launch(
                        kernel=update_rigid_particle_info,
                        dim=self.n,
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

    # ----------------- NumPy host implementations for boundary volumes -----------------
    @staticmethod
    def _cubic_kernel_numpy(r, h, dim=3):
        """Vectorized cubic spline kernel (matching Taichi version) for numpy arrays.

        r: scalar or numpy array of distances (not squared)
        h: smoothing length
        dim: 1/2/3
        returns: kernel value(s)
        """
        r = np.array(r, copy=False)
        q = r / h
        # normalization constant k
        if dim == 1:
            k = 4.0 / 3.0
        elif dim == 2:
            k = 40.0 / 7.0 / np.pi
        else:
            k = 8.0 / np.pi
        k = k / (h ** dim)

        res = np.zeros_like(q, dtype=np.float64)
        mask1 = q <= 1.0
        if np.any(mask1):
            q1 = q[mask1]
            mask2 = q1 <= 0.5
            if np.any(mask2):
                q2 = q1[mask2]
                res_mask2 = k * (6.0 * q2 ** 3 - 6.0 * q2 ** 2 + 1.0)
                res[mask1.nonzero()[0][mask2]] = res_mask2
            if np.any(~mask2):
                q3 = q1[~mask2]
                res_mask3 = k * 2.0 * np.power(1.0 - q3, 3.0)
                res[mask1.nonzero()[0][~mask2]] = res_mask3

        # if input was scalar, return scalar
        if res.shape == ():
            return float(res)
        return res

    def print_rigid_info(self):
        if self.num_rigid_bodies > 0:
            masses = self.rbs.rigid_mass.numpy()
            pos = self.rbs.rigid_x.numpy()
            vel = self.rbs.rigid_v.numpy()
            omega = self.rbs.rigid_omega.numpy()
            quat = self.rbs.rigid_quaternion.numpy()
            rest_cm = self.rbs.rigid_rest_cm.numpy()

            print(f"[rbs] num={self.num_rigid_bodies}")
            for i in range(self.num_objects):
                print(
                    f"  id={i} mass={masses[i]:.6f} pos={pos[i]} vel={vel[i]} "
                    f"omega={omega[i]} quat={quat[i]} rest_cm={rest_cm[i]}"
                )

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

            np_pos = example.x.numpy()
            # print(container.object_collection)
            writer = ti.tools.PLYWriter(num_vertices=np_pos.shape[0])
            writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
            writer.export_frame_ascii(cnt_ply, series_prefix.format(0))
            cnt_ply+=1
            # example.partio_export()