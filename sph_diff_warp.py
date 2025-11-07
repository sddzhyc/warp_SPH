# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Smoothed Particle Hydrodynamics
#
# Shows how to implement a SPH fluid simulation.
#
# Neighbors are found using the wp.HashGrid class, and
# wp.hash_grid_query(), wp.hash_grid_query_next() kernel methods.
#
# Reference Publication
# Matthias Müller, David Charypar, and Markus H. Gross.
# "Particle-based fluid simulation for interactive applications."
# Symposium on Computer animation. Vol. 2. 2003.
#
###########################################################################
import taichi as ti 

import numpy as np
import os

import warp as wp
from particle_system import ParticleSystem
# from particle_system_np import ParticleSystem

# import partio


@wp.func
def square(x: float):
    return x * x


@wp.func
def cube(x: float):
    return x * x * x


@wp.func
def fifth(x: float):
    return x * x * x * x * x


@wp.func
def density_kernel(xyz: wp.vec3, smoothing_length: float):
    # calculate distance
    distance = wp.dot(xyz, xyz)

    return wp.max(cube(square(smoothing_length) - distance), 0.0)


@wp.func
def diff_pressure_kernel(
    xyz: wp.vec3, pressure: float, neighbor_pressure: float,rho: float , neighbor_rho: float, smoothing_length: float
):
    # calculate distance
    distance = wp.sqrt(wp.dot(xyz, xyz))

    if distance < smoothing_length:
        # calculate terms of kernel
        term_1 = -xyz / distance # 单位距离向量
        term_2 = (neighbor_pressure + pressure) / (2.0 * neighbor_rho)
        # term_2 = neighbor_pressure / (neighbor_rho * neighbor_rho) + pressure / (rho * rho)
        term_3 = square(smoothing_length - distance)  # gradient of SPH kernel (grad W); TODO: use another kernel
        return term_1 * term_2 * term_3
    else:
        return wp.vec3()


@wp.func
def diff_viscous_kernel(xyz: wp.vec3, v: wp.vec3, neighbor_v: wp.vec3, neighbor_rho: float, smoothing_length: float):
    # calculate distance
    distance = wp.sqrt(wp.dot(xyz, xyz))

    # calculate terms of kernel
    if distance < smoothing_length:
        term_1 = (neighbor_v - v) / neighbor_rho
        term_2 = smoothing_length - distance
        return term_1 * term_2
    else:
        return wp.vec3()


@wp.kernel
def compute_density(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    density_normalization: float,
    smoothing_length: float,
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # get local particle variables
    x = particle_x[i]

    # store density
    rho = float(0.0)

    # particle contact
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)

    # loop through neighbors to compute density
    for index in neighbors:
        # compute distance
        distance = x - particle_x[index]

        # compute kernel derivative
        rho += density_kernel(distance, smoothing_length)

    # add external potential
    particle_rho[i] = density_normalization * rho

@wp.kernel
def get_acceleration(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    particle_a: wp.array(dtype=wp.vec3),
    isotropic_exp: float,
    base_density: float,
    gravity: float,
    pressure_normalization: float,
    viscous_normalization: float,
    smoothing_length: float,
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # get local particle variables
    x = particle_x[i]
    v = particle_v[i]
    rho = particle_rho[i]
    pressure = isotropic_exp * (rho - base_density)

    # store forces
    pressure_force = wp.vec3()
    viscous_force = wp.vec3()

    # particle contact
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)

    # loop through neighbors to compute acceleration
    for index in neighbors:
        if index != i:
            # get neighbor velocity
            neighbor_v = particle_v[index]

            # get neighbor density and pressures
            neighbor_rho = particle_rho[index]
            neighbor_pressure = isotropic_exp * (neighbor_rho - base_density)

            # compute relative position
            relative_position = particle_x[index] - x

            # calculate pressure force
            pressure_force += diff_pressure_kernel(
                relative_position, pressure, neighbor_pressure, rho, neighbor_rho, smoothing_length
            )

            # compute kernel derivative
            viscous_force += diff_viscous_kernel(relative_position, v, neighbor_v, neighbor_rho, smoothing_length)

    # sum all forces
    force = pressure_normalization * pressure_force + viscous_normalization * viscous_force

    # add external potential
    particle_a[i] = force / rho + wp.vec3(0.0, gravity, 0.0)


@wp.kernel
def apply_bounds(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    damping_coef: float,
    width: float,
    height: float,
    length: float,
):
    tid = wp.tid()

    # get pos and velocity
    x = particle_x[tid]
    v = particle_v[tid]

    # clamp x left
    if x[0] < 0.0:
        x = wp.vec3(0.0, x[1], x[2])
        v = wp.vec3(v[0] * damping_coef, v[1], v[2])

    # clamp x right
    if x[0] > width:
        x = wp.vec3(width, x[1], x[2])
        v = wp.vec3(v[0] * damping_coef, v[1], v[2])

    # clamp y bot
    if x[1] < 0.0:
        x = wp.vec3(x[0], 0.0, x[2])
        v = wp.vec3(v[0], v[1] * damping_coef, v[2])

    # clamp z left
    if x[2] < 0.0:
        x = wp.vec3(x[0], x[1], 0.0)
        v = wp.vec3(v[0], v[1], v[2] * damping_coef)

    # clamp z right
    if x[2] > length:
        x = wp.vec3(x[0], x[1], length)
        v = wp.vec3(v[0], v[1], v[2] * damping_coef)

    # apply clamps
    particle_x[tid] = x
    particle_v[tid] = v


@wp.kernel
def kick(particle_v: wp.array(dtype=wp.vec3), particle_a: wp.array(dtype=wp.vec3), dt: float):
    tid = wp.tid()
    v = particle_v[tid]
    particle_v[tid] = v + particle_a[tid] * dt


@wp.kernel
def drift(particle_x: wp.array(dtype=wp.vec3), particle_v: wp.array(dtype=wp.vec3), dt: float):
    tid = wp.tid()
    x = particle_x[tid]
    particle_x[tid] = x + particle_v[tid] * dt


@wp.kernel
def initialize_particles(
    particle_x: wp.array(dtype=wp.vec3), smoothing_length: float, width: float, height: float, length: float
):
    tid = wp.tid()

    # grid size
    # particle_diameter = 4 * smoothing_length
    nr_x = wp.int32(width / 4.0 / smoothing_length)
    nr_y = wp.int32(height / smoothing_length)
    nr_z = wp.int32(length / 4.0 / smoothing_length)

    # calculate particle position
    z = wp.float(tid % nr_z)
    y = wp.float((tid // nr_z) % nr_y)
    x = wp.float((tid // (nr_z * nr_y)) % nr_x)
    pos = smoothing_length * wp.vec3(x, y, z)

    # add small jitter
    state = wp.rand_init(123, tid)
    pos = pos + 0.001 * smoothing_length * wp.vec3(wp.randn(state), wp.randn(state), wp.randn(state))

    # set position
    particle_x[tid] = pos


class SimSPH:
    def from_ti_to_warp(self,):
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
            
            self.material = wp.array(self.ps.material.to_numpy()[: self.n].astype(np.int32), dtype=wp.int32)
            self.is_dynamic = wp.array(self.ps.is_dynamic.to_numpy()[: self.n].astype(np.int32), dtype=wp.int32)

            grid_size = int(self.ps.grid_num[0]) if hasattr(self.ps, 'grid_num') else max(1, int(self.height / (4.0 * self.smoothing_length)))
            self.grid = wp.HashGrid(grid_size, grid_size, grid_size)


    def move_np_to_warp(self,):
            # use container values
            self.n = int(self.ps.particle_num)
            # convert container numpy arrays (up to self.n) to Warp arrays
            # ensure shapes and types
            px = self.ps.x[: self.n].astype(np.float32)
            pv = self.ps.v[: self.n].astype(np.float32)
            prho = self.ps.density[: self.n].astype(np.float32)

            self.x = wp.array(px, dtype=wp.vec3)
            self.v = wp.array(pv, dtype=wp.vec3)
            self.rho = wp.array(prho, dtype=float)
            self.a = wp.array(np.zeros((self.n, 3), dtype=np.float32), dtype=wp.vec3)
            print(f"x shape: {self.x.shape}")

            grid_size = int(self.ps.grid_num[0]) if hasattr(self.ps, 'grid_num') else max(1, int(self.height / (4.0 * self.smoothing_length)))
            self.grid = wp.HashGrid(grid_size, grid_size, grid_size)

    def initialize(self, ps):

        # wp.launch(
        #         kernel=initialize_particles,
        #         dim=self.n,
        #         inputs=[self.x, self.smoothing_length, self.width, self.height, self.length],
        #     )
        # ps.initialize_particle_system()
        for r_obj_id in ps.object_id_rigid_body:
            self.compute_rigid_rest_cm(r_obj_id)
        # ps.initialize_rigid_info()
        for r_obj_id in ps.object_id_rigid_body:
            self.compute_rigid_mass_info(r_obj_id)
        # solver.compute_static_boundary_volume()
        # solver.compute_moving_boundary_volume()
        # self.compute_static_boundary_volume_numpy()
        # self.compute_moving_boundary_volume_numpy()

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
            self.particle_mass = 0.01 * self.smoothing_length**3
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
        self.viscous_normalization = (45.0 * self.dynamic_visc * self.particle_mass) / (
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
            self.initialize(container)
            self.from_ti_to_warp()
            grid_size = int(self.height / (4.0 * self.smoothing_length))
            self.grid = wp.HashGrid(grid_size, grid_size, grid_size)
        
                # Material
        
        # Used for fluid-solid distinction
        self.material_solid = 0
        self.material_fluid = 1
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
    def compute_com(self, object_id):
        sum_m = 0.0
        cm = np.array([0.0, 0.0, 0.0])
        for p_i in range(self.ps.particle_num):
            if self.ps.object_id[p_i] == object_id:
                mass = self.ps.m_V0 * self.ps.density[p_i]
                cm += mass * self.ps.x[p_i]
                sum_m += mass
        cm /= sum_m
        return cm

    def compute_rigid_rest_cm(self, object_id: int):
        self.ps.rigid_rest_cm[object_id] = self.compute_com(object_id)


    #@ti.kernel  #TODO:待内核化
    def compute_rigid_mass_info(self, object_id: int):
        sum_m = 0.0
        sum_inertia = np.zeros((3, 3), dtype=np.float32)
        n = int(self.ps.particle_num)
        for p_i in range(n):
            if self.ps.object_id[p_i] == object_id:
                mass = self.ps.m_V0 * self.ps.density[p_i]
                sum_m += mass
                r = self.ps.x[p_i] - self.ps.rigid_x[object_id]
                sum_inertia += mass * (r.dot(r) * np.identity(3, dtype=np.float32) - np.outer(r, r))
        self.ps.rigid_mass[object_id] = sum_m
        self.ps.rigid_inertia0[object_id] = sum_inertia
        self.ps.rigid_inertia[object_id] = sum_inertia
        self.ps.rigid_inv_mass[object_id] = 1.0 / sum_m
        self.ps.rigid_inv_inertia[object_id] = np.linalg.inv(sum_inertia)

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
                        inputs=[self.grid.id, self.x, self.rho, self.density_normalization, self.smoothing_length],
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
                            self.pressure_normalization,
                            self.viscous_normalization,
                            self.smoothing_length,
                        ],
                    )

                    # apply bounds
                    wp.launch(
                        kernel=apply_bounds,
                        dim=self.n,
                        inputs=[self.x, self.v, self.damping_coef, self.width, self.height, self.length],
                    )

                    # kick
                    wp.launch(kernel=kick, dim=self.n, inputs=[self.v, self.a, self.dt])

                    # drift
                    wp.launch(kernel=drift, dim=self.n, inputs=[self.x, self.v, self.dt])

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

    #TODO:待移植
    def compute_static_boundary_volume_numpy(self):
        """Compute boundary/rest volumes for static rigid particles using numpy on container.

        Writes results into `container.particle_rest_volumes` and `container.particle_masses`.
        Assumptions: `self.container` is a BaseContainer-like object with host numpy arrays:
          - particle_positions (N x dim)
          - particle_materials (N,)
          - particle_is_dynamic (N,)
          - particle_densities (N,)
          - particle_rest_volumes (N,)
          - particle_masses (N,)
          - V0 (scalar default volume)
          - dh (support radius) or use self.smoothing_length
        """
        c = self.ps
        n = int(c.particle_num)
        if n == 0:
            return

        pos = c.particle_positions[:n].astype(np.float64)
        materials = c.particle_materials[:n]
        is_dyn = c.particle_is_dynamic[:n]
        densities = c.particle_densities[:n]

        mat_rigid = c.material_rigid
        h = getattr(c, "dh", self.smoothing_length)
        dim = pos.shape[1]

        delta0 = float(self._cubic_kernel_numpy(0.0, h, dim=dim))

        # iterate static rigid particles
        mask_static = (materials == mat_rigid) & (is_dyn == 0)
        indices = np.nonzero(mask_static)[0]
        # for performance we vectorize per-particle but keep simple O(N^2) loop here
        for i in indices:
            # distances to all rigid particles
            diffs = pos - pos[i : i + 1]
            dists = np.linalg.norm(diffs, axis=1)
            neighbor_mask = (dists < h) & (materials == mat_rigid) & (np.arange(n) != i)
            if np.any(neighbor_mask):
                delta = delta0 + np.sum(self._cubic_kernel_numpy(dists[neighbor_mask], h, dim=dim))
            else:
                delta = delta0

            if delta <= 1e-12:
                rest_vol = float(getattr(c, "V0", 1e-6))
            else:
                rest_vol = 1.0 / delta * 3.0

            c.particle_rest_volumes[i] = rest_vol
            # update mass = rest_vol * density
            c.particle_masses[i] = rest_vol * float(densities[i])

    def compute_moving_boundary_volume_numpy(self):
        """Compute boundary/rest volumes for dynamic rigid particles using numpy on container.

        Similar assumptions to compute_static_boundary_volume_numpy.
        """
        if self.ps is None:
            raise RuntimeError("No container provided to compute_moving_boundary_volume_numpy")

        c = self.ps
        n = int(c.particle_num)
        if n == 0:
            return

        pos = c.particle_positions[:n].astype(np.float64)
        materials = c.particle_materials[:n]
        is_dyn = c.particle_is_dynamic[:n]
        densities = c.particle_densities[:n]

        mat_rigid = c.material_rigid
        h = getattr(c, "dh", self.smoothing_length)
        dim = pos.shape[1]

        delta0 = float(self._cubic_kernel_numpy(0.0, h, dim=dim))

        mask_dyn = (materials == mat_rigid) & (is_dyn == 1)
        indices = np.nonzero(mask_dyn)[0]
        for i in indices:
            diffs = pos - pos[i : i + 1]
            dists = np.linalg.norm(diffs, axis=1)
            neighbor_mask = (dists < h) & (materials == mat_rigid) & (np.arange(n) != i)
            if np.any(neighbor_mask):
                delta = delta0 + np.sum(self._cubic_kernel_numpy(dists[neighbor_mask], h, dim=dim))
            else:
                delta = delta0

            if delta <= 1e-12:
                rest_vol = float(getattr(c, "V0", 1e-6))
            else:
                rest_vol = 1.0 / delta * 3.0

            c.particle_rest_volumes[i] = rest_vol
            c.particle_masses[i] = rest_vol * float(densities[i])
    
    @wp.func
    def is_dynamic_rigid_body(self, p):
        return self.material[p] == self.material_solid and self.is_dynamic[p]
    # -----------------------------------------------------------------------------------


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
