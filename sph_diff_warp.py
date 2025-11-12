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
import warp as wp

from rigid_fluid_coupling import MaterialMarks, MaterialType, RigidBodies, is_dynamic_rigid_body
# from particle_system_np import ParticleSystem
from kernel_func import *
# import partio

@wp.func
def diff_pressure_kernel(
    xyz: wp.vec3, pressure: float, neighbor_pressure: float, rho: float , neighbor_rho: float, smoothing_length: float
):
    # calculate distance
    distance = wp.sqrt(wp.dot(xyz, xyz))

    if distance < smoothing_length: # 默认smoothing_length即支持半径？
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
    density_normalization: float, # use density_normalization_with_rho
    smoothing_length: float,
    mtr : MaterialMarks,
    m_V: wp.array(dtype=float),
    base_density: float
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

    if mtr.material[i] == MaterialType.FLUID:
        # loop through neighbors to compute density
        for index in neighbors:
            if mtr.material[index] == MaterialType.FLUID:
                # compute distance
                distance = x - particle_x[index]
                # compute kernel derivative, the cube term in poly6 kernel
                rho += m_V[index] * density_kernel(distance, smoothing_length)
            elif mtr.material[index] == MaterialType.SOLID:
                # compute distance
                distance = x - particle_x[index]
                # compute kernel derivative, the cube term in poly6 kernel
                rho += m_V[index] * density_kernel(distance, smoothing_length)
        # add external potential
        particle_rho[i] = density_normalization * base_density * rho
    elif mtr.material[i] == MaterialType.SOLID:
        pass

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
    pressure_normalization_no_mass: float,
    viscous_normalization: float,
    smoothing_length: float,
    mtr : MaterialMarks,
    m_V: wp.array(dtype=float),
    object_id: wp.array(dtype=wp.int32),
    rbs :RigidBodies
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

    if mtr.material[i] == MaterialType.FLUID:
        # loop through neighbors to compute acceleration
        for index in neighbors:
            if index != i:
                # get neighbor velocity
                neighbor_v = particle_v[index]

                # get neighbor density and pressures
                neighbor_rho = particle_rho[index]
                neighbor_pressure = isotropic_exp * (neighbor_rho - base_density) # TODO: 更新EOS方程形式

                # compute relative position
                relative_position = particle_x[index] - x
                if mtr.material[index] == MaterialType.FLUID:
                    # calculate pressure force
                    pressure_force += base_density * m_V[index] * diff_pressure_kernel(
                        relative_position, pressure, neighbor_pressure, rho, neighbor_rho, smoothing_length
                    )
                    # compute kernel derivative
                    viscous_force += base_density * m_V[index] * diff_viscous_kernel(relative_position, v, neighbor_v, neighbor_rho, smoothing_length)
                elif mtr.material[index] == MaterialType.SOLID:
                    fp = base_density * m_V[index] * diff_pressure_kernel(
                        relative_position, pressure, pressure, rho, base_density, smoothing_length
                    )
                    pressure_force += fp
                    if  is_dynamic_rigid_body(mtr, index):
                        # debug: print fp for a small subset to avoid flooding output
                        # if fp != wp.vec3():
                        #     wp.printf("fp=(%f, %f, %f), i=%d, neighbor=%d\n", fp[0], fp[1], fp[2], i, index)
                        # keep particle-level acceleration update (existing behavior) 疑似shape matching的写法？
                        # TODO: 为什么多乘了一个self.density_0？
                        # self.ps.acceleration[p_j] += -f_p * self.density_0 / self.ps.density[p_j]
                        # also aggregate force/torque to the rigid body (Akinci2012 style)
                        r_id = object_id[index]
                        # convert contribution to a force compatible with DFSPH's convention
                        force = -fp * rho * m_V[i]
                        rbs.rigid_force[r_id] += force
                        rbs.rigid_torque[r_id] += wp.cross(x - rbs.rigid_x[r_id], force)

        # sum all forces
        force = pressure_normalization_no_mass * pressure_force + viscous_normalization * viscous_force

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
    mtr : MaterialMarks
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