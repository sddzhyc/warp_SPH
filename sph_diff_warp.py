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

from kernel_func import diff_pressure_kernel, diff_viscous_kernel
from rigid_fluid_coupling import MaterialMarks, MaterialType, RigidBodies, is_dynamic_rigid_body
# from particle_system_np import ParticleSystem
from kernel_func import *

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

    # init density with self-contribution
    rho = m_V[i] * cubic_kernel(wp.vec3(.0, .0, .0), smoothing_length)

    # particle contact
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)

    if mtr.material[i] == MaterialType.FLUID:
        # loop through neighbors to compute density
        for index in neighbors:
            if mtr.material[index] == MaterialType.FLUID:
                # compute distance
                distance = x - particle_x[index]
                rho += m_V[index] * cubic_kernel(distance, smoothing_length)
            elif mtr.material[index] == MaterialType.SOLID:
                distance = x - particle_x[index]
                rho += m_V[index] * cubic_kernel(distance, smoothing_length)
        # add external potential
        # particle_rho[i] = density_normalization * base_density * rho
        particle_rho[i] = base_density * rho
    # 密度下限设为base_density
    # particle_rho[i] = wp.max(particle_rho[i], base_density)

@wp.kernel
def compute_pressure(
    particle_rho: wp.array(dtype=float),
    particle_p: wp.array(dtype=float),
    stiffness: float,
    exponent : float,
    base_density: float
):
    tid = wp.tid()

    # get local particle variables
    rho = particle_rho[tid]

    # 采用Tait方程计算压强
    pressure = stiffness * (wp.pow(rho / base_density, exponent) - 1.0)
    # pressure = isotropic_exp * (rho - base_density)

    # store pressure
    particle_p[tid] = pressure

@wp.kernel
def get_acceleration(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    particle_p: wp.array(dtype=float),
    particle_a: wp.array(dtype=wp.vec3),
    stiffness: float,
    exponent : float,
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
    # 采用新的EOS公式计算压强 
    #pressure = stiffness * (wp.pow(rho / base_density, exponent) - 1.0)
    pressure = particle_p[i]
    # pressure = isotropic_exp * (rho - base_density)

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
                # neighbor_pressure = stiffness * (wp.pow(neighbor_rho / base_density, exponent) - 1.0) # TODO: 考虑存储压强以节省计算
                neighbor_pressure = particle_p[index]
                # neighbor_pressure = isotropic_exp * (neighbor_rho - base_density) 

                # compute relative position
                relative_position = particle_x[index] - x
                if mtr.material[index] == MaterialType.FLUID:
                    # calculate pressure force
                    pressure_force += base_density * m_V[index] * diff_pressure_kernel_cubic(
                        relative_position, pressure, neighbor_pressure, rho, neighbor_rho, smoothing_length
                    )
                    # compute kernel derivative
                    viscous_force += base_density * m_V[index] * diff_viscous_kernel_cubic(relative_position, v, neighbor_v, neighbor_rho, smoothing_length)
                elif mtr.material[index] == MaterialType.SOLID:
                    fp = base_density * m_V[index] * diff_pressure_kernel_cubic(
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
                        force = - fp * rho * m_V[i]
                        rbs.rigid_force[r_id] += force
                        rbs.rigid_torque[r_id] += wp.cross(x - rbs.rigid_x[r_id], force)

        # sum all forces
        force = pressure_force + viscous_normalization * viscous_force

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