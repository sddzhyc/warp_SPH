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
    density_normalization: float, # use density_normalization_with_rho
    smoothing_length: float,
    mtr : MaterialMarks,
    m_V: wp.array(dtype=float),
    base_density: float,
    particle_rho_out: wp.array(dtype=float)
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # get local particle variables
    x = particle_x[i]

    # init density with self-contribution
    # rho = wp.cast(0.0, wp.float32)
    rho = m_V[i] * cubic_kernel(wp.vec3(.0, .0, .0), smoothing_length)

    # particle contact
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)

    if mtr.material[i] == MaterialType.FLUID:
        # loop through neighbors to compute density
        for index in neighbors:
            # skip self to avoid double-counting the self-contribution
            if index == i:
                continue
            if wp.length(x - particle_x[index]) > smoothing_length:
                continue
            distance = x - particle_x[index]
            if mtr.material[index] == MaterialType.FLUID:
                rho += m_V[index] * cubic_kernel(distance, smoothing_length)
            elif mtr.material[index] == MaterialType.SOLID:
                rho += m_V[index] * cubic_kernel(distance, smoothing_length)
        # add external potential
        particle_rho_out[i] = density_normalization * base_density * rho
        # particle_rho[i] = base_density * rho
    # 密度下限设为base_density
    # particle_rho[i] = wp.max(particle_rho[i], base_density)

@wp.kernel
def compute_pressure(
    particle_rho: wp.array(dtype=float),
    mtr : MaterialMarks,
    stiffness: float,
    exponent : float,
    base_density: float,
    particle_p_out: wp.array(dtype=float)
):
    tid = wp.tid()
    if mtr.material[tid] == MaterialType.FLUID:
        # get local particle variables
        rho = particle_rho[tid]

        # clamp density to base_density to avoid invalid/too-small densities
        rho = wp.max(rho, base_density)
        particle_rho[tid] = rho
        # 采用Tait方程计算压强
        pressure = stiffness * (wp.pow(rho / base_density, exponent) - 1.0)
        # pressure = isotropic_exp * (rho - base_density)

        # store pressure
        particle_p_out[tid] = pressure

@wp.kernel
def compute_non_presure_forces(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    viscous_normalization: float,
    smoothing_length: float,
    mtr : MaterialMarks,
    m_V: wp.array(dtype=float),
    base_density: float,
    object_id: wp.array(dtype=wp.int32),
    rbs :RigidBodies,
    particle_viscous_force_out: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    
    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    
    if mtr.material[i] != MaterialType.FLUID:
        particle_viscous_force_out[i] = wp.vec3(0.0, 0.0, 0.0)
        return

    # get local particle variables
    x = particle_x[i]
    v = particle_v[i]
    
    viscous_force = wp.vec3(0.0, 0.0, 0.0)
    
    # particle contact
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)
    
    for index in neighbors:
        if index != i:
            d = wp.length(x - particle_x[index])
            if d < smoothing_length:
                # get neighbor velocity
                neighbor_v = particle_v[index]
                neighbor_rho = particle_rho[index]
                
                relative_position = particle_x[index] - x
                
                if mtr.material[index] == MaterialType.FLUID:
                    viscous_force += base_density * m_V[index] * diff_viscous_kernel_cubic(relative_position, v, neighbor_v, neighbor_rho, smoothing_length)
                # elif mtr.material[index] == MaterialType.SOLID:
                #     term = base_density * m_V[index] * diff_viscous_kernel_cubic(relative_position, v, neighbor_v, neighbor_rho, smoothing_length)
                #     viscous_force += term
                #     if is_dynamic_rigid_body(mtr, index):
                #         r_id = object_id[index]
                #         force = -viscous_normalization * term
                #         rbs.rigid_force[r_id] += force
                #         rbs.rigid_torque[r_id] += wp.cross(x - rbs.rigid_x[r_id], force)
                
    particle_viscous_force_out[i] = viscous_normalization * viscous_force

@wp.kernel
def get_acceleration(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    particle_p: wp.array(dtype=float),
    stiffness: float,
    exponent : float,
    base_density: float,
    gravity: float,
    pressure_normalization_no_mass: float,
    viscous_normalization: float,
    smoothing_length: float,
    mtr : MaterialMarks,
    m_V: wp.array(dtype=float),
    particle_pressure_force: wp.array(dtype=wp.vec3),
    particle_viscous_force: wp.array(dtype=wp.vec3),
    neibor_nums: wp.array(dtype=wp.int32),
    object_id: wp.array(dtype=wp.int32),
    particle_a_out: wp.array(dtype=wp.vec3),
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
    # viscous_force = wp.vec3()

    # particle contact
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)

    if mtr.material[i] == MaterialType.FLUID:
        count = wp.int32(0)
        # loop through neighbors to compute acceleration
        for index in neighbors:
            if index != i and wp.length(x - particle_x[index])  < smoothing_length:
                count += 1
                # get neighbor velocity
                # neighbor_v = particle_v[index]
                # get neighbor density and pressures
                neighbor_rho = particle_rho[index]
                # neighbor_pressure = stiffness * (wp.pow(neighbor_rho / base_density, exponent) - 1.0) # TODO: 考虑存储压强以节省计算
                neighbor_pressure = particle_p[index]
                # neighbor_pressure = isotropic_exp * (neighbor_rho - base_density) 

                # compute relative position
                relative_position = particle_x[index] - x
                if mtr.material[index] == MaterialType.FLUID:
                    # calculate pressure force
                    #  pressure_force += -base_density * m_V[index] * diff_pressure_kernel(
                    pressure_force += base_density * m_V[index] * diff_pressure_kernel_cubic(
                        relative_position, pressure, neighbor_pressure, rho, neighbor_rho, smoothing_length
                    )
                    # compute kernel derivative
                    # viscous_force += base_density * m_V[index] * diff_viscous_kernel_cubic(relative_position, v, neighbor_v, neighbor_rho, smoothing_length)
                # elif mtr.material[index] == MaterialType.SOLID:
                #     fp = base_density * m_V[index] * diff_pressure_kernel_cubic(
                #     # fp = -base_density * m_V[index] * diff_pressure_kernel(
                #         relative_position, pressure, pressure, rho, base_density, smoothing_length
                #     )
                #     pressure_force += fp
                #     if  is_dynamic_rigid_body(mtr, index):
                #         r_id = object_id[index]
                #         # convert contribution to a force compatible with DFSPH's convention
                #         force = - fp * rho * m_V[i]
                #         # force = -pressure_normalization_no_mass * fp * rho * m_V[i]
                #         rbs.rigid_force[r_id] += force
                #         rbs.rigid_torque[r_id] += wp.cross(x - rbs.rigid_x[r_id], force)

        # store neighbor count used for pressure computation
        neibor_nums[i] = wp.cast(count, wp.int32)
        # write per-particle pressure/viscous contributions for diagnostics
        #pressure_force = -pressure_force # TODO：cubic需要添加，而diff_pressure_kernel不需要添加（pressure_normalization_no_mass已负）

        # particle_viscous_force[i] = viscous_normalization * viscous_force
        # force = pressure_force + viscous_normalization * viscous_force
        pressure_force = pressure_force * pressure_normalization_no_mass
        particle_pressure_force[i] = pressure_force
        # add external potential
        particle_a_out[i] = pressure_force + particle_viscous_force[i] + wp.vec3(0.0, gravity, 0.0)
        # particle_a[i] = pressure_force / rho + particle_viscous_force[i] / rho + wp.vec3(0.0, gravity, 0.0) # 粘性力除以密度会导致粘性力过小！！


@wp.kernel
def compute_rigid_force_torque(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    particle_p: wp.array(dtype=float),
    base_density: float,
    pressure_normalization_no_mass: float,
    smoothing_length: float,
    mtr : MaterialMarks,
    m_V: wp.array(dtype=float),
    object_id: wp.array(dtype=wp.int32),
    rigid_x: wp.array(dtype=wp.vec3), # 刚体质心位置
    rigid_force: wp.array(dtype=wp.vec3),
    rigid_torque: wp.array(dtype=wp.vec3),
    particle_a_out: wp.array(dtype=wp.vec3),
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
    # viscous_force = wp.vec3()

    # particle contact
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)

    if mtr.material[i] == MaterialType.FLUID:
        count = wp.int32(0)
        # loop through neighbors to compute acceleration
        for index in neighbors:
            if index != i and wp.length(x - particle_x[index])  < smoothing_length:
                count += 1
                # get neighbor velocity
                # neighbor_v = particle_v[index]
                # get neighbor density and pressures
                # neighbor_rho = particle_rho[index]
                # # neighbor_pressure = stiffness * (wp.pow(neighbor_rho / base_density, exponent) - 1.0) # TODO: 考虑存储压强以节省计算
                # neighbor_pressure = particle_p[index] 
                # compute relative position
                relative_position = particle_x[index] - x
                if mtr.material[index] == MaterialType.SOLID:
                    fp = base_density * m_V[index] * diff_pressure_kernel_cubic(
                    # fp = -base_density * m_V[index] * diff_pressure_kernel(
                        relative_position, pressure, pressure, rho, base_density, smoothing_length
                    )
                    pressure_force += fp
                    if  is_dynamic_rigid_body(mtr, index):
                        r_id = object_id[index]
                        # convert contribution to a force compatible with DFSPH's convention
                        force = - fp * rho * m_V[i]
                        # force = -pressure_normalization_no_mass * fp * rho * m_V[i]
                        rigid_force[r_id] += force
                        rigid_torque[r_id] += wp.cross(x - rigid_x[r_id], force)

        # particle_viscous_force[i] = viscous_normalization * viscous_force
        # force = pressure_force + viscous_normalization * viscous_force
        pressure_force = pressure_force * pressure_normalization_no_mass
        # add external potential
        particle_a_out[i] += pressure_force

@wp.kernel
def apply_bounds(
    damping_coef: float,
    width: float,
    height: float,
    length: float,
    mtr : MaterialMarks,
    particle_x_out: wp.array(dtype=wp.vec3),
    particle_v_out: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()

    # get pos and velocity
    x = particle_x_out[tid]
    v = particle_v_out[tid]

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
    particle_x_out[tid] = x
    particle_v_out[tid] = v


@wp.kernel
def kick(particle_a: wp.array(dtype=wp.vec3), dt: float, particle_v: wp.array(dtype=wp.vec3), 
         particle_v_out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    v = particle_v[tid]
    particle_v_out[tid] = v + particle_a[tid] * dt


@wp.kernel
def drift(particle_x: wp.array(dtype=wp.vec3), particle_v: wp.array(dtype=wp.vec3), dt: float,
          particle_x_out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    x = particle_x[tid]
    particle_x_out[tid] = x + particle_v[tid] * dt


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


# refactored collision response function
@wp.func
def simulate_collisions_warp(particle_v: wp.array(dtype=wp.vec3), idx: int, n: wp.vec3):
    # Collision factor, assume roughly (1-c_f)*velocity loss after collision
    c_f = 0.5
    v = particle_v[idx]
    particle_v[idx] = v - (1.0 + c_f) * wp.dot(v, n) * n


@wp.kernel
def enforce_boundary_3D_warp(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    mtr : MaterialMarks,
    domain_size: wp.vec3,
    padding: float,
):
    tid = wp.tid()

    # only handle particles of the requested type that are dynamic
    if mtr.material[tid] != MaterialType.FLUID or not mtr.is_dynamic[tid]:
        return

    pos = particle_x[tid]
    collision_normal = wp.vec3(0.0, 0.0, 0.0)

    # x axis
    if pos[0] > domain_size[0] - padding:
        collision_normal = collision_normal + wp.vec3(1.0, 0.0, 0.0)
        pos = wp.vec3(domain_size[0] - padding, pos[1], pos[2])
    if pos[0] <= padding:
        collision_normal = collision_normal + wp.vec3(-1.0, 0.0, 0.0)
        pos = wp.vec3(padding, pos[1], pos[2])

    # y axis
    if pos[1] > domain_size[1] - padding:
        collision_normal = collision_normal + wp.vec3(0.0, 1.0, 0.0)
        pos = wp.vec3(pos[0], domain_size[1] - padding, pos[2])
    if pos[1] <= padding:
        collision_normal = collision_normal + wp.vec3(0.0, -1.0, 0.0)
        pos = wp.vec3(pos[0], padding, pos[2])

    # z axis
    if pos[2] > domain_size[2] - padding:
        collision_normal = collision_normal + wp.vec3(0.0, 0.0, 1.0)
        pos = wp.vec3(pos[0], pos[1], domain_size[2] - padding)
    if pos[2] <= padding:
        collision_normal = collision_normal + wp.vec3(0.0, 0.0, -1.0)
        pos = wp.vec3(pos[0], pos[1], padding)

    # write back position
    particle_x[tid] = pos

    # if collided, apply collision response
    cn_len = wp.length(collision_normal)
    if cn_len > 1e-6:
        n = collision_normal / cn_len
        simulate_collisions_warp(particle_v, tid, n)
