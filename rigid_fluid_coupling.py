import warp as wp
from enum import IntEnum
from kernel_func import *

# Used for fluid-solid distinction


class MaterialType(IntEnum):
    SOLID = 0
    FLUID = 1

@wp.struct
class MaterialMarks():
    # store material id per particle (int) and dynamic flag (int)
    material: wp.array(dtype=int)
    is_dynamic: wp.array(dtype=int)


@wp.struct
class RigidBodies():
        rigid_rest_cm: wp.array(dtype=wp.vec3)
        rigid_x: wp.array(dtype=wp.vec3)
        rigid_v0: wp.array(dtype=wp.vec3)
        rigid_v: wp.array(dtype=wp.vec3)
        rigid_quaternion: wp.array(dtype=wp.quat)
        rigid_omega: wp.array(dtype=wp.vec3)
        rigid_omega0: wp.array(dtype=wp.vec3)
        rigid_force: wp.array(dtype=wp.vec3)
        rigid_torque: wp.array(dtype=wp.vec3)
        rigid_mass: wp.array(dtype=wp.float32)
        # inertia matrices are flattened to 9 floats per body (row-major)
        rigid_inertia: wp.array(dtype=wp.mat33)
        rigid_inertia0: wp.array(dtype=wp.mat33)
        rigid_inv_inertia: wp.array(dtype=wp.mat33)
        rigid_inv_mass: wp.array(dtype=wp.float32)
@wp.func
def is_dynamic_rigid_body(mtr: MaterialMarks, idx: int) -> bool:
    return mtr.material[idx] == MaterialType.SOLID and mtr.is_dynamic[idx] == 1
@wp.func
def is_static_rigid_body(mtr: MaterialMarks, idx: int) -> bool:
    return mtr.material[idx] == MaterialType.SOLID and (not mtr.is_dynamic[idx])

@wp.kernel
def compute_static_boundary_volume(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    m_V : wp.array(dtype=wp.float32),
    density_normalization: float, # constant term in poly6 kernel multi mass of particle
    smoothing_length: float,
    mtr : MaterialMarks
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid) # order threads by cell

    if is_static_rigid_body(mtr, i):
        x = particle_x[i]
        neighbors = wp.hash_grid_query(grid, x, smoothing_length)
        rho = float(0.0)
        if mtr.material[i] == MaterialType.FLUID:
            # loop through neighbors to compute density
            for index in neighbors:
                if mtr.material[index] == MaterialType.SOLID:
                    # compute distance
                    distance = x - particle_x[index]
                    # compute kernel derivative, the cube term in poly6 kernel
                    rho += density_kernel(distance, smoothing_length)
            # add external potential
            rho *= density_normalization
            m_V[i] = 1.0 / rho * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly


@wp.kernel
def compute_moving_boundary_volume(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    m_V : wp.array(dtype=wp.float32),
    density_normalization: float, # constant term in poly6 kernel multi mass of particle
    smoothing_length: float,
    mtr : MaterialMarks
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid) # order threads by cell

    if is_dynamic_rigid_body(mtr, i):
        x = particle_x[i]
        neighbors = wp.hash_grid_query(grid, x, smoothing_length)
        rho = float(0.0)
        if mtr.material[i] == MaterialType.FLUID:
            # loop through neighbors to compute density
            for index in neighbors:
                if mtr.material[index] == MaterialType.SOLID:
                    # compute distance
                    distance = x - particle_x[index]
                    # compute kernel derivative, the cube term in poly6 kernel
                    rho += density_kernel(distance, smoothing_length)
            # add external potential
            rho *= density_normalization
            m_V[i] = 1.0 / rho * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly

@wp.kernel
def solve_rigid_body(
    bodies: RigidBodies,
    g: wp.vec3,
    dt: float,
):
    tid = wp.tid()

    f = bodies.rigid_force[tid] + g * bodies.rigid_mass[tid]
    v = bodies.rigid_v[tid] + dt * f / bodies.rigid_mass[tid]
    bodies.rigid_force[tid] = wp.vec3(0.0, 0.0, 0.0)

    bodies.rigid_x[tid] += dt * v

    I_inv = bodies.rigid_inv_inertia[tid]
    omega = bodies.rigid_omega[tid] + dt * (I_inv @ bodies.rigid_torque[tid])
    bodies.rigid_torque[tid] = wp.vec3(0.0, 0.0, 0.0)

    q = bodies.rigid_quaternion[tid]
    dq = 0.5 * wp.quat(omega[0], omega[1], omega[2], 0.0) * q
    q = q + dt * dq
    q = wp.normalize(q)
    bodies.rigid_quaternion[tid] = q

    R = wp.quat_to_matrix(q)
    I0 = bodies.rigid_inertia0[tid]
    I = R @ I0 @ wp.transpose(R)
    bodies.rigid_inertia[tid] = I
    bodies.rigid_inv_inertia[tid] = wp.inverse(I)

    bodies.rigid_v[tid] = v
    bodies.rigid_omega[tid] = omega


@wp.kernel
def update_rigid_particle_info(
    particles_x: wp.array(dtype=wp.vec3),        
    particles_v: wp.array(dtype=wp.vec3),       
    particles_x0: wp.array(dtype=wp.vec3),       
    object_id: wp.array(dtype=int),            
    mtr: MaterialMarks,        
    bodies: RigidBodies                
):
    tid = wp.tid()

    if is_dynamic_rigid_body(mtr, tid) == 1:
        r = object_id[tid]

        x_rel = particles_x0[tid] - bodies.rigid_rest_cm[r]
        R = wp.quat_to_matrix(bodies.rigid_quaternion[r])
        
        particles_x[tid] = bodies.rigid_x[r] + R @ x_rel
        particles_v[tid] = bodies.rigid_v[r] + wp.cross(bodies.rigid_omega[r], x_rel)
