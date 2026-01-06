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
        rigid_inertia: wp.array(dtype=wp.mat33)
        rigid_inertia0: wp.array(dtype=wp.mat33)
        rigid_inv_inertia: wp.array(dtype=wp.mat33)
        rigid_inv_mass: wp.array(dtype=wp.float32)
@wp.func
def is_dynamic_rigid_body(mtr: MaterialMarks, idx: int) -> bool:
    return mtr.material[idx] == MaterialType.SOLID and mtr.is_dynamic[idx] == 1
@wp.func
def is_static_rigid_body(mtr: MaterialMarks, idx: int) -> bool:
    return mtr.material[idx] == MaterialType.SOLID and (mtr.is_dynamic[idx] == 0)

@wp.kernel
def compute_static_boundary_volume(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    m_V : wp.array(dtype=wp.float32),
    density_normalization_no_mass: float, # constant term in poly6 kernel multi mass of particle
    smoothing_length: float,
    mtr : MaterialMarks
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid) # order threads by cell

    if is_static_rigid_body(mtr, i):
        x = particle_x[i]
        neighbors = wp.hash_grid_query(grid, x, smoothing_length)
        rho = cubic_kernel(wp.vec3(), smoothing_length)  # self-contribution
        # loop through neighbors to compute density
        for index in neighbors:
            if mtr.material[index] == MaterialType.SOLID:
                # compute distance
                distance = x - particle_x[index]
                # compute kernel derivative, the cube term in poly6 kernel
                rho += cubic_kernel(distance, smoothing_length)
        # add external potential
        #rho *= density_normalization_no_mass
        m_V[i] = 1.0 / rho * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly

@wp.kernel
def compute_moving_boundary_volume(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    m_V : wp.array(dtype=wp.float32),
    density_normalization_no_mass: float, # constant term in poly6 kernel multi mass of particle
    smoothing_length: float,
    mtr : MaterialMarks
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid) # order threads by cell

    if is_dynamic_rigid_body(mtr, i):
        x = particle_x[i]
        neighbors = wp.hash_grid_query(grid, x, smoothing_length)
        rho = cubic_kernel(wp.vec3(), smoothing_length)  # self-contribution
            # loop through neighbors to compute density
        for index in neighbors:
            if mtr.material[index] == MaterialType.SOLID:
                # compute distance
                distance = x - particle_x[index]
                # compute kernel derivative, the cube term in poly6 kernel
                rho += cubic_kernel(distance, smoothing_length)
        # add external potential
        # rho *= density_normalization_no_mass
        m_V[i] = 1.0 / rho * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly

#TODO: 实现 compute_rigid_rest_cm、compute_rigid_mass_info

@wp.kernel
def solve_rigid_body(
    bodies: RigidBodies,
    g: wp.vec3,
    dt: float,
    bodies_out: RigidBodies
):
    tid = wp.tid()

    f = bodies.rigid_force[tid] + g * bodies.rigid_mass[tid]
    v = bodies.rigid_v[tid] + dt * f / bodies.rigid_mass[tid]
    bodies_out.rigid_force[tid] = wp.vec3(0.0, 0.0, 0.0)

    bodies_out.rigid_x[tid] = bodies.rigid_x[tid] + dt * v

    I_inv = bodies.rigid_inv_inertia[tid]
    omega = bodies.rigid_omega[tid] + dt * (I_inv @ bodies.rigid_torque[tid])
    bodies_out.rigid_torque[tid] = wp.vec3(0.0, 0.0, 0.0)

    q = bodies.rigid_quaternion[tid]
    dq = 0.5 * wp.quat(omega[0], omega[1], omega[2], 0.0) * q
    q = q + dt * dq
    q = wp.normalize(q)
    bodies_out.rigid_quaternion[tid] = q

    R = wp.quat_to_matrix(q)
    I0 = bodies.rigid_inertia0[tid]
    I = R @ I0 @ wp.transpose(R)
    bodies_out.rigid_inertia[tid] = I
    bodies_out.rigid_inv_inertia[tid] = wp.inverse(I)

    bodies_out.rigid_v[tid] = v
    bodies_out.rigid_omega[tid] = omega

@wp.kernel
def solve_rigid_body_diff(
    rigid_x: wp.array(dtype=wp.vec3),
    rigid_v: wp.array(dtype=wp.vec3),
    rigid_force: wp.array(dtype=wp.vec3),
    rigid_mass: wp.array(dtype=wp.float32),
    rigid_quaternion: wp.array(dtype=wp.quat),
    rigid_omega: wp.array(dtype=wp.vec3),
    rigid_torque: wp.array(dtype=wp.vec3),
    rigid_inertia0: wp.array(dtype=wp.mat33),
    rigid_inv_inertia: wp.array(dtype=wp.mat33),
    
    g: wp.vec3,
    dt: float,
    
    rigid_x_out: wp.array(dtype=wp.vec3),
    rigid_v_out: wp.array(dtype=wp.vec3),
    rigid_force_out: wp.array(dtype=wp.vec3), #TODO: 检查是否有必要置0
    rigid_quaternion_out: wp.array(dtype=wp.quat),
    rigid_omega_out: wp.array(dtype=wp.vec3),
    rigid_torque_out: wp.array(dtype=wp.vec3),
    rigid_inertia_out: wp.array(dtype=wp.mat33),
    rigid_inv_inertia_out: wp.array(dtype=wp.mat33)
):
    tid = wp.tid()

    f = rigid_force[tid] + g * rigid_mass[tid]
    v = rigid_v[tid] + dt * f / rigid_mass[tid]
    rigid_force_out[tid] = wp.vec3(0.0, 0.0, 0.0)

    rigid_x_out[tid] = rigid_x[tid] + dt * v

    I_inv = rigid_inv_inertia[tid]
    omega = rigid_omega[tid] + dt * (I_inv @ rigid_torque[tid])
    rigid_torque_out[tid] = wp.vec3(0.0, 0.0, 0.0)

    q = rigid_quaternion[tid]
    dq = 0.5 * wp.quat(omega[0], omega[1], omega[2], 0.0) * q
    q = q + dt * dq
    q = wp.normalize(q)
    rigid_quaternion_out[tid] = q

    R = wp.quat_to_matrix(q)
    I0 = rigid_inertia0[tid]
    I = R @ I0 @ wp.transpose(R)
    rigid_inertia_out[tid] = I
    rigid_inv_inertia_out[tid] = wp.inverse(I)

    rigid_v_out[tid] = v
    rigid_omega_out[tid] = omega


@wp.kernel
def update_rigid_particle_info(
    particles_x: wp.array(dtype=wp.vec3),        
    particles_v: wp.array(dtype=wp.vec3),       
    particles_x0: wp.array(dtype=wp.vec3),       
    object_id: wp.array(dtype=int),            
    mtr: MaterialMarks,        
    bodies: RigidBodies,              
):
    tid = wp.tid()
    # update dynamic rigid body particle transforms
    if is_dynamic_rigid_body(mtr, tid):
        r = object_id[tid]

        # rest-space relative position (assumes rest orientation is identity)
        x_rel = particles_x0[tid] - bodies.rigid_rest_cm[r]
        R = wp.quat_to_matrix(bodies.rigid_quaternion[r])
        x_rel_world = R @ x_rel
        # position and velocity must use the SAME world-space lever arm to avoid artifacts
        particles_x[tid] = bodies.rigid_x[r] + x_rel_world
        particles_v[tid] = bodies.rigid_v[r] + wp.cross(bodies.rigid_omega[r], x_rel_world)
        # particles_v[tid] = bodies.rigid_v[r] + wp.cross(bodies.rigid_omega[r], x_rel)