import warp as wp
from enum import IntEnum
from kernel_func import *
from norm_grad_utils import norm_grad_quat, norm_grad_vec3

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

@wp.kernel
def compute_rigid_cm_mass_kernel(
    object_id: wp.array(dtype=int),
    x: wp.array(dtype=wp.vec3),
    m_V: wp.array(dtype=float),
    rho: wp.array(dtype=float),
    rigid_mass: wp.array(dtype=float),
    rigid_rest_cm: wp.array(dtype=wp.vec3),
    m_V0: float
):
    tid = wp.tid()
    obj_id = object_id[tid]
    
    vol = m_V[tid]
    if vol == 0.0:
        vol = m_V0

    mass = m_V0 * rho[tid]
    wp.atomic_add(rigid_mass, obj_id, mass)
    
    pos = x[tid]
    pos_mass = pos * mass
    
    wp.atomic_add(rigid_rest_cm, obj_id, pos_mass)

@wp.kernel
def finalize_rigid_cm_kernel(
    rigid_mass: wp.array(dtype=float),
    rigid_rest_cm: wp.array(dtype=wp.vec3),
    num_objects: int
):
    tid = wp.tid()
    if tid < num_objects:
        mass = rigid_mass[tid]
        if mass > 0.0:
            sum_pos = rigid_rest_cm[tid]
            rigid_rest_cm[tid] = sum_pos / mass
        else:
            rigid_rest_cm[tid] = wp.vec3(0.0, 0.0, 0.0)

@wp.kernel
def compute_rigid_inertia_kernel(
    object_id: wp.array(dtype=int),
    x: wp.array(dtype=wp.vec3),
    m_V: wp.array(dtype=float),
    rho: wp.array(dtype=float),
    rigid_rest_cm: wp.array(dtype=wp.vec3),
    rigid_inertia_accum_flat: wp.array(dtype=float), # Flattened 3x3 per object -> 9 floats
    m_V0: float
):
    tid = wp.tid()
    obj_id = object_id[tid]
    
    # vol = m_V[tid]
    # if vol == 0.0:
    #     vol = m_V0
    
    mass = m_V0 * rho[tid]
    r = x[tid] - rigid_rest_cm[obj_id]
    
    # Inertia tensor calculation: I = sum( m * ( (r^2) * Eye - r_outer_r ) )
    r_sq = wp.dot(r, r)
    
    # Component-wise accumulation
    base_idx = obj_id * 9
    
    # Row 0: (y^2+z^2, -xy, -xz)
    # Row 1: (-yx, x^2+z^2, -yz)
    # Row 2: (-zx, -zy, x^2+y^2)
    
    # Using explicit formulation
    xx = r[0] * r[0]; yy = r[1] * r[1]; zz = r[2] * r[2]
    xy = r[0] * r[1]; xz = r[0] * r[2]; yz = r[1] * r[2]
    
    # 0,0: yy+zz
    wp.atomic_add(rigid_inertia_accum_flat, base_idx + 0, mass * (yy + zz))
    # 0,1: -xy
    wp.atomic_add(rigid_inertia_accum_flat, base_idx + 1, mass * (-xy))
    # 0,2: -xz
    wp.atomic_add(rigid_inertia_accum_flat, base_idx + 2, mass * (-xz))
    
    # 1,0: -xy
    wp.atomic_add(rigid_inertia_accum_flat, base_idx + 3, mass * (-xy))
    # 1,1: xx+zz
    wp.atomic_add(rigid_inertia_accum_flat, base_idx + 4, mass * (xx + zz))
    # 1,2: -yz
    wp.atomic_add(rigid_inertia_accum_flat, base_idx + 5, mass * (-yz))
    
    # 2,0: -xz
    wp.atomic_add(rigid_inertia_accum_flat, base_idx + 6, mass * (-xz))
    # 2,1: -yz
    wp.atomic_add(rigid_inertia_accum_flat, base_idx + 7, mass * (-yz))
    # 2,2: xx+yy
    wp.atomic_add(rigid_inertia_accum_flat, base_idx + 8, mass * (xx + yy))

@wp.kernel
def finalize_rigid_inertia_kernel(
    rigid_mass: wp.array(dtype=float),
    rigid_inertia_accum_flat: wp.array(dtype=float),
    rigid_inertia: wp.array(dtype=wp.mat33),
    rigid_inv_inertia: wp.array(dtype=wp.mat33),
    rigid_inv_mass: wp.array(dtype=float),
    rigid_inertia0: wp.array(dtype=wp.mat33),
    num_objects: int
):
    tid = wp.tid()
    if tid < num_objects:
        mass = rigid_mass[tid]
        if mass > 0.0:
            rigid_inv_mass[tid] = 1.0 / mass
            
            base_idx = tid * 9
            # Construct mat33
            # Use transpose constructor layout (row-major) if mat33 constructor expects column vectors?
            # Warp mat33 constructor takes 3 column vectors or 9 scalars (row-major)?
            # Checking documentation or behavior: usually row-major for 9 args
            
            # Row 0
            i00 = rigid_inertia_accum_flat[base_idx+0]
            i01 = rigid_inertia_accum_flat[base_idx+1]
            i02 = rigid_inertia_accum_flat[base_idx+2]
            
            # Row 1
            i10 = rigid_inertia_accum_flat[base_idx+3]
            i11 = rigid_inertia_accum_flat[base_idx+4]
            i12 = rigid_inertia_accum_flat[base_idx+5]
            
            # Row 2
            i20 = rigid_inertia_accum_flat[base_idx+6]
            i21 = rigid_inertia_accum_flat[base_idx+7]
            i22 = rigid_inertia_accum_flat[base_idx+8]
            
            I = wp.mat33(
               i00, i01, i02,
               i10, i11, i12,
               i20, i21, i22
            )
            
            rigid_inertia[tid] = I
            rigid_inertia0[tid] = I 
            
            # Determine if matrix is invertible
            det = wp.determinant(I)
            if wp.abs(det) > 1e-6:
                rigid_inv_inertia[tid] = wp.inverse(I)
            else:
                 # fallback for singular matrix
                 rigid_inv_inertia[tid] = wp.mat33(0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0)

        else:
            rigid_inv_mass[tid] = 0.0
            z_mat = wp.mat33(0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0)
            rigid_inertia[tid] = z_mat
            rigid_inertia0[tid] = z_mat
            rigid_inv_inertia[tid] = z_mat

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

    rigid_x_out[tid] = norm_grad_vec3(rigid_x[tid] + dt * v) 

    I_inv = rigid_inv_inertia[tid]
    omega = rigid_omega[tid] + dt * (I_inv @ rigid_torque[tid])
    rigid_torque_out[tid] = wp.vec3(0.0, 0.0, 0.0)

    q = rigid_quaternion[tid]
    dq = 0.5 * wp.quat(omega[0], omega[1], omega[2], 0.0) * q
    q_normalized = wp.normalize(q + dt * dq)
    rigid_quaternion_out[tid] = norm_grad_quat(q_normalized)

    R = wp.quat_to_matrix(q_normalized)
    I0 = rigid_inertia0[tid]
    I = R @ I0 @ wp.transpose(R)
    rigid_inertia_out[tid] = I
    rigid_inv_inertia_out[tid] = wp.inverse(I)

    rigid_v_out[tid] = norm_grad_vec3(v)
    rigid_omega_out[tid] =norm_grad_vec3(omega)


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
        # particles_v[tid] = bodies.rigid_v[r] + wp.cross(bodies.rigid_omega[r], x_rel_world)
        # particles_v[tid] = bodies.rigid_v[r] + wp.cross(bodies.rigid_omega[r], x_rel)

@wp.kernel
def update_rigid_particle_info_diff(
    particles_x: wp.array(dtype=wp.vec3),        
    particles_v: wp.array(dtype=wp.vec3),       
    particles_x0: wp.array(dtype=wp.vec3),       
    object_id: wp.array(dtype=int),            
    mtr: MaterialMarks,
    rigid_rest_cm: wp.array(dtype=wp.vec3),        
    rigid_x: wp.array(dtype=wp.vec3),
    rigid_quaternion: wp.array(dtype=wp.quat),
    rigid_v: wp.array(dtype=wp.vec3),
    rigid_omega: wp.array(dtype=wp.vec3),        
):
    tid = wp.tid()
    # update dynamic rigid body particle transforms
    if is_dynamic_rigid_body(mtr, tid):
        r = object_id[tid]

        # rest-space relative position (assumes rest orientation is identity)
        x_rel = particles_x0[tid] - rigid_rest_cm[r]
        R = wp.quat_to_matrix(rigid_quaternion[r])
        x_rel_world = R @ x_rel
        # position and velocity must use the SAME world-space lever arm to avoid artifacts
        particles_x[tid] = rigid_x[r] + x_rel_world
        particles_v[tid] = rigid_v[r] + wp.cross(rigid_omega[r], x_rel_world)
        # particles_v[tid] = bodies.rigid_v[r] + wp.cross(bodies.rigid_omega[r], x_rel)