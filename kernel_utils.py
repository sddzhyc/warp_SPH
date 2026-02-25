# optional dependency for flexible PLY export with custom attributes

from rigid_fluid_coupling import MaterialMarks, MaterialType
from sph_kernel import wp


import warp as wp


@wp.kernel
def add_particles_kernel(
    offset: int,
    # target arrays
    x: wp.array(dtype=wp.vec3),
    x_0: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    rho: wp.array(dtype=float),
    material: wp.array(dtype=int),
    is_dynamic: wp.array(dtype=int),
    object_id: wp.array(dtype=int),
    m_V: wp.array(dtype=float),
    color: wp.array(dtype=wp.vec3i),
    # source arrays/values
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    density: wp.array(dtype=float),
    mat: wp.array(dtype=int),
    is_dyn: wp.array(dtype=int),
    obj_id: int,
    m_V0: float,
    col: wp.array(dtype=wp.vec3i)
):
    i = wp.tid()
    idx = offset + i
    x[idx] = pos[i]
    x_0[idx] = pos[i]
    v[idx] = vel[i]
    rho[idx] = density[i]
    material[idx] = mat[i]
    is_dynamic[idx] = is_dyn[i]
    object_id[idx] = obj_id
    m_V[idx] = m_V0
    color[idx] = col[i]


# @wp.kernel
# def init_fluid_particles_kernel(
#     x: wp.array(dtype=wp.vec3),
#     v: wp.array(dtype=wp.vec3),
#     rho: wp.array(dtype=float),
#     mask: wp.array(dtype=int),
#     materials: MaterialMarks,
#     fluid_size: int,
#     base_rho: float,
#     object_ids: wp.array(dtype=int),
# ):
#     tid = wp.tid()
#     if tid < fluid_size:
#         # Inactive fluid particles hidden far away
#         x[tid] = wp.vec3(-1000.0, -1000.0, -1000.0)
#         v[tid] = wp.vec3(0.0, 0.0, 0.0)
#         rho[tid] = base_rho
#         mask[tid] = 0 # Inactive
#         # Mark as Static Solid so they are ignored by fluid solver/boundary enforcement
#         materials.material[tid] = MaterialType.SOLID 
#         materials.is_dynamic[tid] = 0 
#         object_ids[tid] = -1

@wp.kernel
def add_ball_kernel(
    x: wp.array(dtype=wp.vec3),
    x_0: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    rho: wp.array(dtype=float),
    mask: wp.array(dtype=int),
    materials: MaterialMarks,
    object_ids: wp.array(dtype=int),
    # fluid_size: int,
    start_idx: int,
    jelly_size: int,
    # n_balls: int,
    stand_ball: wp.array(dtype=wp.vec3),
    ball_center: wp.vec3,
    jelly_rho: float,
    obj_id: int
):
    tid = wp.tid()
    # if tid >= fluid_size:
    #     # Jelly particles
    #     idx_global = tid - fluid_size
    #     ball_idx = idx_global // jelly_size

    if tid < jelly_size:
        idx_global = start_idx + tid

        if idx_global < x.shape[0]:
            # offset_idx = idx_global % jelly_size
            offset_idx = tid

            object_ids[idx_global] = obj_id # Map to rigid body index
            # Position logic
            # ball_center = ball_pos_start + wp.vec3(float(ball_idx)*0.15, -float(ball_idx)*0.02, 0.0)
            x[idx_global] = ball_center + stand_ball[offset_idx]
            x_0[idx_global] = ball_center + stand_ball[offset_idx]
            v[idx_global] = wp.vec3(0.0, 0.0, 0.0)

            # Density/Mass - mass is handled by density * volume outside
            # dens = jelly_rho[ball_idx]
            rho[idx_global] = jelly_rho

            mask[idx_global] = 1 # Active
            materials.material[idx_global] = MaterialType.SOLID
            materials.is_dynamic[idx_global] = 1 # Dynamic Rigid Body