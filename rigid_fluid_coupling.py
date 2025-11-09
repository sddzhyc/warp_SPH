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