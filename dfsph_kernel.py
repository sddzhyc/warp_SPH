import warp as wp
from kernel_func import cubic_kernel_derivative, cubic_kernel_derivative_custom
from rigid_fluid_coupling import MaterialMarks, MaterialType

@wp.kernel
def compute_dfsph_factor_kernel(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    mtr: MaterialMarks,
    m_V: wp.array(dtype=float),
    smoothing_length: float,
    dfsph_factor_out: wp.array(dtype=float)
): 
    tid = wp.tid()
    
    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    
    if mtr.material[i] != MaterialType.FLUID:
        dfsph_factor_out[i] = 0.0
        return

    # get local particle variables
    x_i = particle_x[i]

    sum_grad_p_k = float(0.0)
    grad_p_i = wp.vec3(0.0, 0.0, 0.0)

    # particle contact
    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)

    for index in neighbors:
        if index == i:
            continue
            
        r_vec = x_i - particle_x[index]
        d = wp.length(r_vec)
        
        if d < smoothing_length:
            if mtr.material[index] == MaterialType.FLUID:
                 grad_W = cubic_kernel_derivative(r_vec, smoothing_length)
                 
                 # grad_p_j = -m_V[j] * grad_W
                 # Using the same convention as taichi
                 grad_p_j = -m_V[index] * grad_W
                 
                 sum_grad_p_k += wp.length_sq(grad_p_j)
                 grad_p_i -= grad_p_j # Accumulate -grad_p_j -> + m_V[j]*grad_W
                 
            elif mtr.material[index] == MaterialType.SOLID:
                 grad_W = cubic_kernel_derivative(r_vec, smoothing_length)
                 grad_p_j = -m_V[index] * grad_W
                 grad_p_i -= grad_p_j

    sum_grad_p_k += wp.length_sq(grad_p_i)

    if sum_grad_p_k > 1e-6:
        dfsph_factor_out[i] = -1.0 / sum_grad_p_k # TODO: check sign
    else:
        dfsph_factor_out[i] = 0.0

@wp.kernel
def compute_density_adv_kernel(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    mtr: MaterialMarks,
    m_V: wp.array(dtype=float),
    smoothing_length: float,
    dt: float,
    base_density: float,
    density_adv_out: wp.array(dtype=float)
):
    tid = wp.tid()
    
    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    
    if mtr.material[i] != MaterialType.FLUID:
        density_adv_out[i] = 0.0
        return
        
    # get local particle variables
    x_i = particle_x[i]
    v_i = particle_v[i]
    
    delta = float(0.0)

    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)

    for index in neighbors:
        if index == i:
            continue
            
        r_vec = x_i - particle_x[index]
        d = wp.length(r_vec)
        
        if d < smoothing_length:
            v_j = particle_v[index]
            
            if mtr.material[index] == MaterialType.FLUID or mtr.material[index] == MaterialType.SOLID:
                grad_W = cubic_kernel_derivative(r_vec, smoothing_length)
                
                v_ij = v_i - v_j
                delta += m_V[index] * wp.dot(v_ij, grad_W)
    
    density_ratio = particle_rho[i] / base_density
    adv_val = density_ratio + dt * delta
    density_adv_out[i] = wp.max(adv_val, 1.0)


@wp.kernel
def compute_density_change_kernel(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    mtr: MaterialMarks,
    m_V: wp.array(dtype=float),
    smoothing_length: float,
    dim: int,
    density_change_out: wp.array(dtype=float)
):
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)

    if mtr.material[i] != MaterialType.FLUID:
        density_change_out[i] = 0.0
        return

    x_i = particle_x[i]
    v_i = particle_v[i]

    density_adv = float(0.0)
    num_neighbors = wp.int32(0)

    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)

    for index in neighbors:
        if index == i:
            continue

        r_vec = x_i - particle_x[index]
        d = wp.length(r_vec)

        if d < smoothing_length:
            if mtr.material[index] == MaterialType.FLUID or mtr.material[index] == MaterialType.SOLID:
                v_j = particle_v[index]
                grad_w = cubic_kernel_derivative(r_vec, smoothing_length)
                density_adv += m_V[index] * wp.dot(v_i - v_j, grad_w)
                num_neighbors += 1

    density_adv = wp.max(density_adv, 0.0)
    
    # Do not perform divergence solve when particle deficiency happens
    if dim == 3:
        if num_neighbors < 20:
            density_adv = 0.0
    else:
        if num_neighbors < 7:
            density_adv = 0.0

    density_change_out[i] = density_adv

@wp.kernel
def pressure_solve_iteration_kernel(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    density_adv: wp.array(dtype=float),
    dfsph_factor: wp.array(dtype=float),
    mtr: MaterialMarks,
    m_V: wp.array(dtype=float),
    smoothing_length: float,
    dt: float,
    base_density: float,
    particle_v_out: wp.array(dtype=wp.vec3),
    object_id: wp.array(dtype=wp.int32),
    rigid_force: wp.array(dtype=wp.vec3),
    rigid_torque: wp.array(dtype=wp.vec3),
    rigid_x: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    
    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    
    if mtr.material[i] != MaterialType.FLUID:
        # For non-fluid particles, we might not update velocity here, but we should copy it
        particle_v_out[i] = particle_v[i]
        return
        
    # get local particle variables
    x_i = particle_x[i]
    v_i = particle_v[i]
    
    # Evaluate rhs
    # b_i = self.ps.density_adv[p_i] - 1.0
    # k_i = b_i * self.ps.dfsph_factor[p_i]
    # NOTE: dfpsh_factor needs to be scaled by 1/dt^2
    inv_dt2 = 1.0 / (dt * dt)
    
    b_i = density_adv[i] - 1.0
    k_i = b_i * dfsph_factor[i] * inv_dt2
    
    m_eps = 1e-5
    
    # particle contact
    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)

    vel_change_sum = wp.vec3(0.0, 0.0, 0.0)

    for index in neighbors:
        if index == i:
            continue
            
        r_vec = x_i - particle_x[index]
        d = wp.length(r_vec)
        
        if d < smoothing_length:
            
            if mtr.material[index] == MaterialType.FLUID:

                b_j = density_adv[index] - 1.0
                k_j = b_j * dfsph_factor[index] * inv_dt2
                
                k_sum = k_i + k_j # assuming density_0 ratio is 1
                
                if wp.abs(k_sum) > m_eps:
                    grad_W = cubic_kernel_derivative(r_vec, smoothing_length)
                    grad_p_j = -m_V[index] * grad_W
                    
                    force = -dt * k_sum * grad_p_j
                    vel_change_sum += force

            elif mtr.material[index] == MaterialType.SOLID:
                 if wp.abs(k_i) > m_eps:
                    grad_W = cubic_kernel_derivative(r_vec, smoothing_length)
                    grad_p_j = -m_V[index] * grad_W
                    
                    vel_change = -dt * k_i * grad_p_j
                    
                    vel_change_sum += vel_change 
                    
                    if mtr.is_dynamic[index] != 0:
                        r_id = object_id[index]
                        
                        rho_i = density_adv[i] * base_density # Approximate current density
                        force_rigid = -vel_change * (1.0/dt) * rho_i * m_V[i]
                        
                        wp.atomic_add(rigid_force, r_id, force_rigid)
                        wp.atomic_add(rigid_torque, r_id, wp.cross(particle_x[index] - rigid_x[r_id], force_rigid))

    wp.atomic_add(particle_v_out, i, vel_change_sum)

@wp.kernel
def divergence_solve_iteration_kernel(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    density_change: wp.array(dtype=float),
    dfsph_factor: wp.array(dtype=float),
    mtr: MaterialMarks,
    m_V: wp.array(dtype=float),
    smoothing_length: float,
    dt: float,
    object_id: wp.array(dtype=wp.int32),
    rigid_x: wp.array(dtype=wp.vec3),
    particle_v_out: wp.array(dtype=wp.vec3),
    rigid_force: wp.array(dtype=wp.vec3),
    rigid_torque: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    i = wp.hash_grid_point_id(grid, tid)

    if mtr.material[i] != MaterialType.FLUID:
        particle_v_out[i] = particle_v[i]
        return

    x_i = particle_x[i]

    inv_dt = 1.0 / dt
    m_eps = 1e-5

    b_i = density_change[i]
    k_i = b_i * dfsph_factor[i] * inv_dt

    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)

    vel_change_sum = wp.vec3(0.0, 0.0, 0.0)

    for index in neighbors:
        if index == i:
            continue

        r_vec = x_i - particle_x[index]
        d = wp.length(r_vec)

        if d < smoothing_length:
            if mtr.material[index] == MaterialType.FLUID:
                b_j = density_change[index]
                k_j = b_j * dfsph_factor[index] * inv_dt
                k_sum = k_i + k_j

                if wp.abs(k_sum) > m_eps:
                    grad_w = -cubic_kernel_derivative(r_vec, smoothing_length)
                    grad_p_j = m_V[index] * grad_w
                    vel_change_sum += -dt * k_sum * grad_p_j

            elif mtr.material[index] == MaterialType.SOLID:
                if wp.abs(k_i) > m_eps:
                    grad_w = -cubic_kernel_derivative(r_vec, smoothing_length)
                    grad_p_j = m_V[index] * grad_w
                    vel_change = -dt * k_i * grad_p_j
                    vel_change_sum += vel_change

                    if mtr.is_dynamic[index] != 0:
                        r_id = object_id[index]
                        force_rigid = -vel_change * inv_dt * particle_rho[i] * m_V[i]
                        wp.atomic_add(rigid_force, r_id, force_rigid)
                        wp.atomic_add(rigid_torque, r_id, wp.cross(particle_x[index] - rigid_x[r_id], force_rigid))

    wp.atomic_add(particle_v_out, i, vel_change_sum)


@wp.kernel
def pressure_solve_iteration_kernel_fluid(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    density_adv: wp.array(dtype=float),
    dfsph_factor: wp.array(dtype=float),
    mtr: MaterialMarks,
    m_V: wp.array(dtype=float),
    smoothing_length: float,
    dt: float,
    use_custom_grad: bool,
    particle_v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)

    if mtr.material[i] != MaterialType.FLUID:
        return

    x_i = particle_x[i]
    inv_dt2 = 1.0 / (dt * dt)
    m_eps = 1e-5
    b_i = density_adv[i] - 1.0
    k_i = b_i * dfsph_factor[i] * inv_dt2

    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)
    vel_change_sum = wp.vec3(0.0, 0.0, 0.0)

    for index in neighbors:
        if index == i:
            continue

        r_vec = x_i - particle_x[index]
        d = wp.length(r_vec)
        if d < smoothing_length and mtr.material[index] == MaterialType.FLUID:
            b_j = density_adv[index] - 1.0
            k_j = b_j * dfsph_factor[index] * inv_dt2
            k_sum = k_i + k_j

            if wp.abs(k_sum) > m_eps:
                if use_custom_grad:
                    grad_w = -cubic_kernel_derivative_custom(r_vec, smoothing_length)
                else:
                    grad_w = -cubic_kernel_derivative(r_vec, smoothing_length)
                grad_p_j = m_V[index] * grad_w
                vel_change_sum += -dt * k_sum * grad_p_j

    wp.atomic_add(particle_v_out, i, vel_change_sum)


@wp.kernel
def pressure_solve_iteration_kernel_solid(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    density_adv: wp.array(dtype=float),
    dfsph_factor: wp.array(dtype=float),
    mtr: MaterialMarks,
    m_V: wp.array(dtype=float),
    smoothing_length: float,
    dt: float,
    base_density: float,
    use_custom_grad: bool,
    particle_v_out: wp.array(dtype=wp.vec3),
    object_id: wp.array(dtype=wp.int32),
    rigid_force: wp.array(dtype=wp.vec3),
    rigid_torque: wp.array(dtype=wp.vec3),
    rigid_x: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)

    if mtr.material[i] != MaterialType.FLUID:
        return

    x_i = particle_x[i]
    inv_dt = 1.0 / dt
    inv_dt2 = 1.0 / (dt * dt)
    m_eps = 1e-5
    b_i = density_adv[i] - 1.0
    k_i = b_i * dfsph_factor[i] * inv_dt2

    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)
    vel_change_sum = wp.vec3(0.0, 0.0, 0.0)

    for index in neighbors:
        if index == i:
            continue

        r_vec = x_i - particle_x[index]
        d = wp.length(r_vec)
        if d < smoothing_length and mtr.material[index] == MaterialType.SOLID:
            if wp.abs(k_i) > m_eps:
                if use_custom_grad:
                    grad_w = -cubic_kernel_derivative_custom(r_vec, smoothing_length)
                else:
                    grad_w = -cubic_kernel_derivative(r_vec, smoothing_length)
                grad_p_j = m_V[index] * grad_w
                vel_change = -dt * k_i * grad_p_j
                vel_change_sum += vel_change

                if mtr.is_dynamic[index] != 0:
                    r_id = object_id[index]
                    rho_i = density_adv[i] * base_density
                    force_rigid = -vel_change * inv_dt * rho_i * m_V[i]
                    wp.atomic_add(rigid_force, r_id, force_rigid)
                    wp.atomic_add(rigid_torque, r_id, wp.cross(particle_x[index] - rigid_x[r_id], force_rigid))

    wp.atomic_add(particle_v_out, i, vel_change_sum)


@wp.kernel
def divergence_solve_iteration_kernel_fluid(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    density_change: wp.array(dtype=float),
    dfsph_factor: wp.array(dtype=float),
    mtr: MaterialMarks,
    m_V: wp.array(dtype=float),
    smoothing_length: float,
    dt: float,
    use_custom_grad: bool,
    particle_v_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)

    if mtr.material[i] != MaterialType.FLUID:
        return

    x_i = particle_x[i]
    inv_dt = 1.0 / dt
    m_eps = 1e-5
    b_i = density_change[i]
    k_i = b_i * dfsph_factor[i] * inv_dt

    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)
    vel_change_sum = wp.vec3(0.0, 0.0, 0.0)

    for index in neighbors:
        if index == i:
            continue

        r_vec = x_i - particle_x[index]
        d = wp.length(r_vec)
        if d < smoothing_length and mtr.material[index] == MaterialType.FLUID:
            b_j = density_change[index]
            k_j = b_j * dfsph_factor[index] * inv_dt
            k_sum = k_i + k_j

            if wp.abs(k_sum) > m_eps:
                if use_custom_grad:
                    grad_w = cubic_kernel_derivative_custom(r_vec, smoothing_length)
                else:
                    grad_w = cubic_kernel_derivative(r_vec, smoothing_length)
                grad_p_j = -m_V[index] * grad_w
                vel_change_sum += -dt * k_sum * grad_p_j

    wp.atomic_add(particle_v_out, i, vel_change_sum)


@wp.kernel
def divergence_solve_iteration_kernel_solid(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    density_change: wp.array(dtype=float),
    dfsph_factor: wp.array(dtype=float),
    mtr: MaterialMarks,
    m_V: wp.array(dtype=float),
    smoothing_length: float,
    dt: float,
    object_id: wp.array(dtype=wp.int32),
    rigid_x: wp.array(dtype=wp.vec3),
    use_custom_grad: bool,
    particle_v_out: wp.array(dtype=wp.vec3),
    rigid_force: wp.array(dtype=wp.vec3),
    rigid_torque: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)

    if mtr.material[i] != MaterialType.FLUID:
        return

    x_i = particle_x[i]
    inv_dt = 1.0 / dt
    m_eps = 1e-5
    b_i = density_change[i]
    k_i = b_i * dfsph_factor[i] * inv_dt

    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)
    vel_change_sum = wp.vec3(0.0, 0.0, 0.0)

    for index in neighbors:
        if index == i:
            continue

        r_vec = x_i - particle_x[index]
        d = wp.length(r_vec)
        if d < smoothing_length and mtr.material[index] == MaterialType.SOLID:
            if wp.abs(k_i) > m_eps:
                if use_custom_grad:
                    grad_w = cubic_kernel_derivative_custom(r_vec, smoothing_length)
                else:
                    grad_w = cubic_kernel_derivative(r_vec, smoothing_length)
                grad_p_j = -m_V[index] * grad_w
                vel_change = -dt * k_i * grad_p_j
                vel_change_sum += vel_change

                if mtr.is_dynamic[index] != 0:
                    r_id = object_id[index]
                    force_rigid = -vel_change * inv_dt * particle_rho[i] * m_V[i]
                    wp.atomic_add(rigid_force, r_id, force_rigid)
                    wp.atomic_add(rigid_torque, r_id, wp.cross(particle_x[index] - rigid_x[r_id], force_rigid))

    wp.atomic_add(particle_v_out, i, vel_change_sum)


@wp.kernel
def compute_density_error_kernel(
    density_adv: wp.array(dtype=float),
    mtr: MaterialMarks,
    base_density: float,
    offset: float,
    error_sum: wp.array(dtype=float)
):
    tid = wp.tid()
    if mtr.material[tid] == MaterialType.FLUID:
        # Error = rho_adv * rho_0 - rho_0 
        # But density_adv is ratio. So (ratio - 1) * rho_0.
        # DFSPH.py: density_error += density_0 * density_adv - offset(=density_0)
        # So it is density_0 * (density_adv - 1)
        err = base_density * density_adv[tid] - offset
        # Only counting positive error (compression)? DFSPH usually corrects density > rho0.
        # In DFSPH.py: density_error += ...
        # And density_adv is max(..., 1.0) in compute_density_adv.
        # So density_adv >= 1.0. 
        # So err >= 0.
        wp.atomic_add(error_sum, 0, err)
