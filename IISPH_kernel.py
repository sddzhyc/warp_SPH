import warp as wp
from kernel_func import cubic_kernel, cubic_kernel_derivative, diff_pressure_kernel_cubic, diff_viscous_kernel_cubic
from rigid_fluid_coupling import MaterialMarks, MaterialType, is_dynamic_rigid_body

@wp.kernel
def compute_non_pressure_forces(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    viscous_normalization: float,
    smoothing_length: float,
    mtr: MaterialMarks,
    m_V: wp.array(dtype=float),
    base_density: float,
    gravity: float,
    particle_a_out: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    
    # Initialize with gravity
    particle_a_out[i] = wp.vec3(0.0, gravity, 0.0)

    if mtr.material[i] != MaterialType.FLUID:
        return

    x = particle_x[i]
    v = particle_v[i]
    
    viscous_force = wp.vec3(0.0, 0.0, 0.0)
    
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)
    
    for index in neighbors:
        if index != i:
            d = wp.length(x - particle_x[index])
            if d < smoothing_length:
                # Use x_ij (x_i - x_j)
                relative_position = x - particle_x[index]
                
                # Fluid viscosity
                if mtr.material[index] == MaterialType.FLUID:
                    viscous_force += base_density * m_V[index] * diff_viscous_kernel_cubic(
                        relative_position, v, particle_v[index], particle_rho[index], smoothing_length
                    )
                # Boundary viscosity (optional, can add if needed)

    particle_a_out[i] += viscous_normalization * viscous_force

@wp.kernel
def predict_velocity(
    a_non_p: wp.array(dtype=wp.vec3),
    gravity: float,
    dt: float,
    mtr: MaterialMarks,
    particle_v: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    # if mtr.material[tid] == MaterialType.FLUID:
    particle_v[tid] += a_non_p[tid] * dt

@wp.kernel
def compute_aii_and_density_deviation(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    m_V: wp.array(dtype=float),
    mtr: MaterialMarks,
    smoothing_length: float,
    base_density: float,
    dt: float,
    aii_out: wp.array(dtype=float),
    density_deviation_out: wp.array(dtype=float)
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    
    if mtr.material[i] != MaterialType.FLUID:
        aii_out[i] = 0.0
        density_deviation_out[i] = 0.0
        return

    x_i = particle_x[i]
    v_i = particle_v[i]
    rho_i = particle_rho[i]
    
    sum_term1 = wp.vec3(0.0, 0.0, 0.0) # Sum(m_k / rho_k^2 * Grad W_ik)
    sum_term2 = wp.vec3(0.0, 0.0, 0.0) # Sum(m_j * Grad W_ij)
    sum_term3 = float(0.0)             # Sum(m_j * |Grad W_ij|^2)
    
    divergence = float(0.0)
    
    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)
    
    for index in neighbors:
        if index == i:
            continue
            
        r_ji = particle_x[index] - x_i
        d = wp.length(r_ji)
        
        if d < smoothing_length:
            grad_W = cubic_kernel_derivative(r_ji, smoothing_length)
            
            # For a_ii
            rho_k = particle_rho[index]
            if rho_k > 1e-6:
                rho_val = rho_k
                if mtr.material[index] == MaterialType.SOLID:
                    rho_val = base_density
                
                sum_term1 += (m_V[index] * base_density / (rho_val * rho_val)) * grad_W
            
            sum_term2 += (m_V[index] * base_density) * grad_W
            sum_term3 += (m_V[index] * base_density) * wp.dot(grad_W, grad_W)
            
            # For divergence
            v_j = particle_v[index]
            divergence += (m_V[index] * base_density) * wp.dot(v_i - v_j, grad_W)

    term1_dot_term2 = wp.dot(sum_term1, sum_term2)
    sum_neighbor = -term1_dot_term2
    
    factor = 0.0
    if rho_i > 1e-6:
        factor = m_V[i] / (rho_i * rho_i)
        
    sum_neighbor_of_neighbor = -sum_term3 * factor
    
    aii = (sum_neighbor + sum_neighbor_of_neighbor) * (dt * dt * base_density * base_density)
    aii_out[i] = aii
    
    # Matches Taichi reference logic: self.dt[None] * divergence * self.density_0
    density_deviation_out[i] = base_density - rho_i - dt * divergence * base_density 


@wp.kernel
def update_pressure_and_compute_avg_error(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    pressure_accel: wp.array(dtype=wp.vec3),
    m_V: wp.array(dtype=float),
    a_ii: wp.array(dtype=float),
    density_deviation: wp.array(dtype=float),
    particle_p: wp.array(dtype=float), # In/Out
    mtr: MaterialMarks,
    smoothing_length: float,
    base_density: float,
    dt: float,
    omega: float, # Relaxation
    avg_error_out: wp.array(dtype=float) # Atomic add
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    
    if mtr.material[i] != MaterialType.FLUID:
        return

    x_i = particle_x[i]
    accel_p_i = pressure_accel[i]
    
    Ap = float(0.0)
    dt2 = dt * dt
    
    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)
    
    for index in neighbors:
        if index == i:
            continue
            
        r_ji = particle_x[index] - x_i
        d = wp.length(r_ji)
        
        if d < smoothing_length:
            grad_W = cubic_kernel_derivative(r_ji, smoothing_length)
            # Fluid
            if mtr.material[index] == MaterialType.FLUID:
                accel_p_j = pressure_accel[index]
                Ap += m_V[index] * wp.dot(accel_p_i - accel_p_j, grad_W)
            # Solid
            elif mtr.material[index] == MaterialType.SOLID:
                accel_p_j = pressure_accel[index] 
                Ap += m_V[index] * wp.dot(accel_p_i - accel_p_j, grad_W)
                
    Ap *= dt2 * base_density
    
    aii_val = a_ii[i]
    old_p = particle_p[i]
    new_p = 0.0
    
    if wp.abs(aii_val) > 1e-6:
        new_p = wp.max(old_p + omega * (density_deviation[i] - Ap) / aii_val, 0.0)
    else:
        new_p = 0.0
        
    particle_p[i] = new_p
    
    if new_p > 1e-6 or density_deviation[i] > 1e-6:
        # Error calculation
        error = wp.abs(Ap - density_deviation[i]) / base_density
        wp.atomic_add(avg_error_out, 0, error)

@wp.kernel
def compute_pressure_a(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    particle_p: wp.array(dtype=float), # last_pressure
    m_V: wp.array(dtype=float),
    mtr: MaterialMarks,
    smoothing_length: float,
    base_density: float,
    pressure_accel_out: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    
    pressure_accel_out[i] = wp.vec3(0.0, 0.0, 0.0)

    if mtr.material[i] != MaterialType.FLUID:
        return

    x_i = particle_x[i]
    rho_i = particle_rho[i]
    p_i = particle_p[i]
    
    if rho_i < 1e-6:
        return

    dpi = p_i / (rho_i * rho_i)
    
    d_v = wp.vec3(0.0, 0.0, 0.0)
    
    neighbors = wp.hash_grid_query(grid, x_i, smoothing_length)
    
    for index in neighbors:
        if index == i:
            continue
        
        r_ji = particle_x[index] - x_i
        d = wp.length(r_ji)
        
        if d < smoothing_length:
            grad_W = cubic_kernel_derivative(r_ji, smoothing_length)
            
            # Fluid neighbors
            if mtr.material[index] == MaterialType.FLUID:
                rho_j = particle_rho[index]
                if rho_j > 1e-6:
                    p_j = particle_p[index]
                    dpj = p_j / (rho_j * rho_j)
                    d_v += -base_density * m_V[index] * (dpi + dpj) * grad_W
            
            # Solid neighbors
            elif mtr.material[index] == MaterialType.SOLID:
                dpj = p_i / (base_density * base_density)
                d_v += -base_density * m_V[index] * (dpi + dpj) * grad_W

    pressure_accel_out[i] = d_v
