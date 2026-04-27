import math
import warp as wp

use_const_2order_grad = True # 该项是否启用对于diffFR也有影响
custom_h_scale, custom_grad_scale = 0., 0. # 4.0, 0.1 # 1.0, 0.0001

use_replay_kernel = True
replay_scale = 0.0


@wp.func
def square(x: float):
    return x * x


@wp.func
def cube(x: float):
    return x * x * x


@wp.func
def fifth(x: float):
    return x * x * x * x * x


@wp.func
def density_kernel(xyz: wp.vec3, smoothing_length: float):
    # calculate distance
    distance = wp.dot(xyz, xyz)

    return wp.max(cube(square(smoothing_length) - distance), 0.0)

@wp.func
def diff_pressure_kernel(
    xyz: wp.vec3, pressure: float, neighbor_pressure: float, rho: float , neighbor_rho: float, smoothing_length: float
):
    # calculate distance
    distance = wp.sqrt(wp.dot(xyz, xyz))

    if distance < smoothing_length: # 默认smoothing_length即支持半径？
        # calculate terms of kernel
        term_1 = -xyz / distance # 单位距离向量：注意反向了
        term_2 = (neighbor_pressure + pressure) / (2.0 * neighbor_rho)
        # term_2 = neighbor_pressure / (neighbor_rho * neighbor_rho) + pressure / (rho * rho)
        term_3 = square(smoothing_length - distance)  # gradient of SPH kernel (grad W); TODO: use another kernel
        return term_1 * term_2 * term_3
    else:
        return wp.vec3()

@wp.func
def diff_pressure_kernel_cubic(
    xyz: wp.vec3, pressure: float, neighbor_pressure: float, rho: float , neighbor_rho: float, smoothing_length: float
):
    # calculate distance
    distance = wp.sqrt(wp.dot(xyz, xyz))

    if distance < smoothing_length: # 默认smoothing_length即支持半径？
        # calculate terms of kernel
        # term_2 = (neighbor_pressure + pressure) / (2.0 * neighbor_rho)
        term_2 = neighbor_pressure / (neighbor_rho * neighbor_rho) + pressure / (rho * rho)
        term_3 = cubic_kernel_derivative(xyz, smoothing_length)  # gradient of SPH kernel (grad W); TODO: use another kernel
        return term_2 * term_3
    else:
        return wp.vec3()

@wp.func
def far_rigid_kernel(xyz: wp.vec3):
    return wp.vec3(0.,0.,0.)

@wp.func_grad(far_rigid_kernel)
def adj_far_rigid_kernel(xyz: wp.vec3, adj_ret: wp.vec3):
    d = wp.length(xyz)
    # functional form: f(x) = x^2 / (1 + x^2) 
    # soft_coeff = 0.001 * d * d / (1.0 + d * d)  # failed on diff-demo2 scene
    soft_coeff = 1. * d * d / (1.0 + d * d)
    
    # Original gradient term (normalized direction)
    normalized_xyz = xyz / (d + 1e-8)
    # wp.printf("adj_ret: %f, %f, %f\n", adj_ret.x, adj_ret.y, adj_ret.z)
    # wp.printf("soft_coeff: %f\n", soft_coeff)
    # wp.adjoint[xyz] += soft_coeff * wp.cw_mul(normalized_xyz, adj_ret)
    wp.adjoint[xyz] += soft_coeff * normalized_xyz

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

@wp.func
def diff_viscous_kernel_cubic(r: wp.vec3,  v: wp.vec3, neighbor_v: wp.vec3 , neighbor_rho: float, smoothing_length: float):
    # calculate distance
    distance = wp.sqrt(wp.dot(r, r))
    dim = 3
    v_xy = wp.dot(v - neighbor_v, r)
    # calculate terms of kernel
    res = float(2 * (dim + 2)) * v_xy / (
        distance**2. + 0.01 * smoothing_length**2.) / neighbor_rho * cubic_kernel_derivative(r, smoothing_length)
    return res



@wp.func
def cubic_kernel(xyz: wp.vec3, h: wp.float32):
    distance = wp.length(xyz)
    res = wp.cast(0.0, wp.float32)
    dim = 3 # TODO: support different dimensions
    # value of cubic spline smoothing kernel
    k = 1.0
    if dim == 1:
        k = 4.0 / 3.
    elif dim == 2:
        k = 40.0 / 7. / wp.pi
    elif dim == 3:
        k = 8.0 / wp.pi
    k /= h ** float(dim)
    q = distance / h
    if q <= 1.0: # 支持域半径即h
        if q <= 0.5:
            q2 = q * q
            q3 = q2 * q
            res = k * (6.0 * q3 - 6.0 * q2 + 1.0)
        else:
            res = k * 2. * wp.pow(1. - q, 3.0)
    return res

@wp.func
def cubic_kernel_derivative(r: wp.vec3, support_radius: float):
    dim = 3
    h = support_radius
    # derivative of cubic spline smoothing kernel
    k = 1.0
    if dim == 1:
        k = 4. / 3.
    elif dim == 2:
        k = 40. / 7. / wp.pi
    elif dim == 3:
        k = 8. / wp.pi
    k = 6. * k / h ** float(dim)
    r_norm = wp.length(r)
    q = r_norm / h
    res = wp.vec3()
    if r_norm > 1e-5 and q <= 1.0:
        grad_q = r / (r_norm * h)
        if q <= 0.5:
            res = k * q * (3.0 * q - 2.0) * grad_q
        else:
            factor = 1.0 - q
            res = k * (-factor * factor) * grad_q
    return res


@wp.func
def cubic_kernel_derivative_custom(r: wp.vec3, support_radius: float):
    dim = 3
    h = support_radius
    # derivative of cubic spline smoothing kernel
    k = 1.0
    if dim == 1:
        k = 4. / 3.
    elif dim == 2:
        k = 40. / 7. / wp.pi
    elif dim == 3:
        k = 8. / wp.pi
    k = 6. * k / h ** float(dim)
    r_norm = wp.length(r)
    q = r_norm / h
    res = wp.vec3()
    if r_norm > 1e-5 and q <= 1.0:
        grad_q = r / (r_norm * h)
        if q <= 0.5:
            res = k * q * (3.0 * q - 2.0) * grad_q
        else:
            factor = 1.0 - q
            res = k * (-factor * factor) * grad_q
    return res


# @wp.func_grad(cubic_kernel_derivative_custom)
def adj_cubic_kernel_derivative(r: wp.vec3, support_radius: float, adj_ret: wp.vec3):
    h = support_radius * custom_h_scale
    dim = 3
    # forward replay
    k_base = 1.0
    if dim == 1:
        k_base = 4.0 / 3.0
    elif dim == 2:
        k_base = 40.0 / 7.0 / wp.pi
    elif dim == 3:
        k_base = 8.0 / wp.pi

    k = 6.0 * k_base / wp.pow(h, wp.float32(dim))
    r_norm = wp.length(r)
    q = r_norm / h

    adj_r = wp.vec3()

    if r_norm > 1.0e-5 and q <= 1.0:
        inv_h = 1.0 / h
        inv_r_norm = 1.0 / r_norm

        if q <= 0.5:
            coeff = k * q * (3.0 * q - 2.0)
        else:
            factor = 1.0 - q
            coeff = -k * factor * factor

        scale = inv_r_norm * inv_h
        grad_q = r * scale

        # start adjoint accumulation
        
        adj_coeff = wp.dot(adj_ret, grad_q)
        adj_grad_q = adj_ret * coeff

        adj_scale = wp.dot(adj_grad_q, r)
        adj_r += adj_grad_q * scale

        adj_inv_r_norm = adj_scale * inv_h

        adj_r_norm = -adj_inv_r_norm * inv_r_norm * inv_r_norm

        adj_q = 0.0
        # if q <= 0.5:
        #     adj_q += adj_coeff * k * (6.0 * q - 2.0)
        # else:
        #     factor = 1.0 - q
        #     adj_q += adj_coeff * (2.0 * k * factor)
        factor = 1.0 - q
        adj_q += adj_coeff * (2.0 * k * factor)

        adj_r_norm += adj_q * inv_h
        adj_r += adj_r_norm * (r * inv_r_norm)
    wp.adjoint[r] += adj_r * custom_grad_scale # Add scale param 

if use_const_2order_grad:
    print("Using 0-order gradient for cubic_kernel_derivative_custom")
    @wp.func_grad(cubic_kernel_derivative_custom)
    def adj_cubic_kernel_derivative_2(r: wp.vec3, support_radius: float, adj_ret: wp.vec3):
        wp.adjoint[r] += adj_ret * custom_grad_scale #  (-custom_grad_scale)


# 1. / 13037870 # 162974 # 0.001
# h=0.025: kernel_vals max: 162974.640625 deriv_vals min: -13037870

def auto_replay_scale_from_h_py(support_radius: float) -> float:
    h = float(support_radius)
    return math.pi * h * h * h * h / 16.0


@wp.func
def auto_replay_scale_from_h(support_radius: float):
    # For cubic kernel derivative in dim=3, max |dW/dr| occurs at q = r/h = 1/3.
    # Its closed form is: max |dW/dr| = 16 / (pi * h^4), so scale is its reciprocal.
    h = support_radius
    return wp.pi * h * h * h * h / 16.0

if use_replay_kernel:
    print("Using replay kernel for cubic_kernel_derivative_custom")
    @wp.func_replay(cubic_kernel_derivative_custom)
    def replay_cubic_kernel_derivative_custom(r: wp.vec3, support_radius: float):
        dim = 3
        h = support_radius
        # derivative of cubic spline smoothing kernel
        k = 1.0
        if dim == 1:
            k = 4. / 3.
        elif dim == 2:
            k = 40. / 7. / wp.pi
        elif dim == 3:
            k = 8. / wp.pi
        k = 6. * k / h ** float(dim)
        r_norm = wp.length(r)
        q = r_norm / h
        res = wp.vec3()
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res * auto_replay_scale_from_h(h)


@wp.func
def replay_cubic_kernel_derivative_custom_plot(r: wp.vec3, support_radius: float):
    '''
    for plot_kernel.py to visualize the replay kernel function curve
    '''
    dim = 3
    h = support_radius
    # derivative of cubic spline smoothing kernel
    k = 1.0
    if dim == 1:
        k = 4. / 3.
    elif dim == 2:
        k = 40. / 7. / wp.pi
    elif dim == 3:
        k = 8. / wp.pi
    k = 6. * k / h ** float(dim)
    r_norm = wp.length(r)
    q = r_norm / h
    res = wp.vec3()
    if r_norm > 1e-5 and q <= 1.0:
        grad_q = r / (r_norm * h)
        if q <= 0.5:
            res = k * q * (3.0 * q - 2.0) * grad_q
        else:
            factor = 1.0 - q
            res = k * (-factor * factor) * grad_q
    return res * auto_replay_scale_from_h(h)

