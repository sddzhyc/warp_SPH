import warp as wp

from kernel_func import wp
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


# import partio

@wp.func
def diff_pressure_kernel(
    xyz: wp.vec3, pressure: float, neighbor_pressure: float, rho: float , neighbor_rho: float, smoothing_length: float
):
    # calculate distance
    distance = wp.sqrt(wp.dot(xyz, xyz))

    if distance < smoothing_length: # 默认smoothing_length即支持半径？
        # calculate terms of kernel
        term_1 = -xyz / distance # 单位距离向量
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
        term_3 = cubic_kernel_derivative(xyz, smoothing_length, 3)  # gradient of SPH kernel (grad W); TODO: use another kernel
        return term_2 * term_3
    else:
        return wp.vec3()

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
        distance**2. + 0.01 * smoothing_length**2.) / neighbor_rho * cubic_kernel_derivative(r, smoothing_length, dim)
    return res



@wp.func
def cubic_kernel(xyz: wp.vec3, h: wp.float32):
    distance = wp.sqrt(wp.dot(xyz, xyz))
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
def cubic_kernel_derivative(r: wp.vec3, support_radius: wp.float32, dim: int):
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
    r_norm = wp.sqrt(wp.dot(r, r))
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
