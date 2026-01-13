import warp as wp

from kernel_func import wp


@wp.func
def norm_grad_vec3(state_t: wp.vec3):
    return state_t

@wp.func_grad(norm_grad_vec3)
def adj_norm_grad_vec3(state_t: wp.vec3, adj_ret: wp.vec3):
    alpha = wp.length(adj_ret)
    if alpha == 0.0:
        wp.adjoint[state_t] += adj_ret
    else:
        grad_a = adj_ret * (1.0 / (alpha))  # 归一化梯度  

        wp.adjoint[state_t] += grad_a

@wp.func
def norm_grad_quat(state_t: wp.quat):
    return state_t

@wp.func_grad(norm_grad_quat)
def adj_norm_grad_quat(state_t: wp.quat, adj_ret: wp.quat):
    alpha = wp.length(adj_ret)
    if alpha == 0.0:
        wp.adjoint[state_t] += adj_ret
    else:
        grad_a = adj_ret * (1.0 / (alpha))  # 归一化梯度  

        wp.adjoint[state_t] += grad_a


@wp.func
def kick_step(v: wp.vec3, a: wp.vec3, dt: float):
    return v + a * dt

@wp.func_grad(kick_step)
def adj_kick_step(v: wp.vec3, a: wp.vec3, dt: float, adj_ret: wp.vec3):

    grad_a = adj_ret * dt
    alpha = wp.length(grad_a)

    grad_a = grad_a * (1.0 / (alpha))  # 归一化梯度  

    wp.adjoint[v] += adj_ret 
    wp.adjoint[a] += grad_a


@wp.func
def drift_step(x: wp.vec3, v: wp.vec3, dt: float):
    return x + v * dt

@wp.func_grad(drift_step)
def adj_drift_step(x: wp.vec3, v: wp.vec3, dt: float, adj_ret: wp.vec3):
    grad_v = dt * adj_ret
    alpha = wp.length(grad_v)

    grad_v = grad_v * (1.0 / (alpha))  # 归一化梯度  

    wp.adjoint[v] += adj_ret 
    wp.adjoint[x] += grad_v
