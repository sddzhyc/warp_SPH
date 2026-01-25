import warp as wp

from kernel_func import wp


@wp.func
def norm_grad_vec3(state_t: wp.vec3):
    return state_t

# @wp.func_grad(norm_grad_vec3) # depracated
def adj_norm_grad_vec3(state_t: wp.vec3, adj_ret: wp.vec3):
    alpha = wp.length(adj_ret)
    if alpha <= 100.0:
        wp.adjoint[state_t] += adj_ret
    else:
        grad_a = adj_ret * (1.0 / (alpha))  # 归一化梯度  

        wp.adjoint[state_t] += grad_a

@wp.func
def norm_grad_quat(state_t: wp.quat):
    return state_t

# @wp.func_grad(norm_grad_quat) # depracated
def adj_norm_grad_quat(state_t: wp.quat, adj_ret: wp.quat):
    alpha = wp.length(adj_ret)
    if alpha <= 100.0:
        wp.adjoint[state_t] += adj_ret
    else:
        grad_a = adj_ret * (1.0 / (alpha))  # 归一化梯度  

        wp.adjoint[state_t] += grad_a


# @wp.func
# def norm_grad_sum(state_t: wp.vec3, states_length: float):
#     weight = 1.0 / (states_length + 1e-10)
#     return state_t * weight

@wp.kernel
def sum_L2_states_t(
    grad_array: wp.array(dtype=wp.vec3), sum_L2_out: wp.array(dtype=float)
):
    tid = wp.tid()
    wp.atomic_add(sum_L2_out, 0, wp.dot(grad_array[tid], grad_array[tid]))

@wp.kernel
def norm_states_grad(states_grad: wp.array(dtype=wp.vec3), states_length: float):
    tid = wp.tid()
    states_grad[tid] = norm_grad_vec3(states_grad[tid]) / (states_length + 1e-10)