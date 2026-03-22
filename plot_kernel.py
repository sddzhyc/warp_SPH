import numpy as np
import matplotlib.pyplot as plt
import warp as wp
import kernel_func

wp.init()

@wp.kernel
def compute_kernel_derivative(
    x_vals: wp.array(dtype=float),
    h: float,
    out_grads: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    x = x_vals[tid]
    r_vec = wp.vec3(x, 0.0, 0.0)
    
    # Forward evaluation
    out_grads[tid] = kernel_func.cubic_kernel_derivative_custom(r_vec, h)


@wp.kernel
def compute_cubic_kernel_value(
    x_vals: wp.array(dtype=float),
    h: float,
    out_vals: wp.array(dtype=float)
):
    tid = wp.tid()
    x = x_vals[tid]
    r_vec = wp.vec3(x, 0.0, 0.0)
    out_vals[tid] = kernel_func.cubic_kernel(r_vec, h)

def np_adj_cubic_kernel_derivative(x_vals, support_radius):
    h = support_radius * 1.0
    dim = 3
    k_base = 8.0 / np.pi
    k = 6.0 * k_base / (h ** dim)
    
    r_norms = np.abs(x_vals)
    q = r_norms / h
    
    adj_r_x = np.zeros_like(x_vals)
    adj_ret_x = 1.0
    
    mask = (r_norms > 1e-5) & (q <= 1.0)
    
    r_norm = r_norms[mask]
    q_mask = q[mask]
    x_mask = x_vals[mask]
    
    inv_h = 1.0 / h
    inv_r_norm = 1.0 / r_norm
    
    coeff = np.zeros_like(q_mask)
    mask_q_leq_half = q_mask <= 0.5
    mask_q_gt_half = ~mask_q_leq_half
    
    coeff[mask_q_leq_half] = k * q_mask[mask_q_leq_half] * (3.0 * q_mask[mask_q_leq_half] - 2.0)
    factor = 1.0 - q_mask[mask_q_gt_half]
    coeff[mask_q_gt_half] = -k * factor * factor
    
    scale = inv_r_norm * inv_h
    grad_q_x = x_mask * scale
    
    adj_coeff = adj_ret_x * grad_q_x
    adj_grad_q_x = adj_ret_x * coeff
    adj_scale = adj_grad_q_x * x_mask
    adj_r_curr_x = adj_grad_q_x * scale
    
    adj_inv_r_norm = adj_scale * inv_h
    adj_r_norm_val = -adj_inv_r_norm * inv_r_norm * inv_r_norm
    
    adj_q = np.zeros_like(q_mask)
    adj_q[mask_q_leq_half] = adj_coeff[mask_q_leq_half] * k * (6.0 * q_mask[mask_q_leq_half] - 2.0)
    factor2 = 1.0 - q_mask[mask_q_gt_half]
    adj_q[mask_q_gt_half] = adj_coeff[mask_q_gt_half] * (2.0 * k * factor2)
    
    adj_r_norm_val += adj_q * inv_h
    adj_r_curr_x += adj_r_norm_val * (x_mask * inv_r_norm)
    
    adj_r_x[mask] = adj_r_curr_x
    
    return adj_r_x

def wp_adj_cubic_kernel_derivative(x_vals_np, h):
    sample_count = x_vals_np.shape[0]
    x_vals = wp.array(x_vals_np, dtype=float, requires_grad=True)
    out_grads = wp.zeros(sample_count, dtype=wp.vec3, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch(
            kernel=compute_kernel_derivative,
            dim=sample_count,
            inputs=[x_vals, h, out_grads]
        )

    seed_np = np.zeros((sample_count, 3), dtype=np.float32)
    seed_np[:, 0] = 1.0
    out_seed = wp.array(seed_np, dtype=wp.vec3)
    tape.backward(grads={out_grads: out_seed})

    grad_x_vals = out_grads.numpy()[:, 0]
    adj_grad_x_vals = x_vals.grad.numpy().astype(np.float32)
    return grad_x_vals, adj_grad_x_vals


def wp_eval_cubic_kernel(x_vals_np, h):
    sample_count = x_vals_np.shape[0]
    x_vals = wp.array(x_vals_np, dtype=float)
    out_vals = wp.zeros(sample_count, dtype=float)

    wp.launch(
        kernel=compute_cubic_kernel_value,
        dim=sample_count,
        inputs=[x_vals, h, out_vals]
    )

    return out_vals.numpy().astype(np.float32)

def plot_kernel_derivative():
    h = 1.0
    # Sample points along the positive x-axis only (x > 0)
    x_vals_np = np.linspace(1e-3, 2.5 * h, 500).astype(np.float32)

    # Extract the evaluations (both forward gradient and adjoint gradient from Warp autograd)
    grad_x_vals, adj_grad_x_vals = wp_adj_cubic_kernel_derivative(x_vals_np, h)
    np_adj_grad_x_vals = np_adj_cubic_kernel_derivative(x_vals_np, h)
    kernel_vals = wp_eval_cubic_kernel(x_vals_np, h)

    
    # kernel_vals *= 0.01
    np_adj_grad_x_vals *= 0.5
    adj_grad_x_vals *= 1.0

    adj_min = float(np.min(adj_grad_x_vals))
    adj_max = float(np.max(adj_grad_x_vals))
    adj_span = adj_max - adj_min
    y_pad = adj_span * 0.1

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals_np, grad_x_vals, label='Gradient $\\nabla W$ (x-component)', linewidth=2)
    plt.plot(x_vals_np, adj_grad_x_vals, label='new Gradient 50*$\\nabla^2 W$', linewidth=2, linestyle='-.')
    plt.plot(x_vals_np, np_adj_grad_x_vals, label='Original Gradient 0.5*$\\nabla^2 W$', linewidth=2, linestyle='--')
    plt.plot(x_vals_np, kernel_vals, label='Kernel Value $W(x)$', linewidth=2, color='tab:purple')
    
    # Mark key regions
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=h, color='r', linestyle='--', label='Support Radius (h)')
    plt.axvline(x=0.5*h, color='g', linestyle=':', label='0.5h (Spline knot)')
    
    plt.title(f'Cubic Kernel Derivative Profile (dim=3, h={h})')
    plt.xlabel('Position x')
    plt.ylabel('Derivative Value (Gradient along x)')
    plt.ylim(adj_min - y_pad, adj_max + y_pad)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # output_filename = 'cubic_kernel_derivative_plot.png'
    # plt.savefig(output_filename)
    # print(f"Plot saved to {output_filename}")
    print("Showing plot...")
    plt.show()

if __name__ == "__main__":
    plot_kernel_derivative()
