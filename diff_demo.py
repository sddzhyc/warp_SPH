import warp as wp

@wp.kernel
def kernel_func(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    tid = wp.tid()
    y[tid] = x[tid] ** 2.0 + 3.0 * x[tid] + 1.0

@wp.kernel
def loss_kernel(x: wp.array(dtype=float), loss: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(loss, 0, x[tid])

x = wp.array([1.0, 2.0, 3.0], dtype=float, requires_grad=True)
y = wp.zeros_like(x)
loss = wp.zeros(1, dtype=float, requires_grad=True)

tape = wp.Tape()
with tape:
    wp.launch(kernel_func, x.shape, inputs=[x], outputs=[y])
    wp.launch(loss_kernel, y.shape, inputs=[y, loss])
    # wp.launch(kernel=loss, inputs=[y, l], device="cuda")
# tape.backward(grads={y: wp.ones_like(x)})
tape.backward(loss)
# tape.backward() #输出恒为0
# tape.backward(grads={y: x}) # [ 5. 14. 27.] ？？
print("loss:", loss.numpy())
print("y.grad: ",y.grad)
print("x.grad: ",x.grad)