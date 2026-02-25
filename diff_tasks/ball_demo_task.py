import warp as wp
import numpy as np
from norm_grad_utils import norm_grad_vec3
from rigid_fluid_coupling import MaterialMarks, MaterialType
from .base_task import Task

@wp.kernel
def assign_initial_fluid_velocity(
    v: wp.array(dtype=wp.vec3),
    v_fluid_val: wp.array(dtype=wp.vec3),
    mtr: MaterialMarks
):
    tid = wp.tid()
    if mtr.material[tid] == MaterialType.FLUID:
        v[tid] = norm_grad_vec3(v_fluid_val[0])

@wp.kernel
def compute_loss_kernel(
    x: wp.array(dtype=wp.vec3),
    target_x: wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=float)
):
    tid = wp.tid()
    diff = x[tid] - target_x[tid]
    l = wp.dot(diff, diff)
    wp.atomic_add(loss, 0, l)

@wp.kernel
def compute_rigid_loss_kernel(
    rigid_x: wp.array(dtype=wp.vec3),
    target_rigid_x: wp.array(dtype=wp.vec3),
    rigid_q: wp.array(dtype=wp.quat),
    target_rigid_q: wp.array(dtype=wp.quat),
    loss: wp.array(dtype=float)
):
    tid = wp.tid()

    if tid == 0: # 排除rigid body0（流体块）
        return
    # Position loss per rigid body
    diff_pos = norm_grad_vec3(rigid_x[tid] - target_rigid_x[tid])
    l_pos = wp.dot(diff_pos, diff_pos)
    
    # Rotation loss (0.5 * ||q - tq||^2)
    q = rigid_q[tid]
    tq = target_rigid_q[tid]
    # wp.printf("Rigid body %d: q = (%.9f, %.9f, %.9f, %.9f), tq = (%.9f, %.9f, %.9f, %.9f)\n", tid, q.x, q.y, q.z, q.w, tq.x, tq.y, tq.z, tq.w)
    l_rot = 0.5 * (wp.dot(q, q) + wp.dot(tq, tq) - 2.0 * wp.dot(q, tq))
    
    # Combine losses (add weights here if needed)
    total_loss = l_pos
    # wp.printf("Rigid body %d: Rotation loss = %.9e, Position loss = %.9e, Total loss = %.9e\n", tid, l_rot, l_pos, total_loss)

    wp.atomic_add(loss, 0, total_loss)

class BallDemoTask(Task):
    def __init__(self, sim):
        super().__init__(sim)
        self.target_x = None
        self.target_rigid_x = None
        self.target_rigid_q = None
        self.optimizer = None
        self.opt_v_fluid = None
        
        # Initialize targets
        self.init_targets()
        self.init_optimizer()

    def init_targets(self):
        
        # Rigid targets
        if self.sim.num_objects > 0:
            self.target_rigid_x = wp.zeros_like(self.sim.rbs.rigid_x)
            self.target_rigid_q = wp.zeros_like(self.sim.rbs.rigid_quaternion)
            
            # Default behavior: Initialize with current state
            wp.copy(self.target_rigid_x, self.sim.rbs.rigid_x)
            

            # Print for verification
            print("BallDemoTask initialized with target rigid quaternions:\n", self.target_rigid_q.numpy())

    def init_optimizer(self):
        # Optimize a single variable for fluid initial velocity
        # Assume first particle is fluid or representative
        self.opt_v_fluid = wp.array([self.sim.v.numpy()[0]], dtype=wp.vec3, requires_grad=True) 

        # Optimizer
        self.optimizer = wp.optim.Adam([self.opt_v_fluid], lr=self.sim.train_rate)
        
        # Also set Sim property (backwards compatibility or usage inside sim)
        self.sim.opt_v_fluid = self.opt_v_fluid
        self.sim.opt_var = self.opt_v_fluid

    def compute_loss(self):
        if self.sim.num_objects > 0:
            wp.launch(
                compute_rigid_loss_kernel,
                dim=self.sim.num_objects,
                inputs=[
                    self.sim.rigid_x_arrays[self.sim.sim_steps], 
                    self.target_rigid_x,
                    self.sim.rigid_quaternion_arrays[self.sim.sim_steps],
                    self.target_rigid_q,
                    self.sim.loss
                ]
            )
            
    def get_optimized_vars(self):
        return [self.opt_v_fluid]

    def init_simulation_state(self, t):
        if t == 0:
            with self.sim.tapes[-1]:
                wp.launch(
                    kernel=assign_initial_fluid_velocity,
                    dim=self.sim.particle_max_num,
                    inputs=[self.sim.v_arrays[0], self.opt_v_fluid, self.sim.materialMarks]
                )

    def clear_grad(self):
        if self.opt_v_fluid.grad:
            self.opt_v_fluid.grad.zero_()
