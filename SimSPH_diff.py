import time
from SimSPH import SimSPH
from rigid_fluid_coupling import MaterialMarks, MaterialType, RigidBodies, compute_moving_boundary_volume, compute_static_boundary_volume, solve_rigid_body_diff, update_rigid_particle_info_diff
from sim_utils import export_ply_points
from sph_kernel_diff import *
from norm_grad_utils import sum_L2_states_t, norm_states_grad

import numpy as np
import warp as wp
import warp.optim
import math
import os
from diff_tasks.ball_demo_task import BallDemoTask
from diff_tasks.skip_stone_task import SkipStoneTask

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
def sum_grad_fluid(
    v_grad: wp.array(dtype=wp.vec3),
    fluid_grad_out: wp.array(dtype=wp.vec3),
    mtr: MaterialMarks
):
    tid = wp.tid()
    if mtr.material[tid] == MaterialType.FLUID:
        wp.atomic_add(fluid_grad_out, 0, v_grad[tid])

class SimSPH_diff(SimSPH):

    def __init__(self,config = None, container = None, stage_path="example_sph.usd", sim_steps=100, ply_path=None, lr=0.01, h_scale=1.0, use_custom_grad=False, use_norm_grad=False, verbose=False):
        super().__init__(config, container, 0, stage_path, ply_path)
        self.ply_path = ply_path
        self.sim_steps = sim_steps
        self.train_rate = lr
        self.use_custom_grad = use_custom_grad
        self.use_norm_grad = use_norm_grad
        self.init_diff_phys(self.sim_steps)
        self.init_diff_task()
        self.smoothing_length *= h_scale

    def init_diff_phys(self, sim_steps):
        self.sim_steps = sim_steps

        self.x_arrays = []
        self.v_arrays = []
        self.rho_arrays = []
        self.pressure_arrays = []
        self.a_arrays = []
        self.viscous_forces_arrays = []
        self.pressure_forces_arrays = []
        self.debug_val_arrays = []

        # Initialize arrays for each time step (t=0..sim_steps)
        for _ in range(self.sim_steps + 1):
            self.x_arrays.append(wp.zeros_like(self.x, requires_grad=True))
            self.v_arrays.append(wp.zeros_like(self.v, requires_grad=True))
            self.rho_arrays.append(wp.zeros_like(self.rho, requires_grad=True))
            self.pressure_arrays.append(wp.zeros_like(self.pressure, requires_grad=True))
            self.a_arrays.append(wp.zeros_like(self.a, requires_grad=True))
            # self.viscous_forces_arrays.append(wp.zeros_like(self.v, requires_grad=True))
            # self.pressure_forces_arrays.append(wp.zeros_like(self.v, requires_grad=True))
            # self.debug_val_arrays.append(wp.zeros_like(wp.zeros(self.particle_max_num, dtype=wp.float32)))
        print(f"Initialized differentiable simulation for {self.sim_steps} steps (no segments).")
        # Copy initial state to first arrays
        wp.copy(self.x_arrays[0], self.x)
        wp.copy(self.v_arrays[0], self.v)
        wp.copy(self.rho_arrays[0], self.rho)
        # Optimize memory: Use a single buffer for all steps (Warning: Gradients will be incorrect if used in backward pass)
        temp_visc_buffer = wp.zeros_like(self.v, requires_grad=False)
        temp_pres_buffer = wp.zeros_like(self.v, requires_grad=False)
        temp_debug_buffer = wp.zeros(self.particle_max_num, dtype=wp.float32)
        self.viscous_forces_arrays = [temp_visc_buffer] * (self.sim_steps + 1)
        self.pressure_forces_arrays = [temp_pres_buffer] * (self.sim_steps + 1)
        self.debug_val_arrays = [temp_debug_buffer] * (self.sim_steps + 1)

        # No segment checkpoints or saved grads

        # Loss
        self.loss = wp.zeros((1,), dtype=float, requires_grad=True)

        self.tapes = [] # List of tapes for each step
        self.loss_tape = None # Tape for loss computation
        self.forward_graph = None
        self.backward_graph = None
        self.zero_tape_graph = None

        # Rigid body checkpointing
        self.rigid_x_arrays = []
        self.rigid_v_arrays = []
        self.rigid_omega_arrays = []
        self.rigid_quaternion_arrays = []

        self.rigid_force_arrays = []
        self.rigid_torque_arrays = []
        self.rigid_inertia_arrays = []
        self.rigid_inv_inertia_arrays = []
        
        # No segment checkpoints or saved grads for rigid bodies

        if self.num_objects > 0:
            for _ in range(self.sim_steps + 1):
                self.rigid_x_arrays.append(wp.zeros_like(self.rbs.rigid_x, requires_grad=True))
                self.rigid_v_arrays.append(wp.zeros_like(self.rbs.rigid_v, requires_grad=True))
                self.rigid_omega_arrays.append(wp.zeros_like(self.rbs.rigid_omega, requires_grad=True))
                self.rigid_quaternion_arrays.append(wp.zeros_like(self.rbs.rigid_quaternion, requires_grad=True))
                self.rigid_force_arrays.append(wp.zeros_like(self.rbs.rigid_force, requires_grad=True))
                self.rigid_torque_arrays.append(wp.zeros_like(self.rbs.rigid_torque, requires_grad=True))
                self.rigid_inertia_arrays.append(wp.zeros_like(self.rbs.rigid_inertia, requires_grad=True))
                self.rigid_inv_inertia_arrays.append(wp.zeros_like(self.rbs.rigid_inv_inertia, requires_grad=True))

            wp.copy(self.rigid_x_arrays[0], self.rbs.rigid_x)
            wp.copy(self.rigid_v_arrays[0], self.rbs.rigid_v)
            wp.copy(self.rigid_omega_arrays[0], self.rbs.rigid_omega)
            wp.copy(self.rigid_quaternion_arrays[0], self.rbs.rigid_quaternion)
            wp.copy(self.rigid_inertia_arrays[0], self.rbs.rigid_inertia)
            wp.copy(self.rigid_inv_inertia_arrays[0], self.rbs.rigid_inv_inertia)

    def init_diff_task(self):
        # Initialize Task (which handles targets and optimizer)
        if self.ply_path and ("skip_stone" in self.ply_path):
            print("Initializing SkipStoneTask based on ply_path.")
            self.task = SkipStoneTask(self)
        else:
            self.task = BallDemoTask(self)
            
        self.optimizer = self.task.optimizer
        # self.opt_var maintained by self.opt_v_fluid set by task

    def clear_grad(self):
        # 1. 重置 Tape：防止计算图在多次 backward 中累积，导致内存爆炸和错误回传
        for tape in self.tapes:
            tape.reset()
            tape.zero()
        self.tapes = []
        if self.loss_tape:
            self.loss_tape.reset()
            self.loss_tape.zero()
            self.loss_tape = None

        # 2. 清零 Loss：防止上一轮 Loss 累积
        self.loss.zero_()
        
        # 3. 清零优化变量（初始条件）的梯度
        # Call task-specific gradient clearing
        self.task.clear_grad()

        # 4. 清零所有中间状态数组的梯度
        # 注意：Warp 中如果数组 create 时设置了 requires_grad=True，Warp 会分配 grad 内存。
        # 虽然 Tape reset 会清空图，但为了保险起见，清空历史梯度值防止累加
        for arrs in [self.x_arrays, self.v_arrays, self.rho_arrays, self.pressure_arrays, self.a_arrays,
                     self.viscous_forces_arrays, self.pressure_forces_arrays]:
            for arr in arrs:
                if arr.grad:
                    arr.grad.zero_()
        
        if self.num_objects > 0:
            for arrs in [self.rigid_x_arrays, self.rigid_v_arrays, self.rigid_omega_arrays, 
                         self.rigid_quaternion_arrays, self.rigid_force_arrays, self.rigid_torque_arrays, 
                         self.rigid_inertia_arrays, self.rigid_inv_inertia_arrays]:
                for arr in arrs:
                    if arr.grad:
                        arr.grad.zero_()
        # print("Gradients cleared.")
        
    def reset(self):
        # Reset initial state
        super().reset()
        wp.copy(self.x_arrays[0], self.x)
        wp.copy(self.v_arrays[0], self.v)
        wp.copy(self.rho_arrays[0], self.rho)
        
        if self.num_objects > 0:
            wp.copy(self.rigid_x_arrays[0], self.rbs.rigid_x)
            wp.copy(self.rigid_v_arrays[0], self.rbs.rigid_v)
            wp.copy(self.rigid_omega_arrays[0], self.rbs.rigid_omega)
            wp.copy(self.rigid_quaternion_arrays[0], self.rbs.rigid_quaternion)
            wp.copy(self.rigid_force_arrays[0], self.rbs.rigid_force)
            wp.copy(self.rigid_torque_arrays[0], self.rbs.rigid_torque)
            wp.copy(self.rigid_inertia_arrays[0], self.rbs.rigid_inertia)
            wp.copy(self.rigid_inv_inertia_arrays[0], self.rbs.rigid_inv_inertia)

        for t in range(0, self.sim_steps + 1):
            self.pressure_arrays[t].zero_()
            self.a_arrays[t].zero_()
            # self.viscous_forces_arrays[t].zero_()
            # self.pressure_forces_arrays[t].zero_()
            self.rigid_force_arrays[t].zero_()
            self.rigid_torque_arrays[t].zero_()
            self.debug_val_arrays[t].zero_()

    def normalize_single_state_grad(self, grad_array):
        """Helper to normalize a single state gradient array using L2 norm."""
        if grad_array.grad:
            sum_sq = wp.zeros(1, dtype=float)
            wp.launch(sum_L2_states_t, dim=self.particle_max_num, inputs=[grad_array.grad, sum_sq])
            l2 = np.sqrt(sum_sq.numpy()[0])
            #print(f"Normalizing gradient with L2 norm: {l2}")
            if l2 > 1e-10:
                wp.launch(norm_states_grad, dim=self.particle_max_num, inputs=[grad_array.grad, l2])
# 15.136797 -180.5951     15.135387
# 15.137694 -180.59273    15.134274
    def backward(self):
        # If I want to split tapes, I should also follow this pattern or rely on forward() being called before backward().
        # Standard PyTorch/Warp pattern: 
        #   optimizer.zero_grad() -> clear_grad
        #   loss = model.forward() -> builds graph
        #   loss.backward() -> traverses graph
        # Yes: `with self.tape: ... step(t) ...`
        # This is inefficient if forward was already run.
        # But maybe `forward` method was just for inference?
        # The user's `backward` method:
        #   self.clear_grad()
        #   self.reset()
        #   with self.tape: ... step(t) ...

        self.clear_grad()
        self.reset()
        self.tapes = []

        t_forward_start = time.perf_counter()

        for t in range(self.sim_steps):
            # advance state from t -> t+1
            current_tape = wp.Tape()
            self.tapes.append(current_tape)
            # Set optimized initial rigid velocities if applicable
            self.task.init_simulation_state(t)

            self.step(t)
            # print(f"Completed forward step {t+1}/{self.sim_steps}")
        
        # Loss
        self.loss_tape = wp.Tape()
        with self.loss_tape:
            self.task.compute_loss()

        wp.synchronize()  # 强制等待 GPU 完成并刷新输出
        t_forward_end = time.perf_counter()
        print(f"Forward simulation time: {t_forward_end - t_forward_start:.4f}s")
        print(f"Completed differentiable forward pass {self.sim_steps} steps, Starting backward pass...")
        t_backward_start = time.perf_counter()
        
        # Backward Loop
        self.loss_tape.backward(self.loss)
        
        # We iterate backwards from end to start
        for t in reversed(range(self.sim_steps)):
            # 1. Normalize gradients at t+1 before propagating to t?
            # Actually, gradients flow from t+1 to t.
            # After loss_tape.backward(), grads at step `sim_steps` are populated.
            # We should normalize them before backing through step `sim_steps-1 to sim_steps`.
            
            # Normalize gradients at t+1 (x_arrays[t+1], v_arrays[t+1])
            if self.use_norm_grad:
                self.normalize_single_state_grad(self.x_arrays[t+1])
                self.normalize_single_state_grad(self.v_arrays[t+1])
            
            # 2. Propagate through step t (t -> t+1) to get grads at t
            self.tapes[t].backward()
            
            # (Loop continues to t-1, where we will normalize grads at t, which we just computed)
        
        # Finally, normalize grads at t=0
        if self.use_norm_grad:
            self.normalize_single_state_grad(self.x_arrays[0])
            self.normalize_single_state_grad(self.v_arrays[0])

        wp.synchronize()
        t_backward_end = time.perf_counter()
        print(f"Backward propagation time: {t_backward_end - t_backward_start:.4f}s")
        print(f"Total backward() time: {t_backward_end - t_forward_start:.4f}s")

        # Backward through initialization
        # init_tape.backward()  # Removed as initialization is now handled in step(0)
        
        # Result logic...
        # self.tape.visualize("sph_graph.dot") # Cannot visualize split tapes easily as one graph



    def norm_final_grad(self):
        sum_sq = wp.zeros(1, dtype=float)

        # TODO: 检查代码冗余
        # 1. Normalize x_arrays[0].grad
        if self.x_arrays[0].grad:
            wp.launch(sum_L2_states_t, dim=self.particle_max_num, inputs=[self.x_arrays[0].grad, sum_sq])
            l2_x = np.sqrt(sum_sq.numpy()[0])
            if l2_x > 1e-10:
                wp.launch(norm_states_grad, dim=self.particle_max_num, inputs=[self.x_arrays[0].grad, l2_x])

        # 2. Normalize v_arrays[0].grad
        sum_sq.zero_()
        if self.v_arrays[0].grad:
            wp.launch(sum_L2_states_t, dim=self.particle_max_num, inputs=[self.v_arrays[0].grad, sum_sq])
            l2_v = np.sqrt(sum_sq.numpy()[0])
            if l2_v > 1e-10:
                wp.launch(norm_states_grad, dim=self.particle_max_num, inputs=[self.v_arrays[0].grad, l2_v])
            
            # Compute final opt_var gradient if it's fluid velocity
            # We delegate this to the task if needed, or handle it generically
            if hasattr(self.task, 'norm_final_grad'):
                self.task.norm_final_grad(self.v_arrays[0].grad, self.materialMarks)

    def step(self, t):
            self.time_step = t
            current_tape = self.tapes[-1]
            # use state at time t as input, write results to time t+1
            x_in = self.x_arrays[t]
            # wp.copy(x_out, x_in)
            # wp.copy(v_out, v_in)

            if self.num_objects > 0:
                # change rbs pointer to use current step arrays
                self.rbs.rigid_x = self.rigid_x_arrays[t]
                self.rbs.rigid_v = self.rigid_v_arrays[t]
                self.rbs.rigid_omega = self.rigid_omega_arrays[t] 
                self.rbs.rigid_quaternion = self.rigid_quaternion_arrays[t]
                self.rbs.rigid_force = self.rigid_force_arrays[t]
                self.rbs.rigid_torque = self.rigid_torque_arrays[t]
                self.rbs.rigid_inertia = self.rigid_inertia_arrays[t]
                self.rbs.rigid_inv_inertia = self.rigid_inv_inertia_arrays[t]
                # for _ in range(self.sim_step_to_frame_ratio):
           
            # Common Grid Build
            with wp.ScopedTimer("grid build", active=self.verbose):
                # build grid
                self.grid.build(x_in, self.grid_size)
            
            # Substep (Physics Solver)
            self.sub_step(t)
            
            # Common Post-Processing: Boundary Handling & Rigid Body Update
            x_out = self.x_arrays[t+1]
            v_out = self.v_arrays[t+1]
            
            # Enforce boundary on t+1
            wp.launch(
                kernel=enforce_boundary_3D_warp,
                dim=self.particle_max_num,
                inputs=[
                    x_out,
                    v_out,
                    self.materialMarks,
                    wp.vec3(*self.domain_start),
                    wp.vec3(*self.domain_end),
                    self.padding,
                ]
            )
            
            with current_tape:
                # Rigid Body Update (t -> t+1)
                if self.num_objects > 0:
                    g = wp.vec3(0.0, self.gravity, 0.0)
                    wp.launch(
                        kernel=solve_rigid_body_diff,
                        dim=self.num_objects,
                        inputs=[
                            self.rigid_x_arrays[t],
                            self.rigid_v_arrays[t],
                            self.rigid_force_arrays[t],
                            self.rbs.rigid_mass,
                            self.rigid_quaternion_arrays[t],
                            self.rigid_omega_arrays[t],
                            self.rigid_torque_arrays[t],
                            self.rbs.rigid_inertia0,
                            self.rigid_inv_inertia_arrays[t],
                            g,
                            self.dt,
                        ],
                        outputs=[                            
                            self.rigid_x_arrays[t+1],
                            self.rigid_v_arrays[t+1],
                            self.rigid_force_arrays[t+1],
                            self.rigid_quaternion_arrays[t+1],
                            self.rigid_omega_arrays[t+1],
                            self.rigid_torque_arrays[t+1],
                            self.rigid_inertia_arrays[t+1],
                            self.rigid_inv_inertia_arrays[t+1]]
                    )
                    
                    wp.launch(
                        kernel=update_rigid_particle_info_diff,
                        dim=self.particle_max_num,
                        inputs=[x_out, v_out, self.x_0,
                            self.object_id,
                            self.materialMarks,
                            self.rbs.rigid_rest_cm,
                            self.rigid_x_arrays[t+1],
                            self.rigid_quaternion_arrays[t+1],
                            self.rigid_v_arrays[t+1],
                            self.rigid_omega_arrays[t+1],
                        ]
                    )
                    
            self.sim_time += self.frame_dt

            # print(f"self.rigid_quaternion_arrays[{t}]:", self.rigid_quaternion_arrays[t].numpy()[1])

    def sub_step(self, t):
        current_tape = self.tapes[-1]
        x_in = self.x_arrays[t]
        v_in = self.v_arrays[t]
        v_out = self.v_arrays[t+1]
        rho_out = self.rho_arrays[t+1]
        pressure_out = self.pressure_arrays[t+1]
        a_out = self.a_arrays[t+1]
        # with current_tape:
        with wp.ScopedTimer("forces", active=self.verbose):
            wp.launch(
                kernel=compute_moving_boundary_volume,
                dim=self.particle_max_num,
                inputs=[self.grid.id, x_in, self.m_V, self.density_normalization_no_mass, self.smoothing_length,
                        self.materialMarks],
            )
        with wp.ScopedTimer("compute density", active=self.verbose):
            # compute density of points
            wp.launch(
                kernel=compute_density,
                dim=self.particle_max_num,
                inputs=[self.grid.id, self.x_arrays[t],
                        1.0, # cubic kernel don't need normalization
                        self.smoothing_length,
                self.materialMarks, self.m_V, self.base_density],
            outputs=[rho_out]
            )

            wp.launch(
                kernel=compute_pressure,
                dim=self.particle_max_num,
            inputs=[rho_out, self.materialMarks,
                        self.stiffness, self.exponent, self.base_density],
                outputs=[pressure_out]
            )

        with wp.ScopedTimer("compute non pressure forces", active=self.verbose):
            wp.launch(
                kernel=compute_non_pressure_forces,
                dim=self.particle_max_num,
                inputs=[
                    self.grid.id,
                    self.x_arrays[t],
                    self.v_arrays[t],
                    self.rho_arrays[t],
                    self.dynamic_visc,
                    self.smoothing_length,
                    self.materialMarks,
                    self.m_V,
                    self.base_density,
                    self.object_id,
                    self.rbs
                ],
                outputs=[self.viscous_forces_arrays[t]]
            )

        with wp.ScopedTimer("compute pressure force and acceleration", active=self.verbose):
            # get new acceleration
            wp.launch(
                kernel=get_acceleration,
                dim=self.particle_max_num,
                inputs=[
                    self.grid.id,
                    self.x_arrays[t],
                    self.v_arrays[t],
                    rho_out,
                    pressure_out,
                    self.stiffness,
                    self.exponent,
                    self.base_density,
                    self.gravity,
                    1.0,  # cubic kernel don't need normalization
                    self.dynamic_visc, # cubic kernel only use dynamic_visc
                    self.smoothing_length,
                    self.materialMarks,
                    self.m_V,
                    self.pressure_forces_arrays[t],
                    self.viscous_forces_arrays[t],
                    self.debug_val_arrays[t],
                    self.object_id,
                ],
                outputs=[self.a_arrays[t]]
            )
        with current_tape:                
            with wp.ScopedTimer("compute rigid force and torque", active=self.verbose):
                wp.launch(
                    kernel=compute_rigid_force_torque,
                    dim=self.particle_max_num,
                    inputs=[
                        self.grid.id,
                        self.x_arrays[t],
                        self.v_arrays[t],
                        rho_out,
                        pressure_out,
                        self.base_density,
                        1.0,  # cubic kernel don't need normalization
                        self.smoothing_length,
                        self.materialMarks,
                        self.m_V,
                        self.object_id,
                        self.debug_val_arrays[t],
                        self.rigid_x_arrays[t],
                        self.use_custom_grad,
                    ],
                    outputs=[
                        self.rigid_force_arrays[t],
                        self.rigid_torque_arrays[t],
                        self.a_arrays[t]]
                )

            with wp.ScopedTimer("advection", active=self.verbose):
                # kick
                wp.launch(
                    kernel=kick,
                    dim=self.particle_max_num,
                    inputs=[self.a_arrays[t], self.dt, self.v_arrays[t]],
                    outputs=[self.v_arrays[t+1]]
                )

                # drift
                wp.launch(
                    kernel=drift,
                    dim=self.particle_max_num,
                    inputs=[self.x_arrays[t], self.v_arrays[t+1], self.dt],
                    outputs=[self.x_arrays[t+1]]
                )

            # print(f"self.rigid_quaternion_arrays[{t}]:", self.rigid_quaternion_arrays[t].numpy()[1])
        # with wp.ScopedTimer("render"):
        #     self.renderer.begin_frame(self.sim_time)
        #     self.renderer.render_points(
        #         points=self.x.numpy(), radius=self.smoothing_length, name="points", colors=(0.8, 0.3, 0.2)
        #     )
        #     self.renderer.end_frame()

    def export_ply_from_diff(self, series_prefix, time_step, cnt_ply, verbose=False):
        np_pos = self.x_arrays[time_step].numpy()
        np_rho = self.rho_arrays[time_step].numpy()
        # m_V: use computed per-particle if available, else fallback to constant m_V0
        np_mV = self.m_V.numpy()
        np_obj_id = self.object_id.numpy()
        # also export per-particle force diagnostics (split vec3 into scalar components)
        pf = self.pressure_arrays[time_step].numpy()
        vf = self.viscous_forces_arrays[time_step].numpy()
        np_a = self.a_arrays[time_step].numpy()
        np_v = self.v_arrays[time_step].numpy()
        np_vel = np.linalg.norm(np_v, axis=1)
        debug_val = self.debug_val_arrays[time_step].numpy()

        # Gradients
        grad_x = self.x_arrays[time_step].grad.numpy()
        grad_v = self.v_arrays[time_step].grad.numpy()
        grad_rho = self.rho_arrays[time_step].grad.numpy()
        grad_p = self.pressure_arrays[time_step].grad.numpy()
        grad_a = self.a_arrays[time_step].grad.numpy()

        out_path = series_prefix.format(cnt_ply)
        export_ply_points(out_path, np_pos.astype(np.float32), {
            'rho': np_rho.astype(np.float32),
            'pressure': pf.astype(np.float32),
            'mV': np_mV.astype(np.float32),
            'object_id': np_obj_id.astype(np.int32),
            'debug_val': debug_val.astype(np.float32),
            # 'pressure_fx': pf[:,0].astype(np.float32),
            # 'pressure_fy': pf[:,1].astype(np.float32),
            # 'pressure_fz': pf[:,2].astype(np.float32),
            'viscous_fx': vf[:,0].astype(np.float32),
            'viscous_fy': vf[:,1].astype(np.float32),
            'viscous_fz': vf[:,2].astype(np.float32),
            'ax': np_a[:,0].astype(np.float32),
            'ay': np_a[:,1].astype(np.float32),
            'az': np_a[:,2].astype(np.float32),
            'vx': np_v[:,0].astype(np.float32),
            'vy': np_v[:,1].astype(np.float32),
            'vz': np_v[:,2].astype(np.float32),
            'vel': np_vel.astype(np.float32),
            'material': self.materialMarks.material.numpy().astype(np.int32),
            'is_dynamic': self.materialMarks.is_dynamic.numpy().astype(np.int32),

            # Gradients
            'grad_x_x': grad_x[:,0].astype(np.float32),
            'grad_x_y': grad_x[:,1].astype(np.float32),
            'grad_x_z': grad_x[:,2].astype(np.float32),
            'grad_v_x': grad_v[:,0].astype(np.float32),
            'grad_v_y': grad_v[:,1].astype(np.float32),
            'grad_v_z': grad_v[:,2].astype(np.float32),
            'grad_rho': grad_rho.astype(np.float32),
            'grad_pressure': grad_p.astype(np.float32),
            'grad_a_x': grad_a[:,0].astype(np.float32),
            'grad_a_y': grad_a[:,1].astype(np.float32),
            'grad_a_z': grad_a[:,2].astype(np.float32),
        })
    
        if verbose:
            print(f"Exporting frame {cnt_ply} to PLY on time step {time_step}.")

    def rigid_grad_print(self, rigid_id, time_step):
        print(f"--- Rigid Body {rigid_id} Gradients at Step {time_step} ---")
        
        # Helper to safely get and print gradient
        def print_grad(name, array_list):
            if time_step < len(array_list):
                wp_array = array_list[time_step]
                if wp_array.grad is not None:
                    grad_data = wp_array.grad.numpy()
                    if rigid_id < len(grad_data):
                        print(f"{name}: {grad_data[rigid_id]}")
                    else:
                        print(f"{name}: ID out of range")
                else:
                    print(f"{name}: No gradient")
            else:
                print(f"{name}: Time step out of range")

        print_grad("Pos Grad", self.rigid_x_arrays)
        print_grad("Vel Grad", self.rigid_v_arrays)
        print_grad("Omega Grad", self.rigid_omega_arrays)
        print_grad("Quat Grad", self.rigid_quaternion_arrays)
        print_grad("Force Grad", self.rigid_force_arrays)
        print_grad("Torque Grad", self.rigid_torque_arrays)

    def print_all_rigid_grads(self):
        """打印所有模拟步中刚体质心位置的梯度，以及力的值和梯度"""
        print(f"=== All Rigid Body Center of Mass Gradients ({len(self.rigid_x_arrays)} steps) ===")
        for t in range(len(self.rigid_x_arrays)):
            # 打印刚体 1 (假设 ID 为 1)
            target_id = 1
            if target_id >= self.num_objects:
                continue

            grad_str = "No Grad"
            if self.rigid_x_arrays[t].grad is not None:
                grad_data = self.rigid_x_arrays[t].grad.numpy()
                grad_str = f"{grad_data[target_id]}"
            
            print(f"Step {t:03d} | Body {target_id} X Grad: {grad_str}")

        print("="*50)    
        print(f"=== All Rigid Force Values and Gradients ({len(self.rigid_force_arrays)} steps) ===")
        for t in range(len(self.rigid_force_arrays)):
            target_id = 1
            if target_id >= self.num_objects:
                continue

            force_val = self.rigid_force_arrays[t].numpy()[target_id]
            
            grad_str = "No Grad"
            if self.rigid_force_arrays[t].grad is not None:
                grad_data = self.rigid_force_arrays[t].grad.numpy()
                grad_str = f"{grad_data[target_id]}"
            
            print(f"Step {t:03d} | Body {target_id} Force Val: {force_val} | Force Grad: {grad_str}")       