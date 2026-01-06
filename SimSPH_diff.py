import time
from SimSPH import SimSPH
from particle_system import ParticleSystem
from rigid_fluid_coupling import MaterialMarks, RigidBodies, compute_moving_boundary_volume, compute_static_boundary_volume, solve_rigid_body, solve_rigid_body_diff, update_rigid_particle_info
from sim_utils import export_ply_points
from sph_kernel_diff import *

import numpy as np
import warp as wp
import warp.optim
import math
import os

@wp.kernel
def compute_loss(
    x: wp.array(dtype=wp.vec3),
    target_x: wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=float)
):
    tid = wp.tid()
    diff = x[tid] - target_x[tid]
    l = wp.dot(diff, diff)
    wp.atomic_add(loss, 0, l)

@wp.kernel
def compute_rigid_loss(
    rigid_x: wp.array(dtype=wp.vec3),
    target_rigid_x: wp.array(dtype=wp.vec3),
    rigid_q: wp.array(dtype=wp.quat),
    target_rigid_q: wp.array(dtype=wp.quat),
    loss: wp.array(dtype=float)
):
    tid = wp.tid()

    # Position loss per rigid body
    # diff_pos = rigid_x[tid] - target_rigid_x[tid]
    # l_pos = wp.dot(diff_pos, diff_pos)
    
    # Rotation loss (0.5 * ||q - tq||^2)
    q = rigid_q[tid]
    tq = target_rigid_q[tid]
    # wp.printf("Rigid body %d: q = (%.9f, %.9f, %.9f, %.9f), tq = (%.9f, %.9f, %.9f, %.9f)\n", tid, q.x, q.y, q.z, q.w, tq.x, tq.y, tq.z, tq.w)
    l_rot = 0.5 * (wp.dot(q, q) + wp.dot(tq, tq) - 2.0 * wp.dot(q, tq))
    
    # Combine losses (add weights here if needed)
    total_loss = l_rot
    # wp.printf("Rigid body %d: Rotation loss = %.9e, Total loss = %.9e\n", tid, l_rot, total_loss)

    wp.atomic_add(loss, 0, total_loss)

class SimSPH_diff(SimSPH):

    def __init__(self,config = None, container: ParticleSystem = None, stage_path="example_sph.usd", sim_steps=100, ply_path=None):
        super().__init__(config, container, stage_path, ply_path)
        self.sim_steps = sim_steps
        self.init_diff_phys(self.sim_steps)


    def init_diff_phys(self, sim_steps):
        self.sim_steps = sim_steps

        self.x_arrays = []
        self.v_arrays = []
        self.rho_arrays = []
        self.pressure_arrays = []
        self.a_arrays = []
        
        # Initialize arrays for each time step (t=0..sim_steps)
        for _ in range(self.sim_steps + 1):
            self.x_arrays.append(wp.zeros_like(self.x, requires_grad=True))
            self.v_arrays.append(wp.zeros_like(self.v, requires_grad=True))
            self.rho_arrays.append(wp.zeros_like(self.rho, requires_grad=True))
            self.pressure_arrays.append(wp.zeros_like(self.pressure, requires_grad=True))
            self.a_arrays.append(wp.zeros_like(self.a, requires_grad=True))
        print(f"Initialized differentiable simulation for {self.sim_steps} steps (no segments).")
        # Copy initial state to first arrays
        wp.copy(self.x_arrays[0], self.x)
        wp.copy(self.v_arrays[0], self.v)
        wp.copy(self.rho_arrays[0], self.rho)
        # pressure and a are computed, so 0 is fine or copy if initialized

        # No segment checkpoints or saved grads

        # Loss
        self.loss = wp.zeros((1,), dtype=float, requires_grad=True)
        # Target (dummy for now, should be set properly)
        self.target_x = wp.zeros_like(self.x) 
        
        # Rigid targets
        if self.num_objects > 0:
            self.target_rigid_x = wp.zeros_like(self.rbs.rigid_x)
            self.target_rigid_q = wp.zeros_like(self.rbs.rigid_quaternion)
        
        # Optimizer
        self.train_rate = 0.01

        self.tape = wp.Tape()
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
            # 初始化完整的刚体缓冲区，确保所有字段都有有效的设备数组
            self.rbs_buffer = RigidBodies()
            # 常量/不随步变化的字段直接共享底层数组
            self.rbs_buffer.rigid_rest_cm = self.rbs.rigid_rest_cm
            self.rbs_buffer.rigid_mass = self.rbs.rigid_mass
            self.rbs_buffer.rigid_inv_mass = self.rbs.rigid_inv_mass
            self.rbs_buffer.rigid_inertia0 = self.rbs.rigid_inertia0
            # 需要在仿真中更新的字段先用拷贝或零数组占位，后续每步切换到逐步数组
            self.rbs_buffer.rigid_x = wp.zeros_like(self.rbs.rigid_x, requires_grad=True)
            self.rbs_buffer.rigid_v0 = self.rbs.rigid_v0
            self.rbs_buffer.rigid_v = wp.zeros_like(self.rbs.rigid_v, requires_grad=True)
            self.rbs_buffer.rigid_quaternion = wp.zeros_like(self.rbs.rigid_quaternion, requires_grad=True)
            self.rbs_buffer.rigid_omega = wp.zeros_like(self.rbs.rigid_omega, requires_grad=True)
            self.rbs_buffer.rigid_omega0 = self.rbs.rigid_omega0
            self.rbs_buffer.rigid_force = wp.zeros_like(self.rbs.rigid_force, requires_grad=True)
            self.rbs_buffer.rigid_torque = wp.zeros_like(self.rbs.rigid_torque, requires_grad=True)
            self.rbs_buffer.rigid_inertia = wp.zeros_like(self.rbs.rigid_inertia, requires_grad=True)
            self.rbs_buffer.rigid_inv_inertia = wp.zeros_like(self.rbs.rigid_inv_inertia, requires_grad=True)
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

            # No per-segment start buffers or saved grads required
            
            # Initialize optimizer for rigid body velocity
            self.optimizer = warp.optim.Adam([self.rigid_v_arrays[0]], lr=self.train_rate)
        else:
            # Fallback for fluid only
            self.optimizer = warp.optim.Adam([self.v_arrays[0]], lr=self.train_rate)

    def forward(self):
        self.loss.zero_()

        for t in range(self.sim_steps):
            # advance state from t -> t+1
            self.step(t)

        # Compute loss
        if self.num_objects > 0:
            wp.launch(
                compute_rigid_loss,
                dim=self.num_objects,
                inputs=[
                    self.rigid_x_arrays[self.sim_steps], 
                    self.target_rigid_x,
                    self.rigid_quaternion_arrays[self.sim_steps],
                    self.target_rigid_q,
                    self.loss
                ]
            )
            
            # wp.synchronize() # synchronize to ensure kernel finished, then print relevant arrays and loss
            # rx = self.rigid_x_arrays[self.sim_steps].numpy()
            # trx = self.target_rigid_x.numpy()
            # rq = self.rigid_quaternion_arrays[self.sim_steps].numpy()
            # trq = self.target_rigid_q.numpy()
            # loss_val = self.loss.numpy()

            # n_show = min(5, rx.shape[0])

            # print("compute_rigid_loss - summary:")
            # print(" rigid_x shape:", rx.shape, " target_rigid_x shape:", trx.shape)
            # print(" rigid_quaternion shape:", rq.shape, " target_rigid_q shape:", trq.shape)
            # print(" loss array shape:", loss_val.shape)

            # print(" rigid_x (first rows):\n", rx[:n_show])
            # print(" target_rigid_x (first rows):\n", trx[:n_show])
            # print(" rigid_quaternion (first rows):\n", rq[:n_show])
            # print(" target_rigid_q (first rows):\n", trq[:n_show])
            # print(" loss value:", loss_val)
        else:
            wp.launch(
                compute_loss,
                dim=self.particle_max_num,
                inputs=[self.x_arrays[self.sim_steps], self.target_x, self.loss]
            )

    def backward(self):
        for t in range(self.sim_steps):
            # advance state from t -> t+1
            self.step(t)
            # print(f"self.rigid_quaternion_arrays[{t}]:", self.rigid_quaternion_arrays[t].numpy()[1])
            # print(f"Completed forward step {t+1}/{self.sim_steps}")

        # self.loss.grad.fill_(1.0)
        # print("self.rigid_quaternion_arrays[0]:", self.rigid_quaternion_arrays[0].numpy())
        # print("self.rigid_quaternion_arrays[self.sim_steps]:", self.rigid_quaternion_arrays[self.sim_steps].numpy())
        # print("self.rigid_quaternion_arrays[self.sim_steps].grad before backward:", self.rigid_quaternion_arrays[self.sim_steps].grad)
        # print("target_rigid_q:", self.target_rigid_q)
        # print("num_objects:", self.num_objects)
        with self.tape:
            if self.num_objects > 0:
                wp.launch(
                    compute_rigid_loss,
                    dim=self.num_objects,
                    inputs=[
                        self.rigid_x_arrays[self.sim_steps], 
                        self.target_rigid_x,
                        self.rigid_quaternion_arrays[self.sim_steps],
                        self.target_rigid_q,
                    ],
                    outputs = [self.loss],
                    # adj_inputs=[
                    #     self.rigid_x_arrays[self.sim_steps].grad, 
                    #     None,
                    #     self.rigid_quaternion_arrays[self.sim_steps].grad,
                    #     None,
                    #     self.loss.grad
                    # ],
                    # adjoint=True
                )
            else:
                wp.launch(
                    compute_loss,
                    dim=self.particle_max_num,
                    inputs=[self.x_arrays[self.sim_steps], self.target_x, self.loss],
                    adj_inputs=[self.x_arrays[self.sim_steps].grad, None, self.loss.grad],
                    adjoint=True
                )
        wp.synchronize()  # 强制等待 GPU 完成并刷新输出
        print("Starting backward pass...")
        self.tape.backward(self.loss)
        self.tape.visualize("sph_graph.dot")

    def step(self, t):
            self.time_step = t
            # use state at time t as input, write results to time t+1
            x_in = self.x_arrays[t]
            v_in = self.v_arrays[t]
            v_out = self.v_arrays[t+1]
            rho_out = self.rho_arrays[t+1]
            pressure_out = self.pressure_arrays[t+1]
            a_out = self.a_arrays[t+1]
            
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

                self.rbs_buffer.rigid_x = self.rigid_x_arrays[t+1]
                self.rbs_buffer.rigid_v = self.rigid_v_arrays[t+1]
                self.rbs_buffer.rigid_omega = self.rigid_omega_arrays[t+1]
                self.rbs_buffer.rigid_quaternion = self.rigid_quaternion_arrays[t+1]
                self.rbs_buffer.rigid_force = self.rigid_force_arrays[t+1]
                self.rbs_buffer.rigid_torque = self.rigid_torque_arrays[t+1]
                self.rbs_buffer.rigid_inertia = self.rigid_inertia_arrays[t+1]
                self.rbs_buffer.rigid_inv_inertia = self.rigid_inv_inertia_arrays[t+1]
                # for _ in range(self.sim_step_to_frame_ratio):
                with wp.ScopedTimer("grid build", active=self.verbose):
                    # build grid
                    self.grid.build(x_in, self.grid_size)
            with self.tape:
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
                    kernel=compute_non_presure_forces,
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
                    outputs=[self.viscous_forces]
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
                        self.pressure_forces,
                        self.viscous_forces,
                        self.neibor_nums,
                        self.object_id,
                    ],
                    outputs=[self.a_arrays[t]]
                )
            with self.tape:
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
                            self.rigid_x_arrays[t]
                        ],
                        outputs=[
                            self.rigid_force_arrays[t],
                            self.rigid_torque_arrays[t],
                            self.a_arrays[t]]
                    )

            wp.launch(
                    kernel=enforce_boundary_3D_warp,
                    dim=self.particle_max_num,
                    inputs=[
                        self.x_arrays[t],
                        self.v_arrays[t],
                            self.materialMarks,
                            self.domain_size,
                            self.padding,
                    ]
            )
            with self.tape:
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

                with wp.ScopedTimer("rigid body update", active=self.verbose):
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
                    # wp.launch(kernel=solve_rigid_body, dim=self.num_rigid_bodies, inputs=[self.rbs, g, self.dt]) # 该实现有问题
                    wp.launch(
                        kernel=update_rigid_particle_info,
                        dim=self.particle_max_num,
                        inputs=[self.x_arrays[t+1], self.v_arrays[t+1], self.x_0,
                            self.object_id,
                            self.materialMarks,
                            self.rbs_buffer,
                        ]
                    )
            self.sim_time += self.frame_dt

            # print(f"self.rigid_quaternion_arrays[{t}]:", self.rigid_quaternion_arrays[t].numpy()[1])
        # with wp.ScopedTimer("render"):
        #     self.renderer.begin_frame(self.sim_time)
        #     self.renderer.render_points(
        #         points=self.x.numpy(), radius=self.smoothing_length, name="points", colors=(0.8, 0.3, 0.2)
        #     )
        #     self.renderer.end_frame()

    def export_ply_from_diff(self, series_prefix, time_step, cnt_ply):
        np_pos = self.x_arrays[time_step].numpy()
        np_rho = self.rho_arrays[time_step].numpy()
        # m_V: use computed per-particle if available, else fallback to constant m_V0
        np_mV = self.m_V.numpy()
        np_obj_id = self.object_id.numpy()
        # also export per-particle force diagnostics (split vec3 into scalar components)
        pf = self.pressure_arrays[time_step].numpy()
        vf = self.viscous_forces.numpy()
        np_a = self.a_arrays[time_step].numpy()
        np_v = self.v_arrays[time_step].numpy()

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
            'neighbor_num': self.neibor_nums.numpy().astype(np.int32),
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