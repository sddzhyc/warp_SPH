from SimSPH import SimSPH
from particle_system import ParticleSystem
from rigid_fluid_coupling import MaterialMarks, RigidBodies, compute_moving_boundary_volume, compute_static_boundary_volume, solve_rigid_body, update_rigid_particle_info
from sph_kernel import *

import numpy as np
import warp as wp
import warp.optim
import math
import os
# optional dependency for flexible PLY export with custom attributes
from plyfile import PlyData, PlyElement

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
    
    if tid == 1: # 跳过流体
        # Position loss
        diff_pos = rigid_x[tid] - target_rigid_x[tid]
        l_pos = wp.dot(diff_pos, diff_pos)
        
        # Rotation loss (quaternion difference)
        # User requested: 0.5 * (norm(final_q_rod - target_q_rod)**2)
        # We use the expansion: 0.5 * (dot(q,q) + dot(tq,tq) - 2*dot(q,tq))
        # This is equivalent to 0.5 * ||q - tq||^2
        
        q = rigid_q[tid]
        tq = target_rigid_q[tid]
        l_rot = 0.5 * (wp.dot(q, q) + wp.dot(tq, tq) - 2.0 * wp.dot(q, tq))
        
        # Combine losses (can add weights)
        # total_loss = l_pos + l_rot * 0.1 # Weight rotation loss
        total_loss = l_rot # Weight rotation loss
    
        wp.atomic_add(loss, 0, total_loss)

class SimSPH_diff_with_segment(SimSPH):

    def __init__(self,config = None, container: ParticleSystem = None, stage_path="example_sph.usd", sim_steps=100, ply_path=None):
        super().__init__(config, container, stage_path, ply_path)
        self.sim_steps = sim_steps
        self.init_diff_phys(self.sim_steps)


    def init_diff_phys(self, sim_steps):
        self.sim_steps = sim_steps
        # Memory usage is minimized when the segment size is approx. sqrt(sim_steps)
        # self.segment_size = math.ceil(math.sqrt(sim_steps))
        self.segment_size = sim_steps # 不分段，直接全部存储
        self.num_segments = math.ceil(sim_steps / self.segment_size)
        self.sim_steps = self.segment_size * self.num_segments # Adjust sim_steps

        self.x_arrays = []
        self.v_arrays = []
        self.rho_arrays = []
        self.pressure_arrays = []
        self.a_arrays = []
        
        # Initialize arrays for each step in a segment
        # We need segment_size + 1 arrays to store state from t=0 to t=segment_size
        for _ in range(self.segment_size + 1):
            self.x_arrays.append(wp.zeros_like(self.x, requires_grad=True))
            self.v_arrays.append(wp.zeros_like(self.v, requires_grad=True))
            self.rho_arrays.append(wp.zeros_like(self.rho, requires_grad=True))
            self.pressure_arrays.append(wp.zeros_like(self.pressure, requires_grad=True))
            self.a_arrays.append(wp.zeros_like(self.a, requires_grad=True))
        print(f"Initialized differentiable simulation with {self.num_segments} segments of size {self.segment_size} steps each.")
        # Copy initial state to first arrays
        wp.copy(self.x_arrays[0], self.x)
        wp.copy(self.v_arrays[0], self.v)
        wp.copy(self.rho_arrays[0], self.rho)
        # pressure and a are computed, so 0 is fine or copy if initialized

        # Segment start states
        self.segment_start_x = []
        self.segment_start_v = []
        self.segment_start_rho = []
        
        for _ in range(self.num_segments):
            self.segment_start_x.append(wp.zeros_like(self.x))
            self.segment_start_v.append(wp.zeros_like(self.v))
            self.segment_start_rho.append(wp.zeros_like(self.rho))

        # Gradients saved
        self.x_grad_saved = wp.zeros_like(self.x)
        self.v_grad_saved = wp.zeros_like(self.v)

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

        self.tape = None
        self.forward_graph = None
        self.backward_graph = None
        self.zero_tape_graph = None

        # Rigid body checkpointing
        self.rigid_x_arrays = []
        self.rigid_v_arrays = []
        self.rigid_omega_arrays = []
        self.rigid_quaternion_arrays = []
        
        self.segment_start_rigid_x = []
        self.segment_start_rigid_v = []
        self.segment_start_rigid_omega = []
        self.segment_start_rigid_quaternion = []

        self.rigid_x_grad_saved = None
        self.rigid_v_grad_saved = None
        self.rigid_omega_grad_saved = None
        self.rigid_quaternion_grad_saved = None

        if self.num_objects > 0:
            for _ in range(self.segment_size + 1):
                self.rigid_x_arrays.append(wp.zeros_like(self.rbs.rigid_x, requires_grad=True))
                self.rigid_v_arrays.append(wp.zeros_like(self.rbs.rigid_v, requires_grad=True))
                self.rigid_omega_arrays.append(wp.zeros_like(self.rbs.rigid_omega, requires_grad=True))
                self.rigid_quaternion_arrays.append(wp.zeros_like(self.rbs.rigid_quaternion, requires_grad=True))
            
            self.rigid_x_arrays[0].requires_grad = True
            self.rigid_v_arrays[0].requires_grad = True
            self.rigid_omega_arrays[0].requires_grad = True
            self.rigid_quaternion_arrays[0].requires_grad = True

            wp.copy(self.rigid_x_arrays[0], self.rbs.rigid_x)
            wp.copy(self.rigid_v_arrays[0], self.rbs.rigid_v)
            wp.copy(self.rigid_omega_arrays[0], self.rbs.rigid_omega)
            wp.copy(self.rigid_quaternion_arrays[0], self.rbs.rigid_quaternion)

            for _ in range(self.num_segments):
                self.segment_start_rigid_x.append(wp.zeros_like(self.rbs.rigid_x))
                self.segment_start_rigid_v.append(wp.zeros_like(self.rbs.rigid_v))
                self.segment_start_rigid_omega.append(wp.zeros_like(self.rbs.rigid_omega))
                self.segment_start_rigid_quaternion.append(wp.zeros_like(self.rbs.rigid_quaternion))
            
            self.rigid_x_grad_saved = wp.zeros_like(self.rbs.rigid_x)
            self.rigid_v_grad_saved = wp.zeros_like(self.rbs.rigid_v)
            self.rigid_omega_grad_saved = wp.zeros_like(self.rbs.rigid_omega)
            self.rigid_quaternion_grad_saved = wp.zeros_like(self.rbs.rigid_quaternion)
            
            # Initialize optimizer for rigid body velocity
            self.optimizer = warp.optim.Adam([self.rigid_v_arrays[0]], lr=self.train_rate)
        else:
            # Fallback for fluid only
            self.optimizer = warp.optim.Adam([self.v_arrays[0]], lr=self.train_rate)

    def forward(self):
        self.loss.zero_()
        
        for segment_index in range(self.num_segments):
            # Save start
            wp.copy(self.segment_start_x[segment_index], self.x_arrays[0])
            wp.copy(self.segment_start_v[segment_index], self.v_arrays[0])
            wp.copy(self.segment_start_rho[segment_index], self.rho_arrays[0])
            
            if hasattr(self, 'rbs') and self.num_objects > 0:
                wp.copy(self.segment_start_rigid_x[segment_index], self.rigid_x_arrays[0])
                wp.copy(self.segment_start_rigid_v[segment_index], self.rigid_v_arrays[0])
                wp.copy(self.segment_start_rigid_omega[segment_index], self.rigid_omega_arrays[0])
                wp.copy(self.segment_start_rigid_quaternion[segment_index], self.rigid_quaternion_arrays[0])

            for t in range(1, self.segment_size + 1):
                self.step(t) # This step(t) should read from t-1 and write to t
            
            # Prepare for next segment
            if segment_index < self.num_segments - 1:
                wp.copy(self.x_arrays[0], self.x_arrays[-1])
                wp.copy(self.v_arrays[0], self.v_arrays[-1])
                wp.copy(self.rho_arrays[0], self.rho_arrays[-1])
                
                if hasattr(self, 'rbs') and self.num_objects > 0:
                    wp.copy(self.rigid_x_arrays[0], self.rigid_x_arrays[-1])
                    wp.copy(self.rigid_v_arrays[0], self.rigid_v_arrays[-1])
                    wp.copy(self.rigid_omega_arrays[0], self.rigid_omega_arrays[-1])
                    wp.copy(self.rigid_quaternion_arrays[0], self.rigid_quaternion_arrays[-1])

        # Compute loss
        if self.num_objects > 0:
            wp.launch(
                compute_rigid_loss,
                dim=self.num_objects,
                inputs=[
                    self.rigid_x_arrays[self.segment_size], 
                    self.target_rigid_x,
                    self.rigid_quaternion_arrays[self.segment_size],
                    self.target_rigid_q,
                    self.loss
                ]
            )
            
            wp.synchronize() # synchronize to ensure kernel finished, then print relevant arrays and loss
            rx = self.rigid_x_arrays[self.segment_size].numpy()
            trx = self.target_rigid_x.numpy()
            rq = self.rigid_quaternion_arrays[self.segment_size].numpy()
            trq = self.target_rigid_q.numpy()
            loss_val = self.loss.numpy()

            n_show = min(5, rx.shape[0])

            print("compute_rigid_loss - summary:")
            print(" rigid_x shape:", rx.shape, " target_rigid_x shape:", trx.shape)
            print(" rigid_quaternion shape:", rq.shape, " target_rigid_q shape:", trq.shape)
            print(" loss array shape:", loss_val.shape)

            print(" rigid_x (first rows):\n", rx[:n_show])
            print(" target_rigid_x (first rows):\n", trx[:n_show])
            print(" rigid_quaternion (first rows):\n", rq[:n_show])
            print(" target_rigid_q (first rows):\n", trq[:n_show])
            print(" loss value:", loss_val)
        else:
            wp.launch(
                compute_loss,
                dim=self.particle_max_num,
                inputs=[self.x_arrays[self.segment_size], self.target_x, self.loss]
            )

    def backward(self):
        for segment_index in range(self.num_segments - 1, -1, -1):
            # Restore
            wp.copy(self.x_arrays[0], self.segment_start_x[segment_index])
            wp.copy(self.v_arrays[0], self.segment_start_v[segment_index])
            wp.copy(self.rho_arrays[0], self.segment_start_rho[segment_index])
            
            if hasattr(self, 'rbs') and self.num_objects > 0:
                wp.copy(self.rigid_x_arrays[0], self.segment_start_rigid_x[segment_index])
                wp.copy(self.rigid_v_arrays[0], self.segment_start_rigid_v[segment_index])
                wp.copy(self.rigid_omega_arrays[0], self.segment_start_rigid_omega[segment_index])
                wp.copy(self.rigid_quaternion_arrays[0], self.segment_start_rigid_quaternion[segment_index])

            with wp.Tape() as self.tape:
                for t in range(1, self.segment_size + 1):
                    self.step(t)
            
            if segment_index == self.num_segments - 1:
                self.loss.grad.fill_(1.0)
                
                if self.num_objects > 0:
                    wp.launch(
                        compute_rigid_loss,
                        dim=self.num_objects,
                        inputs=[
                            self.rigid_x_arrays[self.segment_size], 
                            self.target_rigid_x,
                            self.rigid_quaternion_arrays[self.segment_size],
                            self.target_rigid_q,
                            self.loss
                        ],
                        adj_inputs=[
                            self.rigid_x_arrays[self.segment_size].grad, 
                            None,
                            self.rigid_quaternion_arrays[self.segment_size].grad,
                            None,
                            self.loss.grad
                        ],
                        adjoint=True
                    )
                else:
                    wp.launch(
                        compute_loss,
                        dim=self.particle_max_num,
                        inputs=[self.x_arrays[self.segment_size], self.target_x, self.loss],
                        adj_inputs=[self.x_arrays[self.segment_size].grad, None, self.loss.grad],
                        adjoint=True
                    )
            else:
                # Restore gradients
                wp.copy(self.x_arrays[-1].grad, self.x_grad_saved)
                wp.copy(self.v_arrays[-1].grad, self.v_grad_saved)
                
                if hasattr(self, 'rbs') and self.num_objects > 0:
                    wp.copy(self.rigid_x_arrays[-1].grad, self.rigid_x_grad_saved)
                    wp.copy(self.rigid_v_arrays[-1].grad, self.rigid_v_grad_saved)
                    wp.copy(self.rigid_omega_arrays[-1].grad, self.rigid_omega_grad_saved)
                    wp.copy(self.rigid_quaternion_arrays[-1].grad, self.rigid_quaternion_grad_saved)
            
            self.tape.backward()
            
            if segment_index > 0:
                # Save gradients
                wp.copy(self.x_grad_saved, self.x_arrays[0].grad)
                wp.copy(self.v_grad_saved, self.v_arrays[0].grad)
                
                if hasattr(self, 'rbs') and self.num_objects > 0:
                    wp.copy(self.rigid_x_grad_saved, self.rigid_x_arrays[0].grad)
                    wp.copy(self.rigid_v_grad_saved, self.rigid_v_arrays[0].grad)
                    wp.copy(self.rigid_omega_grad_saved, self.rigid_omega_arrays[0].grad)
                    wp.copy(self.rigid_quaternion_grad_saved, self.rigid_quaternion_arrays[0].grad)

                self.tape.zero()

    def step(self, t):
        self.time_step = t
        
        if len(self.x_arrays) > 0 and 0 < t <= self.segment_size:
            x_in = self.x_arrays[t-1]
            x_out = self.x_arrays[t]
            v_in = self.v_arrays[t-1]
            v_out = self.v_arrays[t]
            # rho_in = self.rho_arrays[t-1]
            rho_out = self.rho_arrays[t]
            # pressure_in = self.pressure_arrays[t-1]
            pressure_out = self.pressure_arrays[t]
            a_in = self.a_arrays[t-1]
            a_out = self.a_arrays[t]
            
            wp.copy(x_out, x_in)
            wp.copy(v_out, v_in)
            
            current_x = x_out
            current_v = v_out
            current_rho = rho_out
            current_pressure = pressure_out
            current_a = a_out

            if self.num_objects > 0:
                # Copy previous state to current state
                wp.copy(self.rigid_x_arrays[t], self.rigid_x_arrays[t-1])
                wp.copy(self.rigid_v_arrays[t], self.rigid_v_arrays[t-1])
                wp.copy(self.rigid_omega_arrays[t], self.rigid_omega_arrays[t-1])
                wp.copy(self.rigid_quaternion_arrays[t], self.rigid_quaternion_arrays[t-1])
                
                # Update rbs to point to current arrays
                self.rbs.rigid_x = self.rigid_x_arrays[t]
                self.rbs.rigid_v = self.rigid_v_arrays[t]
                self.rbs.rigid_omega = self.rigid_omega_arrays[t]
                self.rbs.rigid_quaternion = self.rigid_quaternion_arrays[t]

        else:
            current_x = self.x
            current_v = self.v
            current_rho = self.rho
            current_pressure = self.pressure
            current_a = self.a

        with wp.ScopedTimer("step", active=self.verbose):
            for _ in range(self.sim_step_to_frame_ratio):
                # Zero out intermediate arrays that are reused
                current_rho.zero_()
                current_pressure.zero_()
                current_a.zero_()
                
                # Also zero out force buffers if they exist and are used
                if hasattr(self, 'viscous_forces'):
                    self.viscous_forces.zero_()
                if hasattr(self, 'pressure_forces'):
                    self.pressure_forces.zero_()
                
                # Ensure rigid body forces are cleared
                if self.num_objects > 0:
                    self.rbs.rigid_force.zero_()
                    self.rbs.rigid_torque.zero_()

                with wp.ScopedTimer("grid build", active=self.verbose):
                    # build grid
                    self.grid.build(current_x, self.grid_size)

                with wp.ScopedTimer("forces", active=self.verbose):
                    wp.launch(
                        kernel=compute_moving_boundary_volume,
                        dim=self.particle_max_num,
                        inputs=[self.grid.id, current_x, self.m_V, self.density_normalization_no_mass, self.smoothing_length,
                                self.materialMarks],
                    )
                    # compute density of points
                    wp.launch(
                        kernel=compute_density,
                        dim=self.particle_max_num,
                        inputs=[self.grid.id, current_x, current_rho, 
                                1.0, # cubic kernel don't need normalization
                                self.smoothing_length,
                                self.materialMarks, self.m_V, self.base_density],
                    )

                    wp.launch(
                        kernel=compute_pressure,
                        dim=self.particle_max_num,
                        inputs=[current_rho, current_pressure, self.materialMarks,
                                self.stiffness, self.exponent, self.base_density],
                    )

                    wp.launch(
                        kernel=compute_non_presure_forces,
                        dim=self.particle_max_num,
                        inputs=[
                            self.grid.id,
                            current_x,
                            current_v,
                            current_rho,
                            self.dynamic_visc,
                            self.smoothing_length,
                            self.materialMarks,
                            self.m_V,
                            self.base_density,
                            self.viscous_forces,
                            self.object_id,
                            self.rbs
                        ],
                    )

                    # get new acceleration
                    wp.launch(
                        kernel=get_acceleration,
                        dim=self.particle_max_num,
                        inputs=[
                            self.grid.id,
                            current_x,
                            current_v,
                            current_rho,
                            current_pressure,
                            current_a,
                            self.stiffness,
                            self.exponent,
                            self.base_density,
                            self.gravity,
                            # poly6核函数相关参数
                            # self.pressure_normalization_no_mass,
                            # self.viscous_normalization_no_mass,
                            1.0,  # cubic kernel don't need normalization
                            self.dynamic_visc, # cubic kernel only use dynamic_visc
                            self.smoothing_length,
                            self.materialMarks,
                            self.m_V,
                            self.pressure_forces,
                            self.viscous_forces,
                            self.neibor_nums,
                            self.object_id,
                            self.rbs
                        ],
                    )
                    wp.launch(
                        kernel=enforce_boundary_3D_warp,
                        dim=self.particle_max_num,
                        inputs=[current_x, current_v,
                                self.materialMarks,
                                self.domain_size,
                                self.padding,
                        ]
                    )

                    # kick
                    wp.launch(kernel=kick, dim=self.particle_max_num, inputs=[current_v, current_a, self.dt])

                    # drift
                    wp.launch(kernel=drift, dim=self.particle_max_num, inputs=[current_x, current_v, self.dt])

                    g = wp.vec3(0.0, self.gravity, 0.0)

                    wp.launch(kernel=solve_rigid_body, dim=self.num_objects, inputs=[self.rbs, g, self.dt])
                    # wp.launch(kernel=solve_rigid_body, dim=self.num_rigid_bodies, inputs=[self.rbs, g, self.dt]) # 该实现有问题
                    wp.launch(
                        kernel=update_rigid_particle_info,
                        dim=self.particle_max_num,
                        inputs=[current_x, current_v, self.x_0,
                            self.object_id,
                            self.materialMarks,
                            self.rbs,
                        ]
                    )

            self.sim_time += self.frame_dt

        # with wp.ScopedTimer("render"):
        #     self.renderer.begin_frame(self.sim_time)
        #     self.renderer.render_points(
        #         points=self.x.numpy(), radius=self.smoothing_length, name="points", colors=(0.8, 0.3, 0.2)
        #     )
        #     self.renderer.end_frame()