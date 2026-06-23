import warp as wp
import numpy as np
from SimSPH_diff import SimSPH_diff
from dfsph_kernel import compute_density_error_kernel, compute_dfsph_factor_kernel, compute_density_adv_kernel, pressure_solve_iteration_kernel_fluid, pressure_solve_iteration_kernel_solid, compute_density_change_kernel, divergence_solve_iteration_kernel_fluid, divergence_solve_iteration_kernel_solid
from sph_kernel_diff import compute_density, enforce_boundary_3D_warp, compute_non_pressure_forces, kick, drift
from rigid_fluid_coupling import compute_moving_boundary_volume, solve_rigid_body_diff, update_rigid_particle_info_diff


class SimDFSPH_diff(SimSPH_diff):
    def __init__(self, config=None, container=None, stage_path="example_sph.usd", sim_steps=100, ply_path=None, lr=0.01, h_scale=1.0, use_custom_grad=False, use_norm_grad=False, use_diffFR=False, verbose=False, use_v_loss=False, no_pose_loss=False, use_x_loss=False, use_tail_periodic_loss=False, tail_periodic_freq=2.0, tail_periodic_amp=0.5, tail_periodic_weight=1.0, control_interval=1260, control_start_step=0):
        super().__init__(config, container, stage_path, sim_steps, ply_path, lr, h_scale, use_custom_grad, use_norm_grad, use_diffFR, verbose, use_v_loss, no_pose_loss, use_x_loss, use_tail_periodic_loss, tail_periodic_freq, tail_periodic_amp, tail_periodic_weight, control_interval, control_start_step)
        self.m_max_iterations = 5  # Reduced iterations for differentiability
        self.max_error = 0.05
        self.max_error_V = 0.1
        self.m_max_iterations_v = 5
        self.enable_divergence_solver = True
        self.enable_pressure_solver = True  # Disable for initial testing of differentiability
        
    def init_diff_phys(self, sim_steps):
        super().init_diff_phys(sim_steps)
        self.density_error_accum = wp.zeros(1, dtype=float)

        self.dfsph_factor_arrays = []
        self.density_change_arrays = [[] for _ in range(self.sim_steps + 1)]
        self.density_adv_arrays = [[] for _ in range(self.sim_steps + 1)]

        for _ in range(self.sim_steps + 1):
            self.dfsph_factor_arrays.append(wp.zeros(self.particle_max_num, dtype=float, requires_grad=False))

    def _append_iter_buffer(self, buffer_2d, t):
        buf = wp.zeros(self.particle_max_num, dtype=float, requires_grad=True)
        buffer_2d[t].append(buf)
        return buf

    def clear_grad(self):
        super().clear_grad()

        for arr in self.dfsph_factor_arrays:
            if arr.grad:
                arr.grad.zero_()

        for arrs_t in [self.density_change_arrays, self.density_adv_arrays]:
            for arrs in arrs_t:
                for arr in arrs:
                    if arr.grad:
                        arr.grad.zero_()

        self.density_error_accum.zero_()
        # self.print_all_rigid_grads()
        # def print_dfsph_factor_max_grad_t0(self):
        #     grads = self.dfsph_factor_arrays[0].grad.numpy()
        #     max_grad = np.max(grads)
        #     idx = np.argmax(grads)
        #     print(f"t=0 dfsph_factor max grad: {max_grad} at index {idx}")
        # print_dfsph_factor_max_grad_t0(self)

    def _reset_iter_buffers_at_t(self, t):
        self.density_change_arrays[t] = []
        self.density_adv_arrays[t] = []

    def sub_step(self, t):
        current_tape = self.tapes[-1] if len(self.tapes) > 0 else None

        self._reset_iter_buffers_at_t(t)

        # Input state at t
        x_in = self.x_arrays[t]
        v_in = self.v_arrays[t]
        # Output state at t+1 (will be written to)
        v_out = self.v_arrays[t+1]
        x_out = self.x_arrays[t+1]
        # Intermediate/Aux buffers for this step
        # rho_out: density at current config x_in
        rho_out = self.rho_arrays[t+1]
        # a_non_p: Non-pressure acceleration. Used for prediction.
        a_non_p = self.a_arrays[t+1]
        # viscous_force: For visualization/debugging
        # viscous_force = self.viscous_forces_arrays[t+1]
        # dfsph_factor for this step
        dfsph_fac = self.dfsph_factor_arrays[t]

        eta_div = (1.0 / float(self.frame_dt)) * self.max_error_V * 0.01 * self.base_density
        eta_press = self.max_error * 0.01 * self.base_density

        # Note: rbs pointers and grid build are handled in SimSPH_diff.step

        # Start Tape for Differentiable execution

        # 0. Rigid Body Volume (Coupling)
        with wp.ScopedTimer("compute density", active=self.verbose):
            if self.num_objects > 0:
                wp.launch(
                    kernel=compute_moving_boundary_volume,
                    dim=self.particle_max_num,
                    inputs=[self.grid.id, x_in, self.m_V, self.density_normalization_no_mass, self.smoothing_length,
                            self.materialMarks],
                )

            # 1. Compute Density (at x_in)
            wp.launch(
                kernel=compute_density,
                dim=self.particle_max_num,
                inputs=[
                    self.grid.id,
                    x_in,
                    1.0, # Normalization 1.0 for cubic kernel (DFSPH)
                    self.smoothing_length,
                    self.materialMarks,
                    self.m_V,
                    self.base_density,
                    rho_out
                ]
            )
        # with current_tape:
        # 2. Compute DFSPH Factor
        with wp.ScopedTimer("compute dfsph factor", active=self.verbose):
            wp.launch(
                kernel=compute_dfsph_factor_kernel,
                dim=self.particle_max_num,
                inputs=[
                    self.grid.id,
                    x_in,
                    self.materialMarks,
                    self.m_V,
                    self.smoothing_length,
                    self.use_custom_grad,
                    dfsph_fac
                ]
            )

        with wp.ScopedTimer("divergence_solver", active=self.verbose):
            if self.enable_divergence_solver:
                if current_tape is not None:
                    with current_tape:
                        density_change_buf = self._append_iter_buffer(self.density_change_arrays, t)
                        wp.launch(
                            kernel=compute_density_change_kernel,
                            dim=self.particle_max_num,
                            inputs=[
                                self.grid.id,
                                x_in,
                                v_in,
                                self.materialMarks,
                                self.m_V,
                                self.smoothing_length,
                                int(self.dim),
                                self.use_custom_grad,
                            ],
                            outputs=[
                                density_change_buf,
                            ],
                        )
                else:
                    density_change_buf = self._append_iter_buffer(self.density_change_arrays, t)
                    wp.launch(
                        kernel=compute_density_change_kernel,
                        dim=self.particle_max_num,
                        inputs=[
                            self.grid.id,
                            x_in,
                            v_in,
                            self.materialMarks,
                            self.m_V,
                            self.smoothing_length,
                            int(self.dim),
                            self.use_custom_grad,
                        ],
                        outputs=[
                            density_change_buf,
                        ],
                    )

                for m_iterations_v in range(self.m_max_iterations_v):
                    wp.launch(
                        kernel=divergence_solve_iteration_kernel_fluid,
                        dim=self.particle_max_num,
                        inputs=[
                            self.grid.id,
                            x_in,
                            density_change_buf,
                            dfsph_fac,
                            self.materialMarks,
                            self.m_V,
                            self.smoothing_length,
                            self.frame_dt,
                            self.use_custom_grad,
                            v_in,
                        ],
                    )

                    if current_tape is not None:
                        with current_tape:
                            wp.launch(
                                kernel=divergence_solve_iteration_kernel_solid,
                                dim=self.particle_max_num,
                                inputs=[
                                    self.grid.id,
                                    x_in,
                                    rho_out,
                                    density_change_buf,
                                    dfsph_fac,
                                    self.materialMarks,
                                    self.m_V,
                                    self.smoothing_length,
                                    self.frame_dt,
                                    self.object_id,
                                    self.rbs.rigid_x,
                                    self.use_custom_grad,
                                ],
                                outputs=[v_in, self.rbs.rigid_force, self.rbs.rigid_torque, self.debug_val_arrays[t]]
                            )
                            density_change_buf = self._append_iter_buffer(self.density_change_arrays, t)
                            wp.launch(
                                kernel=compute_density_change_kernel,
                                dim=self.particle_max_num,
                                inputs=[
                                    self.grid.id,
                                    x_in,
                                    v_in,
                                    self.materialMarks,
                                    self.m_V,
                                    self.smoothing_length,
                                    int(self.dim),
                                    self.use_custom_grad,
                                ],
                                outputs=[
                                    density_change_buf,
                                ],
                            )
                    else:
                        wp.launch(
                            kernel=divergence_solve_iteration_kernel_solid,
                            dim=self.particle_max_num,
                            inputs=[
                                self.grid.id,
                                x_in,
                                rho_out,
                                density_change_buf,
                                dfsph_fac,
                                self.materialMarks,
                                self.m_V,
                                self.smoothing_length,
                                self.frame_dt,
                                self.object_id,
                                self.rbs.rigid_x,
                                self.use_custom_grad,
                            ],
                            outputs=[v_in, self.rbs.rigid_force, self.rbs.rigid_torque, self.debug_val_arrays[t]]
                        )
                        density_change_buf = self._append_iter_buffer(self.density_change_arrays, t)
                        wp.launch(
                            kernel=compute_density_change_kernel,
                            dim=self.particle_max_num,
                            inputs=[
                                self.grid.id,
                                x_in,
                                v_in,
                                self.materialMarks,
                                self.m_V,
                                self.smoothing_length,
                                int(self.dim),
                                self.use_custom_grad,
                            ],
                            outputs=[
                                density_change_buf,
                            ],
                        )

                    self.density_error_accum.zero_()
                    wp.launch(
                        kernel=compute_density_error_kernel,
                        dim=self.particle_max_num,
                        inputs=[density_change_buf, self.materialMarks, self.base_density, 0.0, self.density_error_accum]
                    )
                    avg_err_div = self.density_error_accum.numpy()[0] / max(1, self.fluid_particle_num)
                    if avg_err_div <= eta_div:
                        break
                    if m_iterations_v == self.m_max_iterations_v - 1 and avg_err_div > eta_div:
                        print(
                            f"[DFSPH warning] divergence_solve reached m_max_iterations_v={self.m_max_iterations_v} "
                            f"with avg_err_div={avg_err_div:.4f} (eta_div={eta_div:.4f})"
                        )

        # 3. Compute Non-Pressure Forces (Viscosity, Gravity)
        wp.launch(
            kernel=compute_non_pressure_forces,
            dim=self.particle_max_num,
            inputs=[
                self.grid.id,
                x_in,
                v_in,
                rho_out,
                self.dynamic_visc,
                self.smoothing_length,
                self.materialMarks,
                self.m_V,
                self.base_density,
                self.gravity,
                self.object_id,
                self.rbs,
                a_non_p,
            ]
        )
        if current_tape is not None:
            with current_tape:
                # 4. Predict Velocity v*
                # v* = v + dt * a_non_p
                # Write initial v* to v_out
                wp.launch(
                    kernel=kick,
                    dim=self.particle_max_num,
                    inputs=[
                        a_non_p,
                        self.frame_dt,
                        v_in,
                    ],
                    outputs=[
                        v_out
                    ]
                )
        else:
            wp.launch(
                kernel=kick,
                dim=self.particle_max_num,
                inputs=[
                    a_non_p,
                    self.frame_dt,
                    v_in,
                ],
                outputs=[
                    v_out
                ]
            )

        with wp.ScopedTimer("pressure_solver", active=self.verbose):
            if self.enable_pressure_solver:
                if current_tape is not None:
                    # 5. Pressure Solve Loop (Iterative Velocity Correction, ping-pong)
                    with current_tape:
                        density_adv_buf = self._append_iter_buffer(self.density_adv_arrays, t)
                        wp.launch(
                            kernel=compute_density_adv_kernel,
                            dim=self.particle_max_num,
                            inputs=[
                                self.grid.id,
                                x_in,
                                v_out,
                                rho_out,
                                self.materialMarks,
                                self.m_V,
                                self.smoothing_length,
                                self.frame_dt,
                                self.base_density,
                                self.use_custom_grad,
                            ],
                            outputs=[
                                density_adv_buf
                            ]
                        )
                else:
                    density_adv_buf = self._append_iter_buffer(self.density_adv_arrays, t)
                    wp.launch(
                        kernel=compute_density_adv_kernel,
                        dim=self.particle_max_num,
                        inputs=[
                            self.grid.id,
                            x_in,
                            v_out,
                            rho_out,
                            self.materialMarks,
                            self.m_V,
                            self.smoothing_length,
                            self.frame_dt,
                            self.base_density,
                            self.use_custom_grad,
                        ],
                        outputs=[
                            density_adv_buf
                        ]
                    )

                for m_iterations in range(self.m_max_iterations):
                    if current_tape is not None:
                        with current_tape:
                            wp.launch(
                                kernel=pressure_solve_iteration_kernel_fluid,
                                dim=self.particle_max_num,
                                inputs=[
                                    self.grid.id,
                                    x_in,
                                    v_out,
                                    density_adv_buf,
                                    dfsph_fac,
                                    self.materialMarks,
                                    self.m_V,
                                    self.smoothing_length,
                                    self.frame_dt,
                                    self.use_custom_grad,
                                    v_out,
                                ]
                            )
                    else:
                        wp.launch(
                            kernel=pressure_solve_iteration_kernel_fluid,
                            dim=self.particle_max_num,
                            inputs=[
                                self.grid.id,
                                x_in,
                                v_out,
                                density_adv_buf,
                                dfsph_fac,
                                self.materialMarks,
                                self.m_V,
                                self.smoothing_length,
                                self.frame_dt,
                                self.use_custom_grad,
                                v_out,
                            ]
                        )

                    wp.launch(
                        kernel=pressure_solve_iteration_kernel_solid,
                        dim=self.particle_max_num,
                        inputs=[
                            self.grid.id,
                            x_in,
                            density_adv_buf,
                            dfsph_fac,
                            self.materialMarks,
                            self.m_V,
                            self.smoothing_length,
                            self.frame_dt,
                            self.base_density,
                            self.use_custom_grad,
                            self.object_id,
                            self.rbs.rigid_x,
                        ],
                        outputs=[
                            v_out,
                            self.rbs.rigid_force,
                            self.rbs.rigid_torque,
                            self.debug_val_arrays[t],
                        ]
                    )

                    if current_tape is not None:
                        with current_tape:
                            density_adv_buf = self._append_iter_buffer(self.density_adv_arrays, t)
                            wp.launch(
                                kernel=compute_density_adv_kernel,
                                dim=self.particle_max_num,
                                inputs=[
                                    self.grid.id,
                                    x_in,
                                    v_out,
                                    rho_out,
                                    self.materialMarks,
                                    self.m_V,
                                    self.smoothing_length,
                                    self.frame_dt,
                                    self.base_density,
                                    self.use_custom_grad,
                                ],
                                outputs=[
                                    density_adv_buf
                                ]
                            )
                    else:
                        density_adv_buf = self._append_iter_buffer(self.density_adv_arrays, t)
                        wp.launch(
                            kernel=compute_density_adv_kernel,
                            dim=self.particle_max_num,
                            inputs=[
                                self.grid.id,
                                x_in,
                                v_out,
                                rho_out,
                                self.materialMarks,
                                self.m_V,
                                self.smoothing_length,
                                self.frame_dt,
                                self.base_density,
                                self.use_custom_grad,
                            ],
                            outputs=[
                                density_adv_buf
                            ]
                        )

                    self.density_error_accum.zero_()
                    wp.launch(
                        kernel=compute_density_error_kernel,
                        dim=self.particle_max_num,
                        inputs=[density_adv_buf, self.materialMarks, self.base_density, self.base_density, self.density_error_accum]
                    )
                    avg_err_press = self.density_error_accum.numpy()[0] / max(1, self.fluid_particle_num)
                    if avg_err_press <= eta_press:
                        break
                    if m_iterations == self.m_max_iterations - 1 and avg_err_press > eta_press:
                        print(
                            f"[DFSPH warning] pressure_solve reached m_max_iterations={self.m_max_iterations} "
                            f"with avg_err_press={avg_err_press:.4f} (eta_press={eta_press:.4f})"
                        )

        if current_tape is not None:
            with current_tape:
                # 6. Advect
                # x_out = x_in + dt * v_out
                wp.launch(
                    kernel=drift,
                    dim=self.particle_max_num,
                    inputs=[
                        x_in,
                        v_out,
                        self.frame_dt,
                    ],
                    outputs=[
                        x_out
                    ]
                )
        else:
            wp.launch(
                kernel=drift,
                dim=self.particle_max_num,
                inputs=[
                    x_in,
                    v_out,
                    self.frame_dt,
                ],
                outputs=[
                    x_out
                ]
            )

