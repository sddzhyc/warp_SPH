import warp as wp
import numpy as np
from SimSPH_diff import SimSPH_diff
from dfsph_kernel import compute_density_error_kernel, compute_dfsph_factor_kernel, compute_density_adv_kernel, pressure_solve_iteration_kernel_fluid, pressure_solve_iteration_kernel_solid, compute_density_change_kernel, divergence_solve_iteration_kernel_fluid, divergence_solve_iteration_kernel_solid
from sph_kernel_diff import compute_density, enforce_boundary_3D_warp, compute_non_pressure_forces, kick, drift
from rigid_fluid_coupling import compute_moving_boundary_volume, solve_rigid_body_diff, update_rigid_particle_info_diff


class SimDFSPH_diff(SimSPH_diff):
    def __init__(self, config=None, container=None, stage_path="example_sph.usd", sim_steps=100, ply_path=None, lr=0.01, h_scale=1.0, use_custom_grad=False, use_norm_grad=False, verbose=False):
        super().__init__(config, container, stage_path, sim_steps, ply_path, lr, h_scale, use_custom_grad, use_norm_grad, verbose)
        self.m_max_iterations = 1  # Reduced iterations for differentiability
        self.max_error = 0.05
        self.max_error_V = 0.1
        self.m_max_iterations_v = 1
        self.enable_divergence_solver = True
        self.enable_pressure_solver = False  # Disable for initial testing of differentiability
        
    def init_diff_phys(self, sim_steps):
        super().init_diff_phys(sim_steps)
        self.density_error_accum = wp.zeros(1, dtype=float)

        self.dfsph_factor_arrays = []
        self.density_change_buf_arrays = []
        self.density_adv_buf_arrays = []

        for _ in range(self.sim_steps + 1):
            self.dfsph_factor_arrays.append(wp.zeros(self.particle_max_num, dtype=float, requires_grad=True))
            self.density_change_buf_arrays.append(wp.zeros(self.particle_max_num, dtype=float, requires_grad=True))
            self.density_adv_buf_arrays.append(wp.zeros(self.particle_max_num, dtype=float, requires_grad=True))

    def clear_grad(self):
        super().clear_grad()

        for arrs in [self.dfsph_factor_arrays, self.density_change_buf_arrays, self.density_adv_buf_arrays]:
            for arr in arrs:
                if arr.grad:
                    arr.grad.zero_()

        self.density_error_accum.zero_()

    def compute_density_change(self, x_state, v_state, density_change_out):
        wp.launch(
            kernel=compute_density_change_kernel,
            dim=self.particle_max_num,
            inputs=[
                self.grid.id,
                x_state,
                v_state,
                self.materialMarks,
                self.m_V,
                self.smoothing_length,
                int(self.dim),
                density_change_out,
            ],
        )

    def sub_step(self, t):
        current_tape = self.tapes[-1]
        
        # Input state at t
        x_in = self.x_arrays[t]
        v_in = self.v_arrays[t]
        
        # Output state at t+1 (will be written to)
        v_out = self.v_arrays[t+1]
        x_out = self.x_arrays[t+1]

        # Preallocated per-step buffers
        density_change_buf = self.density_change_buf_arrays[t]
        density_adv_buf = self.density_adv_buf_arrays[t]
        
        # Intermediate/Aux buffers for this step
        # rho_out: density at current config x_in
        rho_out = self.rho_arrays[t+1]
        
        # a_non_p: Non-pressure acceleration. Used for prediction.
        a_non_p = self.a_arrays[t+1]
        
        # viscous_force: For visualization/debugging
        viscous_force = self.viscous_forces_arrays[t+1]
        
        # dfsph_factor for this step
        dfsph_fac = self.dfsph_factor_arrays[t]

        eta_div = (1.0 / float(self.frame_dt)) * self.max_error_V * 0.01 * self.base_density
        eta_press = self.max_error * 0.01 * self.base_density
        
        # Note: rbs pointers and grid build are handled in SimSPH_diff.step

        # Start Tape for Differentiable execution

        # 0. Rigid Body Volume (Coupling)
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
        with current_tape:        
            # 2. Compute DFSPH Factor
            wp.launch(
                kernel=compute_dfsph_factor_kernel,
                dim=self.particle_max_num,
                inputs=[
                    self.grid.id,
                    x_in,
                    self.materialMarks,
                    self.m_V,
                    self.smoothing_length,
                    dfsph_fac
                ]
            )

        if self.enable_divergence_solver:
            with current_tape:
                self.compute_density_change(x_in, v_in, density_change_buf)
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
                        outputs=[v_in, self.rbs.rigid_force, self.rbs.rigid_torque]
                    )
                self.compute_density_change(x_in, v_in, density_change_buf)

                self.density_error_accum.zero_()
                wp.launch(
                    kernel=compute_density_error_kernel,
                    dim=self.particle_max_num,
                    inputs=[density_change_buf, self.materialMarks, self.base_density, 0.0, self.density_error_accum]
                )
                avg_err_div = self.density_error_accum.numpy()[0] / max(1, self.fluid_particle_num)
                if avg_err_div <= eta_div:
                    break
                print(f"DFSPH - iteration V: {m_iterations_v} Avg density err: {avg_err_div}")

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
                    v_out
                ]
            )
        
        if self.enable_pressure_solver:
            # 5. Pressure Solve Loop (Iterative Velocity Correction, ping-pong)
            # v_press_curr = v_out
            with current_tape:
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
                        density_adv_buf
                    ]
                )

            for _ in range(self.m_max_iterations):
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
                with current_tape:
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
                            v_out,
                            self.object_id,
                            self.rbs.rigid_force,
                            self.rbs.rigid_torque,
                            self.rbs.rigid_x,
                        ]
                    )
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
                    x_out
                ]
            )

