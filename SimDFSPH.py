import warp as wp
import numpy as np
from SimSPH import SimSPH
from dfsph_kernel import compute_density_error_kernel, compute_dfsph_factor_kernel, compute_density_adv_kernel, compute_density_change_kernel, divergence_solve_iteration_kernel, pressure_solve_iteration_kernel
from sph_kernel import compute_non_presure_forces, kick, drift
from sph_kernel_diff import compute_density
from sim_utils import export_ply_points

class SimDFSPH(SimSPH):
    def __init__(self, config=None, container=None, method=0, stage_path="example_sph.usd", ply_path=None):
        super().__init__(config, container, method, stage_path, ply_path)
        self.surface_tension = 0. # 0.01
        self.max_error = 0.05
        self.m_max_iterations = 100
        self.max_error_V = 0.1
        self.m_max_iterations_v = 100
        self.enable_divergence_solver = False
        self.enable_pressure_solver = True
        self.init_DFSPH()
        # Error reduction buffer
        self.density_error_accum = wp.zeros(1, dtype=float)
        # Temporary storage for non-pressure forces if needed, or reuse self.a
        self.viscous_force_temp = None

    def init_DFSPH(self):
        self.dfsph_factor = wp.zeros(self.particle_max_num, dtype=float)
        self.density_adv = wp.zeros(self.particle_max_num, dtype=float)
        self.density_change = wp.zeros(self.particle_max_num, dtype=float)
        self.viscous_force_temp = wp.zeros(self.particle_max_num, dtype=wp.vec3)

    def sub_step(self):
        # SimSPH.step structure:
        # 1. build grid
        # 2. handle rigid bodies
        # 3. substep loop
        # 4. update bodies
        
        # We rewrite data flow relevant to fluid solver logic.
        
        with wp.ScopedTimer("step_dfsph", active=self.verbose):
            
            # 2. Rigid body update (Coupling) - Pre-step
            # (Assuming RigidBodies logic is handled or we rely on SimSPH utils)
            # SimSPH.step handles `solve_rigid_body` etc.
            # We should probably replicate that structure if we override step.
            self.substep_DFSPH()
            

    def substep_DFSPH(self):
        # 1. Compute densities
        wp.launch(
            kernel=compute_density,
            dim=self.particle_max_num,
            inputs=[
                self.grid.id,
                self.x,
                1.0, # density_normalization set to 1.0 as cubic_kernel is already normalized
                self.smoothing_length,
                self.materialMarks,
                self.m_V,
                self.base_density,
                self.rho,
            ]
        )
        
        # 2. Compute DFSPH factor
        wp.launch(
            kernel=compute_dfsph_factor_kernel,
            dim=self.particle_max_num,
            inputs=[
                self.grid.id,
                self.x,
                self.materialMarks,
                self.m_V,
                self.smoothing_length,
                self.dfsph_factor
            ]
        )
        
        # 3. Divergence Solve
        if self.enable_divergence_solver:
            self.divergence_solve()
        
        # 4. Compute Non-Pressure Forces (Viscosity, Gravity)
        wp.launch(
        kernel=compute_non_presure_forces,
        dim=self.particle_max_num,
        inputs=[
            self.grid.id,
            self.x,
            self.v,
            self.rho,
            self.dynamic_visc,
            self.smoothing_length,
            self.materialMarks,
            self.m_V,
            self.base_density,
            self.viscous_forces,
            self.object_id,
            self.rbs,
            self.surface_tension,
            self.gravity,
            self.a
        ],
        )
        
        # 5. Predict Velocity
        # v* = v + dt * a_non_p
        wp.launch(
            kernel=kick,
            dim=self.particle_max_num,
            inputs=[
                self.v,
                self.a,
                float(self.dt) 
            ]
        )
        
        if self.enable_pressure_solver:
        # 6. Pressure Solve
            self.pressure_solve()
        
        # 7. Advect
        # x += v * dt
        wp.launch(
            kernel=drift,
            dim=self.particle_max_num,
            inputs=[
                self.x,
                self.v,
                float(self.dt)
            ]
        )
        
    def pressure_solve(self):
        m_iterations = 0
        avg_density_err = 0.0
        converged = False
        eta = self.max_error * 0.01 * self.base_density

        self.compute_density_adv()
        
        # Loop
        while m_iterations < self.m_max_iterations:
            # Solve iteration (updates velocity)
            self.pressure_solve_iteration()
            
            # Recompute density prediction for error evaluation
            self.compute_density_adv()
            
            # Error check
            self.density_error_accum.zero_()
            wp.launch(
                kernel=compute_density_error_kernel,
                dim=self.particle_max_num,
                inputs=[self.density_adv, self.materialMarks, self.base_density, self.base_density, self.density_error_accum]
            )
            
            total_err = self.density_error_accum.numpy()[0]
            avg_err = total_err / max(1, self.fluid_particle_num)
            avg_density_err = avg_err
            
            if avg_err <= eta:
                converged = True
                break
                
            m_iterations += 1

        if not converged:
            print(
                f"[DFSPH warning] pressure_solve reached m_max_iterations={self.m_max_iterations} "
                f"but avg_density_err={avg_density_err:.6f} > eta={eta:.6f}"
            )

    def divergence_solve(self):
        m_iterations_v = 0
        avg_density_err = 0.0
        converged = False
        eta = (1.0 / float(self.dt)) * self.max_error_V * 0.01 * self.base_density

        self.compute_density_change()

        while m_iterations_v < self.m_max_iterations_v:
            self.divergence_solve_iteration()
            self.compute_density_change()

            self.density_error_accum.zero_()
            wp.launch(
                kernel=compute_density_error_kernel,
                dim=self.particle_max_num,
                inputs=[self.density_change, self.materialMarks, self.base_density, 0.0, self.density_error_accum]
            )

            total_err = self.density_error_accum.numpy()[0]
            avg_err = total_err / max(1, self.fluid_particle_num)
            avg_density_err = avg_err

            if avg_err <= eta:
                converged = True
                break

            m_iterations_v += 1
        print(f"DFSPH - iteration V: {m_iterations_v} Avg density err: {avg_density_err}")
        if not converged:
            print(
                f"[DFSPH warning] divergence_solve reached m_max_iterations_v={self.m_max_iterations_v} "
                f"but avg_density_err={avg_density_err:.6f} > eta={eta:.6f}"
            )

    def compute_density_change(self):
         wp.launch(
            kernel=compute_density_change_kernel,
            dim=self.particle_max_num,
            inputs=[
                self.grid.id,
                self.x,
                self.v,
                self.materialMarks,
                self.m_V,
                self.smoothing_length,
                int(self.dim),
                self.density_change
            ]
        )

    def divergence_solve_iteration(self):
        wp.launch(
            kernel=divergence_solve_iteration_kernel,
            dim=self.particle_max_num,
            inputs=[
                self.grid.id,
                self.x,
                self.v,
                self.rho,
                self.density_change,
                self.dfsph_factor,
                self.materialMarks,
                self.m_V,
                self.smoothing_length,
                float(self.dt),
                self.object_id,
                self.rbs.rigid_x
            ],
            outputs=[
                self.v,
                self.rbs.rigid_force,
                self.rbs.rigid_torque,
            ]
        )

    def export_ply(self, series_prefix, cnt_ply):
        np_pos = self.x.numpy()
        np_rho = self.rho.numpy()
        np_mV = self.m_V.numpy()
        np_obj_id = self.object_id.numpy()
        pf = self.pressure_forces.numpy()
        vf = self.viscous_forces.numpy()
        np_a = self.a.numpy()
        np_v = self.v.numpy()

        np_dfsph_factor = self.dfsph_factor.numpy().astype(np.float32)
        np_density_adv = self.density_adv.numpy().astype(np.float32)
        np_density_change = self.density_change.numpy().astype(np.float32)

        out_path = series_prefix.format(cnt_ply)
        export_ply_points(out_path, np_pos.astype(np.float32), {
            'rho': np_rho.astype(np.float32),
            'pressure': self.pressure.numpy().astype(np.float32),
            'mV': np_mV.astype(np.float32),
            'object_id': np_obj_id.astype(np.int32),
            'neighbor_num': self.neibor_nums.numpy().astype(np.int32),
            'pressure_fx': pf[:,0].astype(np.float32),
            'pressure_fy': pf[:,1].astype(np.float32),
            'pressure_fz': pf[:,2].astype(np.float32),
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
            'dfsph_factor': np_dfsph_factor,
            'density_adv': np_density_adv,
            'density_change': np_density_change,
        })
            
        # print(f"DFSPH - iterations: {m_iterations} Avg density Err: {avg_err:.4f}")

    def compute_density_adv(self):
         wp.launch(
            kernel=compute_density_adv_kernel,
            dim=self.particle_max_num,
            inputs=[
                self.grid.id,
                self.x,
                self.v,
                self.rho,
                self.materialMarks,
                self.m_V,
                self.smoothing_length,
                float(self.dt),
                self.base_density,
                self.density_adv
            ]
        )
        
    def pressure_solve_iteration(self):
        wp.launch(
            kernel=pressure_solve_iteration_kernel,
            dim=self.particle_max_num,
            inputs=[
                self.grid.id,
                self.x,
                self.v,
                self.density_adv,
                self.dfsph_factor,
                self.materialMarks,
                self.m_V,
                self.smoothing_length,
                float(self.dt),
                self.base_density,
                self.v, # Output v (in-place)
                self.object_id,
                self.rbs.rigid_force, 
                self.rbs.rigid_torque,
                self.rbs.rigid_x
            ]
        )
