import warp as wp
import numpy as np
from SimSPH import SimSPH
from dfsph_kernel import compute_density_error_kernel, compute_dfsph_factor_kernel, compute_density_adv_kernel, compute_pressure_solve_iteration_kernel
from sph_kernel import compute_density, compute_non_presure_forces, kick, drift

class SimDFSPH(SimSPH):
    def __init__(self, config=None, container=None, method=0, stage_path="example_sph.usd", ply_path=None):
        super().__init__(config, container, method, stage_path, ply_path)
        self.max_error = 0.05
        self.m_max_iterations = 100
        
        self.dfsph_factor = None
        self.density_adv = None
        
        # Error reduction buffer
        self.density_error_accum = wp.zeros(1, dtype=float)
        
        # Temporary storage for non-pressure forces if needed, or reuse self.a
        self.viscous_force_temp = None

    def init_DFSPH(self):
        if self.dfsph_factor is None:
            self.dfsph_factor = wp.zeros(self.particle_max_num, dtype=float)
            self.density_adv = wp.zeros(self.particle_max_num, dtype=float)
            self.viscous_force_temp = wp.zeros(self.particle_max_num, dtype=wp.vec3)

    def sub_step(self):
        self.init_DFSPH()
        # Hash grid update (done in SimSPH usually? No, SimSPH step does it.)
        # SimSPH.step calls self.grid.build(self.x, self.smoothing_length)
        # So we should do it here or call super().step/substep?
        # SimSPH.step structure:
        # 1. build grid
        # 2. handle rigid bodies
        # 3. substep loop
        # 4. update bodies
        
        # We rewrite data flow relevant to fluid solver logic.
        
        with wp.ScopedTimer("step_dfsph"):
            
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
                self.rho,
                1.0, # density_normalization set to 1.0 as cubic_kernel is already normalized
                self.smoothing_length,
                self.materialMarks,
                self.m_V,
                self.base_density
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
        
        # 3. Divergence Solve (Skipped as requested/missing kernels)
        # TODO: Implement divergence solve
        
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
        
        # Loop
        while m_iterations < self.m_max_iterations:
            # Update density adv based on current v and pos
            self.compute_density_adv()
            
            # Solve iteration (updates velocity)
            self.pressure_solve_iteration()
            
            # Compute density adv again to check error?
            # Or check error from previous density_adv?
            # DFSPH.py: pressure_solve_iteration_kernel -> compute_density_adv -> compute_density_error
            self.compute_density_adv()
            
            # Error check
            self.density_error_accum.zero_()
            wp.launch(
                kernel=compute_density_error_kernel,
                dim=self.particle_max_num,
                inputs=[self.density_adv, self.materialMarks, self.base_density, self.density_error_accum]
            )
            
            total_err = self.density_error_accum.numpy()[0]
            avg_err = total_err / max(1, self.fluid_particle_num)
            
            eta = self.max_error * 0.01 * self.base_density
            if avg_err <= eta:
                break
                
            m_iterations += 1
            
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
            kernel=compute_pressure_solve_iteration_kernel,
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
