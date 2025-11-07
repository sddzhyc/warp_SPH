import taichi as ti
from sph_base_diff import SPHBase
from particle_system_diff import ParticleSystem

class WCSPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        # Pressure state function parameters(WCSPH)
        self.exponent = 7.0
        self.exponent = self.ps.cfg.get_cfg("exponent")

        self.stiffness = 50000.0
        self.stiffness = self.ps.cfg.get_cfg("stiffness")
        
        self.surface_tension = 0.01
        self.dt[None] = self.ps.cfg.get_cfg("timeStepSize")
    

    @ti.func
    def compute_densities_task(self, step, iter, p_i, p_j, ret):
        x_i = self.ps.x[step, p_i]
        if self.ps.material[step, p_j] == self.ps.material_fluid:
            # Fluid neighbors
            x_j = self.ps.x[step, p_j]
            ret += self.ps.m_V[step, p_j] * self.cubic_kernel((x_i - x_j).norm())
        elif self.ps.material[step, p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            x_j = self.ps.x[step, p_j]
            ret += self.ps.m_V[step, p_j] * self.cubic_kernel((x_i - x_j).norm())


    @ti.kernel
    def compute_densities(self, step: int, iter: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[step, p_i] != self.ps.material_fluid:
                continue
            self.ps.density[step, p_i] = self.ps.m_V[step, p_i] * self.cubic_kernel(0.0)
            den = 0.0
            self.ps.for_all_neighbors(step, iter, p_i, self.compute_densities_task, den)
            self.ps.density[step, p_i] += den
            self.ps.density[step, p_i] *= self.density_0
    

    @ti.func
    def compute_pressure_forces_task(self, step, iter, p_i, p_j, ret):
        x_i = self.ps.x[step, p_i]
        dpi = self.ps.pressure[step, p_i] / self.ps.density[step, p_i] ** 2
        # Fluid neighbors
        if self.ps.material[step, p_j] == self.ps.material_fluid:
            x_j = self.ps.x[step, p_j]
            density_j = self.ps.density[step, p_j] * self.density_0 / self.density_0  # TODO: The density_0 of the neighbor may be different when the fluid density is different
            dpj = self.ps.pressure[step, p_j] / (density_j * density_j)
            # Compute the pressure force contribution, Symmetric Formula
            ret += -self.density_0 * self.ps.m_V[step, p_j] * (dpi + dpj) \
                * self.cubic_kernel_derivative(x_i-x_j)
        elif self.ps.material[step, p_j] == self.ps.material_solid:
            # Boundary neighbors
            dpj = self.ps.pressure[step, p_i] / self.density_0 ** 2
            ## Akinci2012
            x_j = self.ps.x[step, p_j]
            # Compute the pressure force contribution, Symmetric Formula
            f_p = -self.density_0 * self.ps.m_V[step, p_j] * (dpi + dpj) \
                * self.cubic_kernel_derivative(x_i-x_j)
            ret += f_p
            if self.ps.is_dynamic_rigid_body(p_j, step):
                self.ps.acceleration[step, p_j] += -f_p * self.density_0 / self.ps.density[step, p_j]
                # aggregate to rigid force/torque (time-indexed)
                r_id = self.ps.object_id[step, p_j]
                force = -f_p * self.ps.density[step, p_i] * self.ps.m_V[step, p_i]
                self.ps.rigid_force[step, r_id] += force
                self.ps.rigid_torque[step, r_id] += (self.ps.x[step, p_j] - self.ps.rigid_x[step, r_id]).cross(force)
    
    @ti.kernel
    def compute_pressure_forces(self, step: int, iter: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.material[step, p_i] != self.ps.material_fluid:
                continue
            self.ps.density[step, p_i] = ti.max(self.ps.density[step, p_i], self.density_0)
            self.ps.pressure[step, p_i] = self.stiffness * (ti.pow(self.ps.density[step, p_i] / self.density_0, self.exponent) - 1.0)
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_static_rigid_body(p_i, step):
                self.ps.acceleration[step, p_i].fill(0)
                continue
            elif self.ps.is_dynamic_rigid_body(p_i, step):
                continue
            dv = ti.Vector([0.0 for _ in range(self.ps.dim)])
            self.ps.for_all_neighbors(step, iter, p_i, self.compute_pressure_forces_task, dv)
            self.ps.acceleration[step, p_i] += dv


    @ti.func
    def compute_non_pressure_forces_task(self, step, iter, p_i, p_j, ret):
        x_i = self.ps.x[step, p_i]

        ############## Surface Tension ###############
        if self.ps.material[step, p_j] == self.ps.material_fluid:
            # Fluid neighbors
            diameter2 = self.ps.particle_diameter * self.ps.particle_diameter
            x_j = self.ps.x[step, p_j]
            r = x_i - x_j
            r2 = r.dot(r)
            if r2 > diameter2:
                ret -= self.surface_tension / self.ps.m[step, p_i] * self.ps.m[step, p_j] * r * self.cubic_kernel(r.norm())
            else:
                ret -= self.surface_tension / self.ps.m[step, p_i] * self.ps.m[step, p_j] * r * self.cubic_kernel(ti.Vector([self.ps.particle_diameter, 0.0, 0.0]).norm())

        ############### Viscosity Force ###############
        d = 2 * (self.ps.dim + 2)
        x_j = self.ps.x[step, p_j]
        # Compute the viscosity force contribution
        r = x_i - x_j
        v_xy = (self.ps.v[step, iter, p_i] - self.ps.v[step, iter, p_j]).dot(r)

        if self.ps.material[step, p_j] == self.ps.material_fluid:
            f_v = d * self.viscosity * (self.ps.m[step, p_j] / (self.ps.density[step, p_j])) * v_xy / (
                r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(r)
            ret += f_v
        elif self.ps.material[step, p_j] == self.ps.material_solid:
            boundary_viscosity = 0.0
            # Boundary neighbors
            ## Akinci2012
            f_v = d * boundary_viscosity * (self.density_0 * self.ps.m_V[step, p_j] / (self.ps.density[step, p_i])) * v_xy / (
                r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(r)
            ret += f_v
            if self.ps.is_dynamic_rigid_body(p_j, step):
                self.ps.acceleration[step, p_j] += -f_v * self.density_0 / self.ps.density[step, p_j]
                # aggregate to rigid force/torque (time-indexed)
                r_id = self.ps.object_id[step, p_j]
                force = -f_v * self.ps.density[step, p_i] * self.ps.m_V[step, p_i]
                self.ps.rigid_force[step, r_id] += force
                self.ps.rigid_torque[step, r_id] += (self.ps.x[step, p_j] - self.ps.rigid_x[step, r_id]).cross(force)


    @ti.kernel
    def compute_non_pressure_forces(self, step: int, iter: int):
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_static_rigid_body(p_i, step):
                self.ps.acceleration[step, p_i].fill(0.0)
                continue
            ############## Body force ###############
            # Add body force
            d_v = ti.Vector(self.g)
            self.ps.acceleration[step, p_i] = d_v
            if self.ps.material[step, p_i] == self.ps.material_fluid:
                self.ps.for_all_neighbors(step, iter, p_i, self.compute_non_pressure_forces_task, d_v)
                self.ps.acceleration[step, p_i] = d_v


    @ti.kernel
    def advect(self, step: int, iter: int):
        # Symplectic Euler
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic[step, p_i]:
                self.ps.v[step, iter + 1, p_i] = self.ps.v[step, iter, p_i] + self.dt[None] * self.ps.acceleration[step, p_i]
                # update position buffer
                self.ps.x_buffer[step, p_i] = self.ps.x[step, p_i] + self.dt[None] * self.ps.v[step, iter + 1, p_i]


    @ti.ad.grad_replaced
    def substep(self):
        # Forward: run the standard WCSPH substep
        step = self.step_num
        iter = self.iter_num[self.step_num]
        self.compute_densities(step, iter)
        self.compute_non_pressure_forces(step, iter)
        self.compute_pressure_forces(step, iter)
        self.advect(step, iter)

    @ti.ad.grad_for(substep)
    def substep_grad(self):
        # Reverse-mode: call adjoints of the kernels in reverse order
        step = self.step_num
        iter = self.iter_num[self.step_num]
        # call grads in reverse
        self.advect.grad(step, iter)
        self.compute_pressure_forces.grad(step, iter)
        self.compute_non_pressure_forces.grad(step, iter)
        self.compute_densities.grad(step, iter)
