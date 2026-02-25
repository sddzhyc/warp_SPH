import warp as wp
import numpy as np
import math
import sys
import os

from SimSPH import SimSPH
from kernel_utils import add_ball_kernel

# Add parent directories to path to find modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir)) # d:\code\warp_SPH
sys.path.append(parent_dir)

from SimDFSPH import SimDFSPH
from rigid_fluid_coupling import RigidBodies, MaterialMarks, MaterialType

@wp.kernel
def pipe_kinematic_update(
    center: wp.array(dtype=wp.vec3),
    angle: wp.array(dtype=float),
    vel: wp.array(dtype=wp.vec2),
    omega: wp.array(dtype=float),
    dt: float,
    bx_min: float,
    bx_max: float,
    z_min: float = 1.4,
    z_max: float = 2.0
):
    idx = wp.tid()
    # Kinematic update logic from solve_pipe
    
    # Update state
    # Integrate angle
    angle[idx] += dt * omega[idx]
    
    # Integrate position
    c = center[idx]
    v = vel[idx]
    center[idx] = wp.vec3(c[0] + dt * v[0], c[1], c[2] + dt * v[1])
    
    # pipe Boundary checks
    c_new = center[idx]
    if c_new[0] > bx_max + 0.05:
        center[idx] = wp.vec3(bx_max + 0.05, c_new[1], c_new[2])
        vel[idx] = wp.vec2(0.0, vel[idx][1])
    elif c_new[0] < bx_min - 0.05:
        center[idx] = wp.vec3(bx_min - 0.05, c_new[1], c_new[2])
        vel[idx] = wp.vec2(0.0, vel[idx][1])

    if c_new[2] < z_min:
        center[idx] = wp.vec3(c_new[0], c_new[1], z_min)
        vel[idx] = wp.vec2(vel[idx][0], 0.0)
    elif c_new[2] > z_max:
        center[idx] = wp.vec3(c_new[0], c_new[1], z_max)
        vel[idx] = wp.vec2(vel[idx][0], 0.0)

    if angle[idx] < 1.0:
        angle[idx] = 1.0
        omega[idx] = 0.0
    elif angle[idx] > 2.2:
        angle[idx] = 2.2
        omega[idx] = 0.0

@wp.func
def random_point_in_unit_sphere(state: wp.uint32):
    rx = float(0.0)
    ry = float(0.0)
    while True:
        rx = wp.randf(state) * 2.0 - 1.0
        ry = wp.randf(state) * 2.0 - 1.0
        if rx*rx + ry*ry <= 1.0:
            break
    return wp.vec2(rx, ry)

@wp.kernel
def pipe_emitter_kernel(
    x: wp.array(dtype=wp.vec3),
    x_0: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    rho: wp.array(dtype=float),
    mV: wp.array(dtype=float),
    mask: wp.array(dtype=int),
    material: MaterialMarks,
    object_id: wp.array(dtype=int),
    color: wp.array(dtype=wp.vec3i),
    # Pipe properties
    pipe_center: wp.array(dtype=wp.vec3),
    pipe_angle: wp.array(dtype=float),
    pipe_width: float,
    pipe_length: float,
    dt: float,
    # Emitter state
    start_idx: int,
    num_to_emit: int,
    rng_seed: int,
    fluid_size: int,
    fluid_offset: int,
    base_rho: float,
    m_V0: float,
    emit_color: wp.vec3i
):
    tid = wp.tid()
    if tid < num_to_emit:
        state = wp.rand_init(rng_seed, tid)
        p_idx = fluid_offset + (start_idx + tid) % fluid_size
        
        # Simplified geometry: emit from center
        idx_pipe = 0 # Assuming pipe 0
        pc = pipe_center[idx_pipe]
        pa = pipe_angle[idx_pipe]
        
        vel = 10.0

        c = wp.cos(pa)
        s = wp.sin(pa)
        # Assuming angle is from X axis towards Y axis
        forward = wp.vec3(c, s, 0.0) 
        
        # Right (Perpendicular in XY plane)
        right = wp.vec3(-s, c, 0.0)
        
        # Up (Orthogonal to both, i.e., Z axis)
        up = wp.vec3(0.0, 0.0, 1.0)

        # random point in unit sphere (disk)
        # We fetch 'ra' first to avoid correlation if state is not updated by func call in caller scope
        ra = wp.randf(state)

        # Use helper function
        disk_pt = random_point_in_unit_sphere(state)
        rx = disk_pt[0]
        ry = disk_pt[1]
        
        # Position logic
        # Taichi: corner + [a * len * dt_ratio, r0 * 0.5 * w, r1 * 0.5 * w] (rotated)
        l_offset = ra * pipe_length * dt / 0.004
        radius = pipe_width * 0.5
        
        pos = pc + forward * l_offset + right * (rx * radius) + up * (ry * radius)

        x[p_idx] = pos
        x_0[p_idx] = pos
        v[p_idx] = forward * vel # Reduce emission velocity
        rho[p_idx] = base_rho
        
        mask[p_idx] = 1
        material.material[p_idx] = MaterialType.FLUID
        material.is_dynamic[p_idx] = 1
        mV[p_idx] = m_V0
        object_id[p_idx] = 0
        color[p_idx] = emit_color 

def generate_standard_sphere(num_pts, radius):
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    
    x *= radius
    y *= radius
    z *= radius
    return np.stack((x, y, z), axis=1).astype(np.float32)


class PipeEnvSolver(SimSPH):
    def __init__(self, config, container=None, ply_path=None, quality=1.0, E_=1000.0, mass=[1.0], rho=[1000.0], G=[0.0, -9.8, 0.0], P=None, n_ball=1, n_pipe=1):
        
        # Configuration setup
        self.quality = quality
        self.n_balls = n_ball
        self.n_pipes = n_pipe
        # self.balls_E = E_
        self.balls_rho = rho
        self.balls_mass = mass
        
        # Parameters
        ball_p_num = 700
        
        # Particle Counts
        # self.fluid_size = self.compute_emitted_fluid_particle_num(self.n_pipes, self.quality)
        # We need n_particles first.
        # total_particles = int(self.n_balls * ball_p_num * self.quality ** 3) + self.fluid_size
        # self.jelly_size = (total_particles - self.fluid_size) // self.n_balls
        
        # self.particle_max_num = total_particles
        # self.solid_particle_num = self.jelly_size * self.n_balls
        
        # Domain and Grid
        self.domain_size_x = 1.6
        self.x_grid = int(80 * self.quality)
        self.y_grid = int(48 * self.quality)
        self.dx = self.domain_size_x / self.x_grid
        # self.p_vol = (self.dx * 0.5) ** 3
        
        # self.sim_dt = 2.0e-4 / self.quality 
        # super().__init__(config=config, container=container, sim_steps=100, ply_path=ply_path) # diff version
        super().__init__(config=config, container=container, ply_path=ply_path)
        
        self.sim_dt = self.dt
        # Override dt with computed sim_dt
        # self.dt = self.sim_dt

        self.active_mask = wp.zeros(self.particle_max_num, dtype=int)

        # Override particle initialization
        # self.init_scene_particles()
        
        # Initialize Pipes
        self.init_pipes()

        # Emitter state
        self.num_now = 0

    def init_scene_particles(self):
        # 1. Generate Standard Sphere Geometry
        stand_ball_np = generate_standard_sphere(self.jelly_size, 0.0125 * 4)
        self.stand_ball = wp.array(stand_ball_np, dtype=wp.vec3)
        
        self.jelly_rho_arr = wp.array(np.array(self.balls_mass, dtype=np.float32), dtype=float) 
        
        self.object_id.zero_() 
        self.ball_pos_start = (0.75, 0.75, 0.32)
        if hasattr(self, 'use_pipe') and not self.use_pipe:
            self.ball_pos_start = (0.1, 0.75, 0.32)
       
        # Initialize arrays
        # wp.launch(
        #     kernel=init_fluid_particles_kernel,
        #     dim=self.particle_max_num,
        #     inputs=[
        #         self.x, self.v, self.rho, self.active_mask, self.materialMarks,
        #         self.fluid_size, self.base_density, self.object_id
        #     ]
        # )
        # wp.launch(
        #     kernel=init_jelly_particles_kernel,
        #     dim=self.particle_max_num,
        #     inputs=[
        #         self.x, self.v, self.rho, self.active_mask, self.materialMarks, self.object_id,
        #         self.fluid_size, self.jelly_size, self.n_balls,
        #         self.stand_ball, wp.vec3(*self.ball_pos_start), self.jelly_rho_arr
        #     ]
        # )
        
        # Initialize Rigid Bodies
        rigid_x_np = self.rbs.rigid_x.numpy()
        rigid_mass_np = self.rbs.rigid_mass.numpy()
        rigid_inv_mass_np = self.rbs.rigid_inv_mass.numpy()
        rigid_quat_np = self.rbs.rigid_quaternion.numpy()
        rigid_inertia_np = self.rbs.rigid_inertia.numpy()
        rigid_inv_inertia_np = self.rbs.rigid_inv_inertia.numpy()
        rigid_inertia0_np = self.rbs.rigid_inertia0.numpy()

        for i in range(self.n_balls):
            pos = [self.ball_pos_start[0] + i*0.15, self.ball_pos_start[1] - i*0.02, self.ball_pos_start[2]]
            
            # Add particles for this ball
            start_idx = i * self.jelly_size
            wp.launch(
                kernel=add_ball_kernel,
                dim=self.jelly_size,
                inputs=[
                    self.x, self.x_0, self.v, self.rho, self.active_mask, self.materialMarks, self.object_id,
                    start_idx, self.jelly_size,
                    self.stand_ball, wp.vec3(*pos), float(self.balls_mass[i]), i
                ]
            )

            rigid_x_np[i] = pos
            pmass = self.balls_mass[i] * self.p_vol * self.jelly_size
            rigid_mass_np[i] = pmass
            rigid_inv_mass_np[i] = 1.0/pmass if pmass > 0 else 0.0
            rigid_quat_np[i] = [0, 0, 0, 1]
            
            r = 0.05 
            I = 2/5 * pmass * r**2
            I_mat = np.diag([I, I, I])
            rigid_inertia0_np[i] = I_mat
            rigid_inv_inertia_np[i] = np.linalg.inv(I_mat)
            rigid_inertia_np[i] = I_mat
        
        # Copy back to warp
        self.rbs.rigid_x = wp.array(rigid_x_np, dtype=wp.vec3, requires_grad=True)
        self.rbs.rigid_mass = wp.array(rigid_mass_np, dtype=float)
        self.rbs.rigid_inv_mass = wp.array(rigid_inv_mass_np, dtype=float)
        self.rbs.rigid_quaternion = wp.array(rigid_quat_np, dtype=wp.quat, requires_grad=True)
        self.rbs.rigid_inertia = wp.array(rigid_inertia_np, dtype=wp.mat33)
        self.rbs.rigid_inv_inertia = wp.array(rigid_inv_inertia_np, dtype=wp.mat33)
        self.rbs.rigid_inertia0 = wp.array(rigid_inertia0_np, dtype=wp.mat33)

        # self.init_diff_phys(self.sim_steps) # TODO: Re-init arrays to copy correct initial state

    def init_pipes(self):
        self.pipe_center = wp.array(np.array([[1.5, 0.16, 1.5]] * self.n_pipes, dtype=np.float32), dtype=wp.vec3)
        self.pipe_angle = wp.array(np.full(self.n_pipes, 1.570796, dtype=np.float32), dtype=float) # PI/2 for vertical up
        self.pipe_omega = wp.zeros(self.n_pipes, dtype=float)
        self.pipe_vel = wp.zeros(self.n_pipes, dtype=wp.vec2)
        self.width = 0.4
        self.length = 0.08

        # Recalculate computed value, for consistence
        # In SimSPH, m_V0 = 0.8 * d^3
        # Here we should use the same standard
        # self.m_V0 = 0.8 * self.particle_diameter ** 3
        
        self.bx = [0, self.domain_size[0] * 0.5] # For boundary checks in pipe kinematics
        self.z_min = 0 # 1.4
        self.z_max = self.domain_size[2] # 2.0
        
    def add_cube_emit(self, idx, alpha):
        num_emit = int(20 * self.quality**3 * alpha * alpha * self.sim_dt / 4e-3)
        emit_offset = self.particle_max_num - self.emitted_particle_num
        seed = np.random.randint(0, 1000000)
        wp.launch(
            kernel=pipe_emitter_kernel,
            dim=num_emit,
            inputs=[
                self.x, 
                self.x_0,
                self.v,
                self.rho,
                self.m_V,
                self.active_mask, 
                self.materialMarks,
                self.object_id,
                self.color,
                self.pipe_center,
                self.pipe_angle,
                self.width * alpha, self.length, self.sim_dt,
                self.num_now,
                num_emit,
                seed, 
                self.emitted_particle_num,
                emit_offset,
                self.base_density,
                self.m_V0,
                wp.vec3i(50, 50, 255)
            ]
        )
        self.num_now = (self.num_now + num_emit) % self.emitted_particle_num

    def step(self, t):
        # Emit particles
        # self.add_cube_emit(0, 1.0)

        # 1. Update Pipes (Kinematic)
        wp.launch(
            kernel=pipe_kinematic_update,
            dim=self.n_pipes,
            inputs=[
                self.pipe_center, self.pipe_angle, self.pipe_vel, self.pipe_omega,
                self.sim_dt, self.bx[0], self.bx[1], self.z_min, self.z_max 
            ]
        )
        
        super().step(t)
