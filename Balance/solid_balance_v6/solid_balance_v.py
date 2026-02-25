import os
import time
import numpy as np
import math
import random
import warp as wp
import gym
import torch
import sys
from torchvision import transforms

# Add parent directories to path to find modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir)) # d:\code\warp_SPH
sys.path.append(parent_dir)

from Balance.solid_balance_v6.PipeEnvSolver import PipeEnvSolver
from Balance.solid_balance_v6.AE_CNN import AutoEncoder

torch.set_num_threads(4)

# Seed
seed = 1024
random.seed(seed)
np.random.seed(seed)

@wp.kernel
def rasterize_velocity_kernel(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=int),
    grid_v: wp.array(dtype=float), # Flat array size 80*48*32*3
    grid_counts: wp.array(dtype=int), # Flat array size 80*48*32
    inv_dx: float,
    grid_dim: wp.vec3i
):
    tid = wp.tid()
    if mask[tid] == 0:
        return
        
    p_x = x[tid]
    p_v = v[tid]
    
    # Grid coordinates
    i = int(p_x[0] * inv_dx)
    j = int(p_x[1] * inv_dx)
    k = int(p_x[2] * inv_dx)
    
    # Check bounds
    if i >= 0 and i < grid_dim[0] and j >= 0 and j < grid_dim[1] and k >= 0 and k < grid_dim[2]:
        # Manual indexing for flattened grid_v
        # Shape (80, 48, 32, 3)
        flat_idx = (i * grid_dim[1] + j) * grid_dim[2] + k
        idx_base = flat_idx * 3
        
        wp.atomic_add(grid_v, idx_base + 0, p_v[0])
        wp.atomic_add(grid_v, idx_base + 1, p_v[1])
        wp.atomic_add(grid_v, idx_base + 2, p_v[2])
        
        wp.atomic_add(grid_counts, flat_idx, 1)

class MusicPipeEnvs(PipeEnvSolver):
    def __init__(self, n_balls, n_pipes, show, config=None):
        self.show = show
        self.n_tasks = 1
        self.config = config

        path = os.path.join(current_dir, "autoencoder.pkl")
        self.model = torch.load(path, map_location=torch.device('cuda'), weights_only=False)
        self.model.eval()

        self._max_episode_steps = 1000
        self.cnt = 0
        self.flag = 0

        self.qualitys = np.random.randint(40, 41, self.n_tasks) * 0.025
        self.Es = np.random.randint(800, 801, self.n_tasks)
        self.quality = float(self.qualitys[0])
        self.E = self.Es[0]

        srand = np.random.randint(1, 2, size=(self.n_tasks, n_balls))
        self.projections = np.random.randint(1, 2, size=self.n_tasks)
        self.projection = self.projections[0]
        self.jelly_masses = srand * 4.0
        self.jelly_mass = self.jelly_masses[0] # array for n_balls? No, srand is size (n_tasks, n_balls). jelly_masses[0] is array of masses.

        srand = np.random.randint(100, 101, self.n_tasks)
        self.fluid_rhos = srand * 0.01
        self.fluid_rho = self.fluid_rhos[0]

        srand = np.random.randint(60, 61, self.n_tasks)
        self.gravitys = srand
        self.gravity = self.gravitys[0]
        
        g_vec = [0.0, -float(self.gravity), 0.0]
        
        super().__init__(config, container=None, ply_path=None, quality=self.quality, E_=self.E, mass=self.jelly_mass, rho=self.fluid_rho, G=g_vec, P=self.projection, n_ball=n_balls, n_pipe=n_pipes)

        # Observation Arrays
        self.grid_dim = (self.x_grid, self.y_grid, 32)
        
        self.grid_v_arr = wp.zeros(80*48*32*3, dtype=float) 
        self.grid_counts_arr = wp.zeros(80*48*32, dtype=int)
        
        self.outflux = np.zeros(self.n_pipes)
        self.out = np.zeros(self.n_balls)

        # PLY Export Setup
        self.ply_cnt = 0
        self.time_step = 0
        self.export_dir = os.path.join(parent_dir, "outputs", "debug_ply")
        os.makedirs(self.export_dir, exist_ok=True)

    def render(self, close=True):
        if self.show:
            pass # Removed Taichi GUI

    @property
    def observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf, (self.n_pipes*8 + self.n_balls*6 + 64,))
        
    @property
    def action_space(self):
        return gym.spaces.Box(-np.ones(self.n_pipes*4), np.ones(self.n_pipes*4))

    def get_all_task_idx(self):
        return range(self.n_tasks)

    # Note: reset method overloaded to return observation
    def reset(self):
        self.reset_env()
        return self.get_observation_space()

    def set_seed(self, seed):
        np.random.seed(seed)
        self.projections = np.random.randint(1, 2, size=self.n_tasks)

    def get_observation_space(self):
        # Retrieve raw state (3, 80, 48, 32)
        raw_state = self.get_state()
        
        # Pass raw 3D grid directly (likely 3D CNN)
        # Shape: (1, 3, 80, 48, 32)
        device = next(self.model.parameters()).device
        inp = torch.from_numpy(raw_state).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            latent = self.model.encoder(inp).reshape(-1).cpu().numpy()
            
        # Helper to ensure 64 dim (if architecture doesn't guarantee it for this input size)
        if len(latent) != 64:
             if len(latent) > 64: latent = latent[:64]
             else: latent = np.pad(latent, (0, 64-len(latent)))

        # Combine states
        jelly = self.get_ball_state()
        rigid = self.get_pipe_state()
        
        return np.concatenate([jelly, rigid, latent])

    def get_state(self):

        self.grid_v_arr.zero_()
        self.grid_counts_arr.zero_()
        
        wp.launch(
            kernel=rasterize_velocity_kernel,
            dim=self.particle_max_num,
            inputs=[
                self.x, # self.x_arrays[0], 
                self.v, # self.v_arrays[0], 
                self.active_mask,
                self.grid_v_arr,
                self.grid_counts_arr,
                1.0/self.dx,
                wp.vec3i(80, 48, 32)
            ]
        )
        
        v_np = self.grid_v_arr.numpy().reshape((80, 48, 32, 3))
        c_np = self.grid_counts_arr.numpy().reshape((80, 48, 32))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            v_avg = v_np / c_np[..., None] 
        v_avg = np.nan_to_num(v_avg)
        
        # (80, 48, 32, 3) -> (3, 80, 48, 32) for Torch CNN
        return np.moveaxis(v_avg, -1, 0)

    def get_ball_state(self):
        x = self.rbs.rigid_x.numpy()
        v = self.rbs.rigid_v.numpy()
        
        # Iterate over all rigid bodies, except the first one (usually fluid)
        # Assuming rbs index 0 is fluid or boundary if not used?
        # Actually in SimSPH load_rigid_body, object_id 0 was fluid particles, 
        # but rigid bodies start from index 1 in self.rbs usually?
        # SimSPH implementation: 
        # self.rbs.rigid_x = wp.zeros(self.num_objects, ...)
        # self.num_objects includes fluids.
        # Usually object 0 is fluid, so rigid bodies are 1..num_objects.
        
        # Fix: Separate positions and velocities into blocks to match cost_np_vec expectation
        # Expected structure: [All Positions (n*3), All Velocities (n*3)]
        pos_list = []
        vel_list = []
        for i in range(1, self.num_objects):
            pos_list.extend(x[i])
            vel_list.extend(v[i])
            
        return np.concatenate([pos_list, vel_list])

    def get_pipe_state(self):
        c = self.pipe_center.numpy()
        a = self.pipe_angle.numpy()
        v = self.pipe_vel.numpy()
        o = self.pipe_omega.numpy()
        f = self.outflux
        
        res = []
        for i in range(self.n_pipes):
            res.extend(c[i])
            res.append(a[i])
            res.extend(v[i])
            res.append(o[i])
            res.append(f[i])
        return np.array(res)

    def solve_pipe_logic(self, idx, a, b, c):
        vel = self.pipe_vel.numpy()
        omega = self.pipe_omega.numpy()
        
        dt = self.sim_dt
        vel[idx][0] += dt * a
        vel[idx][1] += dt * b
        omega[idx] += dt * c
        
        vel[idx][0] = np.clip(vel[idx][0], -20.0, 20.0)
        vel[idx][1] = np.clip(vel[idx][1], -5.0, 5.0)
        omega[idx] = np.clip(omega[idx], -25.0, 25.0)
        
        self.pipe_vel = wp.array(vel, dtype=wp.vec2)
        self.pipe_omega = wp.array(omega, dtype=float)

    def step(self, action):
        old_outflux = self.outflux.copy()
        for i in range(self.n_pipes):
            self.outflux[i] = (action[4*i+3] + 1) * 0.5
            self.outflux[i] = np.clip(self.outflux[i], 0.0, 1.0)
            
        num_substeps = int(4e-3 / self.sim_dt)
        if num_substeps < 1: num_substeps = 1

        for s in range(num_substeps):
            for i in range(self.n_pipes):
                ax = action[4*i] * 1600
                ay = action[4*i+1] * 400
                a_omega = action[4*i+2] * 2000
                self.solve_pipe_logic(i, ax, ay, a_omega)
                
                flux_val = self.outflux[i]
                if s * self.sim_dt < 1e-3:
                   ratio = (s * self.sim_dt / 1e-3)
                   flux_val = old_outflux[i] - ratio * (old_outflux[i] - self.outflux[i])
                
                self.add_cube_emit(i, 1,)
            
            # Physics Step (reading 0, writing 1)
            super().step(self.time_step)
            self.time_step += 1
        if self.time_step % 10 == 0:
            ply_path = os.path.join(self.export_dir, "step_{}.ply")
            # Using time_step=0 because we just copied updated state to 0
            self.export_ply(ply_path, self.ply_cnt)
            print(f"Exported {ply_path.format(self.ply_cnt)}")
            self.ply_cnt += 1

        self.rest_time1 = getattr(self, 'rest_time1', 3.0) - 0.1
        self.rest_time1 = max(self.rest_time1, -2.0)
        
        obs = self.get_observation_space()
        reward = self.get_reward() / 50
        done = 0
        self.cnt += 1
        print(f"time_step: {self.time_step}, Pipe {i}, Pos: {self.pipe_center.numpy()[i]}, Flux: {flux_val:.3f}, Reward: {reward:.3f}")

        if self.cnt == self._max_episode_steps:
            self.flag = 2
        if self.flag in [1, 2, 3]:
            done = 1
            self.cnt = 0
            
        return obs, reward, done, dict(reward=reward)

    def cost_np_vec(self, obs, acts, next_obs):
        # obs shape is (batch, size). Here batch=1
        reward = np.zeros(len(obs))
        # Jelly state starts at 0 for MusicPipeEnvs?
        # get_jelly_state returns [x(n*3), v(n*3), t(n*3)]
        # get_rigid_state returns [pipe data]
        # obs = [jelly, rigid, encode]
        
        # Start indices
        n = self.n_balls
        idx_pos = 0   # Pos is at the start of obs
        idx_vel = n * 3 # Vel is after Pos
        
        
        for i in range(self.n_balls):
            px = obs[:, idx_pos + i*3]
            py = obs[:, idx_pos + i*3 + 1]
            pz = obs[:, idx_pos + i*3 + 2]
            
            tx = self.target_pos[i][0]
            ty = self.target_pos[i][1]
            tz = self.target_pos[i][2]
            
            dist = 2 * (px - tx)**2 + (py - ty)**2 + (pz - tz)**2
            
            vx = obs[:, idx_vel + i*3]
            vy = obs[:, idx_vel + i*3 + 1]
            vz = obs[:, idx_vel + i*3 + 2]
            v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
            
            reward += 2.5 * np.exp(-30 * dist) + 1.5 * np.exp(-2 * v_mag)

        # Boundary penalty
        for i in range(len(reward)):
            for j in range(self.n_balls):
                px = obs[i, idx_pos + j*3]
                py = obs[i, idx_pos + j*3 + 1]
                pz = obs[i, idx_pos + j*3 + 2]
                
                if px <= 0.13 or px >= 1.51: reward[i] -= 25
                if py <= 0.30 or py >= 0.80: reward[i] -= 25
                if pz <= 0.17 or pz >= 0.47: reward[i] -= 25

        reward = reward * 0.1
        return -reward

    def get_reward(self):
        obs = self.get_observation_space()
        return self.cost_np_vec(obs.reshape(1, -1), None, None)[0]

    def reset_env(self):
        self.flag = 0
        
        # Re-initialize scene particles to reset positions/velocities/masks based on current parameters
        # This updates self.x, self.v, self.active_mask, etc.
        # self.init_scene_particles()
        self.init_pipes()
        
        # Now reset simulation state (t=0) from the updated self.x/v
        super().reset() 
        
        # Capture initial positions of rigid bodies as targets
        x_init = self.rbs.rigid_x.numpy()
        self.target_pos = []
        for i in range(1, self.num_objects):
            self.target_pos.append(x_init[i])
        self.target_pos = np.array(self.target_pos)
        
        # Reset counters
        self.cnt = 0
        self.num_now = 0
        self.time_step = 0
        self.outflux = np.zeros(self.n_pipes)
        
        # SimSPH.reset() usually resets x_arrays[0] to initial state.
        # But our PipeEnvSolver needs to re-run init_scene_particles to reset specific custom positions?
        # SimSPH stores self.x/self.x_0.
        # If we rely on SimSPH.reset(), it resets x_arrays[0] from self.x (initial).
        # Which is fine if self.x hasn't changed.
        
        # Important: The init logic in PipeEnvSolver.init_scene_particles sets self.x.
        # So we should be good.
        
        return 