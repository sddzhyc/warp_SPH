# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal MLS-MPM fluid example in Warp.

This script follows the structure of the APIC example, but replaces the
FEM-based transfer with a direct MLS-MPM particle-grid-particle update.
"""

from dataclasses import dataclass

import numpy as np

import warp as wp
import warp.render
from warp._src.utils import warn


@wp.func
def cubic_bspline_weights(x: float):
    return wp.vec3(
        0.5 * (1.5 - x) * (1.5 - x),
        0.75 - (x - 1.0) * (x - 1.0),
        0.5 * (x - 0.5) * (x - 0.5),
    )


@wp.func
def grid_index(ix: int, iy: int, iz: int, grid_res: int):
    return ix + grid_res * (iy + grid_res * iz)


@wp.kernel
def clear_grid(grid_mass: wp.array(dtype=float), grid_vel: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    grid_mass[tid] = 0.0
    grid_vel[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def p2g_mls_mpm(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_c: wp.array(dtype=wp.mat33),
    particle_j: wp.array(dtype=float),
    particle_mass: float,
    particle_volume: float,
    grid_mass: wp.array(dtype=float),
    grid_vel: wp.array(dtype=wp.vec3),
    dt: float,
    dx: float,
    grid_res: int,
    bulk_modulus: float,
):
    p = wp.tid()

    x = particle_x[p]
    v = particle_v[p]
    c = particle_c[p]
    j = particle_j[p]

    xp = x / dx
    base_x = int(wp.floor(xp[0] - 0.5))
    base_y = int(wp.floor(xp[1] - 0.5))
    base_z = int(wp.floor(xp[2] - 0.5))

    fx = wp.vec3(xp[0] - float(base_x), xp[1] - float(base_y), xp[2] - float(base_z))

    wx = cubic_bspline_weights(fx[0])
    wy = cubic_bspline_weights(fx[1])
    wz = cubic_bspline_weights(fx[2])

    stress = -dt * 4.0 * bulk_modulus * particle_volume * (j - 1.0) / (dx * dx)

    for i in range(3):
        for jn in range(3):
            for k in range(3):
                weight = wx[i] * wy[jn] * wz[k]

                gx = base_x + i
                gy = base_y + jn
                gz = base_z + k
                if gx < 0 or gy < 0 or gz < 0 or gx >= grid_res or gy >= grid_res or gz >= grid_res:
                    continue

                idx = grid_index(gx, gy, gz, grid_res)
                dpos = wp.vec3(float(gx) - xp[0], float(gy) - xp[1], float(gz) - xp[2]) * dx
                momentum = particle_mass * (v + c @ dpos) + stress * dpos

                wp.atomic_add(grid_mass, idx, weight * particle_mass)
                wp.atomic_add(grid_vel, idx, weight * momentum)


@wp.kernel
def normalize_grid(
    grid_mass: wp.array(dtype=float),
    grid_vel: wp.array(dtype=wp.vec3),
    dt: float,
    gravity: wp.vec3,
    grid_res: int,
    boundary: int,
):
    tid = wp.tid()

    mass = grid_mass[tid]
    vel = wp.vec3(0.0, 0.0, 0.0)

    if mass > 0.0:
        vel = grid_vel[tid] / mass
        vel += dt * gravity

        iz = tid // (grid_res * grid_res)
        rem = tid - iz * grid_res * grid_res
        iy = rem // grid_res
        ix = rem - iy * grid_res

        if ix < boundary and vel[0] < 0.0:
            vel[0] = 0.0
        if ix >= grid_res - boundary and vel[0] > 0.0:
            vel[0] = 0.0
        if iy < boundary and vel[1] < 0.0:
            vel[1] = 0.0
        if iy >= grid_res - boundary and vel[1] > 0.0:
            vel[1] = 0.0
        if iz < boundary and vel[2] < 0.0:
            vel[2] = 0.0
        if iz >= grid_res - boundary and vel[2] > 0.0:
            vel[2] = 0.0

    grid_vel[tid] = vel


@wp.kernel
def g2p_mls_mpm(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_c: wp.array(dtype=wp.mat33),
    particle_j: wp.array(dtype=float),
    grid_vel: wp.array(dtype=wp.vec3),
    dt: float,
    dx: float,
    grid_res: int,
    domain_min: wp.vec3,
    domain_max: wp.vec3,
):
    p = wp.tid()

    x = particle_x[p]
    xp = x / dx
    base_x = int(wp.floor(xp[0] - 0.5))
    base_y = int(wp.floor(xp[1] - 0.5))
    base_z = int(wp.floor(xp[2] - 0.5))

    fx = wp.vec3(xp[0] - float(base_x), xp[1] - float(base_y), xp[2] - float(base_z))

    wx = cubic_bspline_weights(fx[0])
    wy = cubic_bspline_weights(fx[1])
    wz = cubic_bspline_weights(fx[2])

    new_v = wp.vec3(float(0.0), float(0.0), float(0.0))
    c00 = float(0.0)
    c01 = float(0.0)
    c02 = float(0.0)
    c10 = float(0.0)
    c11 = float(0.0)
    c12 = float(0.0)
    c20 = float(0.0)
    c21 = float(0.0)
    c22 = float(0.0)

    for i in range(3):
        for jn in range(3):
            for k in range(3):
                weight = wx[i] * wy[jn] * wz[k]

                gx = base_x + i
                gy = base_y + jn
                gz = base_z + k
                if gx < 0 or gy < 0 or gz < 0 or gx >= grid_res or gy >= grid_res or gz >= grid_res:
                    continue

                idx = grid_index(gx, gy, gz, grid_res)
                g_v = grid_vel[idx]
                dpos = wp.vec3(float(gx) - xp[0], float(gy) - xp[1], float(gz) - xp[2]) * dx

                new_v += weight * g_v

                scale = 4.0 * weight / (dx * dx)
                c00 += scale * g_v[0] * dpos[0]
                c01 += scale * g_v[0] * dpos[1]
                c02 += scale * g_v[0] * dpos[2]
                c10 += scale * g_v[1] * dpos[0]
                c11 += scale * g_v[1] * dpos[1]
                c12 += scale * g_v[1] * dpos[2]
                c20 += scale * g_v[2] * dpos[0]
                c21 += scale * g_v[2] * dpos[1]
                c22 += scale * g_v[2] * dpos[2]

    particle_v[p] = new_v
    particle_c[p] = wp.mat33(c00, c01, c02, c10, c11, c12, c20, c21, c22)

    new_x = x + dt * new_v
    new_x[0] = wp.clamp(new_x[0], domain_min[0], domain_max[0])
    new_x[1] = wp.clamp(new_x[1], domain_min[1], domain_max[1])
    new_x[2] = wp.clamp(new_x[2], domain_min[2], domain_max[2])

    particle_x[p] = new_x
    particle_j[p] = wp.clamp(particle_j[p] * (1.0 + dt * (c00 + c11 + c22)), 0.6, 1.4)


class Example:
    @dataclass
    class State:
        particle_q: wp.array[wp.vec3]
        particle_qd: wp.array[wp.vec3]
        particle_c: wp.array[wp.mat33]
        particle_j: wp.array[float]

    def __init__(self, quiet=False, stage_path="example_mls_mpm_fluid.usd", opengl=False, grid_res=32):
        self._quiet = quiet
        self.grid_res = int(grid_res)
        self.grid_count = self.grid_res * self.grid_res * self.grid_res
        self.dx = 1.0 / float(self.grid_res)
        self.dt = 4.0e-4
        self.sim_substeps = 25
        self.frame_dt = self.dt * self.sim_substeps
        self.current_frame = 0
        self.gravity = wp.vec3(0.0, -9.8, 0.0)
        self.bulk_modulus = 400.0
        self.boundary = 3
        self.domain_min = wp.vec3(0.0, 0.0, 0.0)
        self.domain_max = wp.vec3(1.0, 1.0, 1.0)

        particle_spacing = 0.5 * self.dx
        self.particle_volume = particle_spacing ** 3
        self.particle_mass = self.particle_volume
        self.radius = 0.35 * self.dx

        particle_q = self._spawn_particles(
            bounds_lo=np.array([0.22, 0.22, 0.22], dtype=np.float32),
            bounds_hi=np.array([0.56, 0.82, 0.56], dtype=np.float32),
            spacing=particle_spacing,
        )
        particle_count = particle_q.shape[0]
        particle_qd = wp.zeros(particle_count, dtype=wp.vec3)

        if not self._quiet:
            print("Particle count:", particle_count)
            print("Grid resolution:", self.grid_res)

        self.state_0 = self.State(
            wp.clone(particle_q),
            wp.clone(particle_qd),
            particle_c=wp.zeros(shape=(particle_count,), dtype=wp.mat33),
            particle_j=wp.full(particle_count, 1.0, dtype=float),
        )
        self.state_1 = self.State(
            wp.clone(particle_q),
            wp.clone(particle_qd),
            particle_c=wp.zeros(shape=(particle_count,), dtype=wp.mat33),
            particle_j=wp.full(particle_count, 1.0, dtype=float),
        )

        self.opengl_renderer = None
        self.usd_renderer = None

        try:
            if opengl:
                self.opengl_renderer = warp.render.OpenGLRenderer(
                    screen_width=1024,
                    screen_height=1024,
                )
        except Exception as err:
            warn(f"Could not initialize OpenGL renderer: {err}.")

        try:
            if stage_path:
                self.usd_renderer = warp.render.UsdRenderer(stage_path)
        except Exception as err:
            print(f"Could not initialize Usd renderer '{stage_path}': {err}.")

        self.grid_mass = wp.zeros(self.grid_count, dtype=float)
        self.grid_vel = wp.zeros(self.grid_count, dtype=wp.vec3)

    def step(self):
        self.current_frame += 1

        for _ in range(self.sim_substeps):
            wp.launch(kernel=clear_grid, dim=self.grid_count, inputs=[self.grid_mass, self.grid_vel])

            wp.launch(
                kernel=p2g_mls_mpm,
                dim=self.state_0.particle_q.shape[0],
                inputs=[
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    self.state_0.particle_c,
                    self.state_0.particle_j,
                    self.particle_mass,
                    self.particle_volume,
                    self.grid_mass,
                    self.grid_vel,
                    self.dt,
                    self.dx,
                    self.grid_res,
                    self.bulk_modulus,
                ],
            )

            wp.launch(
                kernel=normalize_grid,
                dim=self.grid_count,
                inputs=[self.grid_mass, self.grid_vel, self.dt, self.gravity, self.grid_res, self.boundary],
            )

            wp.launch(
                kernel=g2p_mls_mpm,
                dim=self.state_0.particle_q.shape[0],
                inputs=[
                    self.state_1.particle_q,
                    self.state_1.particle_qd,
                    self.state_1.particle_c,
                    self.state_1.particle_j,
                    self.grid_vel,
                    self.dt,
                    self.dx,
                    self.grid_res,
                    self.domain_min,
                    self.domain_max,
                ],
            )

            self.state_0, self.state_1 = self.state_1, self.state_0

    @staticmethod
    def _spawn_particles(bounds_lo, bounds_hi, spacing):
        xs = np.arange(bounds_lo[0] + 0.5 * spacing, bounds_hi[0], spacing, dtype=np.float32)
        ys = np.arange(bounds_lo[1] + 0.5 * spacing, bounds_hi[1], spacing, dtype=np.float32)
        zs = np.arange(bounds_lo[2] + 0.5 * spacing, bounds_hi[2], spacing, dtype=np.float32)

        points = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

        rng = np.random.default_rng(42)
        points += rng.uniform(-0.12 * spacing, 0.12 * spacing, size=points.shape)
        points = np.clip(points, bounds_lo + 0.05 * spacing, bounds_hi - 0.05 * spacing)

        return wp.array(np.ascontiguousarray(points), dtype=wp.vec3)

    def render(self):
        if self.usd_renderer is None and self.opengl_renderer is None:
            return

        time = self.current_frame * self.frame_dt

        if self.usd_renderer is not None:
            self.usd_renderer.begin_frame(time)
            self.usd_renderer.render_points(
                "particles",
                self.state_0.particle_q.numpy(),
                radius=self.radius,
            )
            self.usd_renderer.end_frame()

        if self.opengl_renderer is not None:
            self.opengl_renderer.begin_frame(time)
            self.opengl_renderer.render_points(
                "particles",
                self.state_0.particle_q,
                radius=self.radius,
            )
            self.opengl_renderer.end_frame()


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_mls_mpm_fluid.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=250, help="Total number of frames.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--opengl", action="store_true")
    parser.add_argument("--grid-res", type=int, default=32)

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            quiet=args.quiet,
            stage_path=args.stage_path,
            opengl=args.opengl,
            grid_res=args.grid_res,
        )

        for _ in range(args.num_frames):
            with wp.ScopedTimer("step", synchronize=True, print=False):
                example.step()
            with wp.ScopedTimer("render", synchronize=True, print=False):
                example.render()

        if example.usd_renderer is not None:
            example.usd_renderer.save()