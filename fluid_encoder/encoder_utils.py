import os

import numpy as np

def normalize_fluid_field(field, eps=1e-6):
    """Normalize fluid field to zero mean and unit range per channel.
    field: numpy array with shape (N, C, ... ) or (C, ...)
    Returns normalized field in same shape (float32).
    """
    arr = np.array(field, dtype=np.float32)
    # support batch or single
    if arr.ndim >= 3:
        # assume shape (N, C, ...)
        axis = tuple(range(0, arr.ndim))[2:]
        mean = arr.mean(axis=axis, keepdims=True)
        mn = arr.min(axis=axis, keepdims=True)
        mx = arr.max(axis=axis, keepdims=True)
        denom = (mx - mn)
        denom[denom < eps] = 1.0
        return (arr - mean) / denom
    else:
        # fallback
        mean = arr.mean()
        mx = arr.max()
        mn = arr.min()
        denom = (mx - mn) if (mx - mn) >= eps else 1.0
        return (arr - mean) / denom


def _metadata_value(metadata, key, default=None):
    if metadata is None:
        return default
    if key not in metadata:
        return default

    value = metadata[key]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    return value


def _load_fluid_velocity_field_npz(npz_path):
    with np.load(npz_path) as data:
        if "velocity_field" in data:
            velocity_field = data["velocity_field"]
        else:
            velocity_field = data[data.files[0]]

        weight = data["weight"] if "weight" in data else None
        metadata = {key: data[key] for key in data.files if key not in {"velocity_field", "weight"}}

    velocity_field = np.asarray(velocity_field, dtype=np.float32)
    if velocity_field.ndim == 4 and velocity_field.shape[0] != 3 and velocity_field.shape[-1] == 3:
        velocity_field = np.transpose(velocity_field, (3, 0, 1, 2))

    if velocity_field.ndim != 4 or velocity_field.shape[0] != 3:
        raise ValueError(
            f"Expected velocity_field with shape (3, D, H, W) or (D, H, W, 3), got {velocity_field.shape} in {npz_path}"
        )

    if weight is not None:
        weight = np.asarray(weight, dtype=np.float32)

    return velocity_field, weight, metadata


def _fluid_field_to_point_cloud(velocity_field, weight=None, rigid_center=None, grid_shape=None, grid_spacing=1.0):
    velocity_field = np.asarray(velocity_field, dtype=np.float32)
    if velocity_field.ndim != 4 or velocity_field.shape[0] != 3:
        raise ValueError(f"velocity_field must have shape (3, D, H, W), got {velocity_field.shape}")

    if grid_shape is None:
        grid_shape = tuple(int(v) for v in velocity_field.shape[1:])
    else:
        grid_shape = tuple(int(v) for v in grid_shape)

    if rigid_center is None:
        rigid_center = np.zeros(3, dtype=np.float32)
    else:
        rigid_center = np.asarray(rigid_center, dtype=np.float32).reshape(3)

    grid_spacing = float(grid_spacing)
    half_extent = np.asarray(grid_shape, dtype=np.float32) * 0.5 * grid_spacing
    min_corner = rigid_center - half_extent

    grid_indices = np.indices(grid_shape, dtype=np.float32)
    positions = np.stack(
        [grid_indices[0], grid_indices[1], grid_indices[2]],
        axis=-1,
    ).reshape(-1, 3)
    positions = min_corner[None, :] + (positions + 0.5) * grid_spacing

    velocities = velocity_field.reshape(3, -1).T

    if weight is not None:
        weight = np.asarray(weight, dtype=np.float32).reshape(-1)

    return positions, velocities, weight


def write_point_cloud_ply(file_path, positions, velocities, weight=None):
    positions = np.asarray(positions, dtype=np.float32)
    velocities = np.asarray(velocities, dtype=np.float32)

    if positions.shape[0] != velocities.shape[0]:
        raise ValueError("positions and velocities must have the same length")

    if weight is not None:
        weight = np.asarray(weight, dtype=np.float32).reshape(-1)
        if weight.shape[0] != positions.shape[0]:
            raise ValueError("weight must have the same number of samples as positions")

    file_dir = os.path.dirname(file_path)
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {positions.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float vx\n")
        f.write("property float vy\n")
        f.write("property float vz\n")
        if weight is not None:
            f.write("property float weight\n")
        f.write("end_header\n")

        if weight is None:
            for pos, vel in zip(positions, velocities):
                f.write(f"{pos[0]} {pos[1]} {pos[2]} {vel[0]} {vel[1]} {vel[2]}\n")
        else:
            for pos, vel, w in zip(positions, velocities, weight):
                f.write(f"{pos[0]} {pos[1]} {pos[2]} {vel[0]} {vel[1]} {vel[2]} {w}\n")


def fluid_npz_to_houdini_ply(npz_path, ply_path=None, min_weight=0.0):
    """Convert a saved fluid-field NPZ into an ASCII PLY point cloud for Houdini.

    The exported PLY stores one point per grid cell with point position at the
    cell center and custom velocity/weight attributes. Houdini can import the
    resulting PLY as a point cloud and visualize the `v`/`weight` attributes.
    """
    velocity_field, weight, metadata = _load_fluid_velocity_field_npz(npz_path)
    grid_shape = _metadata_value(metadata, "grid_shape", velocity_field.shape[1:])
    grid_spacing = _metadata_value(metadata, "grid_spacing", 1.0)
    rigid_center = _metadata_value(metadata, "rigid_center", np.zeros(3, dtype=np.float32))

    positions, velocities, weight_flat = _fluid_field_to_point_cloud(
        velocity_field=velocity_field,
        weight=weight,
        rigid_center=rigid_center,
        grid_shape=grid_shape,
        grid_spacing=grid_spacing,
    )

    mask = np.ones(positions.shape[0], dtype=bool)
    if weight_flat is not None and min_weight > 0:
        mask &= weight_flat >= float(min_weight)

    positions = positions[mask]
    velocities = velocities[mask]
    if weight_flat is not None:
        weight_flat = weight_flat[mask]

    if ply_path is None:
        ply_path = os.path.splitext(npz_path)[0] + ".ply"

    write_point_cloud_ply(ply_path, positions, velocities, weight=weight_flat)
    return ply_path


def batch_convert_fluid_npz_dir_to_houdini_ply(input_dir, output_dir=None, min_weight=0.0):
    """Convert every fluid-field NPZ file in a directory to a Houdini-friendly PLY."""
    input_dir = os.path.abspath(input_dir)
    if output_dir is None:
        output_dir = input_dir
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    npz_files = sorted(
        file_name for file_name in os.listdir(input_dir) if file_name.lower().endswith(".npz")
    )
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")

    output_paths = []
    for file_name in npz_files:
        npz_path = os.path.join(input_dir, file_name)
        ply_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".ply")
        output_paths.append(fluid_npz_to_houdini_ply(npz_path, ply_path=ply_path, min_weight=min_weight))

    return output_paths


def particles_to_grid_mass_center(particle_pos, particle_vel, center_pos, grid_shape, grid_spacing, keep_channels=3):
    """
    Deposit particle velocities onto a fixed grid centered at `center_pos` using trilinear (PIC-like) weighting.

    particle_pos: (N,3) numpy array
    particle_vel: (N,3) numpy array
    center_pos: (3,) world coordinates for grid center
    grid_shape: (D, H, W) ints
    grid_spacing: scalar distance between grid cells

    Returns:
      grid_vel: numpy array shape (3, D, H, W) with interpolated velocities (zeros where no contribution)
      weight: numpy array shape (D,H,W) of accumulated weights
    """
    particle_pos = np.asarray(particle_pos, dtype=np.float32)
    particle_vel = np.asarray(particle_vel, dtype=np.float32)
    # Use explicit axis names to avoid confusion: grid_shape is (nx, ny, nz)
    nx, ny, nz = tuple(int(v) for v in grid_shape)
    # compute grid corner (min) so grid is centered at center_pos
    half = np.array([nx, ny, nz], dtype=np.float32) * 0.5 * grid_spacing
    min_corner = np.asarray(center_pos, dtype=np.float32) - half

    grid_vel_sum = np.zeros((3, nx, ny, nz), dtype=np.float32)
    weight = np.zeros((nx, ny, nz), dtype=np.float32)

    # map each particle to grid space (x,y,z -> indices along nx,ny,nz)
    rel = (particle_pos - min_corner) / float(grid_spacing)
    # only keep particles inside grid bounds
    inside_mask = np.logical_and.reduce((
        rel[:, 0] >= 0, rel[:, 0] < nx,
        rel[:, 1] >= 0, rel[:, 1] < ny,
        rel[:, 2] >= 0, rel[:, 2] < nz,
    ))
    if not np.any(inside_mask):
        return grid_vel_sum, weight

    rel = rel[inside_mask]
    vel = particle_vel[inside_mask]

    # for each particle, distribute to 8 neighbors
    ix = np.floor(rel).astype(int)
    fx = rel - ix.astype(np.float32)

    for p in range(rel.shape[0]):
        i, j, k = ix[p]
        wx, wy, wz = fx[p]
        # neighbor offsets (x, y, z)
        for dz in (0, 1):
            wzt = (1 - wz) if dz == 0 else wz
            zk = k + dz
            if zk < 0 or zk >= nz:
                continue
            for dy in (0, 1):
                yht = (1 - wy) if dy == 0 else wy
                yk = j + dy
                if yk < 0 or yk >= ny:
                    continue
                for dx in (0, 1):
                    xht = (1 - wx) if dx == 0 else wx
                    xk = i + dx
                    if xk < 0 or xk >= nx:
                        continue
                    w = xht * yht * wzt
                    weight[xk, yk, zk] += w
                    grid_vel_sum[:, xk, yk, zk] += vel[p] * w

    # avoid division by zero
    nz_mask = weight > 0
    grid_vel = np.zeros_like(grid_vel_sum)
    grid_vel[:, nz_mask] = grid_vel_sum[:, nz_mask] / weight[nz_mask]

    return grid_vel, weight


def get_local_fluid_grid_around_rigid(particle_pos, particle_vel, rigid_center, grid_shape=(32,32,32), grid_spacing=0.02, influence_radius=None):
    """
    Convenience wrapper: filters particles to those within the grid extents (or influence_radius if provided)
    and returns a fixed-size grid of velocities centered at `rigid_center`.
    """
    if influence_radius is None:
        # cover whole grid half-extent
        half_extent = np.array(grid_shape) * 0.5 * grid_spacing
        influence_radius = float(np.linalg.norm(half_extent))

    # filter by distance to rigid center
    dists = np.linalg.norm(particle_pos - np.asarray(rigid_center, dtype=np.float32), axis=1)
    mask = dists <= influence_radius
    if not np.any(mask):
        # return empty grids
        vel_grid = np.zeros((3,)+tuple(grid_shape), dtype=np.float32)
        weight = np.zeros(grid_shape, dtype=np.float32)
        return vel_grid, weight

    return particles_to_grid_mass_center(particle_pos[mask], particle_vel[mask], rigid_center, grid_shape, grid_spacing)
