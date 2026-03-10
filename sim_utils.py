import numpy as np
from plyfile import PlyData, PlyElement
import os

def load_ply_points(path: str):
    """Load point cloud from PLY file.

    Returns:
        pos: (N,3) float32 numpy array of positions
        attrs: dict of {name: (N,) array} for other attributes
    """
    plydata = PlyData.read(path)
    vertex = plydata['vertex']

    x = vertex['x']
    y = vertex['y']
    z = vertex['z']
    pos = np.stack([x, y, z], axis=1).astype(np.float32)

    attrs = {}
    for prop in vertex.properties:
        if prop.name not in ['x', 'y', 'z']:
            attrs[prop.name] = np.array(vertex[prop.name])

    return pos, attrs


def export_ply_points(path: str, pos: np.ndarray, attrs: dict, text: bool = False):
    """Export point cloud to PLY with arbitrary per-vertex scalar attributes.

    path: output .ply path
    pos: (N,3) float32 numpy array
    attrs: dict of {name: (N,) array-like} extra per-vertex scalars (e.g., rho, mV)
        Note: field 'particle_id' is always exported and set to [0, 1, ..., N-1].
    """
    pos = np.asarray(pos)
    n = int(pos.shape[0])

    # Keep integer attributes as integer in PLY for both correctness and speed,
    # and keep float attributes in float32.
    attrs = dict(attrs) if attrs is not None else {}
    attrs['particle_id'] = np.arange(n, dtype=np.int32)

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    attr_arrays = {}
    for name, arr in attrs.items():
        arr_np = np.asarray(arr)
        if arr_np.shape[0] != n:
            raise ValueError(f"Attribute '{name}' length {arr_np.shape[0]} != point count {n}")

        if arr_np.dtype.kind in ('i', 'u', 'b'):
            out_dtype = 'i4'
            arr_np = np.asarray(arr_np, dtype=np.int32)
        else:
            out_dtype = 'f4'
            arr_np = np.asarray(arr_np, dtype=np.float32)

        dtype.append((str(name), out_dtype))
        attr_arrays[str(name)] = arr_np

    data = np.empty(n, dtype=dtype)
    data['x'] = np.asarray(pos[:, 0], dtype=np.float32)
    data['y'] = np.asarray(pos[:, 1], dtype=np.float32)
    data['z'] = np.asarray(pos[:, 2], dtype=np.float32)
    for name, arr_np in attr_arrays.items():
        data[name] = arr_np

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    # Binary PLY is significantly faster and smaller than ASCII.
    PlyData([PlyElement.describe(data, 'vertex')], text=text).write(path)