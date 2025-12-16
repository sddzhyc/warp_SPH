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


def export_ply_points(path: str, pos: np.ndarray, attrs: dict):
    """Export point cloud to PLY with arbitrary per-vertex scalar attributes.

    path: output .ply path
    pos: (N,3) float32 numpy array
    attrs: dict of {name: (N,) array-like} extra per-vertex scalars (e.g., rho, mV)
    """
    n = int(pos.shape[0])
    dtype = [('x','f4'),('y','f4'),('z','f4')]
    for name in attrs.keys():
        dtype.append((str(name), 'f4'))

    data = np.empty(n, dtype=dtype)
    data['x'] = pos[:, 0].astype('f4')
    data['y'] = pos[:, 1].astype('f4')
    data['z'] = pos[:, 2].astype('f4')
    for name, arr in attrs.items():
        data[str(name)] = np.asarray(arr, dtype='f4')

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    PlyData([PlyElement.describe(data, 'vertex')], text=True).write(path)