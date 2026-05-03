import csv
import numpy as np
from pathlib import Path

import yaml
from scipy.spatial.transform import Rotation


def load_camera_matrix(calib_path: Path):
    with open(calib_path) as f:
        content = f.read().replace("%YAML 1.2", "")
    data = yaml.safe_load(content)
    cam = data["cameras"][0]
    fx, fy = cam["focal_length"]
    cx, cy = cam["principal_point"]
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    return K, dist

def quat_trans_to_matrix(q, t) -> np.ndarray:
    """Quaternion (qx,qy,qz,qw) + translation → 4x4 SE(3)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q).as_matrix()  # scipy expects (x, y, z, w)
    T[:3, 3] = t
    return T

def load_odometry_poses(gt_path: Path) -> dict:
    """Returns {ts_ns: T_world_camera (4x4)} indexed by timestamp."""
    poses = {}
    with open(gt_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = int(row["ts (ns)"])
            t = np.array([float(row["tx (m)"]), float(row["ty (m)"]), float(row["tz (m)"])])
            q = np.array([float(row["qx"]), float(row["qy"]),
                          float(row["qz"]), float(row["qw"])])
            poses[ts] = quat_trans_to_matrix(q, t)
    return poses