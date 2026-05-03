# standard library
import csv
from pathlib import Path

# third-party
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def load_camera_matrix(calib_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load intrinsic camera parameters from a YAML calibration file.

    Parameters
    ----------
    calib_path : Path
        Path to the calibration YAML file (e.g. calibration.yaml).

    Returns
    -------
    K : np.ndarray, shape (3, 3)
        Intrinsic matrix:
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        fx, fy  – focal lengths in pixels (how strongly the lens converges light).
        cx, cy  – principal point in pixels (where the optical axis hits the image plane,
                  ideally the image centre).
        A 3-D point P_c = [X, Y, Z] in camera frame projects to pixel (u, v) as:
            [u, v, 1]^T  =  (1/Z) * K * P_c

    dist : np.ndarray, shape (4, 1)
        Lens distortion coefficients [k1, k2, p1, p2].
    """
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

def load_camera_poses(gt_path: Path) -> dict:
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