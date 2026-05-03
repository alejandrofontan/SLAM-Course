"""
ArUco marker detector for iPhone recordings (StrayScanner format).
Reads images listed in rgb.csv, detects ArUco markers, and computes:
  - T_cam_marker : marker pose relative to camera  (per frame)
  - T_world_marker: marker pose in world frame using groundtruth camera poses
Live 3-D plot shows trajectory, current camera position, and oriented marker planes.
Saves per-frame detections and per-marker averaged world positions.
"""

import csv
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

DATA_DIR = Path("/home/alejandro/VSLAM-LAB-Benchmark/STRAYSCANNER/bb7e805d7c")
MARKER_LENGTH = 0.15  # physical marker side length in metres — adjust as needed
ARUCO_DICT = cv2.aruco.DICT_6X6_50
PLOT_EVERY = 10        # redraw every N frames for speed
AXIS_LEN = MARKER_LENGTH * 0.6   # length of the drawn frame axes
FRUSTUM_DEPTH = MARKER_LENGTH * 0.8  # near-plane distance for the camera frustum
IMG_W, IMG_H = 640, 480


def load_camera_matrix(calib_path: Path):
    with open(calib_path) as f:
        content = f.read().replace("%YAML 1.2", "")  # PyYAML only supports 1.1
    data = yaml.safe_load(content)
    cam = data["cameras"][0]
    fx, fy = cam["focal_length"]
    cx, cy = cam["principal_point"]
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)  # StrayScanner provides no distortion
    return K, dist


def load_groundtruth(gt_path: Path) -> dict:
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


def quat_trans_to_matrix(q, t) -> np.ndarray:
    """Quaternion (qx,qy,qz,qw) + translation → 4x4 SE(3)."""
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rvec_tvec_to_matrix(rvec, tvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T


def mean_pose(transforms: list) -> tuple[np.ndarray, np.ndarray]:
    """Average a list of 4x4 SE(3) transforms → (R_mean, t_mean).
    Rotation is averaged via SVD projection onto SO(3)."""
    t_mean = np.mean([T[:3, 3] for T in transforms], axis=0)
    R_sum = sum(T[:3, :3] for T in transforms)
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:  # fix reflection
        U[:, -1] *= -1
        R_mean = U @ Vt
    return R_mean, t_mean


def draw_camera_frustum(ax, T_world_cam: np.ndarray, K: np.ndarray):
    """Draw a camera frustum pyramid in world frame using the intrinsics."""
    R = T_world_cam[:3, :3]
    t = T_world_cam[:3, 3]
    d = FRUSTUM_DEPTH

    # unproject image corners to camera frame at depth d
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    corners_img = np.array([[0, 0], [IMG_W, 0], [IMG_W, IMG_H], [0, IMG_H]], dtype=float)
    corners_cam = np.column_stack([
        (corners_img[:, 0] - cx) / fx * d,
        (corners_img[:, 1] - cy) / fy * d,
        np.full(4, d),
    ])
    corners_w = (R @ corners_cam.T).T + t  # (4, 3)

    # 4 lines from camera centre to image-plane corners
    for c in corners_w:
        ax.plot([t[0], c[0]], [t[1], c[1]], [t[2], c[2]], color="orange", linewidth=1)

    # image-plane rectangle
    rect = np.vstack([corners_w, corners_w[0]])
    ax.plot(rect[:, 0], rect[:, 1], rect[:, 2], color="orange", linewidth=1)

    # camera centre dot
    ax.scatter(*t, color="orange", s=30, zorder=5)


def draw_marker_frame(ax, R: np.ndarray, t: np.ndarray, visible: bool):
    """Draw a flat square + XYZ axes for one ArUco marker in world frame."""
    color = "green" if visible else "black"
    alpha = 0.5 if visible else 0.2
    L = MARKER_LENGTH / 2

    # marker square corners in marker frame (Z = 0 plane)
    corners_m = np.array([[-L, -L, 0], [L, -L, 0], [L, L, 0], [-L, L, 0]])
    corners_w = (R @ corners_m.T).T + t  # (4, 3)

    poly = Poly3DCollection([corners_w], alpha=alpha, facecolor=color, edgecolor=color)
    ax.add_collection3d(poly)

    # X (red), Y (green), Z (blue) axes
    for axis_idx, axis_color in enumerate(["red", "lime", "blue"]):
        end = t + R[:, axis_idx] * AXIS_LEN
        ax.plot([t[0], end[0]], [t[1], end[1]], [t[2], end[2]],
                color=axis_color, linewidth=1.5)

    ax.text(t[0], t[1], t[2], f" {'' }", fontsize=0)  # anchor for tight layout


def setup_plot():
    plt.ion()
    fig = plt.figure(figsize=(16, 7))
    ax_img = fig.add_subplot(121)
    ax_img.axis("off")
    ax_img.set_title("Current frame")
    ax_3d = fig.add_subplot(122, projection="3d")
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")
    ax_3d.set_title("World map")
    fig.tight_layout()
    return fig, ax_img, ax_3d


def redraw(ax_img, ax_3d, frame_rgb, corners, ids, traj_pts, T_world_cam, K, marker_poses: dict):
    visible_ids = set(ids.ravel().tolist()) if ids is not None else set()

    # --- image panel ---
    ax_img.cla()
    ax_img.axis("off")
    ax_img.set_title("Current frame")
    if frame_rgb is not None:
        display = frame_rgb.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
        ax_img.imshow(display)

    # --- 3-D map panel ---
    ax_3d.cla()
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")
    ax_3d.set_title("World map")

    if len(traj_pts) > 1:
        traj = np.array(traj_pts)
        ax_3d.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                   color="steelblue", linewidth=1, label="trajectory")

    draw_camera_frustum(ax_3d, T_world_cam, K)

    for marker_id, transforms in marker_poses.items():
        R, t = mean_pose(transforms)
        visible = marker_id in visible_ids
        draw_marker_frame(ax_3d, R, t, visible)
        color = "green" if visible else "black"
        ax_3d.text(t[0], t[1], t[2], f" {marker_id}", fontsize=8, color=color)

    ax_3d.legend(loc="upper left", fontsize=8)
    plt.pause(0.001)


def main():
    K, dist = load_camera_matrix(DATA_DIR / "calibration.yaml")
    gt_poses = load_groundtruth(DATA_DIR / "groundtruth.csv")

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
        cv2.aruco.DetectorParameters(),
    )

    with open(DATA_DIR / "rgb.csv") as f:
        reader = csv.DictReader(f)
        image_paths = [(int(row["ts_rgb_0 (ns)"]), row["path_rgb_0"]) for row in reader]

    print(f"Processing {len(image_paths)} frames …")

    fig, ax_img, ax_3d = setup_plot()

    detections = []
    marker_poses: dict[int, list] = {}  # marker_id → list of T_world_marker (4x4)
    traj_pts = []
    last_frame_rgb = None
    last_corners, last_ids = None, None
    last_T_world_cam = np.eye(4)

    for frame_idx, (ts_ns, rel_path) in enumerate(image_paths):
        T_world_cam = gt_poses.get(ts_ns)
        if T_world_cam is None:
            print(f"  WARNING: no groundtruth for ts={ts_ns}", file=sys.stderr)
            continue

        cam_pos = T_world_cam[:3, 3]
        traj_pts.append(cam_pos.copy())

        img_path = DATA_DIR / rel_path
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  WARNING: could not read {img_path}", file=sys.stderr)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        last_frame_rgb = frame_rgb
        last_corners, last_ids = corners, ids
        last_T_world_cam = T_world_cam

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_LENGTH, K, dist
            )
            for i, marker_id in enumerate(ids.ravel()):
                T_cam_marker = rvec_tvec_to_matrix(rvecs[i], tvecs[i])
                T_world_marker = T_world_cam @ T_cam_marker
                t_world = T_world_marker[:3, 3]
                t_cam = tvecs[i].ravel()
                r_cam = rvecs[i].ravel()

                detections.append({
                    "ts_ns": ts_ns,
                    "image": rel_path,
                    "marker_id": int(marker_id),
                    "tx_cam": t_cam[0], "ty_cam": t_cam[1], "tz_cam": t_cam[2],
                    "rx_cam": r_cam[0], "ry_cam": r_cam[1], "rz_cam": r_cam[2],
                    "tx_world": t_world[0], "ty_world": t_world[1], "tz_world": t_world[2],
                })
                marker_poses.setdefault(int(marker_id), []).append(T_world_marker)

                print(
                    f"  [{rel_path}] id={marker_id:3d}  "
                    f"cam=({t_cam[0]:+.3f}, {t_cam[1]:+.3f}, {t_cam[2]:+.3f}) m  "
                    f"world=({t_world[0]:+.3f}, {t_world[1]:+.3f}, {t_world[2]:+.3f}) m"
                )

        if frame_idx % PLOT_EVERY == 0:
            redraw(ax_img, ax_3d, frame_rgb, corners, ids,
                   traj_pts, T_world_cam, K, marker_poses)

    # final redraw with complete data
    redraw(ax_img, ax_3d, last_frame_rgb, last_corners, last_ids,
           traj_pts, last_T_world_cam, K, marker_poses)
    print(f"\nDetected {len(detections)} observations.")

    if not detections:
        plt.ioff()
        fig.show()
        return

    # --- per-frame detections CSV ---
    det_csv = DATA_DIR / "aruco_detections.csv"
    det_fields = ["ts_ns", "image", "marker_id",
                  "tx_cam", "ty_cam", "tz_cam", "rx_cam", "ry_cam", "rz_cam",
                  "tx_world", "ty_world", "tz_world"]
    with open(det_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=det_fields)
        writer.writeheader()
        writer.writerows(detections)
    print(f"Per-frame detections → {det_csv}")

    # --- averaged world position per marker id CSV ---
    print("\n--- Marker world positions (averaged over all observations) ---")
    marker_rows = []
    for marker_id, transforms in sorted(marker_poses.items()):
        _, mean_t = mean_pose(transforms)
        positions = np.array([T[:3, 3] for T in transforms])
        std_pos = np.std(positions, axis=0)
        marker_rows.append({
            "marker_id": marker_id,
            "n_obs": len(transforms),
            "tx_world": mean_t[0], "ty_world": mean_t[1], "tz_world": mean_t[2],
            "std_x": std_pos[0], "std_y": std_pos[1], "std_z": std_pos[2],
        })
        print(
            f"  id={marker_id:3d}  n={len(transforms):4d}  "
            f"pos=({mean_t[0]:+.4f}, {mean_t[1]:+.4f}, {mean_t[2]:+.4f}) m  "
            f"std=({std_pos[0]:.4f}, {std_pos[1]:.4f}, {std_pos[2]:.4f}) m"
        )

    markers_csv = DATA_DIR / "aruco_world_positions.csv"
    marker_fields = ["marker_id", "n_obs",
                     "tx_world", "ty_world", "tz_world",
                     "std_x", "std_y", "std_z"]
    with open(markers_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=marker_fields)
        writer.writeheader()
        writer.writerows(marker_rows)
    print(f"\nMarker world positions → {markers_csv}")

    plt.ioff()
    fig.show()


if __name__ == "__main__":
    main()
