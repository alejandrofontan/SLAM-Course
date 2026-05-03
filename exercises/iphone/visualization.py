
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import cv2

import gtsam

MARKER_LENGTH = 0.15  # physical marker side length in metres — adjust as needed
PLOT_EVERY = 10        # redraw every N frames for speed
AXIS_LEN = MARKER_LENGTH * 0.6   # length of the drawn frame axes
FRUSTUM_DEPTH = MARKER_LENGTH * 0.8  # near-plane distance for the camera frustum
IMG_W, IMG_H = 640, 480

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

def plot_optimised_result(result, initial_estimate, traj_pts,
                          last_frame_rgb, last_corners, last_ids, last_T_world_cam, K):
    # Extract optimised camera trajectory and landmark poses from the GTSAM result.
    # Keys are decoded with gtsam.Symbol to separate X (camera) from L (landmark) variables.
    opt_traj_pts = []
    opt_marker_poses: dict[int, list] = {}
    for key in sorted(initial_estimate.keys()):
        sym = gtsam.Symbol(key)
        pose = result.atPose3(key)
        T = np.eye(4)
        T[:3, :3] = pose.rotation().matrix()
        T[:3, 3] = pose.translation()
        if chr(sym.chr()) == "x":
            opt_traj_pts.append(T[:3, 3].copy())
        else:
            opt_marker_poses[sym.index()] = [T]

    plt.ioff()
    fig_final = plt.figure(figsize=(16, 7))
    ax_img_f = fig_final.add_subplot(121)
    ax_img_f.axis("off")
    ax_img_f.set_title("Last frame")
    if last_frame_rgb is not None:
        display = last_frame_rgb.copy()
        if last_ids is not None:
            cv2.aruco.drawDetectedMarkers(display, last_corners, last_ids)
        ax_img_f.imshow(display)

    ax_3d_f = fig_final.add_subplot(122, projection="3d")
    ax_3d_f.set_xlabel("X (m)")
    ax_3d_f.set_ylabel("Y (m)")
    ax_3d_f.set_zlabel("Z (m)")
    ax_3d_f.set_title("Final optimised map")

    # odometry trajectory (red)
    if len(traj_pts) > 1:
        gt = np.array(traj_pts)
        ax_3d_f.plot(gt[:, 0], gt[:, 1], gt[:, 2],
                     color="red", linewidth=1, label="odometry")

    # optimised camera trajectory (green)
    if len(opt_traj_pts) > 1:
        opt = np.array(opt_traj_pts)
        ax_3d_f.plot(opt[:, 0], opt[:, 1], opt[:, 2],
                     color="green", linewidth=1, label="optimised")

    # optimised camera frustum at last pose
    draw_camera_frustum(ax_3d_f, last_T_world_cam, K)

    # optimised ArUco landmarks — all visible (green), same style as live plot
    for marker_id, transforms in opt_marker_poses.items():
        R, t = mean_pose(transforms)
        draw_marker_frame(ax_3d_f, R, t, visible=True)
        ax_3d_f.text(t[0], t[1], t[2], f" {marker_id}", fontsize=8, color="green")

    ax_3d_f.legend(loc="upper left", fontsize=8)
    fig_final.tight_layout()
    plt.show()