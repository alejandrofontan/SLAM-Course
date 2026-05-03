
# third-party
import cv2
import gtsam
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

def draw_camera_frustum(ax, T_world_cam: np.ndarray, K: np.ndarray,
                        marker_length: float, img_size: tuple = (640, 480)):
    """Draw a camera frustum pyramid in world frame using the intrinsics."""
    R = T_world_cam[:3, :3]
    t = T_world_cam[:3, 3]
    d = marker_length * 0.8  # frustum depth scaled to marker size

    # unproject image corners to camera frame at depth d
    img_w, img_h = img_size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    corners_img = np.array([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]], dtype=float)
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


def draw_marker_frame(ax, R: np.ndarray, t: np.ndarray, visible: bool, marker_length: float):
    """Draw a flat square + XYZ axes for one ArUco marker in world frame."""
    color = "green" if visible else "black"
    alpha = 0.5 if visible else 0.2
    half = marker_length / 2
    axis_len = marker_length * 0.6

    # marker square corners in marker frame (Z = 0 plane)
    corners_m = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]])
    corners_w = (R @ corners_m.T).T + t  # (4, 3)

    poly = Poly3DCollection([corners_w], alpha=alpha, facecolor=color, edgecolor=color)
    ax.add_collection3d(poly)

    # X (red), Y (green), Z (blue) axes
    for axis_idx, axis_color in enumerate(["red", "lime", "blue"]):
        end = t + R[:, axis_idx] * axis_len
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


def redraw(ax_img, ax_3d, frame_rgb, corners, ids, traj_pts, T_world_cam, K, marker_poses: dict,
           marker_length: float, img_size: tuple = (640, 480)):
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

    draw_camera_frustum(ax_3d, T_world_cam, K, marker_length, img_size)

    for marker_id, transforms in marker_poses.items():
        R, t = mean_pose(transforms)
        visible = marker_id in visible_ids
        draw_marker_frame(ax_3d, R, t, visible, marker_length)
        color = "green" if visible else "black"
        ax_3d.text(t[0], t[1], t[2], f" {marker_id}", fontsize=8, color=color)

    if ax_3d.get_legend_handles_labels()[0]:
        ax_3d.legend(loc="upper left", fontsize=8)
    plt.pause(0.001)

def animate_optimisation(optimizer, graph, initial_estimate, traj_pts,
                         last_T_world_cam, K, marker_length: float,
                         params=None, img_size: tuple = (640, 480)):
    """Run LM iterations one at a time, updating a live split view:
      left  — factor-graph error convergence curve (log scale)
      right — 3-D trajectories morphing toward the optimised solution

    Returns the final optimised Values.
    """
    if params is None:
        params = gtsam.LevenbergMarquardtParams()
    max_iter = params.getMaxIterations()
    rel_tol  = params.getRelativeErrorTol()
    abs_tol  = params.getAbsoluteErrorTol()

    plt.ion()
    fig = plt.figure(figsize=(14, 6))
    ax_err = fig.add_subplot(121)
    ax_3d  = fig.add_subplot(122, projection="3d")
    fig.suptitle("Levenberg–Marquardt optimisation", fontsize=11)

    errors = [graph.error(initial_estimate)]

    for _ in range(max_iter):
        optimizer.iterate()
        curr_error = optimizer.error()
        errors.append(curr_error)
        prev_error = errors[-2]
        if abs(prev_error - curr_error) < abs_tol:
            break
        if prev_error > 0 and abs(prev_error - curr_error) / prev_error < rel_tol:
            break

        current = optimizer.values()

        # extract current camera trajectory and landmark poses
        opt_traj_pts: list = []
        opt_marker_poses: dict = {}
        for key in sorted(initial_estimate.keys()):
            sym  = gtsam.Symbol(key)
            pose = current.atPose3(key)
            t    = np.array(pose.translation())
            R    = pose.rotation().matrix()
            T    = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
            if chr(sym.chr()) == "x":
                opt_traj_pts.append(t.copy())
            else:
                opt_marker_poses[sym.index()] = [T]

        # --- error curve ---
        ax_err.cla()
        ax_err.semilogy(errors, color="steelblue", linewidth=1.5)
        ax_err.set_xlabel("Iteration")
        ax_err.set_ylabel("Factor graph error (log)")
        ax_err.set_title(f"Convergence  —  iter {len(errors) - 1}  |  error {errors[-1]:.4f}")
        ax_err.grid(True, which="both", alpha=0.3)

        # --- 3-D map ---
        ax_3d.cla()
        ax_3d.set_xlabel("X (m)")
        ax_3d.set_ylabel("Y (m)")
        ax_3d.set_zlabel("Z (m)")
        ax_3d.set_title("Trajectory")

        if len(traj_pts) > 1:
            traj = np.array(traj_pts)
            ax_3d.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                       color="steelblue", linewidth=1, alpha=0.4, label="odometry")

        if len(opt_traj_pts) > 1:
            opt = np.array(opt_traj_pts)
            ax_3d.plot(opt[:, 0], opt[:, 1], opt[:, 2],
                       color="green", linewidth=1, label="optimised")

        draw_camera_frustum(ax_3d, last_T_world_cam, K, marker_length, img_size)

        for marker_id, transforms in opt_marker_poses.items():
            R, t = mean_pose(transforms)
            draw_marker_frame(ax_3d, R, t, visible=True, marker_length=marker_length)
            ax_3d.text(t[0], t[1], t[2], f" {marker_id}", fontsize=8, color="green")

        ax_3d.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        plt.waitforbuttonpress()

    plt.ioff()
    plt.close(fig)
    return optimizer.values()


def plot_optimised_result(result, initial_estimate, traj_pts, gt_pts,
                          last_frame_rgb, last_corners, last_ids, last_T_world_cam, K,
                          marker_length: float, img_size: tuple = (640, 480)):
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

    # --- Figure 1: last frame image ---
    fig_img = plt.figure(figsize=(8, 6))
    ax_img_f = fig_img.add_subplot(111)
    ax_img_f.axis("off")
    ax_img_f.set_title("Last frame")
    if last_frame_rgb is not None:
        display = last_frame_rgb.copy()
        if last_ids is not None:
            cv2.aruco.drawDetectedMarkers(display, last_corners, last_ids)
        ax_img_f.imshow(display)
    fig_img.tight_layout()

    # --- Figure 2: 3-D map (same style as live redraw) ---
    fig_3d = plt.figure(figsize=(8, 8))
    ax_3d_f = fig_3d.add_subplot(111, projection="3d")
    ax_3d_f.set_xlabel("X (m)")
    ax_3d_f.set_ylabel("Y (m)")
    ax_3d_f.set_zlabel("Z (m)")
    ax_3d_f.set_title("Final optimised map")

    if len(traj_pts) > 1:
        traj = np.array(traj_pts)
        ax_3d_f.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                     color="steelblue", linewidth=1, label="odometry")

    if len(opt_traj_pts) > 1:
        opt = np.array(opt_traj_pts)
        ax_3d_f.plot(opt[:, 0], opt[:, 1], opt[:, 2],
                     color="green", linewidth=1, label="optimised")

    gt_valid = [p for p in gt_pts if p is not None]
    if len(gt_valid) > 1:
        gt_arr = np.array(gt_valid)
        ax_3d_f.plot(gt_arr[:, 0], gt_arr[:, 1], gt_arr[:, 2],
                     color="red", linewidth=1, linestyle="--", label="ground truth")

    draw_camera_frustum(ax_3d_f, last_T_world_cam, K, marker_length, img_size)

    for marker_id, transforms in opt_marker_poses.items():
        R, t = mean_pose(transforms)
        draw_marker_frame(ax_3d_f, R, t, visible=True, marker_length=marker_length)
        ax_3d_f.text(t[0], t[1], t[2], f" {marker_id}", fontsize=8, color="green")

    ax_3d_f.legend(loc="upper left", fontsize=8)

    # --- Figure 3: ATE per frame vs ground truth ---
    fig_ate = plt.figure(figsize=(10, 4))
    ax_ate = fig_ate.add_subplot(111)
    n = min(len(traj_pts), len(opt_traj_pts), len(gt_pts))
    # keep only frames where a GT position is available
    valid_idx = [i for i in range(n) if gt_pts[i] is not None]
    if valid_idx:
        frames    = np.array(valid_idx)
        odom_arr  = np.array([traj_pts[i]     for i in valid_idx])
        opt_arr   = np.array([opt_traj_pts[i]  for i in valid_idx])
        gt_arr    = np.array([gt_pts[i]        for i in valid_idx])
        ate_odom  = np.linalg.norm(odom_arr - gt_arr, axis=1)
        ate_opt   = np.linalg.norm(opt_arr  - gt_arr, axis=1)
        rmse_odom = float(np.sqrt(np.mean(ate_odom ** 2)))
        rmse_opt  = float(np.sqrt(np.mean(ate_opt  ** 2)))
        ax_ate.plot(frames, ate_odom, color="steelblue", linewidth=1,
                    label=f"odometry  (RMSE = {rmse_odom:.4f} m)")
        ax_ate.plot(frames, ate_opt,  color="green",     linewidth=1,
                    label=f"optimised (RMSE = {rmse_opt:.4f} m)")
        ax_ate.set_title("ATE vs ground truth")
        ax_ate.set_xlabel("Frame")
        ax_ate.set_ylabel("Translation error (m)")
        ax_ate.legend(fontsize=8)
        ax_ate.grid(True, alpha=0.3)
    fig_ate.tight_layout()

    plt.show()