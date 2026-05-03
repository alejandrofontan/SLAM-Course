
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

import gtsam
from gtsam.symbol_shorthand import L, X

from dataset import load_camera_matrix, load_odometry_poses
from math_utilities import rvec_tvec_to_matrix
from visualization import mean_pose, draw_camera_frustum, draw_marker_frame, setup_plot, redraw

DATA_DIR = Path("/home/alejandro/VSLAM-LAB-Benchmark/STRAYSCANNER/bb7e805d7c")
MARKER_LENGTH = 0.15  # physical marker side length in metres — adjust as needed
ARUCO_DICT = cv2.aruco.DICT_6X6_50
PLOT_EVERY = 10        # redraw every N frames for speed

def main():
    K, dist = load_camera_matrix(DATA_DIR / "calibration.yaml")
    gt_poses = load_odometry_poses(DATA_DIR / "groundtruth.csv")

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

    # tight prior on the first camera pose to fix gauge freedom (6-DOF: rx,ry,rz, tx,ty,tz)
    PRIOR_NOISE    = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]))
    ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]))
    MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.05, 0.02, 0.02, 0.02]))  # (rx,ry,rz, tx,ty,tz)

    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    prev_T_world_cam = None

    for frame_idx, (ts_ns, rel_path) in enumerate(image_paths):
        T_world_cam = gt_poses.get(ts_ns)
        if T_world_cam is None:
            print(f"  WARNING: no groundtruth for ts={ts_ns}", file=sys.stderr)
            continue

        cam_pos = T_world_cam[:3, 3]
        traj_pts.append(cam_pos.copy())

        cur_pose_gtsam = gtsam.Pose3(gtsam.Rot3(T_world_cam[:3, :3]), gtsam.Point3(*T_world_cam[:3, 3]))
        initial_estimate.insert(X(frame_idx), cur_pose_gtsam)

        if frame_idx == 0:
            graph.add(gtsam.PriorFactorPose3(X(frame_idx), cur_pose_gtsam, PRIOR_NOISE))
        else:
            T_rel = np.linalg.inv(prev_T_world_cam) @ T_world_cam
            odometry = gtsam.Pose3(gtsam.Rot3(T_rel[:3, :3]), gtsam.Point3(*T_rel[:3, 3]))
            graph.add(gtsam.BetweenFactorPose3(X(frame_idx - 1), X(frame_idx), odometry, ODOMETRY_NOISE))

        prev_T_world_cam = T_world_cam

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
                T_cam_marker_gtsam = gtsam.Pose3(gtsam.Rot3(T_cam_marker[:3, :3]), gtsam.Point3(*T_cam_marker[:3, 3]))
                graph.add(gtsam.BetweenFactorPose3(X(frame_idx), L(marker_id), T_cam_marker_gtsam, MEASUREMENT_NOISE))
                if not initial_estimate.exists(L(marker_id)):
                    T_world_marker_gtsam = gtsam.Pose3(gtsam.Rot3(T_world_marker[:3, :3]), gtsam.Point3(*T_world_marker[:3, 3]))
                    initial_estimate.insert(L(marker_id), T_world_marker_gtsam)


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

    # --- optimise ---
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    print(f"\nInitial error: {graph.error(initial_estimate):.4f}")
    print(f"Final error:   {graph.error(result):.4f}")

    # rebuild trajectory and marker poses from optimised result for final redraw
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

    # --- final static plot ---
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

if __name__ == "__main__":
    main()
