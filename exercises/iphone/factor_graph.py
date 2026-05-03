
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

    # K is the 3x3 intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
    # It maps 3D camera-frame points to 2D pixel coordinates.
    # fx, fy are the focal lengths in pixels; cx, cy is the principal point (image centre).
    # dist holds the lens distortion coefficients (k1, k2, p1, p2, ...).
    K, dist = load_camera_matrix(DATA_DIR / "calibration.yaml")

    # odom_poses is a dict {timestamp_ns: T_world_cam} where T_world_cam is a 4x4 SE(3) matrix.
    # It expresses the position and orientation of the camera in the world frame at each instant.
    # We use it as odometry: the relative motion between consecutive frames becomes
    # a BetweenFactor constraint in the factor graph.
    odom_poses = load_odometry_poses(DATA_DIR / "groundtruth.csv")

    # ArucoDetector identifies ArUco markers in a greyscale image.
    # getPredefinedDictionary selects which set of marker patterns to look for
    # (here DICT_6X6_50: 6x6 bit patterns, 50 unique IDs).
    # DetectorParameters controls thresholds for corner detection and decoding;
    # defaults work well for standard lighting conditions.
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
        cv2.aruco.DetectorParameters(),
    )

    # rgb.csv lists every RGB frame with its nanosecond timestamp and relative file path.
    # We build an ordered list of (timestamp, path) pairs to iterate over the sequence.
    # The timestamp is the key used to look up the matching groundtruth pose in odom_poses.
    with open(DATA_DIR / "rgb.csv") as f:
        reader = csv.DictReader(f)
        image_paths = [(int(row["ts_rgb_0 (ns)"]), row["path_rgb_0"]) for row in reader]

    print(f"Processing {len(image_paths)} frames …")

    # --- Noise models ---
    # Every factor in the graph encodes how much we trust its measurement via a noise model.
    # Diagonal.Sigmas takes one standard deviation (sigma) per DOF — larger sigma = less trust.
    # A Pose3 has 6 DOF ordered as (rx, ry, rz, tx, ty, tz) in the tangent space (Lie algebra).

    # PRIOR_NOISE: very tight (1 mm / 0.001 rad) — anchors the first pose to fix the
    # gauge freedom (the factor graph has no absolute reference without it).
    PRIOR_NOISE    = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]))

    # ODOMETRY_NOISE: moderate (1 cm / 0.01 rad) — reflects typical IMU/wheel odometry drift
    # between consecutive frames. The relative motion T_rel comes from visual-inertial odometry here,
    # so it could be tighter; loosen it if using noisier odometry sources.
    ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]))

    # MEASUREMENT_NOISE: models ArUco pose estimation uncertainty.
    # Rotation (0.05 rad ≈ 3°) is noisier than translation (0.02 m) because
    # small marker detection errors cause larger angular errors than positional ones.
    MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.05, 0.02, 0.02, 0.02]))

    # graph collects all factors (constraints): prior, odometry, and ArUco measurements.
    # It defines the objective function that the optimiser will minimise.
    graph = gtsam.NonlinearFactorGraph()

    # initial_estimate holds the starting value for every variable (X and L)
    # before optimisation. A good initial guess (here: visual-inertial odometry) helps the
    # nonlinear solver converge to the correct solution.
    initial_estimate = gtsam.Values()

    # --- Auxiliary variables ---
    # These accumulate data across frames and are used for visualisation,
    # CSV export, and building the factor graph inside the main loop.
    fig, ax_img, ax_3d = setup_plot()  # live plot figure and axes
    detections = []                    # flat list of per-frame ArUco observations (for CSV)
    marker_poses: dict[int, list] = {}  # marker_id → list of T_world_marker (4x4), used to average poses
    traj_pts = []                      # camera positions in world frame (odometry trajectory)
    last_frame_rgb = None              # last RGB frame shown in the image panel
    last_corners, last_ids = None, None  # ArUco detections of the last frame (for final plot)
    last_T_world_cam = np.eye(4)       # camera pose of the last frame (for final frustum)
    prev_T_world_cam = None            # camera pose of the previous frame (to compute T_rel)

    # --- Main loop: iterate over every RGB frame in chronological order ---
    # For each frame we do three things:
    #   1. Add a camera pose variable X(frame_idx) and an odometry factor to the graph.
    #   2. Detect ArUco markers and add a measurement factor L(marker_id) for each one.
    #   3. Update the live visualisation every PLOT_EVERY frames.
    for frame_idx, (ts_ns, rel_path) in enumerate(image_paths):
        T_world_cam = odom_poses.get(ts_ns)

        # Convert the 4x4 numpy matrix to a GTSAM Pose3 (rotation + translation).
        # Rot3 wraps the 3x3 rotation matrix; Point3 wraps the translation vector.
        # This pose is inserted into initial_estimate as the starting value for
        # variable X(frame_idx) — the camera pose at this frame.
        # * unpacks the numpy array [tx, ty, tz] into three separate float arguments
        # because gtsam.Point3 expects Point3(tx, ty, tz), not Point3(array).
        cur_pose_gtsam = gtsam.Pose3(gtsam.Rot3(T_world_cam[:3, :3]), gtsam.Point3(*T_world_cam[:3, 3]))

        # X(frame_idx) is the symbolic key for this camera pose variable in the graph.
        # X and L are shorthand generators: X(i) produces a unique integer key
        # that GTSAM uses to distinguish camera poses (X) from landmarks (L).
        initial_estimate.insert(X(frame_idx), cur_pose_gtsam)

        if frame_idx == 0:
            # PriorFactorPose3 is a unary (absolute) factor that pins X(0) to cur_pose_gtsam.
            # Without it, the factor graph has no fixed reference frame: the optimizer could
            # rigidly shift or rotate the entire solution without changing any relative error
            # (this ambiguity is called gauge freedom). The tight PRIOR_NOISE (1 mm / 0.001 rad)
            # effectively freezes the first camera pose, anchoring the whole map to the world frame.
            # Only X(0) needs a prior; every other pose is constrained through relative BetweenFactors.
            graph.add(gtsam.PriorFactorPose3(X(frame_idx), cur_pose_gtsam, PRIOR_NOISE))
        else:
            # T_rel is the relative motion from the previous camera pose to the current one,
            # expressed in the previous camera frame: T_rel = inv(T_world_prev) @ T_world_cur.
            # This is the odometry measurement: "how much did the camera move between two frames?"
            # BetweenFactorPose3 encodes this as a binary constraint between X(frame_idx-1) and
            # X(frame_idx): it penalises deviations from T_rel weighted by ODOMETRY_NOISE.
            # Unlike the prior, BetweenFactors are relative — they only constrain the difference
            # between two variables, which is what odometry (and most sensors) naturally measures.
            T_rel = np.linalg.inv(prev_T_world_cam) @ T_world_cam
            odometry = gtsam.Pose3(gtsam.Rot3(T_rel[:3, :3]), gtsam.Point3(*T_rel[:3, 3]))
            graph.add(gtsam.BetweenFactorPose3(X(frame_idx - 1), X(frame_idx), odometry, ODOMETRY_NOISE))

        prev_T_world_cam = T_world_cam  # slide the window: current becomes previous for the next iteration


        # Load the image and prepare two versions for downstream processing:
        # - frame_rgb: colour image (BGR→RGB) used only for display.
        # - gray: single-channel image required by the ArUco detector,
        #   which looks for black-and-white square patterns.
        # detectMarkers returns the pixel corners of every found marker and their IDs;
        # the third return value (rejected candidates) is discarded with _.
        img_path = DATA_DIR / rel_path
        frame = cv2.imread(str(img_path))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        # Cache this frame's data so the final static plot can use the last processed frame.
        last_frame_rgb = frame_rgb
        last_corners, last_ids = corners, ids
        last_T_world_cam = T_world_cam

        # If any markers were found, estimate their 3-D pose relative to the camera.
        # estimatePoseSingleMarkers uses the known physical marker size (MARKER_LENGTH),
        # the camera intrinsics (K), and the detected pixel corners to solve a PnP problem,
        # returning one rotation vector (rvec) and translation vector (tvec) per marker.
        # The inner loop then processes each marker individually.
        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_LENGTH, K, dist
            )
            for i, marker_id in enumerate(ids.ravel()):
                # T_cam_marker: 4x4 SE(3) pose of the marker in the camera frame,
                # built from the Rodrigues rotation vector and translation returned by PnP.
                # T_world_marker: the same pose lifted to the world frame by left-multiplying
                # with the known camera-to-world transform — chain rule of rigid-body transforms:
                #   T_world_marker = T_world_cam @ T_cam_marker
                T_cam_marker = rvec_tvec_to_matrix(rvecs[i], tvecs[i])
                T_world_marker = T_world_cam @ T_cam_marker

                # Accumulate world-frame observations for the live plot only.
                # redraw() calls mean_pose() on this list each frame, averaging all past sightings
                # into a single stable marker position. Without accumulation the marker plane would
                # jitter with each noisy PnP estimate. The final static plot ignores marker_poses
                # and uses opt_marker_poses extracted from the optimiser result instead.
                marker_poses.setdefault(int(marker_id), []).append(T_world_marker)

                T_cam_marker_gtsam = gtsam.Pose3(gtsam.Rot3(T_cam_marker[:3, :3]), gtsam.Point3(*T_cam_marker[:3, 3]))
                graph.add(gtsam.BetweenFactorPose3(X(frame_idx), L(marker_id), T_cam_marker_gtsam, MEASUREMENT_NOISE))
                if not initial_estimate.exists(L(marker_id)):
                    T_world_marker_gtsam = gtsam.Pose3(gtsam.Rot3(T_world_marker[:3, :3]), gtsam.Point3(*T_world_marker[:3, 3]))
                    initial_estimate.insert(L(marker_id), T_world_marker_gtsam)


        cam_pos = T_world_cam[:3, 3]
        traj_pts.append(cam_pos.copy())
        if frame_idx % PLOT_EVERY == 0:
            redraw(ax_img, ax_3d, frame_rgb, corners, ids,
                   traj_pts, T_world_cam, K, marker_poses)

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
