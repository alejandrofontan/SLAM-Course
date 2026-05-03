
"""
Factor-graph SLAM exercise for iPhone recordings (StrayScanner format).

Builds a GTSAM factor graph over a sequence of RGB frames:
  - Camera pose variables X(i)   : one per frame, initialised from visual-inertial odometry.
  - Landmark variables    L(id)  : one per ArUco marker, initialised from the first detection.
  - PriorFactorPose3      : anchors X(0) to fix gauge freedom.
  - BetweenFactorPose3    : odometry edges between consecutive camera poses.
  - BetweenFactorPose3    : ArUco measurement edges between X(i) and L(id).

After processing all frames the graph is optimised with Levenberg-Marquardt.
A live 3-D plot shows the odometry trajectory and detected markers during processing;
a final static plot compares the odometry (red) against the optimised trajectory (green).
"""

# standard library
import csv
from pathlib import Path

# third-party
import cv2
import gtsam
import matplotlib.pyplot as plt
import numpy as np
from gtsam.symbol_shorthand import L, X

# local
from dataset import load_camera_matrix, load_camera_poses
from math_utilities import rvec_tvec_to_matrix
from visualization import animate_optimisation, plot_optimised_result, setup_plot, redraw

# Download and extract the exercise data from HuggingFace:
# https://huggingface.co/datasets/vslamlab/slam_course_data/blob/main/s11_loop.zip

# Adjust DATA_DIR below to point at whichever sequence you want to run.
DATA_DIR = Path("s11_loop")
MARKER_LENGTH = 0.15  # physical marker side length in metres — adjust as needed
ARUCO_DICT = cv2.aruco.DICT_6X6_50
PLOT_EVERY = 10        # redraw every N frames for speed
AVERAGE_MARKER_POSES = False  # True: average all past observations; False: show only the latest


def main():

    # K is the 3x3 intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
    # It maps 3D camera-frame points to 2D pixel coordinates.
    # fx, fy are the focal lengths in pixels; cx, cy is the principal point (image centre).
    # dist holds the lens distortion coefficients (k1, k2, p1, p2, ...).
    K, dist = load_camera_matrix(DATA_DIR / "calibration.yaml")

    # odom_poses is a dict {timestamp_ns: T_world_cam} where T_world_cam is a 4x4 SE(3) matrix.
    # It expresses the position and orientation of the camera in the world frame at each instant,
    # as estimated by the device's visual-inertial odometry (CameraTrajectory.csv).
    # The relative motion between consecutive frames becomes a BetweenFactor in the factor graph.
    # gt_poses holds the same structure but from the reference groundtruth trajectory,
    # used only for evaluation (ATE computation), not as a factor graph constraint.
    odom_poses = load_camera_poses(DATA_DIR / "CameraTrajectory.csv")
    gt_poses = load_camera_poses(DATA_DIR / "groundtruth.csv")

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
    marker_poses: dict[int, list] = {}  # marker_id → list of T_world_marker (4x4), used to average poses
    traj_pts = []                      # camera positions in world frame (odometry trajectory)
    gt_pts   = []                      # ground-truth positions aligned to traj_pts (None if unavailable)
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
        if T_world_cam is None:
            print(f"Warning: no odometry pose for timestamp {ts_ns} (frame {frame_idx}) — skipping")
            continue
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

                # Store world-frame observations for the live plot.
                # AVERAGE_MARKER_POSES=True: accumulate all sightings so mean_pose() can average
                # them into a stable position (less jitter). False: keep only the latest estimate.
                if AVERAGE_MARKER_POSES:
                    marker_poses.setdefault(int(marker_id), []).append(T_world_marker)
                else:
                    marker_poses[int(marker_id)] = [T_world_marker]

                # Add an ArUco measurement factor to the graph.
                # BetweenFactorPose3(X(i), L(id), T_cam_marker) encodes the constraint:
                #   "at frame i, the camera observed marker id at relative pose T_cam_marker."
                # The optimiser will adjust both X(i) and L(id) to best satisfy all such
                # constraints simultaneously, weighted by MEASUREMENT_NOISE.
                # L(marker_id) is initialised only on its first sighting (exists() guard):
                # subsequent detections just add more factors without touching initial_estimate.
                T_cam_marker_gtsam = gtsam.Pose3(gtsam.Rot3(T_cam_marker[:3, :3]), gtsam.Point3(*T_cam_marker[:3, 3]))
                graph.add(gtsam.BetweenFactorPose3(X(frame_idx), L(marker_id), T_cam_marker_gtsam, MEASUREMENT_NOISE))
                if not initial_estimate.exists(L(marker_id)):
                    T_world_marker_gtsam = gtsam.Pose3(gtsam.Rot3(T_world_marker[:3, :3]), gtsam.Point3(*T_world_marker[:3, 3]))
                    initial_estimate.insert(L(marker_id), T_world_marker_gtsam)


        # Append the current camera position to the trajectory and refresh the live plot.
        # redraw is called only every PLOT_EVERY frames to avoid slowing down processing.
        cam_pos = T_world_cam[:3, 3]
        traj_pts.append(cam_pos.copy())
        gt_T = gt_poses.get(ts_ns)
        gt_pts.append(gt_T[:3, 3].copy() if gt_T is not None else None)
        if frame_idx % PLOT_EVERY == 0:
            redraw(ax_img, ax_3d, frame_rgb, corners, ids,
                   traj_pts, T_world_cam, K, marker_poses, MARKER_LENGTH)

    # --- optimise ---
    # LevenbergMarquardtOptimizer minimises the total factor graph error — the sum of
    # squared, noise-weighted residuals from all prior, odometry, and measurement factors.
    # It iterates by linearising the nonlinear cost around the current estimate and solving
    # the resulting linear system, damping the step with a trust-region parameter (lambda).
    # graph.error() evaluates the objective at a given Values; comparing initial vs. final
    # error gives a quick sanity check that the optimisation converged and improved the estimate.
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    print(f"\nInitial error: {graph.error(initial_estimate):.4f}")
    plt.close(fig)
    result = animate_optimisation(optimizer, graph, initial_estimate, traj_pts,
                                  last_T_world_cam, K, MARKER_LENGTH, params)

    print(f"Final error:   {graph.error(result):.4f}")
    plot_optimised_result(result, initial_estimate, traj_pts, gt_pts,
                          last_frame_rgb, last_corners, last_ids, last_T_world_cam, K, MARKER_LENGTH)

if __name__ == "__main__":
    main()
