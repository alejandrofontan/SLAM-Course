import gtsam
import gtsam.utils.plot as gtsam_plot
import numpy as np
import plotly.graph_objects as go
from gtsam.symbol_shorthand import X, L

######################################################################################################
######################################################################################################
# Create an empty nonlinear factor graph
graph = gtsam.NonlinearFactorGraph()

######################################################################################################
######################################################################################################
# Retrieve Calibration file
cal_params_file = gtsam.findExampleDataFile("VO_calibration.txt")

# Read calibration parameters
cal_params = np.loadtxt(cal_params_file)

# Create a Cal3_S2Stereo calibration object
K = gtsam.Cal3_S2Stereo(cal_params)
print(K)

# Define the stereo measurement noise model (isotropic, 1 pixel standard deviation)
measurement_noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)

######################################################################################################
#####################################################################################################
# Odometry data file with camera poses
odometry_data_file = gtsam.findExampleDataFile("VO_camera_poses_large.txt")

# Read poses
camera_poses = np.loadtxt(odometry_data_file)

# Create a gtsam.Values object to hold the pose data
initial_estimates = gtsam.Values()

# Provide an initial guess for each camera pose
# for pose in camera_poses:
# 	pose_id = int(pose[0])
# 	m = pose[1:].copy().reshape(4, 4)
# 	initial_estimates.insert(?, ?)

######################################################################################################
#####################################################################################################
# Factor data file with pixel coordinates and landmark coordinates
factor_data_file = gtsam.findExampleDataFile("VO_stereo_factors_large.txt")

# Each row has 8 columns corresponding to [pose_id, landmark_id, uL, uR, v, X, Y, Z]
data_matrix = np.loadtxt(factor_data_file)

print(f"Loaded {data_matrix.shape[0]} stereo observations.")
print(f"First row (for reference):\n{data_matrix[0]}")

######################################################################################################
#####################################################################################################
# Add a stereo measurement factor for each observation
for entry in data_matrix:
	pose_id = int(entry[0])
	landmark_id = int(entry[1])
	uL, uR, v = entry[2:5]
	# graph.add(gtsam.GenericStereoFactor3D(
	# 	gtsam.StereoPoint2(?, ?, ?),	# Stereo measurement
	# 	?, 		                        # Assumed pixel measurement noise
	# 	?, 					            # Camera pose key
	# 	?, 				                # Landmark key
	# 	?								# Stereo calibration
	# ))

######################################################################################################
#####################################################################################################
# Provide an initial guess for each unique landmark in world coordinates
# for entry in data_matrix:
# 	landmark_id = int(entry[1])
# 	if initial_estimates.exists(L(landmark_id)):
# 		continue	# Skip this landmark if already initialized

# 	pose_id = int(entry[0])
# 	land_X, land_Y, land_Z	= entry[5:8]

# 	cam_pose = initial_estimates.atPose3(?)	# Get pose of observing camera
# 	world_point = cam_pose.transformFrom(				# Convert from camera frame to world frame
# 		gtsam.Point3(?, ?, ?)
# 	)
# 	initial_estimates.insert(?, ?)

######################################################################################################
#####################################################################################################
# Fix the first pose to serve as the world frame origin
first_pose = initial_estimates.atPose3(X(1))
graph.add(gtsam.NonlinearEqualityPose3(X(1), first_pose))

######################################################################################################
#####################################################################################################
# Set up optimizer with optional METIS ordering strategy
params = gtsam.LevenbergMarquardtParams()
params.setOrderingType("METIS")

# Optimize the factor graph to compute maximum posterior estimates
#optimizer = ???????????????
#result = optimizer.optimize()

######################################################################################################
#####################################################################################################
# Function to extract camera coordinates and put them in an N x 3 matrix
def extract_camera_xyz(values: gtsam.Values, prefix='x'):
    coords = []
    for key in values.keys():
        symbol = gtsam.Symbol(key)
        if symbol.chr() == ord(prefix):
            coords.append(values.atPose3(key).translation())
    return np.array([p for p in coords])

# Function to extract landmark coordinates and put them in an N x 3 matrix
def extract_landmark_xyz(values: gtsam.Values, prefix='l'):
    coords = []
    for key in values.keys():
        symbol = gtsam.Symbol(key)
        if symbol.chr() == ord(prefix):
            coords.append(values.atPoint3(key))
    return np.array([p for p in coords])

# Extract both initial and optimized estimates
init_cam = extract_camera_xyz(initial_estimates)
opt_cam = extract_camera_xyz(result)
init_land = extract_landmark_xyz(initial_estimates)
opt_land = extract_landmark_xyz(result)

# Define traces
trace_init_cam = go.Scatter3d(
    x=init_cam[:, 0], y=init_cam[:, 1], z=init_cam[:, 2],
    mode="lines+markers",
    name="Initial Trajectory",
    line=dict(color="lightgray", width=2, dash="dash"),
    marker=dict(size=2, color="black")
)

trace_opt_cam = go.Scatter3d(
    x=opt_cam[:, 0], y=opt_cam[:, 1], z=opt_cam[:, 2],
    mode="lines+markers",
    name="Optimized Trajectory",
    line=dict(color="blue", width=2),
    marker=dict(size=2, color="blue")
)

scatter_init_land = go.Scatter3d(
    x=init_land[:, 0], y=init_land[:, 1], z=init_land[:, 2],
    mode="markers",
    name="Initial Landmarks",
    marker=dict(size=2, color="black", opacity=0.5)
)

scatter_opt_land = go.Scatter3d(
    x=opt_land[:, 0], y=opt_land[:, 1], z=opt_land[:, 2],
    mode="markers",
    name="Optimized Landmarks",
    marker=dict(size=2, color="orange")
)

# Assemble the figure
fig = go.Figure(data=[trace_init_cam, trace_opt_cam, scatter_init_land, scatter_opt_land])

fig.update_layout(
    title="SLAM Result: Initial vs Optimized Estimates",
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data',
        camera=dict(
            up=dict(x=0, y=-1, z=0),
            center=dict(x=0, y=0.1, z=0),
            eye=dict(x=-1, y=-1, z=-1)
		)
	),
    legend=dict(x=0, y=1),
    margin=dict(l=0, r=0, b=0, t=30)
)

fig.show()

######################################################################################################
#####################################################################################################
# Compute total error (cost) before and after optimization
init_cost = graph.error(initial_estimates)
opt_cost = graph.error(result)

fig_cost = go.Figure(data=[
    go.Bar(name="Initial Estimate", x=["Initial"], y=[init_cost], marker_color="gray"),
    go.Bar(name="Optimized Result", x=["Optimized"], y=[opt_cost], marker_color="blue")
])

fig_cost.update_layout(
    title="Total Graph Error: Initial vs. Optimized",
    yaxis_title="Total Error (sum of squared residuals)",
    bargap=0.4,
    showlegend=True
)

fig_cost.show()