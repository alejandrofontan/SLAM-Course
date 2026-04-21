import matplotlib.pyplot as plt
import numpy as np
import graphviz

import gtsam
import gtsam.utils.plot as gp
from gtsam.symbol_shorthand import L, X

# This Example: We'll simulate a robot moving in a 2D plane. The robot has:
#    - Odometry: Measurements of its own motion (how far it moved between steps).
#    - Bearing-Range Sensor: A sensor (like a simple laser scanner) that measures
#    the bearing (angle) and range (distance) to landmarks.
#
# We'll build a factor graph representing the robot's poses, landmark positions,
# odometry measurements, and bearing-range measurements. Then, we'll use GTSAM to
# optimize the graph and find the best estimate of the robot's trajectory and landmark locations.

# We can use shorthand symbols for variable keys
# X(i) represents the i-th pose variable
# L(j) represents the j-th landmark variable

######################################################################################################
######################################################################################################
# Prior noise on the first pose (x, y, theta) - sigmas = [0.3m, 0.3m, 0.1rad]
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))
# Odometry noise (dx, dy, dtheta) - sigmas = [0.2m, 0.2m, 0.1rad]
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
# Measurement noise (bearing, range) - sigmas = [0.1rad, 0.2m]
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.2]))

######################################################################################################
######################################################################################################
# Create an empty nonlinear factor graph
graph = gtsam.NonlinearFactorGraph()

######################################################################################################
######################################################################################################
# Add a prior on pose X(1) at the origin.
# A prior factor consists of a mean (gtsam.Pose2) and a noise model.
graph.add(gtsam.PriorFactorPose2(X(1), gtsam.Pose2(0.0, 0.0, 0.0), PRIOR_NOISE))

######################################################################################################
######################################################################################################
# Add odometry factors between X(1),X(2) and X(2),X(3), respectively.
# The measurement is the relative motion: Pose2(dx, dy, dtheta).

# Between X(1) and X(2): Move forward 2m
graph.add(gtsam.BetweenFactorPose2(X(1), X(2), gtsam.Pose2(2.0, 0.0, 0.0), ODOMETRY_NOISE))
# Between X(2) and X(3): Move forward 2m
graph.add(gtsam.BetweenFactorPose2(X(2), X(3), gtsam.Pose2(2.0, 0.0, 0.0), ODOMETRY_NOISE))

######################################################################################################
######################################################################################################
# Add Range-Bearing measurements to two different landmarks L(1) and L(2).
# Measurements are Bearing (gtsam.Rot2) and Range (float).

# From X(1) to L(1)
graph.add(gtsam.BearingRangeFactor2D(X(1), L(1), gtsam.Rot2.fromDegrees(45), np.sqrt(4.0+4.0), MEASUREMENT_NOISE))
# From X(2) to L(1)
graph.add(gtsam.BearingRangeFactor2D(X(2), L(1), gtsam.Rot2.fromDegrees(90), 2.0, MEASUREMENT_NOISE))
# From X(3) to L(2)
graph.add(gtsam.BearingRangeFactor2D(X(3), L(2), gtsam.Rot2.fromDegrees(90), 2.0, MEASUREMENT_NOISE))

# Print the graph. This shows the factors and the variables they connect.
print("Factor Graph:\n{}".format(graph))

######################################################################################################
######################################################################################################
# Create (deliberately inaccurate) initial estimate.
# gtsam.Values is a container mapping variable keys to their estimated values.
initial_estimate = gtsam.Values()

# Insert initial guesses for poses (Pose2: x, y, theta)
initial_estimate.insert(X(1), gtsam.Pose2(-0.25, 0.20, 0.15))
initial_estimate.insert(X(2), gtsam.Pose2(2.30, 0.10, -0.20))
initial_estimate.insert(X(3), gtsam.Pose2(4.10, 0.10, 0.10))

# Insert initial guesses for landmarks (Point2: x, y)
initial_estimate.insert(L(1), gtsam.Point2(1.80, 2.10))
initial_estimate.insert(L(2), gtsam.Point2(4.10, 1.80))

# Print the initial estimate
print("Initial Estimate:\n{}".format(initial_estimate))

# Now that we have an initial estimate we can also visualize the graph:
graphviz.Source(graph.dot(initial_estimate)).view()

######################################################################################################
######################################################################################################
# Optimize using Levenberg-Marquardt optimization.
# The optimizer accepts optional parameters, but we'll use the defaults here.
params = gtsam.LevenbergMarquardtParams()
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

# Perform the optimization
result = optimizer.optimize()

# Print the final optimized result
# This gtsam.Values object contains the most likely estimates for all variables.
print("\nFinal Result:\n{}".format(result))

fig = plt.figure(1)
axes = fig.add_subplot()
axes = fig.axes[0]

# Plot 2D poses
poses : gtsam.Values = gtsam.utilities.allPose2s(result)
for key in poses.keys():
    pose = poses.atPose2(key)
    gp.plot_pose2_on_axes(axes, pose, axis_length=0.3)

# Plot 2D landmarks
landmarks : np.ndarray = gtsam.utilities.extractPoint2(result) # 2xn array
for landmark in landmarks:
    gp.plot_point2_on_axes(axes, landmark, linespec="b")

axes.set_aspect("equal", adjustable="datalim")

######################################################################################################
######################################################################################################
# Calculate and print marginal covariances for all variables.
# This provides information about the uncertainty of the estimates.
marginals = gtsam.Marginals(graph, result)

# Print the covariance matrix for each variable
print("X1 covariance:\n{}\n".format(marginals.marginalCovariance(X(1))))
print("X2 covariance:\n{}\n".format(marginals.marginalCovariance(X(2))))
print("X3 covariance:\n{}\n".format(marginals.marginalCovariance(X(3))))
print("L1 covariance:\n{}\n".format(marginals.marginalCovariance(L(1))))
print("L2 covariance:\n{}\n".format(marginals.marginalCovariance(L(2))))

fig = plt.figure(2)
axes = fig.add_subplot()
axes = fig.axes[0]

# Plot 2D poses
poses = gtsam.utilities.allPose2s(result)
for key in poses.keys():
    pose = poses.atPose2(key)
    covariance = marginals.marginalCovariance(key)

    gp.plot_pose2_on_axes(axes, pose, covariance=covariance, axis_length=0.3)

# Plot 2D landmarks
landmarks: np.ndarray = gtsam.utilities.extractPoint2(result)  # 2xn array
for j, landmark in enumerate(landmarks):
    gp.plot_point2_on_axes(axes, landmark, linespec="b")
    covariance = marginals.marginalCovariance(L(j+1))
    gp.plot_covariance_ellipse_2d(axes, landmark, covariance=covariance)

axes.set_aspect("equal", adjustable="datalim")

######################################################################################################
######################################################################################################
# Linearize the graph at the optimized result to get the Jacobian and Hessian.
# graph.linearize() returns a GaussianFactorGraph (the local linear approximation).
# .jacobian() returns (A, b) where A is the full stacked Jacobian and b the RHS.
# .hessian() returns (H, eta) where H = A^T A and eta = A^T b.
linear_graph = graph.linearize(result)
A, b = linear_graph.jacobian()
H, eta = linear_graph.hessian()

# Build column tick labels from the key ordering
POSE_DIMS  = ["x", "y", "θ"]
POINT_DIMS = ["x", "y"]
col_labels = []
for key in linear_graph.keyVector():
    sym = gtsam.Symbol(key)
    name = f"{chr(sym.chr())}{sym.index()}"
    dims = POSE_DIMS if chr(sym.chr()) == 'x' else POINT_DIMS
    col_labels.extend([f"{name}[{d}]" for d in dims])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im0 = axes[0].imshow(np.abs(A), aspect="auto", cmap="viridis")
axes[0].set_title("Jacobian |A|")
axes[0].set_xlabel("Variables")
axes[0].set_ylabel("Factors (rows)")
axes[0].set_xticks(range(len(col_labels)))
axes[0].set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(np.abs(H), cmap="viridis")
axes[1].set_title("Hessian |H = AᵀA|")
axes[1].set_xticks(range(len(col_labels)))
axes[1].set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
axes[1].set_yticks(range(len(col_labels)))
axes[1].set_yticklabels(col_labels, fontsize=8)
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()