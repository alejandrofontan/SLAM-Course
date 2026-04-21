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

# Documentation and usage instructions
# https://borglab.github.io/gtsam/user-guide/
# https://borglab.github.io/gtsam/pose2slamexample/#id-3-1-add-prior-factor

######################################################################################################
######################################################################################################
# Create noise models with specified standard deviations (sigmas).
# gtsam.noiseModel.Diagonal.Sigmas takes a numpy array of standard deviations.

# Prior noise on the first pose (x, y, theta) - sigmas = [0.3m, 0.3m, 0.1rad]
#PRIOR_NOISE = ???????
# Odometry noise (dx, dy, dtheta) - sigmas = [0.2m, 0.2m, 0.1rad]
#ODOMETRY_NOISE = ???????
# Measurement noise (bearing, range) - sigmas = [0.1rad, 0.2m]
#MEASUREMENT_NOISE = ??????

######################################################################################################
######################################################################################################
# Create an empty nonlinear factor graph
#graph = ????

######################################################################################################
######################################################################################################
# Add a prior on pose X(1) at the origin.
# A prior factor consists of a mean (gtsam.Pose2) and a noise model.

######################################################################################################
######################################################################################################
# Add odometry factors between X(1),X(2) and X(2),X(3), respectively.
# The measurement is the relative motion: Pose2(dx, dy, dtheta).

# Between X(1) and X(2): Move forward 2m
# graph.add()
# Between X(2) and X(3): Move forward 2m
# graph.add(gtsam.BetweenFactorPose2(X(2), X(3), gtsam.Pose2(2.0, 0.0, 0.0), ODOMETRY_NOISE))

######################################################################################################
######################################################################################################
# Add Range-Bearing measurements to two different landmarks L(1) and L(2).
# Measurements are Bearing (gtsam.Rot2) and Range (float).

# From X(1) to L(1)
#graph.add()
# From X(2) to L(1)
#graph.add()
# From X(3) to L(2)
#graph.add()

# Print the graph. This shows the factors and the variables they connect.
#print("Factor Graph:\n{}".format(graph))

######################################################################################################
######################################################################################################
# Create (deliberately inaccurate) initial estimate.
# gtsam.Values is a container mapping variable keys to their estimated values.
#initial_estimate = gtsam.Values()

# Insert initial guesses for poses (Pose2: x, y, theta)
#initial_estimate.insert(X(1), gtsam.Pose2(-0.25, 0.20, 0.15))
#initial_estimate.insert(X(2), gtsam.Pose2(2.30, 0.10, -0.20))
#initial_estimate.insert(X(3), gtsam.Pose2(4.10, 0.10, 0.10))

# Insert initial guesses for landmarks (Point2: x, y)
#initial_estimate.insert(L(1), gtsam.Point2(1.80, 2.10))
#initial_estimate.insert(L(2), gtsam.Point2(4.10, 1.80))

# Print the initial estimate
#print("Initial Estimate:\n{}".format(initial_estimate))

# Now that we have an initial estimate we can also visualize the graph:
#graphviz.Source(graph.dot(initial_estimate)).view()

######################################################################################################
######################################################################################################
# Optimize using Levenberg-Marquardt optimization.
# The optimizer accepts optional parameters, but we'll use the defaults here.
#params = ????
#optimizer =

# Perform the optimization
#result = optimizer.optimize()

# Print the final optimized result
# This gtsam.Values object contains the most likely estimates for all variables.
#print("\nFinal Result:\n{}".format(result))

# fig = plt.figure(1)
# axes = fig.add_subplot()
# axes = fig.axes[0]

# Plot 2D poses
# poses : gtsam.Values = gtsam.utilities.allPose2s(result)
# for key in poses.keys():
#     pose = poses.atPose2(key)
#     gp.plot_pose2_on_axes(axes, pose, axis_length=0.3)

# Plot 2D landmarks
# landmarks : np.ndarray = gtsam.utilities.extractPoint2(result) # 2xn array
# for landmark in landmarks:
#     gp.plot_point2_on_axes(axes, landmark, linespec="b")

# axes.set_aspect("equal", adjustable="datalim")

######################################################################################################
######################################################################################################
# Calculate and print marginal covariances for all variables.
# This provides information about the uncertainty of the estimates.
# marginals = gtsam.Marginals(graph, result)

# Print the covariance matrix for each variable
# print("X1 covariance:\n{}\n".format(marginals.marginalCovariance(X(1))))
# print("X2 covariance:\n{}\n".format(marginals.marginalCovariance(X(2))))
# print("X3 covariance:\n{}\n".format(marginals.marginalCovariance(X(3))))
# print("L1 covariance:\n{}\n".format(marginals.marginalCovariance(L(1))))
# print("L2 covariance:\n{}\n".format(marginals.marginalCovariance(L(2))))

# fig = plt.figure(2)
# axes = fig.add_subplot()
# axes = fig.axes[0]

# Plot 2D poses
# poses = gtsam.utilities.allPose2s(result)
# for key in poses.keys():
#     pose = poses.atPose2(key)
#     covariance = marginals.marginalCovariance(key)

#     gp.plot_pose2_on_axes(axes, pose, covariance=covariance, axis_length=0.3)

# Plot 2D landmarks
# landmarks: np.ndarray = gtsam.utilities.extractPoint2(result)  # 2xn array
# for j, landmark in enumerate(landmarks):
#     gp.plot_point2_on_axes(axes, landmark, linespec="b")
#     covariance = marginals.marginalCovariance(L(j+1))
#     gp.plot_covariance_ellipse_2d(axes, landmark, covariance=covariance)

# axes.set_aspect("equal", adjustable="datalim")
# plt.show()