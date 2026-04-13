"""
ekf_slam.py

Python port of EKFSLAM.m
Original source: Probabilistic Robotics course, Sapienza University of Rome
Copyright (c) 2016 Bartolomeo Della Corte, Giorgio Grisetti
License: CC Attribution-NonCommercial-ShareAlike 3.0

Dependencies:
    pip install numpy matplotlib

Assumes the following functions are importable from your exercise/solution module:
    prediction(mu, sigma, transition)        -> (mu, sigma)
    correction(mu, sigma, obs, id_to_state, state_to_id) -> (mu, sigma, id_to_state, state_to_id)
    addNewLandmarks(mu, sigma, obs, id_to_state, state_to_id) -> (mu, sigma, id_to_state, state_to_id)

And from your visualization module:
    plot_state(landmarks, mu, sigma, observations_t, trajectory)
"""

import time
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# --- local imports (mirror the addpath / source calls) --------------------
from tools.g2o_wrapper.load_g2o import load_g2o, Landmark   # port of loadG2o.m
from solution.ekf_functions import (             # swap for solution.ekf_functions if needed
    prediction,
    correction,
    add_new_landmarks,
)
from tools.visualization.visualization import plot_state       # port of plotState


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join(os.path.dirname(__file__), "../datasets/dataset_point.g2o")
ID_MAP_SIZE  = 10_000          # buffer size for id <-> state index mappings
PAUSE_SEC    = 0.1             # delay between frames (replaces pause(.1))


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def main() -> None:
    plt.close("all")

    # Load dataset (landmarks slot is intentionally ignored — no prior map)
    _, poses, transitions, observations = load_g2o(DATASET_PATH)

    # Initial pose at the origin
    mu = np.array([0.0,   # x
                   0.0,   # y
                   0.0])  # theta (yaw)
    print(f"initial pose: [{mu[0]:.6f}, {mu[1]:.6f}, {mu[2]:.6f}]")

    # Initial covariance: high uncertainty
    sigma = np.eye(3)

    # Bookkeeping: bidirectional mapping between landmark IDs and state indices
    # Uninitialised slots hold -1 (same convention as the Octave version)
    id_to_state_map = np.full(ID_MAP_SIZE, -1, dtype=int)
    state_to_id_map = np.full(ID_MAP_SIZE, -1, dtype=int)

    # Visualisation setup
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("ekf_slam")
    plt.ion()
    trajectory = [mu[:2].copy()]   # list of [x, y] snapshots

    # ---------------------------------------------------------------------------
    # Main simulation loop
    # ---------------------------------------------------------------------------
    for t, (transition, observations_t) in enumerate(zip(transitions, observations)):

        # EKF predict step
        mu, sigma = prediction(mu, sigma, transition)

        # EKF correct step
        mu, sigma, id_to_state_map, state_to_id_map = correction(
            mu, sigma, observations_t, id_to_state_map, state_to_id_map
        )

        # Add newly seen landmarks into the state vector
        mu, sigma, id_to_state_map, state_to_id_map = add_new_landmarks(
            mu, sigma, observations_t, id_to_state_map, state_to_id_map
        )

        # --- Visualisation ---------------------------------------------------
        n_landmarks = (len(mu) - 3) // 2
        if n_landmarks > 0:
            # Rebuild landmark list from the current state vector
            lm_list = [
                Landmark(
                    id=int(state_to_id_map[i]),
                    position=np.array([mu[3 + 2*i], mu[3 + 2*i + 1]])
                )
                for i in range(n_landmarks)
            ]

            print(
                f"current pose: [{mu[0]:.6f}, {mu[1]:.6f}, {mu[2]:.6f}], "
                f"map size (landmarks): {n_landmarks}"
            )

            trajectory.append(mu[:2].copy())
            traj_array = np.array(trajectory)

            plot_state(ax, lm_list, mu, sigma, observations_t, traj_array)
            plt.pause(PAUSE_SEC)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
