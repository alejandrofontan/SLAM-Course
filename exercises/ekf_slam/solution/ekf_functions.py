"""
ekf_functions.py

Python port of prediction.m, correction.m, addNewLandmarks.m
Original source: Probabilistic Robotics course, Sapienza University of Rome
Copyright (c) 2016 Bartolomeo Della Corte, Giorgio Grisetti
License: CC Attribution-NonCommercial-ShareAlike 3.0
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tools.g2o_wrapper.load_g2o import Observation


# ---------------------------------------------------------------------------
# Transition model  (referenced by prediction, assumed defined elsewhere in
# the original repo — included here for completeness)
# ---------------------------------------------------------------------------

def transition_model(pose: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Apply control input u = [ux, uy, utheta] to pose [x, y, theta]."""
    x, y, theta = pose
    ux, uy, utheta = u
    return np.array([
        x + np.cos(theta)*ux - np.sin(theta)*uy,
        y + np.sin(theta)*ux + np.cos(theta)*uy,
        theta + utheta,
    ])


# ---------------------------------------------------------------------------
# Prediction step
# ---------------------------------------------------------------------------

def prediction(
    mu: np.ndarray,
    sigma: np.ndarray,
    control_input,              # Transition dataclass: .delta = [ux, uy, utheta]
) -> tuple[np.ndarray, np.ndarray]:
    """
    EKF prediction step.

    Parameters
    ----------
    mu            : state mean  (3 + 2*N,)
    sigma         : state covariance  (3+2N, 3+2N)
    control_input : Transition with .delta = [ux, uy, utheta]

    Returns
    -------
    mu, sigma : predicted mean and covariance
    """
    dim_mu = len(mu)
    dim_u  = 3

    u         = control_input.delta          # [ux, uy, utheta]
    mu_theta  = mu[2]

    # Jacobian A: identity everywhere except the 3x3 robot block
    A = np.eye(dim_mu)
    A[0:3, 0:3] = np.array([
        [1, 0, -np.sin(mu_theta)*u[0] - np.cos(mu_theta)*u[1]],
        [0, 1,  np.cos(mu_theta)*u[0] - np.sin(mu_theta)*u[1]],
        [0, 0,  1],
    ])

    # Jacobian B: maps control inputs to state dimensions
    B = np.zeros((dim_mu, dim_u))
    B[0:3, :] = np.array([
        [np.cos(mu_theta), -np.sin(mu_theta), 0],
        [np.sin(mu_theta),  np.cos(mu_theta), 0],
        [0,                 0,                1],
    ])

    # Propagate robot pose through the transition model
    mu = mu.copy()
    mu[0:3] = transition_model(mu[0:3], u)

    # Control noise covariance
    sigma_u = np.diag([0.02**2, 0.02**2, 0.01**2])

    # Predicted covariance
    sigma = A @ sigma @ A.T + B @ sigma_u @ B.T

    sigma = (sigma + sigma.T) / 2
    return mu, sigma


# ---------------------------------------------------------------------------
# Correction step
# ---------------------------------------------------------------------------

def correction(
    mu: np.ndarray,
    sigma: np.ndarray,
    observations: Observation,
    id_to_state_map: np.ndarray,
    state_to_id_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    EKF correction step for all re-observed landmarks.

    Parameters
    ----------
    mu               : state mean  (3 + 2*N,)
    sigma            : state covariance  (3+2N, 3+2N)
    observations     : Observation for the current timestep
                       .observation is an (M, 2) array of [x, y] measurements
    id_to_state_map  : landmark_id  -> 1-based index in mu  (-1 = unseen)
    state_to_id_map  : 1-based index in mu -> landmark_id

    Returns
    -------
    mu, sigma, id_to_state_map, state_to_id_map
    """
    obs_array = np.atleast_2d(observations.observation)  # (M, 2)
    M = obs_array.shape[0]

    if M == 0:
        return mu, sigma, id_to_state_map, state_to_id_map

    dim_state = len(mu)
    mu_t      = mu[0:2]
    mu_theta  = mu[2]

    c   =  np.cos(mu_theta)
    s   =  np.sin(mu_theta)
    R   = np.array([[ c, -s], [ s,  c]])   # rotation matrix
    Rt  = np.array([[ c,  s], [-s,  c]])   # R transposed
    Rtp = np.array([[-s,  c], [-c, -s]])   # d(Rt)/d(theta)

    z_t = []   # stacked measurements
    h_t = []   # stacked predictions
    C_t = []   # stacked Jacobian rows

    n_known = 0

    for i in range(M):
        meas_xy = obs_array[i]                       # [x, y] in robot frame
        lm_id = _get_landmark_id(observations, i)

        n = id_to_state_map[lm_id]                  # 1-based state index, -1 if new

        if n == -1:                                  # new landmark — skip for now
            continue

        id_state = 3 + 2 * (n - 1)                  # 0-based slice start in mu

        n_known += 1
        z_t.extend(meas_xy.tolist())

        # Predicted measurement
        lm_mu        = mu[id_state: id_state + 2]
        delta_t      = lm_mu - mu_t
        meas_pred    = Rt @ delta_t
        h_t.extend(meas_pred.tolist())

        # Jacobian row block (2 x dim_state)
        C_m                          = np.zeros((2, dim_state))
        C_m[0:2, 0:2]                = -Rt
        C_m[0:2, 2]                  = Rtp @ delta_t
        C_m[0:2, id_state:id_state+2] = Rt
        C_t.append(C_m)

    if n_known == 0:
        return mu, sigma, id_to_state_map, state_to_id_map

    z_t = np.array(z_t)                              # (2*n_known,)
    h_t = np.array(h_t)                              # (2*n_known,)
    C_t = np.vstack(C_t)                             # (2*n_known, dim_state)

    noise = 0.1**2
    sigma_z = np.eye(2 * n_known) * noise

    # Kalman gain
    S = C_t @ sigma @ C_t.T + sigma_z
    K = sigma @ C_t.T @ np.linalg.inv(S)

    # Update
    innovation = z_t - h_t
    mu    = mu +  K @ innovation

    sigma = (np.eye(dim_state) - K @ C_t) @ sigma
    #I_KC  = np.eye(dim_state) - K @ C_t
    #sigma = I_KC @ sigma @ I_KC.T + K @ sigma_z @ K.T

    sigma = (sigma + sigma.T) / 2
    return mu, sigma, id_to_state_map, state_to_id_map


# ---------------------------------------------------------------------------
# Add new landmarks
# ---------------------------------------------------------------------------

def add_new_landmarks(
    mu: np.ndarray,
    sigma: np.ndarray,
    measurements: Observation,
    id_to_state_map: np.ndarray,
    state_to_id_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Expand the state vector and covariance with newly observed landmarks.

    Parameters
    ----------
    mu               : state mean  (3 + 2*N,)
    sigma            : state covariance  (3+2N, 3+2N)
    measurements     : Observation for the current timestep
    id_to_state_map  : landmark_id -> 1-based index in mu  (-1 = unseen)
    state_to_id_map  : 1-based index in mu -> landmark_id

    Returns
    -------
    mu, sigma, id_to_state_map, state_to_id_map
    """
    mu_t      = mu[0:2]
    mu_theta  = mu[2]
    c = np.cos(mu_theta)
    s = np.sin(mu_theta)
    R = np.array([[c, -s], [s, c]])

    obs_array = np.atleast_2d(measurements.observation)  # (M, 2)
    M = obs_array.shape[0]

    n = (len(mu) - 3) // 2   # current number of landmarks in the state

    for i in range(M):
        lm_id = _get_landmark_id(measurements, i)
        state_pos = id_to_state_map[lm_id]

        if state_pos != -1:          # already known — skip
            continue

        # Register new landmark
        n += 1
        id_to_state_map[lm_id] = n
        state_to_id_map[n]     = lm_id

        # Landmark position in world frame
        lm_in_robot = obs_array[i]                       # [x, y]
        lm_in_world = mu_t + R @ lm_in_robot             # (2,)

        # Expand mu
        mu = np.append(mu, lm_in_world)

        # Expand sigma: new rows/cols are zero, diagonal block = initial noise
        initial_noise = 2.0
        old_size = sigma.shape[0]
        new_size = old_size + 2

        new_sigma = np.zeros((new_size, new_size))
        new_sigma[0:old_size, 0:old_size] = sigma
        new_sigma[old_size:, old_size:]   = np.eye(2) * initial_noise
        sigma = new_sigma

        print(f"observed new landmark with identifier: {lm_id}")

    return mu, sigma, id_to_state_map, state_to_id_map


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _get_landmark_id(observations: Observation, i: int) -> int:
    """Return the landmark id for the i-th measurement in the grouped struct."""
    return int(observations.landmark_id[i])
