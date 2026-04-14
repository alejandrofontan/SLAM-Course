import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tools.g2o_wrapper.load_g2o import Observation


# ---------------------------------------------------------------------------
# Transition model
# ---------------------------------------------------------------------------

def transition_model(pose: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Apply control input u = [ux, uy, utheta] to pose [x, y, theta]."""
    x, y, theta = pose
    ux, uy, utheta = u
    return np.array()


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
    mu            : state mean  (?,?)
    sigma         : state covariance  (?, ?)
    control_input : Transition with .delta = [ux, uy, utheta]

    Returns
    -------
    mu, sigma : predicted mean and covariance
    """
    dim_mu = 0 # ?
    dim_u  = 0 # ?

    u         = control_input.delta          # [ux, uy, utheta]
    mu_theta  = mu[2]

    # Jacobian G:
    # ??????????????????

    # Jacobian Gu:
    # ??????????????????

    # Propagate robot pose through the transition model
    # ????????????????????

    # Control noise covariance
    # ??????????????????

    # Predicted covariance
    # ??????????????????

    #sigma = (sigma + sigma.T) / 2
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
    mu               : state mean  (?,?)
    sigma            : state covariance  (?, ?)
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

    dim_state = 0 # ??????????????????
    mu_t      = mu[0:2]
    mu_theta  = mu[2]

    c   =  np.cos(mu_theta)
    s   =  np.sin(mu_theta)
    R   = np.array()   # rotation matrix
    Rt  = np.array()   # R transposed
    Rtp = np.array()   # d(Rt)/d(theta)

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

        # id_state = ???????????????               

        n_known += 1
        z_t.extend(meas_xy.tolist())

        # Predicted measurement
        #lm_mu        = mu[id_state: id_state + 2]
        delta_t      = 0 # ??????????????????
        meas_pred    = 0 # ??????????????????
        h_t.extend(meas_pred.tolist())

        # Jacobian row block (2 x dim_state)
        # ??????????????????
        # C_t.append(C_m)

    if n_known == 0:
        return mu, sigma, id_to_state_map, state_to_id_map

    z_t = np.array(z_t)                              # (2*n_known,)
    h_t = np.array(h_t)                              # (2*n_known,)
    C_t = np.vstack(C_t)                             # (2*n_known, dim_state)

    # noise = 0.1**2
    # sigma_z = ?????

    # Kalman gain
    # ??????????????????

    # Update
    # ??????????????????

    # sigma = (sigma + sigma.T) / 2
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
