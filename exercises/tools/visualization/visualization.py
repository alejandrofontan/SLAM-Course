"""
visualization.py

Python port of the visualization toolkit:
  circleAsPolygon.m, clearObservations.m, drawArrow.m, drawBelief.m,
  drawCircle.m, drawLabels.m, drawLandmarks.m, drawLandmarks_slam.m,
  drawLine.m, drawMap.m, drawObservations.m, drawObservationsBayes.m,
  drawParticles.m, drawPolygon.m, drawRect.m, drawRectangle.m,
  drawRobot.m, drawShape.m, drawTrajectoryXY.m, fillPolygon.m,
  plotcov2d.m, plotState.m, plotStateEKFSLAM.m, plotStatePF.m,
  rectAsPolygon.m

Original source: Probabilistic Robotics course, Sapienza University of Rome
License: CC Attribution-NonCommercial-ShareAlike 3.0

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow
from typing import Optional


# ---------------------------------------------------------------------------
# circleAsPolygon
# ---------------------------------------------------------------------------

def circle_as_polygon(circle: np.ndarray, n: int = 64) -> np.ndarray:
    """
    Convert a circle [x0, y0, r] into an (N+1, 2) closed polygon array.

    Parameters
    ----------
    circle : array-like [x0, y0, r]
    n      : number of segments (default 64)

    Returns
    -------
    points : np.ndarray of shape (N+1, 2)
    """
    x0, y0, r = circle[0], circle[1], circle[2]
    t = np.linspace(0, 2 * np.pi, n + 1)
    x = x0 + r * np.cos(t)
    y = y0 + r * np.sin(t)
    return np.column_stack([x, y])


# ---------------------------------------------------------------------------
# drawCircle
# ---------------------------------------------------------------------------

def draw_circle(
    ax: plt.Axes,
    x0: float,
    y0: float,
    r: float,
    n: int = 72,
    **kwargs,
) -> plt.Line2D:
    """
    Draw a circle on *ax* centred at (x0, y0) with radius r.

    Any extra keyword arguments are forwarded to ax.plot().
    """
    t  = np.linspace(0, 2 * np.pi, n + 1)
    xt = x0 + r * np.cos(t)
    yt = y0 + r * np.sin(t)
    (h,) = ax.plot(xt, yt, **kwargs)
    return h


# ---------------------------------------------------------------------------
# drawPolygon
# ---------------------------------------------------------------------------

def draw_polygon(
    ax: plt.Axes,
    coords: np.ndarray,
    **kwargs,
) -> plt.Line2D:
    """
    Draw a closed polygon from an (N, 2) coordinate array.

    Keyword arguments are forwarded to ax.plot().
    """
    kwargs.setdefault("color", "b")
    # close the polygon
    px = np.append(coords[:, 0], coords[0, 0])
    py = np.append(coords[:, 1], coords[0, 1])
    (h,) = ax.plot(px, py, **kwargs)
    return h


# ---------------------------------------------------------------------------
# drawRect  (axis-aligned or rotated)
# ---------------------------------------------------------------------------

def draw_rect(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    theta_deg: float = 0.0,
    **kwargs,
) -> plt.Line2D:
    """
    Draw a rectangle on *ax*.

    Parameters
    ----------
    x, y      : bottom-left corner
    w, h      : width and height
    theta_deg : rotation angle in degrees (default 0)
    """
    theta = np.deg2rad(theta_deg)
    tx = np.array([
        x,
        x + w * np.cos(theta),
        x + w * np.cos(theta) - h * np.sin(theta),
        x               - h * np.sin(theta),
        x,
    ])
    ty = np.array([
        y,
        y + w * np.sin(theta),
        y + w * np.sin(theta) + h * np.cos(theta),
        y               + h * np.cos(theta),
        y,
    ])
    (line,) = ax.plot(tx, ty, **kwargs)
    return line


# ---------------------------------------------------------------------------
# drawRectangle  (map cell helper)
# ---------------------------------------------------------------------------

def draw_rectangle_in_map(
    ax: plt.Axes,
    map_: np.ndarray,
    row: int,
    col: int,
    color: str,
) -> None:
    """Draw a single 1×1 coloured cell at (row, col) in map coordinates."""
    map_rows = map_.shape[0]
    rect = mpatches.Rectangle(
        (col - 1, map_rows - row), 1, 1,
        facecolor=color, edgecolor="none",
    )
    ax.add_patch(rect)


# ---------------------------------------------------------------------------
# drawMap
# ---------------------------------------------------------------------------

def draw_map(ax: plt.Axes, map_: np.ndarray) -> None:
    """
    Draw an occupancy grid map on *ax*.

    Parameters
    ----------
    map_ : 2-D numpy array where 1 = occupied (black), 0 = free (white)
    """
    map_rows, map_cols = map_.shape
    for row in range(1, map_rows + 1):
        for col in range(1, map_cols + 1):
            color = "black" if map_[row - 1, col - 1] == 1 else "white"
            rect = mpatches.Rectangle(
                (col - 1, map_rows - row), 1, 1,
                facecolor=color, edgecolor="grey", linewidth=0.3,
            )
            ax.add_patch(rect)
    ax.set_xlim(0, map_cols)
    ax.set_ylim(0, map_rows)
    ax.set_aspect("equal")


# ---------------------------------------------------------------------------
# drawBelief
# ---------------------------------------------------------------------------

def draw_belief(ax: plt.Axes, state_belief: np.ndarray, map_: np.ndarray) -> None:
    """
    Draw a belief heatmap (grayscale) over the map.

    Parameters
    ----------
    state_belief : 2-D array, values in [0, 1]  (1 = fully confident)
    map_         : occupancy grid (used only for axis sizing)
    """
    # invert: 0 = black (100 % confidence), 1 = white (0 % confidence)
    plotted = np.flipud(1.0 - state_belief)
    ax.imshow(
        plotted,
        cmap="gray",
        vmin=0, vmax=1,
        extent=[0, map_.shape[1], 0, map_.shape[0]],
        origin="upper",
        aspect="equal",
    )


# ---------------------------------------------------------------------------
# drawLabels
# ---------------------------------------------------------------------------

def draw_labels(
    ax: plt.Axes,
    px,
    py,
    labels,
    fmt: str = "%.2f",
) -> list:
    """
    Draw text labels at positions (px, py).

    Parameters
    ----------
    px, py  : scalar or array-like of coordinates
    labels  : scalar, array of numbers, or list of strings
    fmt     : format string used when labels are numeric (default '%.2f')
    """
    px = np.atleast_1d(px)
    py = np.atleast_1d(py)

    if np.isscalar(labels) or (hasattr(labels, '__len__') and not isinstance(labels, str)):
        labels = np.atleast_1d(labels)
        label_strs = [fmt % v for v in labels]
    else:
        label_strs = [str(labels)] * len(px)

    handles = []
    for x, y, s in zip(px, py, label_strs):
        handles.append(ax.text(x + 0.15, y, s))
    return handles


# ---------------------------------------------------------------------------
# drawArrow
# ---------------------------------------------------------------------------

def draw_arrow(
    ax: plt.Axes,
    x1: float, y1: float,
    x2: float, y2: float,
    length: float = 10.0,
    width: float = 5.0,
    color: str = "b",
) -> FancyArrow:
    """
    Draw an arrow from (x1, y1) to (x2, y2) on *ax*.
    """
    dx, dy = x2 - x1, y2 - y1
    h = ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
    )
    return h


# ---------------------------------------------------------------------------
# drawLine  (infinite line clipped to axes)
# ---------------------------------------------------------------------------

def draw_line(
    ax: plt.Axes,
    line: np.ndarray,
    **kwargs,
) -> Optional[plt.Line2D]:
    """
    Draw an infinite line [x0, y0, dx, dy] clipped to the current axes limits.

    Returns the Line2D handle, or None if the line lies outside the axes.
    """
    kwargs.setdefault("color", "b")
    x0, y0, vx, vy = line[0], line[1], line[2], line[3]
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    pts = []
    # intersect with x = xlim[0] and x = xlim[1]
    if abs(vx) > 1e-12:
        for xb in xlim:
            t = (xb - x0) / vx
            yb = y0 + t * vy
            if ylim[0] <= yb <= ylim[1]:
                pts.append((xb, yb))
    # intersect with y = ylim[0] and y = ylim[1]
    if abs(vy) > 1e-12:
        for yb in ylim:
            t = (yb - y0) / vy
            xb = x0 + t * vx
            if xlim[0] <= xb <= xlim[1]:
                pts.append((xb, yb))

    if len(pts) < 2:
        return None

    (h,) = ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], **kwargs)
    return h


# ---------------------------------------------------------------------------
# drawLandmarks
# ---------------------------------------------------------------------------

def draw_landmarks(ax: plt.Axes, landmarks: list, color: str = "r") -> None:
    """
    Draw landmark circles and ID labels on *ax*.

    Parameters
    ----------
    landmarks : list of Landmark dataclass objects (id, position)
    color     : face colour of the landmark circles
    """
    radius = 0.1
    for lm in landmarks:
        x, y = lm.position[0], lm.position[1]
        draw_circle(ax, x, y, radius, color=color)
        draw_labels(ax, x, y, lm.id, fmt="%d")


# ---------------------------------------------------------------------------
# drawLandmarks_slam  (draws from raw mean vector + state_to_id_map)
# ---------------------------------------------------------------------------

def draw_landmarks_slam(
    ax: plt.Axes,
    mean: np.ndarray,
    state_to_id_map: np.ndarray,
) -> None:
    """
    Draw landmarks from the EKF state vector.

    Parameters
    ----------
    mean            : full state vector (2*N,) containing [lx1, ly1, lx2, ly2, ...]
    state_to_id_map : 1-based index -> landmark id
    """
    n = len(mean) // 2
    radius = 0.1
    for i in range(n):
        land_id = state_to_id_map[i + 1]   # 1-based
        lx = mean[2 * i]
        ly = mean[2 * i + 1]
        draw_circle(ax, lx, ly, radius, color="r")
        draw_labels(ax, lx, ly, int(land_id), fmt="%d")


# ---------------------------------------------------------------------------
# drawObservations
# ---------------------------------------------------------------------------

def draw_observations(
    ax: plt.Axes,
    pose: np.ndarray,
    observations,
) -> None:
    """
    Draw observations from the current robot pose.

    Supports both point observations (x, y in robot frame) and
    bearing-only observations.

    Parameters
    ----------
    pose         : [x, y, theta]
    observations : Observation dataclass
    """
    obs_array = np.atleast_2d(observations.observation)
    n = obs_array.shape[0]
    if n == 0:
        return

    # Detect bearing-only mode: observation has 1 column
    bearing_mode = obs_array.shape[1] == 1

    for i in range(n):
        if bearing_mode:
            bearing = obs_array[i, 0]
            l = 1.5
            incr_x = np.cos(bearing)
            incr_y = np.sin(bearing)
            ray_x = pose[0] + incr_x * np.cos(pose[2]) - incr_y * np.sin(pose[2])
            ray_y = pose[1] + incr_y * np.cos(pose[2]) + incr_x * np.sin(pose[2])
            ax.plot([pose[0], ray_x], [pose[1], ray_y], "b", linewidth=1.5)
        else:
            # point observation: transform from robot to world frame
            c, s  = np.cos(pose[2]), np.sin(pose[2])
            R     = np.array([[c, -s], [s, c]])
            lm_robot = obs_array[i]
            lm_world = pose[:2] + R @ lm_robot
            draw_circle(ax, lm_world[0], lm_world[1], 0.2, color="b")


# ---------------------------------------------------------------------------
# drawObservationsBayes  (grid-world directional observations)
# ---------------------------------------------------------------------------

def draw_observations_bayes(
    ax: plt.Axes,
    map_: np.ndarray,
    observations: np.ndarray,
    row: int,
    col: int,
) -> None:
    """
    Highlight observed directions (UP/DOWN/LEFT/RIGHT) on the map.

    Parameters
    ----------
    observations : length-4 boolean/int array [UP, DOWN, LEFT, RIGHT]
    row, col     : current cell position (1-based)
    """
    map_rows = map_.shape[0]

    dirs = [
        (0, (col - 1, map_rows - row + 1,     1, 0.5)),   # UP
        (1, (col - 1, map_rows - row - 0.5,   1, 0.5)),   # DOWN
        (2, (col - 1.5, map_rows - row,      0.5, 1  )),   # LEFT
        (3, (col,      map_rows - row,        0.5, 1  )),   # RIGHT
    ]
    for idx, (bx, by, bw, bh) in dirs:
        if observations[idx]:
            rect = mpatches.Rectangle(
                (bx, by), bw, bh,
                facecolor="blue", edgecolor="none",
            )
            ax.add_patch(rect)


# ---------------------------------------------------------------------------
# clearObservations  (redraw cells to clear observation highlights)
# ---------------------------------------------------------------------------

def clear_observations(
    ax: plt.Axes,
    observations: np.ndarray,
    row: int,
    col: int,
    map_: np.ndarray,
) -> None:
    """
    Redraw adjacent cells to erase directional observation highlights.

    Parameters
    ----------
    observations : length-4 boolean/int array [UP, DOWN, LEFT, RIGHT]
    row, col     : current cell position (1-based)
    map_         : occupancy grid
    """
    map_rows = map_.shape[0]

    def _cell_color(r, c):
        return "black" if map_[r - 1, c - 1] else "white"

    checks = [
        (0, row - 1, col, (col - 1, map_rows - row + 1, 1, 1)),   # UP
        (1, row + 1, col, (col - 1, map_rows - row - 1, 1, 1)),   # DOWN
        (2, row, col - 1, (col - 2, map_rows - row,     1, 1)),   # LEFT
        (3, row, col + 1, (col,     map_rows - row,     1, 1)),   # RIGHT
    ]
    for idx, nr, nc, (bx, by, bw, bh) in checks:
        if observations[idx]:
            color = _cell_color(nr, nc)
            rect = mpatches.Rectangle(
                (bx, by), bw, bh,
                facecolor=color, edgecolor="none",
            )
            ax.add_patch(rect)


# ---------------------------------------------------------------------------
# drawParticles
# ---------------------------------------------------------------------------

def draw_particles(
    ax: plt.Axes,
    samples: np.ndarray,
    weights: np.ndarray,
    best_particle: int,
    gt_pose,
) -> None:
    """
    Draw particle filter samples on *ax*.

    Parameters
    ----------
    samples       : (3, N) array of particle poses [x; y; theta]
    weights       : (N,) weight array (unused visually but kept for API parity)
    best_particle : 0-based index of the best particle (-1 to skip)
    gt_pose       : ground-truth pose object with .x and .y attributes
    """
    ax.plot(samples[0, :], samples[1, :], "b.")
    ax.plot(gt_pose.x, gt_pose.y, "md", markersize=12, markerfacecolor="m")

    if best_particle >= 0:
        bp = samples[:, best_particle]
        draw_circle(ax, bp[0], bp[1], 0.5, color="g")
        # draw_robot would go here — port separately when available


# ---------------------------------------------------------------------------
# rectAsPolygon
# ---------------------------------------------------------------------------

def rect_as_polygon(rect: np.ndarray) -> np.ndarray:
    """
    Convert a rectangle [x, y, w, h] or [x, y, w, h, theta] into a (4, 2)
    vertex array (centred rectangle, theta in radians).

    Mirrors rectAsPolygon.m exactly.
    """
    x, y   = rect[0], rect[1]
    w, h   = rect[2] / 2, rect[3] / 2   # half-extents
    theta  = rect[4] if len(rect) > 4 else 0.0

    v  = np.array([np.cos(theta), np.sin(theta)])
    M  = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]) * np.array([w, h])

    tx = x + M @ v
    # reversed-row trick from the original to get the perpendicular direction
    ty = y + M[::-1, ::-1] @ v
    return np.column_stack([tx, ty])


# ---------------------------------------------------------------------------
# fillPolygon
# ---------------------------------------------------------------------------

def fill_polygon(
    ax: plt.Axes,
    coords: np.ndarray,
    color: str = "b",
) -> plt.Polygon:
    """
    Fill a closed polygon on *ax*.

    Handles NaN-separated multi-polygon arrays by splitting them and filling
    each sub-polygon individually.

    Parameters
    ----------
    coords : (N, 2) array; NaN rows separate disjoint sub-polygons
    color  : fill colour (default 'b')
    """
    nan_rows = np.where(np.isnan(coords[:, 0]))[0]
    if len(nan_rows) == 0:
        return ax.fill(coords[:, 0], coords[:, 1], color=color)[0]

    # split on NaN boundaries and fill each piece
    starts = np.concatenate([[0], nan_rows + 1])
    ends   = np.concatenate([nan_rows, [len(coords)]])
    handles = []
    for s, e in zip(starts, ends):
        piece = coords[s:e]
        if len(piece) > 0:
            handles.append(ax.fill(piece[:, 0], piece[:, 1], color=color)[0])
    return handles[0] if handles else None


# ---------------------------------------------------------------------------
# drawShape
# ---------------------------------------------------------------------------

def draw_shape(
    ax: plt.Axes,
    shape_type: str,
    param: np.ndarray,
    option: str = "draw",
    color: str = "b",
) -> None:
    """
    Draw or fill a named shape on *ax*.

    Parameters
    ----------
    shape_type : 'circle', 'rect', or 'polygon'
    param      : shape parameters
                   circle  -> [x0, y0, r]
                   rect    -> [x0, y0, w, h] or [x0, y0, w, h, theta_rad]
                   polygon -> (N, 2) vertex array
    option     : 'draw' (outline) or 'fill' (filled)
    color      : colour string
    """
    if shape_type == "circle":
        poly = circle_as_polygon(param, n=128)
    elif shape_type == "rect":
        poly = rect_as_polygon(param)
    elif shape_type == "polygon":
        poly = param
    else:
        raise ValueError(f"Unknown shape type: '{shape_type}'")

    if option == "fill":
        fill_polygon(ax, poly, color=color)
    else:
        draw_polygon(ax, poly, color=color)


# ---------------------------------------------------------------------------
# drawRobot
# ---------------------------------------------------------------------------

def draw_robot(
    ax: plt.Axes,
    pose: np.ndarray,
    covariance: Optional[np.ndarray] = None,
) -> None:
    """
    Draw the robot as a small filled green square at *pose* = [x, y, theta].

    Parameters
    ----------
    pose       : [x, y, theta]
    covariance : unused (kept for API parity with drawRobot.m)
    """
    dim = 0.25
    # rect_as_polygon expects a centred rectangle: [cx, cy, w, h, theta]
    param = np.array([pose[0], pose[1], dim, dim, pose[2]])
    draw_shape(ax, "rect", param, option="fill", color="g")


# ---------------------------------------------------------------------------
# drawTrajectoryXY
# ---------------------------------------------------------------------------

def draw_trajectory_xy(
    ax: plt.Axes,
    trajectory: np.ndarray,
) -> None:
    """
    Draw the robot trajectory as a thick black polyline.

    Parameters
    ----------
    trajectory : (T, 2) array of [x, y] positions
    """
    if len(trajectory) < 2:
        return
    traj = np.asarray(trajectory)
    ax.plot(traj[:, 0], traj[:, 1], "k-", linewidth=2)


# ---------------------------------------------------------------------------
# plotcov2d
# ---------------------------------------------------------------------------

def plot_cov_2d(
    ax: plt.Axes,
    center_x: float,
    center_y: float,
    cov: np.ndarray,
    color: str = "k",
    sigma_factor: float = 1.0,
) -> None:
    """
    Plot a 2-D covariance ellipse plus its principal axes.

    Parameters
    ----------
    center_x, center_y : ellipse centre
    cov                : 2×2 covariance matrix
    color              : line colour
    sigma_factor       : scale factor (1 = 1-sigma, 2 = 2-sigma, …)
    """
    cov2 = np.array([[cov[0, 0], cov[0, 1]], [cov[1, 0], cov[1, 1]]])
    eigenvalues, eigenvectors = np.linalg.eig(cov2)

    lam1, lam2 = eigenvalues[0], eigenvalues[1]
    if lam1 < 0 or lam2 < 0:
        print(f"[plot_cov_2d] Non-positive definite matrix:\n{cov2}")
        return

    v1 = eigenvectors[:, 0]
    v2 = eigenvectors[:, 1]
    theta = np.arctan2(v1[1], v1[0])

    a = sigma_factor * np.sqrt(lam1)
    b = sigma_factor * np.sqrt(lam2)

    # ellipse outline
    n  = 500
    ang = np.linspace(0, 2 * np.pi, n + 1)
    R   = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    pts = np.array([center_x, center_y])[:, None] + R @ np.array([np.cos(ang) * a,
                                                                    np.sin(ang) * b])
    ax.plot(pts[0], pts[1], color=color)

    # principal axes
    mu = np.array([center_x, center_y])
    ax.plot(*zip(mu - a * v1, mu + a * v1), color=color)
    ax.plot(*zip(mu - b * v2, mu + b * v2), color=color)


# ---------------------------------------------------------------------------
# plotState  (general EKF state — landmarks + robot)
# ---------------------------------------------------------------------------

def plot_state(
    ax: plt.Axes,
    landmarks: list,
    mu: np.ndarray,
    sigma: np.ndarray,
    observations_t,
    trajectory: Optional[np.ndarray] = None,
    traj_gt_array: Optional[np.ndarray] = None,
) -> None:
    """
    Full EKF state visualisation (port of plotState.m).

    Draws: trajectory, observations, landmarks with covariance ellipses,
    robot pose with covariance ellipse, and fixes axis to [-11, 11].

    Parameters
    ----------
    landmarks     : list of Landmark dataclass objects
    mu            : full state mean [x, y, theta, lx1, ly1, ...]
    sigma         : full state covariance
    observations_t: current Observation
    trajectory    : (T, 2) position history, or None for init step
    """
    ax.cla()
    ax.set_aspect("equal")
    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)

    robot_pose = mu[0:3]
    map_size   = (len(mu) - 3) // 2

    if trajectory is None:
        # init step: just robot + landmarks, no covariance
        draw_landmarks(ax, landmarks)
        draw_robot(ax, robot_pose)
    else:
        draw_trajectory_xy(ax, trajectory)
        if traj_gt_array is not None:
            draw_trajectory_xy(ax, traj_gt_array)
        draw_observations(ax, robot_pose, observations_t)
        draw_landmarks(ax, landmarks)
        
        # robot covariance
        plot_cov_2d(ax, robot_pose[0], robot_pose[1], sigma[0:2, 0:2], color="k", sigma_factor=1)

        # landmark covariances
        for i in range(map_size):
            idx = 3 + 2 * i          # 0-based start of landmark block
            plot_cov_2d(ax, mu[idx], mu[idx + 1],
                        sigma[idx:idx+2, idx:idx+2], color="r", sigma_factor=1)

        draw_robot(ax, robot_pose, sigma)

    plt.draw()


# ---------------------------------------------------------------------------
# plotStateEKFSLAM
# ---------------------------------------------------------------------------

def plot_state_ekf_slam(
    ax: plt.Axes,
    mean: np.ndarray,
    covariance: np.ndarray,
    observations,
    state_to_id_map: np.ndarray,
    trajectory: np.ndarray,
) -> None:
    """
    EKF-SLAM specific state visualisation (port of plotStateEKFSLAM.m).

    Parameters
    ----------
    mean           : full state vector [x, y, theta, lx1, ly1, ...]
    covariance     : full state covariance
    observations   : current Observation
    state_to_id_map: 1-based index -> landmark id
    trajectory     : (T, 2) robot position history
    """
    ax.cla()
    ax.set_aspect("equal")

    robot_pose = mean[0:3]
    map_size   = (len(mean) - 3) // 2

    draw_trajectory_xy(ax, trajectory)

    if map_size > 0:
        draw_landmarks_slam(ax, mean[3:], state_to_id_map)

    draw_robot(ax, robot_pose, covariance)

    if observations is not None and len(np.atleast_1d(observations.observation)) > 0:
        draw_observations(ax, robot_pose, observations)

    # robot covariance ellipse
    plot_cov_2d(ax, robot_pose[0], robot_pose[1], covariance[0:2, 0:2], color="k", sigma_factor=1)

    # landmark covariance ellipses
    for i in range(map_size):
        idx = 3 + 2 * i
        plot_cov_2d(ax, mean[idx], mean[idx + 1],
                    covariance[idx:idx+2, idx:idx+2], color="r", sigma_factor=1)

    plt.draw()


# ---------------------------------------------------------------------------
# plotStatePF
# ---------------------------------------------------------------------------

def plot_state_pf(
    ax: plt.Axes,
    samples: np.ndarray,
    weights: np.ndarray,
    landmarks: list,
    best_particle: int,
    gt_pose,
) -> None:
    """
    Particle filter state visualisation (port of plotStatePF.m).

    Parameters
    ----------
    samples       : (3, N) particle array [x; y; theta]
    weights       : (N,) weight array
    landmarks     : list of Landmark dataclass objects
    best_particle : 0-based index of best particle (-1 to skip)
    gt_pose       : ground-truth pose with .x and .y
    """
    ax.cla()
    ax.set_aspect("equal")

    draw_landmarks(ax, landmarks)
    draw_particles(ax, samples, weights, best_particle, gt_pose)

    plt.draw()
