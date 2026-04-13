"""
load_g2o.py

Python port of loadG2o.m
Original source: Probabilistic Robotics course, Sapienza University of Rome
Copyright (c) 2016 Bartolomeo Della Corte, Giorgio Grisetti
License: CC Attribution-NonCommercial-ShareAlike 3.0
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Data classes (replaces the Octave struct constructors)
# ---------------------------------------------------------------------------

@dataclass
class Landmark:
    id: int
    position: np.ndarray        # [x, y]


@dataclass
class Pose:
    id: int
    x: float
    y: float
    theta: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])


@dataclass
class Transition:
    from_id: int
    to_id: int
    delta: np.ndarray           # [x, y, theta]


@dataclass
class Observation:
    pose_id: int
    landmark_id: list            # one int per observation row
    observation: np.ndarray     # bearing scalar or [x, y] point


# ---------------------------------------------------------------------------
# Helper extractors (mirror the bottom functions in the .m file)
# ---------------------------------------------------------------------------

def _extract_landmark(elements: list[str]) -> Landmark:
    return Landmark(
        id=int(elements[1]),
        position=np.array([float(elements[2]), float(elements[3])]),
    )


def _extract_pose(elements: list[str]) -> Pose:
    return Pose(
        id=int(elements[1]),
        x=float(elements[2]),
        y=float(elements[3]),
        theta=float(elements[4]),
    )


def _extract_transition(elements: list[str]) -> Transition:
    return Transition(
        from_id=int(elements[1]),
        to_id=int(elements[2]),
        delta=np.array([float(elements[3]), float(elements[4]), float(elements[5])]),
    )


def _extract_bearing(elements: list[str]) -> Observation:
    return Observation(
        pose_id=int(elements[1]),
        landmark_id=[int(elements[2])],
        observation=np.array([float(elements[3])]),
    )


def _extract_point(elements: list[str]) -> Observation:
    return Observation(
        pose_id=int(elements[1]),
        landmark_id=[int(elements[2])],
        observation=np.array([float(elements[3]), float(elements[4])]),
    )


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_g2o(filepath: str) -> tuple[
    list[Landmark],
    list[Pose],
    list[Transition],
    list[Observation],
]:
    """
    Parse a .g2o file and return (landmarks, poses, transitions, observations).

    Parameters
    ----------
    filepath : str
        Path to the .g2o file.

    Returns
    -------
    landmarks    : list[Landmark]
    poses        : list[Pose]
    transitions  : list[Transition]
    observations : list[Observation]
        Each Observation groups all measurements taken from the same pose.
        observation.observation is a 1-D or 2-D numpy array depending on type
        (bearing-only → shape (N,), point → shape (N, 2)).
    """
    landmarks:    list[Landmark]    = []
    poses:        list[Pose]        = []
    transitions:  list[Transition]  = []
    observations: list[Observation] = []

    # counters (mirrors the debug vars in the .m file)
    counts = dict(
        vert_xy=0, vert_se2=0, robotlaser=0,
        edge_se2=0, edge_se2_xy=0, edge_bearing_se2_xy=0,
    )

    curr_id = -1  # tracks the last seen pose_id for observation grouping

    with open(filepath, "r") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            elements = line.split()
            tag = elements[0]

            # ---- vertices ------------------------------------------------
            if tag == "VERTEX_XY":
                landmarks.append(_extract_landmark(elements))
                counts["vert_xy"] += 1

            elif tag == "VERTEX_SE2":
                poses.append(_extract_pose(elements))
                counts["vert_se2"] += 1

            # ---- laser (not implemented in original either) ---------------
            elif tag == "ROBOTLASER1":
                counts["robotlaser"] += 1  # TODO

            # ---- odometry edge -------------------------------------------
            elif tag == "EDGE_SE2":
                transitions.append(_extract_transition(elements))
                counts["edge_se2"] += 1

            # ---- bearing-only observation ---------------------------------
            elif tag == "EDGE_BEARING_SE2_XY":
                obs = _extract_bearing(elements)
                if obs.pose_id == curr_id:
                    observations[-1].observation = np.append(
                        observations[-1].observation, obs.observation
                    )
                    observations[-1].landmark_id.append(obs.landmark_id[0])
                else:
                    observations.append(obs)
                    curr_id = obs.pose_id
                    counts["edge_bearing_se2_xy"] += 1

            # ---- 2-D point observation ------------------------------------
            elif tag == "EDGE_SE2_XY":
                obs = _extract_point(elements)
                if obs.pose_id == curr_id:
                    observations[-1].observation = np.vstack(
                        [observations[-1].observation, obs.observation]
                    )
                    observations[-1].landmark_id.append(obs.landmark_id[0])
                else:
                    observations.append(obs)
                    curr_id = obs.pose_id
                    counts["edge_se2_xy"] += 1

            else:
                print(f"[load_g2o] Unknown tag: '{tag}' — skipping line.")

    print(
        f"[G2oWrapper] loading file...\n"
        f"  #landmarks:               {counts['vert_xy']}\n"
        f"  #poses:                   {counts['vert_se2']}\n"
        f"  #transitions:             {counts['edge_se2']}\n"
        f"  #observations (bearing):  {counts['edge_bearing_se2_xy']}\n"
        f"  #observations (point):    {counts['edge_se2_xy']}\n"
        f"  #laser scans:             {counts['robotlaser']}"
    )

    return landmarks, poses, transitions, observations
