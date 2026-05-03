import csv
import numpy as np
from pathlib import Path

import cv2
import yaml
from scipy.spatial.transform import Rotation

def rvec_tvec_to_matrix(rvec, tvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T
