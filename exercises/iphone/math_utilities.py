# third-party
import cv2
import numpy as np

def rvec_tvec_to_matrix(rvec, tvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T
