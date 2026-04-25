"""
head_pose.py
------------
Tracks nose-tip landmark position per frame and computes mean
frame-to-frame displacement (ΔH) as per Eq. (4) in the project paper.
"""

import numpy as np


NOSE_TIP_IDX = 1   # MediaPipe Face Mesh nose tip landmark


def get_nose_tip(face_landmarks, img_w: int, img_h: int) -> np.ndarray:
    """Return nose-tip (x, y) in pixel space."""
    lm = face_landmarks.landmark[NOSE_TIP_IDX]
    return np.array([lm.x * img_w, lm.y * img_h])


class HeadPoseTracker:
    """
    Computes mean frame-to-frame nose-tip displacement over a window.

    ΔH = (1 / N-1) * Σ ||H_i+1 - H_i||

    High ΔH → restlessness → cognitive strain signal.
    """

    def __init__(self):
        self.positions = []   # nose-tip (x, y) per frame

    def update(self, face_landmarks, img_w: int, img_h: int):
        """Record nose-tip position for the current frame."""
        pos = get_nose_tip(face_landmarks, img_w, img_h)
        self.positions.append(pos)

    def compute_variation(self) -> float:
        """
        Returns mean frame-to-frame displacement (pixels).
        Returns 0.0 if fewer than 2 frames recorded.
        """
        if len(self.positions) < 2:
            return 0.0
        pts  = np.array(self.positions)
        diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return float(diffs.mean())

    def reset(self):
        self.positions = []