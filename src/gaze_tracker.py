"""
gaze_tracker.py
---------------
Tracks iris centre position per frame and computes gaze stability
(σ_gaze) over a sliding window as per Eq. (3) in the project paper.
"""

import numpy as np


# MediaPipe iris landmark indices (Face Mesh with refine_landmarks=True)
LEFT_IRIS_IDX  = [474, 475, 476, 477]
RIGHT_IRIS_IDX = [469, 470, 471, 472]


def get_iris_centre(face_landmarks, indices: list, img_w: int, img_h: int) -> np.ndarray:
    """Return mean (x, y) of iris landmarks in pixel space."""
    coords = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        coords.append([lm.x * img_w, lm.y * img_h])
    return np.mean(coords, axis=0)


class GazeTracker:
    """
    Accumulates iris centre positions over a window and computes
    gaze stability as the standard deviation of position.

    Low σ_gaze → stable fixation → high cognitive engagement.
    High σ_gaze → wandering gaze → distraction or low load.
    """

    def __init__(self):
        self.positions = []   # list of (x, y) per frame

    def update(self, face_landmarks, img_w: int, img_h: int):
        """Record iris centre for the current frame."""
        left_centre  = get_iris_centre(face_landmarks, LEFT_IRIS_IDX,  img_w, img_h)
        right_centre = get_iris_centre(face_landmarks, RIGHT_IRIS_IDX, img_w, img_h)
        centre = (left_centre + right_centre) / 2.0
        self.positions.append(centre)

    def compute_stability(self) -> float:
        """
        σ_gaze = sqrt( (1/N) * Σ [(xi - x̄)² + (yi - ȳ)²] )

        Returns:
            Scalar gaze stability value. Lower = more stable = more focused.
        """
        if len(self.positions) < 2:
            return 0.0
        pts  = np.array(self.positions)
        mean = pts.mean(axis=0)
        sigma = np.sqrt(np.mean(np.sum((pts - mean) ** 2, axis=1)))
        return float(sigma)

    def reset(self):
        self.positions = []