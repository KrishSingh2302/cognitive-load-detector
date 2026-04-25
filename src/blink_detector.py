"""
blink_detector.py
-----------------
Computes Eye Aspect Ratio (EAR) from MediaPipe facial landmarks
and detects blinks using a consecutive-frame threshold.

Reference:
    Soukupá & Čech, "Real-time eye blink detection using facial landmarks",
    21st Computer Vision Winter Workshop (CVWW), 2016.
"""

import numpy as np


# ── Constants ────────────────────────────────────────────────────────────────
EAR_THRESHOLD   = 0.21   # Below this → eye is closed (Soukupá & Čech, 2016)
CONSEC_FRAMES   = 3      # Consecutive frames below threshold = one blink


# ── MediaPipe landmark indices for left and right eye ────────────────────────
# Each list contains 6 indices: [P1, P2, P3, P4, P5, P6]
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]


def calc_ear(eye_landmarks: np.ndarray) -> float:
    """
    Compute the Eye Aspect Ratio from 6 eye landmark coordinates.

    Args:
        eye_landmarks: numpy array of shape (6, 2) — (x, y) per landmark
                       ordered as [P1, P2, P3, P4, P5, P6]

    Returns:
        EAR value (float). Typical open-eye range: 0.25–0.35.
        Drops below 0.21 during a blink.
    """
    # Vertical distances
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])  # ||P2 - P6||
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])  # ||P3 - P5||

    # Horizontal distance
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])  # ||P1 - P4||

    ear = (A + B) / (2.0 * C)
    return ear


def get_eye_landmarks(face_landmarks, indices: list, img_w: int, img_h: int) -> np.ndarray:
    """
    Extract pixel-space (x, y) coordinates for a given set of landmark indices.

    Args:
        face_landmarks: MediaPipe face landmark result
        indices:        List of 6 landmark indices
        img_w, img_h:   Frame dimensions for denormalization

    Returns:
        numpy array of shape (6, 2)
    """
    coords = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        coords.append([lm.x * img_w, lm.y * img_h])
    return np.array(coords, dtype=np.float64)


class BlinkDetector:
    """
    Stateful blink detector using EAR over consecutive frames.

    Usage:
        detector = BlinkDetector()
        for each frame:
            ear = detector.update(face_landmarks, img_w, img_h)
            blink_count = detector.blink_count
    """

    def __init__(self):
        self.counter     = 0      # consecutive frames below EAR threshold
        self.blink_count = 0      # total blinks detected in current window
        self.current_ear = 0.0    # EAR value from latest frame

    def update(self, face_landmarks, img_w: int, img_h: int) -> float:
        """
        Process one frame. Updates internal blink counter.

        Returns:
            Average EAR of left and right eye for this frame.
        """
        left_pts  = get_eye_landmarks(face_landmarks, LEFT_EYE_IDX,  img_w, img_h)
        right_pts = get_eye_landmarks(face_landmarks, RIGHT_EYE_IDX, img_w, img_h)

        ear_left  = calc_ear(left_pts)
        ear_right = calc_ear(right_pts)
        ear       = (ear_left + ear_right) / 2.0

        self.current_ear = ear

        if ear < EAR_THRESHOLD:
            self.counter += 1
        else:
            # Eye just opened — if it was closed long enough, count as blink
            if self.counter >= CONSEC_FRAMES:
                self.blink_count += 1
            self.counter = 0

        return ear

    def reset(self):
        """Reset counters at the start of a new window."""
        self.blink_count = 0
        self.counter     = 0