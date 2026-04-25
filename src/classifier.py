"""
classifier.py
-------------
Rule-based cognitive load classifier.
Fuses visual and interaction features into one of three load levels.

Decision logic (per project paper, Section VI):

    Blink rate elevated?
    ├── NO  → Fixation long & gaze stable?
    │         ├── YES → LOW
    │         └── NO  → MEDIUM
    └── YES → Head motion ↑ or typing pauses ↑?
              ├── YES → HIGH
              └── NO  → MEDIUM
"""


# ── Thresholds ───────────────────────────────────────────────────────────────
BLINK_RATE_NORMAL_MIN = 12   # blinks/min  — below this = suppressed (focused)
BLINK_RATE_ELEVATED   = 20   # blinks/min  — above this = elevated
GAZE_STABLE_THRESHOLD = 15.0 # pixels      — σ_gaze below this = stable
FIXATION_MIN_RATIO    = 0.6  # fraction of window with low σ = long fixation
HEAD_MOTION_THRESHOLD = 3.0  # pixels/frame — ΔH above this = elevated motion
PAUSE_COUNT_ELEVATED  = 3    # pauses per window above this = elevated IKI


class CognitiveLoadClassifier:
    """
    Stateless rule-based classifier.
    Call classify() with aggregated window features.
    """

    LOAD_LOW    = "LOW"
    LOAD_MEDIUM = "MEDIUM"
    LOAD_HIGH   = "HIGH"

    def classify(
        self,
        blink_rate_per_min: float,
        gaze_sigma:         float,
        head_delta:         float,
        pause_count:        int,
    ) -> str:
        """
        Classify cognitive load from aggregated window features.

        Args:
            blink_rate_per_min: Blinks per minute over the window
            gaze_sigma:         σ_gaze — iris position std deviation (pixels)
            head_delta:         ΔH — mean frame-to-frame nose displacement (pixels)
            pause_count:        Number of IKI pauses > threshold in the window

        Returns:
            One of "LOW", "MEDIUM", or "HIGH"
        """
        blink_elevated = blink_rate_per_min > BLINK_RATE_ELEVATED

        if not blink_elevated:
            gaze_stable   = gaze_sigma < GAZE_STABLE_THRESHOLD
            if gaze_stable:
                return self.LOAD_LOW
            else:
                return self.LOAD_MEDIUM
        else:
            head_elevated   = head_delta   > HEAD_MOTION_THRESHOLD
            pauses_elevated = pause_count  > PAUSE_COUNT_ELEVATED
            if head_elevated or pauses_elevated:
                return self.LOAD_HIGH
            else:
                return self.LOAD_MEDIUM