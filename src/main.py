"""
main.py
-------
Entry point for the cognitive load detection pipeline.

Run:
    python src/main.py

Controls:
    Q  — quit
    R  — force reset current window early
"""

import time
import cv2
import mediapipe as mp

from blink_detector   import BlinkDetector
from gaze_tracker     import GazeTracker
from head_pose        import HeadPoseTracker
from keystroke_logger import KeystrokeLogger
from classifier       import CognitiveLoadClassifier


# ── Configuration ─────────────────────────────────────────────────────────────
CAMERA_INDEX  = 0      # Change to 1 or 2 if using DroidCam
WINDOW_SEC    = 30     # Sliding window duration in seconds
DISPLAY_W     = 960
DISPLAY_H     = 540

# Load level → display colour (BGR)
LOAD_COLORS = {
    "LOW":    (34,  197, 94),    # green
    "MEDIUM": (245, 158, 11),    # amber
    "HIGH":   (239,  68, 68),    # red
}


def main():
    # ── MediaPipe setup ───────────────────────────────────────────────────────
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,   # required for iris landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ── Module init ───────────────────────────────────────────────────────────
    blink      = BlinkDetector()
    gaze       = GazeTracker()
    head       = HeadPoseTracker()
    keys       = KeystrokeLogger()
    classifier = CognitiveLoadClassifier()

    keys.start()

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}.")
        print("        If using DroidCam, try CAMERA_INDEX = 1 or 2.")
        return

    window_start  = time.time()
    current_load  = "---"
    frame_count   = 0

    print("[INFO] Pipeline running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        img_h, img_w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0]
            blink.update(lm, img_w, img_h)
            gaze.update(lm,  img_w, img_h)
            head.update(lm,  img_w, img_h)
            frame_count += 1

        # ── Window evaluation ─────────────────────────────────────────────────
        elapsed = time.time() - window_start
        if elapsed >= WINDOW_SEC:
            blink_rate = (blink.blink_count / elapsed) * 60   # blinks/min
            gaze_sigma = gaze.compute_stability()
            head_delta = head.compute_variation()
            pause_cnt  = keys.pause_count

            current_load = classifier.classify(
                blink_rate_per_min=blink_rate,
                gaze_sigma=gaze_sigma,
                head_delta=head_delta,
                pause_count=pause_cnt,
            )

            print(f"[WINDOW] blink={blink_rate:.1f}/min  σ_gaze={gaze_sigma:.1f}px  "
                  f"ΔH={head_delta:.2f}px  pauses={pause_cnt}  → {current_load}")

            blink.reset(); gaze.reset(); head.reset(); keys.reset()
            window_start = time.time()
            frame_count  = 0

        # ── Overlay ───────────────────────────────────────────────────────────
        color = LOAD_COLORS.get(current_load, (180, 180, 180))

        cv2.rectangle(frame, (0, 0), (DISPLAY_W, 55), (15, 25, 42), -1)
        cv2.putText(frame, f"Cognitive Load: {current_load}",
                    (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

        cv2.putText(frame, f"EAR: {blink.current_ear:.3f}",
                    (14, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
        cv2.putText(frame, f"Blinks: {blink.blink_count}",
                    (14, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
        cv2.putText(frame, f"Window: {int(WINDOW_SEC - elapsed)}s remaining",
                    (14, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120,120,120), 1)

        cv2.imshow("Cognitive Load Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            blink.reset(); gaze.reset(); head.reset(); keys.reset()
            window_start = time.time()

    keys.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()