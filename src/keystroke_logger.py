"""
keystroke_logger.py
-------------------
Logs inter-keystroke intervals (IKI) in a background thread using pynput.
Flags pauses above a configurable threshold as cognitive pause events.

Reference:
    Epp, Lippold & Mandryk (CHI 2011) — keystroke dynamics for emotional state inference.
"""

import time
import threading
from pynput import keyboard


PAUSE_THRESHOLD = 2.0   # seconds — IKI above this = cognitive pause


class KeystrokeLogger:
    """
    Background keyboard listener that records IKI values.

    Usage:
        logger = KeystrokeLogger()
        logger.start()
        # ... run for one window ...
        pauses = logger.pause_count
        avg_iki = logger.average_iki()
        logger.reset()
    """

    def __init__(self, pause_threshold: float = PAUSE_THRESHOLD):
        self.pause_threshold = pause_threshold
        self.iki_values      = []
        self.pause_count     = 0
        self._last_time      = None
        self._lock           = threading.Lock()
        self._listener       = None

    def _on_press(self, key):
        now = time.time()
        with self._lock:
            if self._last_time is not None:
                iki = now - self._last_time
                self.iki_values.append(iki)
                if iki > self.pause_threshold:
                    self.pause_count += 1
            self._last_time = now

    def start(self):
        """Start listening for keystrokes in a background thread."""
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def stop(self):
        if self._listener:
            self._listener.stop()

    def average_iki(self) -> float:
        """Return mean IKI in seconds. Returns 0.0 if no keystrokes recorded."""
        with self._lock:
            if not self.iki_values:
                return 0.0
            return float(sum(self.iki_values) / len(self.iki_values))

    def reset(self):
        """Reset counters at the start of a new window."""
        with self._lock:
            self.iki_values  = []
            self.pause_count = 0
            self._last_time  = None