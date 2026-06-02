# capture.py


class KeystrokeCapture:
    """Thin event-list container used by the GUI to accumulate keystroke events."""

    def __init__(self):
        self.events = []

    def reset(self):
        self.events = []
