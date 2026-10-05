import math
import numpy as np


# Class: Splits a path into evenly spaced intermediate targets
# Every corner of the path is kept as a target and each straight piece between two
# corners is cut into equal steps of at most spacing, so the targets never leave the
# planned (collision-free) path, not even at its corners.
class WaypointGenerator:
    # Constructor
    # spacing: Maximum distance between two consecutive targets in meters
    def __init__(self, spacing=0.5):
        self.spacing = spacing

    # Targets (M, 2) along a path (N, 2) of corners, ending on its last point.
    # The first point is left out: it is where the robot already stands.
    def generate(self, path):
        path = np.asarray(path, dtype=float)
        if len(path) == 1:
            return path.copy()  # already at the goal
        targets = []
        for a, b in zip(path[:-1], path[1:]):
            steps = max(1, math.ceil(np.linalg.norm(b - a) / self.spacing))
            fractions = np.arange(1, steps + 1)[:, None] / steps
            targets.append(a + fractions * (b - a))
        return np.concatenate(targets)
