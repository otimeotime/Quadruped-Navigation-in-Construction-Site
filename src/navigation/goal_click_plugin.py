import math
from threading import Lock
import numpy as np
from genesis.utils.raycast import plane_raycast
from genesis.vis.keybindings import KeyMod, MouseButton
from genesis.vis.viewer_plugins import RaycasterViewerPlugin

# Modifiers that turn a left drag into a pan or zoom of the camera; a click holding one is not a goal
CAMERA_MODIFIERS = KeyMod.SHIFT | KeyMod.CTRL | KeyMod.ALT


# Class: Viewer plugin that turns a mouse click on the scene into a navigation goal
# A plain left click (press and release without dragging) casts a ray from the cursor into
# the scene, and the point it hits (ground or obstacle) becomes the goal. Dragging still
# orbits the camera as usual: the plugin never consumes mouse events.
# Clicks arrive on the viewer's thread, so the goal is handed over through a lock; the
# simulation loop collects it with pop_goal().
class GoalClickPlugin(RaycasterViewerPlugin):
    # Constructor
    # max_drag: How far the cursor may move between press and release for a click, in pixels
    def __init__(self, max_drag=5):
        super().__init__()
        self.max_drag = max_drag
        self._lock = Lock()
        self._press = None  # cursor position of a pending left press
        self._goal = None   # last clicked goal (x, y) not yet collected

    def on_mouse_press(self, x, y, button, modifiers):
        is_plain_left = button == MouseButton.LEFT and not modifiers & CAMERA_MODIFIERS
        self._press = (x, y) if is_plain_left else None

    def on_mouse_release(self, x, y, button, modifiers):
        press, self._press = self._press, None
        if press is None or button != MouseButton.LEFT or math.dist(press, (x, y)) > self.max_drag:
            return
        ray = self._screen_position_to_ray(x, y)
        hit = self._raycaster.cast(ray.origin, ray.direction)
        if hit is None:  # nothing drawn under the cursor: fall back on the ground plane z = 0
            hit = plane_raycast(np.array([0.0, 0.0, 1.0]), 0.0, ray)
        if hit is not None:
            with self._lock:
                self._goal = np.array(hit.position[:2], dtype=float)

    # The goal clicked since the last call, or None
    def pop_goal(self):
        with self._lock:
            goal, self._goal = self._goal, None
        return goal
