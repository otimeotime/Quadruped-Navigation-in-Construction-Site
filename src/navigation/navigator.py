import numpy as np
from .path_planner import PathPlanner
from .velocity_controller import VelocityController
from .waypoint_generator import WaypointGenerator


# Class: Navigation pipeline from a goal coordinate to velocity commands
# set_goal() plans the shortest collision-free path from the robot to the goal and splits it
# into intermediate targets. compute_command() is then called at every control step with the
# robot's pose: it returns the velocity toward the current target, moving on to the next one
# once the robot is within waypoint_tolerance, until it is within goal_tolerance of the goal.
# If the robot drifts so far that an obstacle stands between it and its target, the path is
# planned again from where it is.
class Navigator:
    # Constructor
    # grid: OccupancyGrid of the site
    # planner: PathPlanner (default: one on grid)
    # waypoint_generator: WaypointGenerator (default: 0.5 m spacing)
    # controller: VelocityController (default settings)
    # waypoint_tolerance: Distance at which an intermediate target counts as reached, in meters
    # goal_tolerance: Distance at which the goal counts as reached, in meters
    def __init__(self, grid, planner=None, waypoint_generator=None, controller=None, waypoint_tolerance=0.3,
                 goal_tolerance=0.15):
        self.grid = grid
        self.planner = planner if planner is not None else PathPlanner(grid)
        self.waypoint_generator = waypoint_generator if waypoint_generator is not None else WaypointGenerator()
        self.controller = controller if controller is not None else VelocityController()
        self.waypoint_tolerance = waypoint_tolerance
        self.goal_tolerance = goal_tolerance

        self.goal = None       # goal (x, y) as given to set_goal
        self.path = None       # planned path corners (N, 2), from the robot to the goal
        self.waypoints = None  # intermediate targets (M, 2), the last one being the goal
        self.target_idx = 0    # index of the current target in waypoints
        self.reached = False   # whether the goal has been reached
        self.num_replans = 0   # replans since the goal was set

    # Navigate from start (x, y) to goal (x, y); returns False if the goal is unreachable
    def set_goal(self, start, goal):
        self.goal = np.asarray(goal, dtype=float)
        self.reached = False
        self.num_replans = 0
        return self._plan(start)

    @property
    def target(self):
        return None if self.waypoints is None else self.waypoints[self.target_idx]

    # Velocity command (lin_vel_x, lin_vel_y, ang_vel) for a robot at position (x, y) with yaw
    # (radians); zero when there is no goal or it has been reached.
    def compute_command(self, position, yaw):
        if self.waypoints is None or self.reached:
            return np.zeros(3)
        position = np.asarray(position, dtype=float)

        last = len(self.waypoints) - 1
        while self.target_idx < last and self._can_advance(position):
            self.target_idx += 1
        if self.target_idx == last and np.linalg.norm(self.target - position) < self.goal_tolerance:
            self.reached = True
            return np.zeros(3)

        # Only check while the robot stands outside every obstacle: on a low one it stepped onto,
        # every straight line is blocked and it would replan at every step.
        if self.grid.clearance_at(position) > 0 and not self.grid.segment_is_free(position, self.target, margin=0.0):
            self.num_replans += 1
            if not self._plan(position):
                return np.zeros(3)
        is_goal = self.target_idx == len(self.waypoints) - 1
        return self.controller.compute(position, yaw, self.target, is_goal=is_goal)

    # Whether the robot at position may move on from the current target to the next: it is within
    # waypoint_tolerance of the current one and sees the next one in a straight line that comes no
    # closer to obstacles than the planned path (or than the robot already is). Aiming early at the
    # target after a corner would otherwise cut the corner, right into the obstacle it bends around.
    # Once the robot stands on the current target (within goal_tolerance) it always moves on.
    def _can_advance(self, position):
        distance = np.linalg.norm(self.target - position)
        if distance < self.goal_tolerance:
            return True
        if distance >= self.waypoint_tolerance:
            return False
        # Two cells of slack: clearance is only known per cell, so a straight line on the path
        # may still clip a cell slightly below the inflation radius.
        margin = min(self.grid.inflation, self.grid.clearance_at(position)) - 2 * self.grid.resolution
        return self.grid.segment_is_free(position, self.waypoints[self.target_idx + 1], margin=margin)

    # Plan a path from start to the goal and reset the targets; False if the goal is unreachable
    def _plan(self, start):
        self.path = self.planner.plan(start, self.goal)
        if self.path is None:
            self.waypoints = None
            return False
        self.waypoints = self.waypoint_generator.generate(self.path)
        self.target_idx = 0
        return True
