import numpy as np


# Class: Source of navigation commands: random goal coordinates in the construction site
# Goals are drawn uniformly over the free cells the robot can reach from where it stands,
# at least min_distance away from it, so every goal has a collision-free path. They also keep
# min_clearance from obstacles and walls, so the robot has room to stop and turn there.
class GoalSampler:
    # Constructor
    # grid: OccupancyGrid of the site
    # min_distance: Minimum straight-line distance between the robot and its goal in meters
    # min_clearance: Minimum distance between a goal and any obstacle or wall in meters
    #                (Go2 sweeps a circle of about 0.38 m radius when it turns in place)
    # seed: Random seed of the goals (None for a new sequence every run)
    def __init__(self, grid, min_distance=2.0, min_clearance=0.4, seed=None):
        self.grid = grid
        self.min_distance = min_distance
        self.min_clearance = min_clearance
        self.rng = np.random.default_rng(seed)

    # A goal (x, y) for a robot at position (x, y)
    def sample(self, position):
        position = np.asarray(position, dtype=float)
        robot_cell = self.grid.nearest_free_cell(self.grid.to_cell(position))
        reachable = self.grid.regions == self.grid.regions[robot_cell]
        goals = self.grid.to_world(np.argwhere(reachable & (self.grid.clearance >= self.min_clearance)))
        if len(goals) == 0:  # no open space in reach: any reachable cell will do
            goals = self.grid.to_world(np.argwhere(reachable))
        far = goals[np.linalg.norm(goals - position, axis=1) >= self.min_distance]
        if len(far) > 0:  # otherwise everything in reach is close, take any of it
            goals = far
        return goals[self.rng.integers(len(goals))]
