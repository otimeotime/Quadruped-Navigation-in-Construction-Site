import math
import numpy as np
from scipy import ndimage

# Obstacle kinds whose footprint is a circle: their half_extents hold the radius twice
ROUND_KINDS = ("barrel", "rubble")


# Class: 2D occupancy grid of the construction site, as seen by an (idealized) overhead camera
# Instead of a real camera, it reads the obstacle layout straight from the environment.
# Each cell stores its clearance, the distance from its center to the nearest obstacle
# footprint or wall. A cell is occupied when its clearance is below the inflation radius,
# so the robot can be planned as a point: any free cell is a safe spot for the robot's center.
# Every obstacle is treated as blocking, even the low ones the robot could step over.
class OccupancyGrid:
    # Constructor
    # obstacles: Obstacle dicts as in Go2EnvConstruction.obstacles (kind, half_extents, x, y, yaw in degrees)
    # site_size: Side length of the square site inside the walls, centered on the origin, in meters
    # resolution: Cell size in meters
    # inflation: Clearance kept between the robot's center and any obstacle or wall, in meters.
    #            Go2 is about 0.31 m wide; keep it below half the site's min_gap so no passage closes.
    def __init__(self, obstacles, site_size, resolution=0.05, inflation=0.25):
        self.site_size = site_size
        self.resolution = resolution
        self.inflation = inflation
        self.num_cells = round(site_size / resolution)
        # World coordinate of the grid's lower edge, in both x and y
        self.origin = -site_size / 2

        # Cell (i, j) is centered on x = centers[i], y = centers[j]
        centers = self.origin + (np.arange(self.num_cells) + 0.5) * resolution
        xs, ys = np.meshgrid(centers, centers, indexing="ij")
        clearance = site_size / 2 - np.maximum(np.abs(xs), np.abs(ys))  # distance to the walls
        for obstacle in obstacles:
            clearance = np.minimum(clearance, _distance_to_obstacle(xs, ys, obstacle))
        self.clearance = clearance
        self.occupied = clearance < inflation

        # Label of the connected free region of each cell (0 if occupied), to tell reachable goals
        self.regions, _ = ndimage.label(~self.occupied, structure=np.ones((3, 3)))
        # Indices of the nearest free cell of each cell, to step out of occupied cells
        self.nearest_free = ndimage.distance_transform_edt(self.occupied, return_distances=False,
                                                           return_indices=True)

    # Grid of the site of a Go2EnvConstruction environment
    @classmethod
    def from_env(cls, env, **kwargs):
        return cls(env.obstacles, env.site_size, **kwargs)

    # Cell indices (..., 2) of world points (..., 2), clipped to the grid
    def to_cell(self, points):
        cells = np.floor((np.asarray(points) - self.origin) / self.resolution).astype(int)
        return np.clip(cells, 0, self.num_cells - 1)

    # World coordinates (..., 2) of cell centers (..., 2)
    def to_world(self, cells):
        return self.origin + (np.asarray(cells) + 0.5) * self.resolution

    # Clearance of the cells under world points (..., 2)
    def clearance_at(self, points):
        i, j = np.moveaxis(self.to_cell(points), -1, 0)
        return self.clearance[i, j]

    # Whether world points (..., 2) are free; points outside the site are not
    def is_free(self, points):
        return self.clearance_at(points) >= self.inflation

    # Whether the straight segment from a to b keeps at least margin clearance
    # (default: the inflation radius, i.e. it only crosses free cells)
    def segment_is_free(self, a, b, margin=None):
        margin = self.inflation if margin is None else margin
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        # Sample twice per cell so no cell along the segment is skipped
        steps = max(1, math.ceil(np.linalg.norm(b - a) / (0.5 * self.resolution)))
        points = a + np.linspace(0.0, 1.0, steps + 1)[:, None] * (b - a)
        return bool(np.all(self.clearance_at(points) >= margin))

    # The cell itself if free, else the nearest free cell, as an (i, j) tuple
    def nearest_free_cell(self, cell):
        i, j = cell
        return int(self.nearest_free[0, i, j]), int(self.nearest_free[1, i, j])

    # Whether two cells are free and joined by free cells
    def connected(self, cell_a, cell_b):
        region = self.regions[tuple(cell_a)]
        return region != 0 and region == self.regions[tuple(cell_b)]


# Distance from points (xs, ys) to an obstacle's footprint (0 or less inside)
def _distance_to_obstacle(xs, ys, obstacle):
    dx, dy = xs - obstacle["x"], ys - obstacle["y"]
    hx, hy = obstacle["half_extents"]
    if obstacle["kind"] in ROUND_KINDS:
        return np.hypot(dx, dy) - hx
    # Offset along the footprint rectangle's own axes
    yaw = math.radians(obstacle["yaw"])
    u = dx * math.cos(yaw) + dy * math.sin(yaw)
    v = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return np.hypot(np.maximum(np.abs(u) - hx, 0.0), np.maximum(np.abs(v) - hy, 0.0))
