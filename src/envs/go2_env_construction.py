import math
import numpy as np
import genesis as gs
from genesis.utils.geom import euler_to_R
from .go2_env_rough import Go2EnvRough

# Colors (RGB in [0, 1])
WALL_COLOR = (0.55, 0.62, 0.70)      # site hoarding
CONCRETE_COLOR = (0.62, 0.62, 0.60)  # concrete blocks
RUBBLE_COLOR = (0.48, 0.45, 0.42)    # broken concrete and bricks
WOOD_COLOR = (0.60, 0.42, 0.22)      # pallets and planks
PIPE_COLOR = (0.90, 0.45, 0.10)      # orange plastic pipes
BARREL_COLOR = (0.95, 0.75, 0.10)    # yellow barrels
BARRIER_COLOR = (0.80, 0.22, 0.15)   # red plastic road barriers

# Obstacles are sunk this deep into the ground so that none floats above a bump
SINK_DEPTH = 0.1

# Obstacle kinds and how often each is drawn. Large obstacles and long barriers
# are common because they block straight paths and force detours.
OBSTACLE_KINDS = {
    "concrete_block": 0.25,
    "barrier": 0.20,
    "pallet": 0.15,
    "pipe": 0.15,
    "barrel": 0.10,
    "plank": 0.05,
    "rubble": 0.10,
}

# Class: Go2 environment in a construction site
# A finite square site of rough ground (see Go2EnvRough, fixed mode), bounded by
# walls and filled with fixed obstacles of many sizes: concrete blocks, road
# barriers, pallet stacks, pipes, barrels, planks and rubble. The robot spawns
# at the center, which is kept clear.
# Every gap (between obstacles, and between an obstacle and a wall) is at least
# min_gap wide. Obstacles are convex, so with min_gap wider than the robot they
# can never close off a region: every free spot stays reachable, only not
# always in a straight line.
class Go2EnvConstruction(Go2EnvRough):
    # Constructor
    # site_size: Side length of the square site inside the walls, in meters
    # wall_height: Height of the boundary walls in meters
    # wall_thickness: Thickness of the boundary walls in meters
    # num_obstacles: Number of obstacles to place (fewer are placed if the site is full)
    # obstacle_seed: Random seed of the obstacle layout
    # min_gap: Minimum free space between two obstacles, and between an obstacle and a wall, in meters.
    #          Keep it wider than the robot (Go2 is about 0.31 m wide and 0.7 m long) so it can pass and turn.
    # spawn_clearance: Radius around the spawn point kept free of obstacles, in meters
    # roughness, seed: Bump height and terrain seed, as in Go2EnvRough
    # Every other argument is forwarded to Go2Env.
    def __init__(self, *args, site_size=12.0, wall_height=1.5, wall_thickness=0.2, num_obstacles=25,
                 obstacle_seed=0, min_gap=0.8, spawn_clearance=1.0, roughness=0.04, seed=0, **kwargs):
        self.site_size = site_size
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness
        self.num_obstacles = num_obstacles
        self.obstacle_seed = obstacle_seed
        self.min_gap = min_gap
        self.spawn_clearance = spawn_clearance
        # The bumpy ground runs 1 m past the walls so they stand on it.
        super().__init__(*args, roughness=roughness, seed=seed, terrain_size=site_size + 2.0, randomize=False,
                         **kwargs)

    def _add_ground(self):
        super()._add_ground()
        self._add_walls()
        self._add_obstacles()

    # Add a static colored entity. Its morph sets batch_fixed_verts=False: the site is the
    # same in every environment, so all environments share one copy of its geometry.
    def _add_fixed(self, morph, color):
        self.scene.add_entity(morph, surface=gs.surfaces.Default(color=color))

    def _add_walls(self):
        half = self.site_size / 2 + self.wall_thickness / 2
        length = self.site_size + 2 * self.wall_thickness
        height = self.wall_height + SINK_DEPTH
        z = height / 2 - SINK_DEPTH
        for x, y, size in [
            (half, 0.0, (self.wall_thickness, length, height)),
            (-half, 0.0, (self.wall_thickness, length, height)),
            (0.0, half, (length, self.wall_thickness, height)),
            (0.0, -half, (length, self.wall_thickness, height)),
        ]:
            self._add_fixed(gs.morphs.Box(pos=(x, y, z), size=size, fixed=True, batch_fixed_verts=False), WALL_COLOR)

    def _add_obstacles(self):
        rng = np.random.default_rng(self.obstacle_seed)
        kinds = list(OBSTACLE_KINDS)
        weights = np.array(list(OBSTACLE_KINDS.values()))
        candidates = [self._sample_obstacle(kind, rng) for kind in rng.choice(kinds, self.num_obstacles, p=weights)]
        # Place the largest first, so smaller ones fill the gaps between them.
        candidates.sort(key=lambda obstacle: obstacle["half_extents"][0] * obstacle["half_extents"][1], reverse=True)

        # Each placed obstacle as a dict: kind, size, footprint half extents, x, y, yaw (degrees).
        # Kept for later use, e.g. navigation.
        self.obstacles = []
        for obstacle in candidates:
            if self._find_free_spot(obstacle, rng):
                self._place_obstacle(obstacle)
                self.obstacles.append(obstacle)

    # Random dimensions of one obstacle. Sizes are mostly log-uniform so that small
    # and large obstacles are equally likely. "half_extents" is the half length and
    # half width of its footprint rectangle.
    @staticmethod
    def _sample_obstacle(kind, rng):
        def log_uniform(low, high):
            return float(np.exp(rng.uniform(np.log(low), np.log(high))))

        if kind == "concrete_block":
            # From curb stones the robot can step on to blocks taller than it
            size = (log_uniform(0.3, 2.0), log_uniform(0.3, 2.0), log_uniform(0.1, 1.0))
        elif kind == "barrier":
            # A row of road barriers: a long, thin wall segment
            size = (rng.uniform(1.5, 3.5), rng.uniform(0.3, 0.5), rng.uniform(0.8, 1.0))
        elif kind == "pallet":
            # Full or half pallets, stacked one to five high
            footprint = (1.2, 1.0) if rng.random() < 0.6 else (0.8, 0.6)
            size = (*footprint, 0.15 * int(rng.integers(1, 6)))
        elif kind == "pipe":
            radius, length = log_uniform(0.04, 0.3), log_uniform(0.8, 4.0)
            return {"kind": kind, "size": (radius, length), "half_extents": (length / 2, radius)}
        elif kind == "barrel":
            radius, height = rng.uniform(0.2, 0.35), rng.uniform(0.5, 1.0)
            return {"kind": kind, "size": (radius, height), "half_extents": (radius, radius)}
        elif kind == "plank":
            size = (log_uniform(0.8, 3.0), rng.uniform(0.15, 0.4), rng.uniform(0.03, 0.08))
        else:  # rubble: a small chunk lying tilted
            size = (log_uniform(0.1, 0.4), log_uniform(0.1, 0.4), log_uniform(0.08, 0.3))
            # Bounding circle of the chunk, since the tilt makes its footprint irregular
            radius = math.hypot(*size) / 2
            tilt = tuple(rng.uniform(-25.0, 25.0, 2))  # roll, pitch in degrees
            return {"kind": kind, "size": size, "half_extents": (radius, radius), "tilt": tilt}
        return {"kind": kind, "size": size, "half_extents": (size[0] / 2, size[1] / 2)}

    # Try random poses until the obstacle clears the walls, the spawn point and the
    # obstacles already placed; store the pose in the obstacle on success.
    def _find_free_spot(self, obstacle, rng, attempts=200):
        half_site = self.site_size / 2
        for _ in range(attempts):
            yaw = rng.uniform(0.0, 180.0)
            hx, hy = obstacle["half_extents"]
            cos, sin = abs(math.cos(math.radians(yaw))), abs(math.sin(math.radians(yaw)))
            # Half size of the footprint's axis-aligned bounding box
            ex, ey = hx * cos + hy * sin, hx * sin + hy * cos
            limit_x, limit_y = half_site - self.min_gap - ex, half_site - self.min_gap - ey
            if limit_x <= 0 or limit_y <= 0:
                return False  # larger than the site
            x, y = rng.uniform(-limit_x, limit_x), rng.uniform(-limit_y, limit_y)
            candidate = dict(obstacle, x=x, y=y, yaw=yaw)
            if _distance_to_footprint(0.0, 0.0, candidate) < self.spawn_clearance:
                continue
            if not any(_footprints_overlap(candidate, other, self.min_gap) for other in self.obstacles):
                obstacle.update(x=x, y=y, yaw=yaw)
                return True
        return False

    def _place_obstacle(self, obstacle):
        kind, size, x, y, yaw = obstacle["kind"], obstacle["size"], obstacle["x"], obstacle["y"], obstacle["yaw"]
        if kind in ("concrete_block", "barrier", "pallet", "plank"):
            height = size[2] + SINK_DEPTH
            color = {"concrete_block": CONCRETE_COLOR, "barrier": BARRIER_COLOR}.get(kind, WOOD_COLOR)
            morph = gs.morphs.Box(pos=(x, y, height / 2 - SINK_DEPTH), size=(size[0], size[1], height),
                                  euler=(0.0, 0.0, yaw), fixed=True, batch_fixed_verts=False)
            self._add_fixed(morph, color)
        elif kind == "rubble":
            euler = (*obstacle["tilt"], yaw)
            # Lowest corner of the tilted chunk, measured from its center
            reach_down = np.abs(euler_to_R(np.array(euler))[2]) @ np.array(size) / 2
            morph = gs.morphs.Box(pos=(x, y, reach_down - SINK_DEPTH / 2), size=size, euler=euler,
                                  fixed=True, batch_fixed_verts=False)
            self._add_fixed(morph, RUBBLE_COLOR)
        elif kind == "pipe":
            radius, length = size
            # Cylinders stand along z; tip it over onto the ground (axis along y), then turn
            # it by yaw - 90 so that its axis follows the footprint's long side at angle yaw.
            morph = gs.morphs.Cylinder(pos=(x, y, radius - min(radius, SINK_DEPTH) / 2), radius=radius, height=length,
                                       euler=(90.0, 0.0, yaw - 90.0), fixed=True, batch_fixed_verts=False)
            self._add_fixed(morph, PIPE_COLOR)
        else:  # barrel
            radius, height = size
            morph = gs.morphs.Cylinder(pos=(x, y, (height + SINK_DEPTH) / 2 - SINK_DEPTH), radius=radius,
                                       height=height + SINK_DEPTH, fixed=True, batch_fixed_verts=False)
            self._add_fixed(morph, BARREL_COLOR)


# Unit axes of an obstacle's footprint rectangle
def _footprint_axes(obstacle):
    yaw = math.radians(obstacle["yaw"])
    return np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])


# Whether two footprint rectangles come closer than gap (separating axis test)
def _footprints_overlap(a, b, gap):
    axes_a, axes_b = _footprint_axes(a), _footprint_axes(b)
    offset = np.array([b["x"] - a["x"], b["y"] - a["y"]])
    for axis in (*axes_a, *axes_b):
        reach_a = sum(h * abs(u @ axis) for h, u in zip(a["half_extents"], axes_a))
        reach_b = sum(h * abs(u @ axis) for h, u in zip(b["half_extents"], axes_b))
        if abs(offset @ axis) >= reach_a + reach_b + gap:
            return False
    return True


# Distance from a point to an obstacle's footprint rectangle (0 inside)
def _distance_to_footprint(px, py, obstacle):
    u, v = _footprint_axes(obstacle)
    offset = np.array([px - obstacle["x"], py - obstacle["y"]])
    outside = [max(abs(offset @ axis) - h, 0.0) for axis, h in zip((u, v), obstacle["half_extents"])]
    return math.hypot(*outside)
