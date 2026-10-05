import numpy as np
import torch
import torch.nn.functional as F
import genesis as gs
from genesis.ext.isaacgym import terrain_utils
from .go2_env import Go2Env

# Height-field resolution: cell size and height step in meters
HORIZONTAL_SCALE = 0.1
VERTICAL_SCALE = 0.005

# Class: Go2 environment on rough ground
# Identical to Go2Env except that the flat plane is replaced by random bumps.
# Two modes:
# - Fixed (randomize=False): one bump field of the given roughness, the robot
#   always spawns at the origin. Used to test the robustness of a policy.
# - Randomized (randomize=True): domain randomization for training. The ground
#   is a grid of tiles, each with its own random roughness and bump spacing,
#   surrounded by a flat border, and every episode starts at a random spot.
class Go2EnvRough(Go2Env):
    # Constructor
    # roughness: Bump heights are sampled in [-roughness, roughness] meters.
    #            When randomized, it is the upper bound of each tile's roughness.
    # bump_spacing: Distance between two random bump peaks in meters (fixed mode)
    # terrain_size: Side length of the square bumpy area in meters, centered on the origin
    # randomize: Enable terrain domain randomization (see above)
    # seed: Terrain random seed. None draws a new terrain every run (randomized mode only).
    # tile_size: Side length of each randomized tile in meters
    # bump_spacing_range: Range of each randomized tile's bump spacing in meters
    # border_size: Width of the flat border around the randomized tiles, so robots
    #              that walk out of the bumpy area do not fall off the edge
    # Every other argument is forwarded to Go2Env.
    def __init__(self, *args, roughness=0.04, bump_spacing=0.4, terrain_size=100.0, randomize=False, seed=0,
                 tile_size=10.0, bump_spacing_range=(0.3, 0.8), border_size=20.0, **kwargs):
        self.roughness = roughness
        self.bump_spacing = bump_spacing
        self.terrain_size = terrain_size
        self.randomize = randomize
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % 2**32)
        self.terrain_seed = seed
        self.tile_size = tile_size
        self.bump_spacing_range = bump_spacing_range
        self.border_size = border_size if randomize else 0.0
        super().__init__(*args, **kwargs)

    def _add_ground(self):
        # Use the global NumPy generator like Genesis does, so the fixed terrain
        # with seed 0 matches the earlier gs.morphs.Terrain(randomize=False) one.
        saved_state = np.random.get_state()
        np.random.seed(self.terrain_seed)
        if self.randomize:
            height_field = self._random_tiles()
        else:
            height_field = self._bumps(self.terrain_size, self.roughness, self.bump_spacing)
        np.random.set_state(saved_state)

        # Flat border around the bumpy area
        border_cells = round(self.border_size / HORIZONTAL_SCALE)
        height_field = np.pad(height_field, border_cells, constant_values=0.0)

        # Terrain grows from pos along +x (rows) and +y (columns), so shift it to center it on the origin.
        half_size = self.terrain_size / 2 + self.border_size
        self.terrain_origin = -half_size
        self.scene.add_entity(
            gs.morphs.Terrain(
                pos=(-half_size, -half_size, 0.0),
                horizontal_scale=HORIZONTAL_SCALE,
                vertical_scale=VERTICAL_SCALE,
                height_field=height_field,
            ),
        )

        # Highest ground within 0.3 m of each cell, used to spawn robots above the bumps
        heights = torch.tensor(height_field * VERTICAL_SCALE, device=self.device, dtype=gs.tc_float)
        radius = round(0.3 / HORIZONTAL_SCALE)
        self.spawn_ground_height = F.max_pool2d(heights[None, None], 2 * radius + 1, stride=1, padding=radius)[0, 0]

    # Height field (in VERTICAL_SCALE units) of a square of random bumps
    def _bumps(self, size, roughness, spacing):
        cells = round(size / HORIZONTAL_SCALE) + 1
        tile = terrain_utils.SubTerrain(
            width=cells, length=cells, vertical_scale=VERTICAL_SCALE, horizontal_scale=HORIZONTAL_SCALE
        )
        terrain_utils.random_uniform_terrain(
            tile, min_height=-roughness, max_height=roughness, step=VERTICAL_SCALE, downsampled_scale=spacing
        )
        return tile.height_field_raw

    # Height field of a grid of tiles with roughness ~ U(0, roughness) and bump spacing ~ U(bump_spacing_range)
    def _random_tiles(self):
        n_tiles = round(self.terrain_size / self.tile_size)
        tile_cells = round(self.tile_size / HORIZONTAL_SCALE)
        height_field = np.zeros((n_tiles * tile_cells + 1,) * 2)
        self.tile_roughness = np.random.uniform(0.0, self.roughness, (n_tiles, n_tiles))

        # Fade the bumps to zero over the outer 0.5 m of each tile, otherwise tiles of
        # different roughness meet in a step up to twice the roughness high.
        edge_dist = np.minimum(np.arange(tile_cells + 1), np.arange(tile_cells, -1, -1)) * HORIZONTAL_SCALE
        ramp = np.clip(edge_dist / 0.5, 0.0, 1.0)
        ramp = ramp * ramp * (3 - 2 * ramp)  # smoothstep
        fade = np.outer(ramp, ramp)

        for i in range(n_tiles):
            for j in range(n_tiles):
                tile = self._bumps(
                    self.tile_size, self.tile_roughness[i, j], np.random.uniform(*self.bump_spacing_range)
                )
                # Neighboring tiles share their edge row/column, which is zero after fading.
                rows = slice(i * tile_cells, (i + 1) * tile_cells + 1)
                cols = slice(j * tile_cells, (j + 1) * tile_cells + 1)
                height_field[rows, cols] = tile * fade
        return height_field

    def reset_idx(self, envs_idx):
        super().reset_idx(envs_idx)
        if not self.randomize or len(envs_idx) == 0:
            return

        # Respawn at a random spot of the bumpy area, just above the local bumps
        half_span = self.terrain_size / 2 - 1.0
        xy = (2 * torch.rand((len(envs_idx), 2), device=self.device) - 1) * half_span
        cell = ((xy - self.terrain_origin) / HORIZONTAL_SCALE).round().long()
        ground = self.spawn_ground_height[cell[:, 0], cell[:, 1]]
        self.base_pos[envs_idx] = torch.cat((xy, (ground + self.base_init_pos[2])[:, None]), dim=1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
