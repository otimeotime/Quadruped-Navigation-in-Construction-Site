import heapq
import math
import numpy as np

# 8-connected moves (di, dj, cost in cells)
MOVES = [(di, dj, math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]


# Class: Shortest collision-free path planner on an occupancy grid
# A* search over the free cells (8-connected), followed by string pulling: corners that
# see each other in a straight line are joined directly. This removes the zigzag of grid
# moves and leaves an any-angle path close to the true shortest one, still clear of
# every obstacle by at least the grid's inflation radius.
class PathPlanner:
    # Constructor
    # grid: OccupancyGrid of the site
    def __init__(self, grid):
        self.grid = grid

    # Shortest path from start to goal (world xy) as an (N, 2) array of corners, starting at
    # start. If the goal is not free, the path ends on the nearest free cell instead.
    # Returns None if the goal cannot be reached.
    def plan(self, start, goal):
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        start_cell = self.grid.nearest_free_cell(self.grid.to_cell(start))
        goal_cell = self.grid.nearest_free_cell(self.grid.to_cell(goal))
        if not self.grid.connected(start_cell, goal_cell):
            return None

        cells = self._search(start_cell, goal_cell)
        points = [start, *self.grid.to_world(cells)]
        if self.grid.is_free(goal):
            points[-1] = goal  # end exactly on the goal rather than on its cell center
        return self._pull_string(np.array(points))

    # A* from start_cell to goal_cell (connected free cells); returns the cells of the path
    def _search(self, start_cell, goal_cell):
        occupied = self.grid.occupied.tolist()  # plain lists index much faster than arrays here
        size = self.grid.num_cells
        gi, gj = goal_cell

        # Octile distance: exact length of the shortest 8-connected path without obstacles
        def heuristic(i, j):
            di, dj = abs(i - gi), abs(j - gj)
            return max(di, dj) + (math.sqrt(2) - 1) * min(di, dj)

        cost = {start_cell: 0.0}
        parent = {start_cell: None}
        frontier = [(heuristic(*start_cell), start_cell)]
        closed = set()
        while frontier:
            _, cell = heapq.heappop(frontier)
            if cell == goal_cell:
                break
            if cell in closed:
                continue
            closed.add(cell)
            i, j = cell
            for di, dj, step in MOVES:
                ni, nj = i + di, j + dj
                if not (0 <= ni < size and 0 <= nj < size) or occupied[ni][nj]:
                    continue
                # Diagonal moves may not cut the corner of an occupied cell
                if di and dj and (occupied[i + di][j] or occupied[i][j + dj]):
                    continue
                new_cost = cost[cell] + step
                if new_cost < cost.get((ni, nj), math.inf):
                    cost[(ni, nj)] = new_cost
                    parent[(ni, nj)] = cell
                    heapq.heappush(frontier, (new_cost + heuristic(ni, nj), (ni, nj)))

        cells = []
        cell = goal_cell
        while cell is not None:
            cells.append(cell)
            cell = parent[cell]
        return cells[::-1]

    # Keep only the corners of the path: from each kept point, jump to the farthest
    # following point that it reaches in a free straight line.
    def _pull_string(self, points):
        kept = [0]
        while kept[-1] < len(points) - 1:
            anchor = kept[-1]
            # The next point is always kept: the start may lie in an occupied cell
            # (robot too close to an obstacle) and must still lead out of it.
            reach = anchor + 1
            while reach + 1 < len(points) and self.grid.segment_is_free(points[anchor], points[reach + 1]):
                reach += 1
            kept.append(reach)
        return points[kept]
