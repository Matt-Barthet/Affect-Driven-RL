from time import sleep

import numpy as np
from queue import PriorityQueue
from PCGRL.PCGUtility import construct_state

from scipy.ndimage import binary_dilation

import heapq

# Tune to your liking
NOISE_STRENGTH = 0
REPULSION_STRENGTH = 0
REPULSION_AREA = 5


def dijkstra_with_path(grid, start, goal):
    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    rows, cols = len(grid), len(grid[0])

    repulsion_grid = np.zeros_like(grid)
    repulsion_grid[np.where(grid != 7)] = REPULSION_STRENGTH / REPULSION_AREA
    for i in range(REPULSION_AREA):
        repulsion_mask = np.zeros_like(repulsion_grid)
        repulsion_mask[np.where(repulsion_grid != 0)] = 1
        repulsion_grid = repulsion_grid + ((i + 1) * REPULSION_STRENGTH / REPULSION_AREA) * binary_dilation(repulsion_mask).astype(repulsion_grid.dtype)


    def is_valid(x, y):
        return 0 <= x < rows and 0 <= y < cols and (grid[int(x)][int(y)] == 7)

    open_set = [(0, start)]
    visited = set()
    came_from = {start: None}

    while open_set:
        current_cost, current = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return current_cost, path[::-1]

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if is_valid(*neighbor) and neighbor not in visited:
                # add travel cost
                cost = current_cost + 1
                # add repulsion force from track
                cost += repulsion_grid[int(neighbor[0]), int(neighbor[1])]
                # add some noise
                cost += np.random.random() * NOISE_STRENGTH

                heapq.heappush(open_set, (cost, neighbor))
                came_from[neighbor] = current

    return float("inf"), []
