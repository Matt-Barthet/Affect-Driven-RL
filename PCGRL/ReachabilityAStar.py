from time import sleep

import numpy as np
from queue import PriorityQueue
from PCGRL.PCGUtility import construct_state

import heapq


def dijkstra_with_path(grid, start, goal):
    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    rows, cols = len(grid), len(grid[0])

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
                heapq.heappush(open_set, (current_cost + 1, neighbor))
                came_from[neighbor] = current

    return float("inf"), []


class AStar:

    def __init__(self, initial, goal, env):
        self.goal = goal
        self.initial_state = initial[0:2]  # Assuming this is a state representation
        self.initial_actions = initial[2].copy()  # Sequence of actions to reach the initial state from the environment's start state
        self.env = env

        self.queue = PriorityQueue()
        self.queue.put((0, (self.initial_state, self.initial_actions)))

        self.grid = None
        self.current_state = None
        self.index = None
        self.facing = None

    @staticmethod
    def distance(a, b):
        """Calculate the Manhattan distance or an appropriate heuristic between two points"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def reset_to_state(self, actions):
        self.env.reset()
        for action in actions:
            next_state, reward, done, _ = self.env.env.step(action)  # Execute the action
            self.current_state, self.grid, self.index, self.facing = construct_state(next_state,
                                                                                     self.env.one_hot_encode)

    def a_star_search(self):

        """Perform A* search to find a path from start to goal"""
        came_from = {}
        g_score = {self.initial_state: 0}
        f_score = {self.initial_state: self.distance(self.initial_state, self.goal)}
        best_g_scores = {self.initial_state: 0}

        while not self.queue.empty():
            _, (current_index, current_actions) = self.queue.get()

            if current_index == self.goal:
                for action in range(5):
                    self.reset_to_state(current_actions.copy())
                    next_state, reward, done, _ = self.env.env.step(action)
                    self.current_state, self.grid, self.index, self.facing = construct_state(next_state, self.env.one_hot_encode)
                    new_actions = current_actions + [action]
                    if self.index == (50, 50):
                        return new_actions

            for action in range(5):
                # print(f"Passing actions to reset {current_actions}")
                self.reset_to_state(current_actions.copy())
                next_state, reward, done, _ = self.env.env.step(action)
                self.current_state, self.grid, self.index, self.facing = construct_state(next_state, self.env.one_hot_encode)

                if self.env.customSideChannel.collision:
                    continue

                new_actions = current_actions + [action]
                tentative_g_score = g_score[current_index] + 1
                came_from[self.index] = self.current_state
                g_score[self.index] = tentative_g_score
                f_score[self.index] = tentative_g_score + self.distance(self.index, self.goal)
                best_g_scores[self.index] = tentative_g_score  # Update the best g_score for this state
                self.queue.put((f_score[self.index], (self.index, new_actions)))
                # print(self.queue.qsize())
        return None
