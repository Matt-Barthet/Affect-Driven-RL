import numpy as np
import hashlib


def get_state_hash(state):
    state_string = "_".join(str(e) for e in state)
    state_hash = hashlib.md5(state_string.encode()).hexdigest()
    return state_hash


class Cell:

    def __init__(self, state, trajectory_dict):

        self.key = get_state_hash(state)
        self.trajectory_dict = trajectory_dict

        self.state = state
        self.human_vector = []

        self.score = 0
        self.arousal = 0
        self.arousal_values = []
        self.uncertainty = 0
        self.arousal_reward = 0
        self.behavior_reward = 0
        self.blended_reward = 0

        self.age = 0
        self.visited = 1
        self.final = False

    def get_cell_length(self):
        return len(self.trajectory_dict['state_trajectory'])

    def normalize_r_a(self):
        return self.arousal_reward / len(self.trajectory_dict['arousal_trajectory'])

    def normalize_r_b(self):
        return self.behavior_reward / len(self.trajectory_dict['behavior_trajectory'])

    def assess_cell(self, weight, normalize_behavior, behavior_function, arousal_function, kNN):
        self.generate_human_arousal(kNN)
        self.behavior_reward = behavior_function(self.trajectory_dict['score_trajectory'], None)
        self.arousal_reward = arousal_function([], [])
        if normalize_behavior:
            self.blended_reward = self.normalize_r_a() * weight + self.normalize_r_b() * (1 - weight)
        else:
            self.blended_reward = self.normalize_r_a() * weight + self.behavior_reward * (1 - weight)

    def generate_human_arousal(self, k):
        self.arousal = 0
        self.arousal_values = [0]
        self.uncertainty = np.std(self.arousal_values)
        self.trajectory_dict['arousal_trajectory'].append(self.arousal)
        self.trajectory_dict['uncertainty_trajectory'].append(self.uncertainty)

    def update_key(self, state):
        self.key = get_state_hash(state)
