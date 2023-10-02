import numpy as np
import hashlib


def get_state_hash(state):
    state_string = "_".join(str(e) for e in state)
    state_hash = hashlib.md5(state_string.encode()).hexdigest()
    return state_hash


class Cell:

    def __init__(self, state, trajectory_dict, properties):
        self.key = get_state_hash(state)
        self.state = state
        self.trajectory_dict = trajectory_dict
        self.properties = properties
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
        self.legalState = True

    def get_cell_length(self):
        return len(self.trajectory_dict['state_trajectory'])

    def normalize_r_a(self):
        return self.arousal_reward / len(self.trajectory_dict['arousal_trajectory'])

    def normalize_r_b(self):
        return self.behavior_reward / len(self.trajectory_dict['behavior_trajectory'])

    def calculate_rewards(self, behavior_function, arousal_function):
        if self.behavior_target == "Imitate":
            self.currentCell.behavior_reward = behavior_function()
        else:
            self.currentCell.behavior_reward += self.score

    def calculate_blended_reward(self, weight, normalize_behavior, behavior_function, arousal_function):
        self.calculate_rewards(behavior_function, arousal_function)
        if normalize_behavior:
            return self.normalize_r_a() * weight + self.normalize_r_b() * (1 - weight)
        else:
            return self.normalize_r_a() * weight + self.behavior_reward * (1 - weight)

    def generate_human_arousal(self, k):
        """
        TODO:Generate the arousal value observed by the closest humans to the state represented by this cell.
        """
        self.arousal = 0
        self.arousal_values = [0]
        self.uncertainty = np.std(self.arousal_values)
        self.trajectory_dict['arousal_trajectory'].append(self.arousal)
        self.trajectory_dict['uncertainty_trajectory'].append(self.uncertainty)

    def update_key(self, state):
        self.key = get_state_hash(state)
