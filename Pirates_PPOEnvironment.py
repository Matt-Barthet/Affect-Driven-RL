from abc import ABC
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from SurrogateModel import KNNSurrogateModel
from Utils.Tensorboard_Callbacks import TensorboardCallback
from BaseEnvironment import BaseEnvironment
import numpy as np
import pickle
from Utils.Scalers import VectorScaler
import torch
import sys

class PPO_Environment(BaseEnvironment, ABC):

    def __init__(self, id_number, graphics, obs_space, path, arousal_model, weight):

        """ ---- Pirates! specific code ---- """
        self.gridWidth = 11
        self.gridHeight = 11
        self.elementSize = 1

        self.last_x = -np.inf
        self.max_x = -np.inf
        self.death_applied = False
        self.previous_score = 0  # maximum possible score of 460
        super().__init__(id_number=id_number, graphics=graphics, obs_space=obs_space, path=path,
                         args=["-gridWidth", f"{self.gridWidth}", "-gridHeight", f"{self.gridHeight}", "-elementSize",
                          f"{self.elementSize}"], capture_fps=60, time_scale=1, arousal_model=arousal_model, weight=weight)

    def calculate_reward(self, state, position):
        current_x = np.round(position)
        health = state[3]
        reward = 0
        if health > 0:
            self.death_applied = False
        if health == 0 and not self.death_applied:
            reward -= 5
            self.death_applied = True
        elif current_x > self.last_x:
            reward += 1
        self.last_x = current_x
        self.current_reward = (self.current_score - self.previous_score) + reward
        self.current_reward /= 31
        self.previous_score = self.current_score

    def reset_condition(self):
        if self.episode_length > 1000:
            self.episode_length = 0
            self.max_x = -np.inf
            self.create_and_send_message("[Cell Name]:Seed")
            self.reset()
        if self.customSideChannel.levelEnd:
            self.handle_level_end()

    def reset(self, **kwargs):
        state = super().reset()
        self.cumulative_reward = 0
        return self.construct_state(state[1], state[0])

    def construct_state(self, state, grid):
        one_hot = self.one_hot_encode(grid, 7)
        flattened_matrix_obs = [vector for sublist in one_hot for item in sublist for vector in item]
        combined_observations = list(state[2:]) + list(flattened_matrix_obs)
        return combined_observations

    def step(self, action):
        transformed_action = (action[0] - 1, action[1])
        state, env_score, arousal, d, info = super().step(transformed_action)
        position = state[1][0]
        state = self.construct_state(state[1], state[0])
        self.calculate_reward(state, position)
        self.cumulative_reward += self.current_reward
        self.best_reward = np.max([self.best_reward, self.current_reward])
        self.reset_condition()
        final_reward = self.current_reward * (1 - self.weight) + (arousal * self.weight)
        return state, final_reward, d, info

    def handle_level_end(self):
        print("End of level reached, resetting environment.")
        self.create_and_send_message("[Cell Name]:Seed")
        self.reset()
        self.customSideChannel.levelEnd = False


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(f"Incorrect number of arguments specified, was expecting 7, found {len(sys.argv)}")
        run = 1
        target_name = "Maximize"
        target_signal_str = "np.ones"
        weight = 0
    else:
        run = int(sys.argv[1])
        weight = float(sys.argv[2])
        print(f"Run {run}")
        target_name = "Maximize"
        target_signal_str = "np.ones"

    function_mapping = {
        'np.ones': np.ones,
        'np.zeros': np.zeros,
        'imitate': None
    }

    preference_task = True
    classification_task = False
    cluster = 0

    vector_length = (852,)
    model = KNNSurrogateModel(5, classification_task, preference_task, cluster, "Pirates")
    env = PPO_Environment(run, graphics=True,
                          obs_space={"low": -np.inf, "high": np.inf, "shape": vector_length},
                          path="./Builds/Pirates/platform.exe", arousal_model=model, weight=weight)
    sideChannel = env.customSideChannel
    model = PPO("MlpPolicy", env=env, tensorboard_log="./Tensorboard", device='cpu')
    model.learn(total_timesteps=1000000, progress_bar=True, callback=TensorboardCallback(), tb_log_name="PPO")

    if weight == 0:
        label = 'optimize'
    elif weight == 0.5:
        label = 'blended'
    else:
        label = 'arousal'
    model.save(f"ppo_solid_{label}_{run}")

