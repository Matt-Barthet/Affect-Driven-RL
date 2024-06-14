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

        """ ---- Heist! specific code ---- """
        self.gridWidth = 9
        self.gridHeight = 9
        self.elementSize = 0.5
        self.position_dict = {}
        self.death_applied = False
        self.previous_health = 0

        self.current_ammo = 0
        self.current_health = 0
        self.previous_angle, self.current_angle = 0, 0

        super().__init__(id_number=id_number, graphics=graphics, obs_space=obs_space, path=path,
                         args=["-agentGridWidth", f"{self.gridWidth}", "-agentGridHeight", f"{self.gridHeight}", "-cellSize",
                         f"{self.elementSize}"], capture_fps=15, time_scale=1, arousal_model=arousal_model, weight=weight)

    def round_to_nearest_5(self, lst):
        rounded_list = [round(x / 3) * 3 for x in lst]
        return rounded_list

    def calculate_reward(self, state, position, action):

        self.current_reward = self.current_score - self.previous_score

        position = self.round_to_nearest_5(position)
        position = f"{position[0]},{position[1]},{position[2]}"
        if position not in self.position_dict:
            self.position_dict.update({position: 0})
            self.current_reward += 1

        # Penalty for taking damage

        self.current_angle = np.abs(state[-5])

        if self.current_angle == 0:
            self.current_reward -= 1
        elif self.previous_angle > self.current_angle:
            self.current_reward += 0.1
        elif self.previous_angle < self.current_angle:
            self.current_reward += -0.1

        if self.current_angle == 0:
            pass
        # elif self.current_angle < 20:
        #     self.current_reward += 5
        elif self.current_angle < 90:
            self.current_reward += (180 - self.current_angle) / 180
        elif self.current_angle < 180:
            self.current_reward += (180 - self.current_angle) / 180
            self.current_reward -= 0.5

        self.previous_angle = self.current_angle
        self.previous_health = self.current_health

    def reset_condition(self):
        self.episode_length += 1
        if self.episode_length > 300 * 4:
            self.episode_length = 0
            self.create_and_send_message("[Cell Name]:Seed")
            self.reset()
        if self.customSideChannel.levelEnd:
            self.handle_level_end()

    def reset(self, **kwargs):
        state = super().reset()
        self.cumulative_reward = 0
        self.previous_score = 0
        self.position_dict.clear()
        return self.construct_state(state[0][0], np.asarray(state[1][0]))

    def construct_state(self, state, grid):
        one_hot = self.one_hot_encode(grid, 4)
        # print(one_hot.shape)
        flattened_matrix_obs = [vector for sublist in one_hot for item in sublist for vector in item]
        combined_observations = list(flattened_matrix_obs) + list(state[3:])
        return combined_observations

    def step(self, action):
        # print(action)
        transformed_action = [
            # np.round(action[0]/2 + 0.5),
            # np.round(action[1]/2 + 0.5),
            action[0] * 4,
            action[1] * 2,
            np.round(action[2]+1),
            np.round(action[3]+1),
            np.round(action[4]/2 + 0.5),
            # np.round(action[7]/2 + 0.5)
        ]

        # print(transformed_action[2])
        state, env_score, arousal, d, info = super().step(transformed_action)
        position = state[0][0][:3]

        self.current_health = state[0][0][3]
        self.current_ammo = state[0][0][1]

        state = self.construct_state(state[0][0], state[1][0])
        self.calculate_reward(state, position, transformed_action)
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

    vector_length = (341,)
    model = KNNSurrogateModel(5, classification_task, preference_task, cluster, "Pirates")
    env = PPO_Environment(run, graphics=True,
                          obs_space={"low": -np.inf, "high": np.inf, "shape": vector_length},
                          path="./Builds/Heist_GoBlend/Top-Down Shooter.exe", arousal_model=model, weight=weight)


    sideChannel = env.customSideChannel
    model = PPO("MlpPolicy", env=env, tensorboard_log="./Tensorboard", device='cpu')
    model.learn(total_timesteps=1000000, progress_bar=True, callback=TensorboardCallback(), tb_log_name="PPO")

    if weight == 0:
        label = 'optimize'
    elif weight == 0.5:
        label = 'blended'
    else:
        label = 'arousal'
    model.save(f"ppo_heist_{label}_{run}")

