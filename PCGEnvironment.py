from time import sleep
import numpy as np
from abc import ABC
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from PCGRL.PCGUtility import construct_state
from Utils.Tensorboard_Callbacks import TensorboardCallback
from BaseEnvironment import BaseEnvironment
from PCGRL.ReachabilityAStar import AStar
from hashlib import md5
import json


class PCGEnvironment(BaseEnvironment, ABC):

    def __init__(self, id_number, graphics, path, shape):

        self.path = None
        self.action_string = None
        self.fixed_grid = []
        self.agent_width = 9
        self.agent_height = 9
        self.cell_size = 40
        self.grid_width = 100
        self.grid_height = 100

        # Arguments for the state representation of the designer agent / track grid
        self.args = ["-agentGridHeight", f"{self.agent_height}",
                     "-agentGridWidth", f"{self.agent_width}",
                     "-gridHeight", f"{self.grid_height}",
                     "-gridWidth", f"{self.grid_width}",
                     "-cellSize", f"{self.cell_size}"]

        obs_space = {"low": -np.inf, "high": np.inf, "shape": shape}
        super().__init__(id_number, graphics, obs_space, path, args=self.args)
        self.max_reward = 0
        self.cumulative_reward = 0
        self.reward = 0
        self.episode_length = 0
        self.action_list = []

        self.current_index = ()
        self.facing = ()
        self.close_circuit = False

        self.archive = {}
        self.track_length = 0
        self.json_text = ""

    def reset(self, **kwargs):
        state = self.env.reset()
        self.score = 0
        self.cumulative_reward = 0
        self.track_length = 0
        self.action_list.clear()
        new_state, self.fixed_grid, self.current_index, self.facing = construct_state(state, self.one_hot_encode)
        return self.tuple_to_vector(new_state)

    def update_stats(self):
        self.track_length += self.reward
        self.cumulative_reward += self.reward
        self.max_score = np.max([self.score, self.max_score])
        self.max_reward = np.max([self.max_reward, self.cumulative_reward])

    def send_load_message(self):
        self.action_string = "[Load]:"
        self.action_string += ','.join(f"{action}" for action in self.action_list)
        self.create_and_send_message(self.action_string)

    def step(self, action):

        # self.create_and_send_message("[Spawn]:49,50") # Spawn placeholder to sanity check coordinate
        """path = self.reachability_solver.a_star_search(self.fixed_grid, (self.current_index[0] + (self.facing[0]), self.current_index[1] + (self.facing[1])), (49, 50) )
        message = f"[Place]:"
        for element in path:
            message += f"{int(element[0])},{int(element[1])}-{2}/"
        self.create_and_send_message(message[:-1])"""
        if self.current_index == (50, 49):
            self.reset() # we are blocking the goal for closing the circuit

        if self.close_circuit:
            state, env_score, d, info = self.env.step(action)
            new_state, self.fixed_grid, self.current_index, self.facing = construct_state(state, self.one_hot_encode)
            print(self.path)
            sleep(100)
        else:
            self.action_list.append(action)
            state, env_score, d, info = self.env.step(action)
            new_state, self.fixed_grid, self.current_index, self.facing = construct_state(state, self.one_hot_encode)
            self.score += env_score
            self.reward = env_score
            self.update_stats()
            # sleep(1)

        if self.max_reward > 20:
            # self.create_and_send_message(f"[Load]:{self.action_string}")
            self.close_circuit = True
            a_star = AStar((self.current_index[1] + (self.facing[1]), self.current_index[0] + (self.facing[0]), self.action_list), (50, 49), self)
            self.path = a_star.a_star_search()
            """for action in self.path:
                self.create_and_send_message(f"[Spawn]: {action[1]}, {action[0]}")  # Spawn placeholder to sanity check coordinate"""

        if self.customSideChannel.collision:
            self.customSideChannel.collision = False
            self.reset()

        return new_state, self.cumulative_reward, d, info


if __name__ == "__main__":
    vector_length = (567,)
    env = DummyVecEnv([lambda: PCGEnvironment(counter, graphics=True, path="./Builds/SolidRallyPCGRL/SolidRallyPCG.exe", shape=vector_length) for counter in [0]])
    model = PPO("MlpPolicy", env=env, tensorboard_log="./Tensorboard")
    model.learn(total_timesteps=1500000, progress_bar=True, callback=TensorboardCallback(), tb_log_name="PCGRL_PPO")
    model.save("ppo_solid_test")
