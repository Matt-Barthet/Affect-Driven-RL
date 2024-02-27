from time import sleep
import numpy as np
from abc import ABC

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from PCGRL.PCGUtility import construct_state
from Solid_PPOEnvironment import load_model, load_scaler
from Utils.Tensorboard_Callbacks import TensorboardCallback, PPOHackCallback
from BaseEnvironment import BaseEnvironment
from PCGRL.Dijkstra import dijkstra_with_path
from hashlib import md5
import json

from custom_learn_ppo import *

class PCGEnvironment(BaseEnvironment, ABC):

    def __init__(self, id_number, graphics, path, shape, model, scaler, dont_simulate=False, name=""):

        self.reset_dijkstra = False
        self.arousal_list = []
        self.path = []
        self.action_string = None
        self.fixed_grid = []
        self.agent_width = 19
        self.agent_height = 19
        self.cell_size = 40
        self.grid_width = 100
        self.grid_height = 100
        self.name = name

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

        self.previousSurrogate = []
        self.currentSurrogate = []
        self.previousTensor = None
        self.arousal_model = model
        self.scaler = scaler

        self.flush_buffer = False
        self.increase, self.decrease = 0, 0
        self.dont_simulate = dont_simulate

        self.tracks = {}
        self.tracks_per_episode = 10
        self.tracks_this_episode = 0
        self.success = False

        self.previousActions, self.previousArousals = [], []

    def reset(self, **kwargs):
        # print("RESETTING")
        state = self.env.reset()
        self.score = 0
        self.cumulative_reward = 0
        self.track_length = 0
        self.customSideChannel.race_ended = False
        new_state, self.fixed_grid, self.current_index, self.facing = construct_state(state, self.one_hot_encode)
        self.previousSurrogate = None
        self.currentSurrogate = None
        return self.tuple_to_vector(new_state)

    def send_load_message(self):
        self.action_string = "[Load]:"
        self.action_string += ','.join(f"{action}" for action in self.action_list)
        self.create_and_send_message(self.action_string)

    def reset_to_state(self, actions):
        self.env.reset()
        for action in actions:
            next_state, reward, done, _ = self.env.step(action)  # Execute the action
            self.current_state, self.grid, self.current_index, self.facing = construct_state(next_state, self.one_hot_encode)

    def simulate_race(self):

        if self.dont_simulate:
            return

        counter = 0
        while not self.customSideChannel.race_ended:

            if counter >= 10000:
                break

            self.previousSurrogate = []
            if self.currentSurrogate is not None:
                self.previousSurrogate = np.array(self.currentSurrogate.copy(), dtype=np.float32)

            for i in range(50):
                _ = self.env.step(-1)
                counter += 1
                if i == 48:
                    self.create_and_send_message("Send Vector")

            self.currentSurrogate = np.array(self.customSideChannel.arousal_vector.copy(), dtype=np.float32)

            if len(self.previousSurrogate) == 0:
                self.previousSurrogate = np.zeros((31,))

            if len(self.currentSurrogate) == 0:
                self.currentSurrogate = np.zeros((31,))

            score = np.ceil(self.currentSurrogate[1])
            combined_scaler = np.concatenate((self.previousSurrogate.copy(), self.currentSurrogate.copy()))
            combined_scaler = self.scaler.transform(combined_scaler.reshape(1, -1))
            combined_scaler = np.clip(combined_scaler, 0, 1)
            tensor = torch.Tensor(combined_scaler)
            self.previousTensor = tensor
            arousal = self.arousal_model(tensor).detach().numpy()
            self.arousal_list[int(score)].append(arousal[0])
            self.previousSurrogate = self.currentSurrogate.copy()

            if arousal < 0.45:
                self.decrease += 1
            elif arousal > 0.55:
                self.increase += 1

    def step(self, action):

        new_state = None # Avoid runtime errors.

        # If the target circuit size is reached, switch to Dijkstra mode.
        if len(self.action_list) > 10 and not self.dont_simulate:
            if len(self.path) == 0:
                _, self.path = dijkstra_with_path(self.fixed_grid,
                                              (self.current_index[1], self.current_index[0]),
                                              (49, 50))
                self.path.append((50, 50))
                self.reset_dijkstra = True
                self.path = self.path[1:]

            previous_len = len(self.action_list)
            if len(self.path) > 0:
                previous_len = len(self.action_list)
                for new_action in range(5):
                    if self.reset_dijkstra:
                        self.reset_to_state(self.action_list.copy())
                    next_state, reward, done, info = self.env.step(new_action)
                    new_state, self.fixed_grid, self.current_index, self.facing = construct_state(next_state, self.one_hot_encode)
                    grid = np.array(self.fixed_grid.copy())
                    for element in self.path:
                        grid[int(element[0])][int(element[1])] = -1
                    if (self.current_index[1], self.current_index[0]) == self.path[0]:
                        self.action_list.append(new_action)
                        self.path = self.path[1:]
                        self.reset_dijkstra = False

                        if len(self.path) == 0:
                            self.create_and_send_message(f"Screenshot:{self.name} Track {len(self.tracks)}.png")
                        break
                    self.reset_dijkstra = True

            # If no change is made after trying to build the Dijkstra path, we have a bogus track.
            if len(self.action_list) == previous_len:
                self.reset()
                self.path.clear()
                self.action_list.clear()
                self.arousal_list.clear()

            # If the Dijkstra path is fully built (i.e. the circuit is closed) simulate a race lap.
            elif len(self.path) == 0:
                self.tracks_this_episode += 1
                if self.tracks_this_episode == self.tracks_per_episode:
                    done = True
                    self.tracks_this_episode = 0

                self.success = True
                self.arousal_list = [[] for _ in range(len(self.action_list)+3)]
                self.simulate_race()

                self.previousActions = self.action_list.copy()
                self.previousArousals = self.arousal_list.copy()

                self.tracks[f"Track {len(self.tracks)}"] = {}
                self.tracks[f"Track {len(self.tracks)-1}"]['Actions'] = self.action_list.copy()
                self.tracks[f"Track {len(self.tracks)-1}"]['Arousals'] = self.arousal_list.copy()
                np.save(f"./{self.name} Tracks.npy", self.tracks)

        # If we have not reached the target track length, keep using PPO to build a track
        else:

            if len(self.action_list) > 1:
                self.success = False

            if self.current_index == (49, 50) and not self.dont_simulate:
                self.reset()

            self.action_list.append(action)
            state, env_score, done, info = self.env.step(action)
            new_state, self.fixed_grid, self.current_index, self.facing = construct_state(state, self.one_hot_encode)

            self.score += env_score
            self.reward = env_score
            self.track_length += self.reward
            self.cumulative_reward += self.reward
            self.max_score = self.score if self.score > self.max_score else self.max_score
            self.max_reward = self.reward if self.score > self.max_reward else self.max_reward

        # If we get a collision flag from unit, reset the environment
        if self.customSideChannel.collision:
            self.customSideChannel.collision = False
            self.action_list.clear()
            self.arousal_list.clear()
            self.reset()

        # If we get an error step, reset the environment to avoid runtime errors.
        if new_state is None:
            state = env.step(-1)
            self.customSideChannel.collision = True
            return state

        return new_state, -1, done, info


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=6)  # precision is optional

    input_size = 62  # Update this with the input size
    hidden_size1 = 16
    hidden_size2 = 2
    output_size = 1

    model_path = 'best_model_scaled.pth'
    scaler_path = 'best_scaler_global.pkl'

    scaler = load_scaler(scaler_path)
    arousal_model = load_model(model_path, input_size, hidden_size1, hidden_size2, output_size)

    vector_length = (2527,)

    experiment_name = "Maximize Arousal Long Tracks"

    env = PCGEnvironment(1, graphics=True, path="./Builds/SolidRallyPCGRL/Racing.exe",
                         shape=vector_length, model=arousal_model, scaler=scaler, dont_simulate=False, name=experiment_name)
    model = CustomPPO("MlpPolicy", env=env, tensorboard_log="./Tensorboard", learning_rate=0.0001)

    callback = PPOHackCallback(None, env)

    model.custom_learn(total_timesteps=50000, progress_bar=True, callback=callback, tb_log_name=f"PCGRL {experiment_name}")
    model.save(f"ppo {experiment_name} pcg")

    # tracks = {}
    # tracks = dict(np.load("./Random_Tracks.npy", allow_pickle=True).item())
    # print(tracks)
    # # Visualize tracks
    # for name, track in list(tracks.items()):
    #     env.action_list.clear()
    #     _ = env.reset()
    #     actions = track['Actions']
    #     for action in range(len(actions)):
    #         if action == len(actions)-1:
    #             env.create_and_send_message(f"Screenshot:{name}.png")
    #         _ = env.step(actions[action])
    #
    # exit()
    #
    # # Build Random Tracks
    # for i in range(len(tracks), 20):
    #     tracks[f"Track {i}"] = {}
    #     env.action_list.clear()
    #     _ = env.reset()
    #     done = False
    #     while not done:
    #         _, _, done, _ = env.step(env.action_space.sample())
    #     tracks[f"Track {i}"]['Actions'] = env.action_list.copy()
    #     tracks[f"Track {i}"]['Arousals'] = env.arousal_list.copy()
    #     print(tracks[f"Track {i}"])
    #
    #     np.save("./Random_Tracks.npy", tracks)

