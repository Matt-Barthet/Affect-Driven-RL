from abc import ABC
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from Utils.Tensorboard_Callbacks import TensorboardCallback
from BaseEnvironment import BaseEnvironment
import numpy as np
import pickle
from Utils.Scalers import VectorScaler


class PPO_Environment(BaseEnvironment, ABC):

    def __init__(self, id_number, graphics, scaler, obs_space, path):
        super().__init__(id_number, graphics, scaler, obs_space, path)

        self.max_reward = 0
        self.cumulative_reward = 0
        self.reward = 0
        self.episode_length = 0
        self.last_x = np.round(self.reset()[0])
        self.max_x = -np.inf
        self.death_applied = False

    def calculate_reward(self, state, env_score, position):
        current_x = np.round(position)
        health = state[3]
        reward = 0
        if health > 0:
            self.death_applied = False
        if health == 0 and not self.death_applied:
            reward -= 5
            self.death_applied = True
        # elif current_x > self.last_x:
            # reward += 1
        self.last_x = current_x
        self.reward = (env_score - self.score) + reward
        self.score = env_score

    def reset_condition(self):
        self.episode_length += 1
        if self.episode_length > 4 * 140:
            self.episode_length = 0
            self.max_x = -np.inf
            self.create_and_send_message("[Cell Name]:Seed")
            self.reset()

    def reset(self, **kwargs):
        state, position = super().reset()
        self.last_x = position
        self.score = 0
        self.cumulative_reward = 0
        return state

    def update_stats(self):
        self.cumulative_reward += self.reward
        self.max_score = np.max([self.score, self.max_score])
        self.max_reward = np.max([self.max_reward, self.reward])

    def step(self, action):
        # Move the env forward 1 tick and receive messages through side-channel.
        state, env_score, d, info = self.env.step((action[0] - 1, action[1]))
        position = state[1][0]
        state = self.construct_state(state[1], state[0])
        self.calculate_reward(state, env_score, position)
        self.update_stats()
        self.reset_condition()
        return state, self.reward, d, info

    def handle_level_end(self):
        print("End of level reached, resetting environment.")
        self.reset()


if __name__ == "__main__":

    load_scaler = False
    if load_scaler:
        with open('../Models_Pkls/MinMaxScaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    else:
        scaler = VectorScaler(49)

    vector_length = (3092,)
    env = DummyVecEnv([lambda: PPO_Environment(counter, graphics=True, scaler=None,
                                               obs_space={"low": -np.inf, "high": np.inf, "shape": vector_length},
                                               path="./Builds/Platformer.app") for counter in [0]])
    sideChannel = env.envs[0].customSideChannel
    model = PPO("MlpPolicy", env=env, tensorboard_log="./Tensorboard")
    model.learn(total_timesteps=1500000, progress_bar=True, callback=TensorboardCallback(), tb_log_name="PPO")
    model.save("ppo_solid_test")

    with open("../Models_Pkls/MinMaxScaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
