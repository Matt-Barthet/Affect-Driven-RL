from abc import ABC

import joblib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from Utils.Tensorboard_Callbacks import TensorboardCallback
from BaseEnvironment import BaseEnvironment
import numpy as np

import torch
from torch import nn
import sklearn


class RankNet(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(RankNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


def load_model(model_path, input_size, hidden_size1, hidden_size2, output_size):
    model = RankNet(input_size, hidden_size1, hidden_size2, output_size).to('cuda:0')
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to inference mode
    return model


class PPO_Environment(BaseEnvironment, ABC):

    def __init__(self, id_number, graphics, obs_space, path, arousal_model, scaler, args=None):
        super().__init__(id_number, graphics, obs_space, path, args)
        self.episode_length = 0
        self.reward = 0
        self.max_reward = 0
        self.cumulative_reward = 0
        self.model = arousal_model
        self.scaler = scaler
        self.previousSurrogate = []
        self.currentSurrogate = []
        self.previousTensor = None

    def calculate_reward(self, state):
        rotation_component = (180 - state[-1]) / 180
        speed_component = np.linalg.norm([state[0], state[1], state[2]]) / 60
        distance_component = (1 + state[-2])
        self.reward = rotation_component * speed_component / distance_component

    def reset_condition(self):
        self.episode_length += 1
        if self.episode_length > 200:
            self.episode_length = 0
            self.reset()

    def update_stats(self):
        self.max_score = np.max([self.score, self.max_score])
        self.max_reward = np.max([self.max_reward, self.reward])

    def reset(self, **kwargs):
        self.currentSurrogate = np.array(self.customSideChannel.arousal_vector.copy())
        return super().reset()

    def step(self, action):

        # Move the env forward 1 tick and receive messages through side-channel.
        self.previousSurrogate = np.array(self.currentSurrogate.copy(), dtype=np.float32)

        state, env_score, d, info = self.env.step(np.asarray([tuple([action[0] - 1, action[1] - 1])]))
        state = self.tuple_to_vector(state[0])
        self.score = env_score
        self.calculate_reward(state)
        self.update_stats()
        self.reset_condition()

        self.currentSurrogate = np.array(self.customSideChannel.arousal_vector.copy(), dtype=np.float32)

        combined_scaler = np.concatenate((self.previousSurrogate.copy(), self.currentSurrogate.copy()))
        # combined_scaler = self.scaler.transform(combined_scaler.reshape(1, -1))

        tensor = torch.Tensor(combined_scaler)
        self.previousTensor = tensor
        arousal = self.model(tensor)
        self.previousSurrogate = self.currentSurrogate.copy()
        return state, self.reward, d, info


def load_scaler(scaler_path):
    scaler = joblib.load(scaler_path)
    return scaler


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=6)  # precision is optional

    input_size = 62  # Update this with the input size
    hidden_size1 = 16
    hidden_size2 = 2
    output_size = 2

    model_path = 'best_model_scaled.pth'
    scaler_path = 'Builds/SolidRallyPCGRL/best_scaler_global.pkl'

    scaler = load_scaler(scaler_path)
    model = load_model(model_path, input_size, hidden_size1, hidden_size2, output_size)

    env = DummyVecEnv([lambda: PPO_Environment(counter,
                                               graphics=True,
                                               obs_space={"low": -np.inf, "high": np.inf, "shape": (49,)},
                                               path="./Builds/Racing.exe",
                                               arousal_model=model,
                                               scaler=scaler) for counter in [1]],)
    sideChannel = env.envs[0].customSideChannel

    model = PPO("MlpPolicy", env=env, tensorboard_log="../Tensorboard")
    model.learn(total_timesteps=1500000, progress_bar=True, callback=TensorboardCallback(), tb_log_name="PPO")
    model.save("ppo_solid_test")
