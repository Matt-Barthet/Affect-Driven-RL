from abc import ABC
import numpy as np
from BaseEnvironment import BaseEnvironment

from abc import ABC
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from Utils.Tensorboard_Callbacks import TensorboardCallback
from BaseEnvironment import BaseEnvironment
import numpy as np



class PCGEnvironment(BaseEnvironment, ABC):

    def __init__(self, id_number, graphics, scaler, path,):
        obs_space = {"low": -np.inf, "high": np.inf, "shape": (415, )}
        super().__init__(id_number, graphics, scaler, obs_space, path)
        
        self.max_reward = 0
        self.cumulative_reward = 0
        self.reward = 0
        self.episode_length = 0

    def reset(self):
        state = self.env.reset()
        self.score = 0
        self.cumulative_reward = 0
        new_state = self.construct_state(state[1], state[0])
        return new_state

    def update_stats(self):
        self.cumulative_reward += self.reward
        self.max_score = np.max([self.score, self.max_score])
        self.max_reward = np.max([self.max_reward, self.cumulative_reward])

    def step(self, action):
        state, env_score, d, info = self.env.step(action)
        new_state = self.construct_state(state[1], state[0])
        self.score += env_score
        self.reward = env_score
        self.update_stats()
        if self.customSideChannel.collision:
            self.customSideChannel.collision = False
            self.reset()
        return new_state, self.cumulative_reward, d, info


if __name__ == "__main__":
    vector_length = (415,)
    env = DummyVecEnv([lambda: PCGEnvironment(counter, graphics=True, scaler=None, path="./Builds/SolidRallyPCG.exe") for counter in [0]])
    model = PPO("MlpPolicy", env=env, tensorboard_log="./Tensorboard")
    model.learn(total_timesteps=1500000, progress_bar=True, callback=TensorboardCallback(), tb_log_name="PPO")
    model.save("ppo_solid_test")
