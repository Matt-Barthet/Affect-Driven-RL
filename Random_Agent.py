from abc import ABC

import pandas as pd
import numpy as np
import gym
from mlagents_envs.side_channel import OutgoingMessage
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityEnvironmentException

from BaseEnvironment import BaseEnvironment
from Solid_PPOEnvironment import PPO_Environment
from SurrogateModel import KNNSurrogateModel

if __name__ == "__main__":

    preference_task = True
    classification_task = False
    cluster = 0
    weight = 0.5
    vector_length = (49,)
    model = KNNSurrogateModel(5, classification_task, preference_task, cluster)

    env = PPO_Environment(0, graphics=True,
                          obs_space={"low": -np.inf, "high": np.inf, "shape": vector_length},
                          path="./Builds/SolidRallyGoBlend_Mac.app", arousal_model=model, weight=0.0, capture_fps=60)
    sideChannel = env.customSideChannel

    model = PPO("MlpPolicy", env=env, tensorboard_log="./Tensorboard", device='cpu')
    model.load("Affectively Experiments/ppo_solid_0.5W_1.zip")

    state = env.reset()
    for i in range(5000):
        action, _ = model.predict(state)
        state, reward, done, info = env.step(action)
        if done:
            state = env.reset()

    # env.step(np.asarray([tuple([0, 0])]))
    env.close()