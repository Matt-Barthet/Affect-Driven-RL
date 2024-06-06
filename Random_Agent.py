from abc import ABC

import pandas as pd
import numpy as np
import gym
from mlagents_envs.side_channel import OutgoingMessage
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityEnvironmentException

from BaseEnvironment import BaseEnvironment

if __name__ == "__main__":

    env = DummyVecEnv([lambda: BaseEnvironment(counter, graphics=True,
                                               scaler=None,
                                               include_affect=False,
                                               obs_space={"low": -np.inf, "high": np.inf, "shape": (13,)},
                                               path="./Builds/Platformer.app") for counter in [1]])
    sideChannel = env.envs[0].customSideChannel

    env.reset()
    for i in range(5000):
        actions = env.action_space.sample()
        print(actions)
        env.step(np.asarray([tuple([actions[0], actions[1]])]))

    # env.step(np.asarray([tuple([0, 0])]))
    env.close()