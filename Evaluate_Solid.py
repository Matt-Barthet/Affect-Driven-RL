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
from scipy import stats


def compute_confidence_interval(data, confidence=0.95):
    data = np.array(data)  # Ensure the input data is a numpy array
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    ci = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)  # Margin of error
    return np.round(mean, 4), np.round(ci, 4)


if __name__ == "__main__":

    preference_task = True
    classification_task = False
    cluster = 0
    weight = 0
    vector_length = (50,)
    model = KNNSurrogateModel(5, classification_task, preference_task, cluster, 'Solid')

    env = PPO_Environment(0, graphics=True,
                          obs_space={"low": -np.inf, "high": np.inf, "shape": vector_length},
                          path="./Builds/SolidRallyGoBlend_Mac.app", arousal_model=model, weight=weight)
    sideChannel = env.customSideChannel

    model = PPO("MlpPolicy", env=env, device='cpu')
    model.load("./Affectively Experiments/Solid Rally/ppo_solid_optimize_1.zip", env=env, force_reset=True)
    model.set_parameters("./Affectively Experiments/Solid Rally/ppo_solid_optimize_1.zip")

    state = env.reset()
    arousals, scores = [], []
    for i in range(30):
        for _ in range(600):
            # action, _ = model.predict(state, deterministic=False)
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if done:
                state = env.reset()

        arousals.append(np.mean(env.arousal_trace))
        scores.append(env.best_score)
        env.best_score = 0
        env.arousal_trace.clear()

    print(f"Best Score: {compute_confidence_interval(scores)}, Mean Arousal: {compute_confidence_interval(arousals)}")
    env.close()
