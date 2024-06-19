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
from Evaluate_Solid import compute_confidence_interval
from Hiest_PPOEnvironment import PPO_Environment
from SurrogateModel import KNNSurrogateModel

if __name__ == "__main__":

    preference_task = True
    classification_task = False
    cluster = 0
    weight = 0
    vector_length = (341,)
    model = KNNSurrogateModel(5, classification_task, preference_task, cluster, 'fps')

    env = PPO_Environment(9, graphics=True,
                          obs_space={"low": -np.inf, "high": np.inf, "shape": vector_length},
                          path="./Builds/Heist_GoBlend/Top-Down Shooter.exe", arousal_model=model, weight=weight)

    sideChannel = env.customSideChannel
    model = PPO("MlpPolicy", env=env, tensorboard_log="./Tensorboard", device='cpu')
    model.load("./Affectively Experiments/Heist/ppo_heist_optimize_1.zip", env=env, force_reset=True)
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
