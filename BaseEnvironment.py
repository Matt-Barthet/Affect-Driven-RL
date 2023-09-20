from abc import ABC

import numpy as np
import gym
from mlagents_envs.side_channel import OutgoingMessage
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from Utils.SideChannels import MySideChannel
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityEnvironmentException


class BaseEnvironment(gym.Env, ABC):
    """
    This is the base unity-gym environment that all environments should inherit from. It sets up the
    unty-gym wrapper, configures the game engine parameters and sets up the custom side channel for
    communicating between our python scripts and unity's update loop.
    """
    def __init__(self, id_number, graphics, scaler, obs_space, include_affect):

        super(BaseEnvironment, self).__init__()

        # Set up the game engine communication channels
        self.engineConfigChannel = EngineConfigurationChannel()
        self.engineConfigChannel.set_configuration_parameters(capture_frame_rate=10)
        self.customSideChannel = MySideChannel()

        # Load the unity build and wrap it in a gym environment
        self.env = self.load_environment(id_number, graphics)
        self.env = UnityToGymWrapper(self.env, uint8_visual=False, allow_multiple_obs=True)

        # Set observation space and action space - important for learning
        self.action_space = self.env.action_space
        self.action_size = self.env.action_size
        self.observation_space = gym.spaces.Box(low=obs_space['low'], high=obs_space['high'], shape=obs_space['shape'])

        # Use the side channel to set whether affect values should be generated
        self.create_and_send_message("[Generate Arousal]:{}".format(include_affect))

        # Important learning variables that should be used in all environments
        self.score = 0 # In-game score
        self.max_score = 0
        self.steps = 0
        self.scaler = scaler # For re-scaling the state space

    def step(self, action):
        # Move the env forward 1 tick and receive messages through side-channel.
        state, env_score, d, info = self.env.step(np.asarray([tuple([action[0] - 1, action[1] - 1])]))
        self.score = env_score
        self.max_score = np.max([self.score, self.max_score])
        self.scaler.update(state)
        return state, env_score, d, info

    @staticmethod
    def tuple_to_vector(s):
        obs = []
        for i in range(len(s[0])):
            obs.append(s[0][i])
        return obs

    def reset(self):
        self.steps = 0
        return self.tuple_to_vector(self.env.reset())

    def create_and_send_message(self, contents):
        message = OutgoingMessage()
        message.write_string(contents)
        self.customSideChannel.queue_message_to_send(message)

    def load_environment(self, path, identifier, graphics):
        try:
            env = UnityEnvironment(f"../Builds/{path}",
                                   side_channels=[self.engineConfigChannel, self.customSideChannel],
                                   worker_id=identifier,
                                   no_graphics=not graphics)
        except UnityEnvironmentException:
            print("Path not found! Please specify the right environment path.")
            return None
        return env
