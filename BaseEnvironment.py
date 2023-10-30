from abc import ABC

import mlagents_envs
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
    def __init__(self, id_number, graphics, scaler, obs_space, path, args=None):

        super(BaseEnvironment, self).__init__()

        # Set up the game engine communication channels
        if args is None:
            args = [""]
        self.engineConfigChannel = EngineConfigurationChannel()
        self.engineConfigChannel.set_configuration_parameters(capture_frame_rate=360, time_scale=1)
        self.customSideChannel = MySideChannel()

        # Load the unity build and wrap it in a gym environment
        self.env = self.load_environment(path, id_number, graphics, args)
        self.env = UnityToGymWrapper(self.env, uint8_visual=False, allow_multiple_obs=True)

        # Set observation space and action space - important for learning
        self.action_space = self.env.action_space
        self.action_size = self.env.action_size

        self.observation_space = gym.spaces.Box(low=obs_space['low'], high=obs_space['high'], shape=obs_space['shape'])

        # Important learning variables that should be used in all environments
        self.score = 0 # In-game score
        self.max_score = 0
        self.steps = 0
        self.scaler = scaler # For re-scaling the state space

    def step(self, action):
        # Move the env forward 1 tick and receive messages through side-channel.
        state, env_score, d, info = self.env.step(np.asarray([tuple([action[0]-1, action[1]])]))
        self.score = env_score
        self.max_score = np.max([self.score, self.max_score])

        if self.scaler is not None:
            self.scaler.update(state)

        if self.customSideChannel.levelEnd:
            self.handle_level_end()

        return self.tuple_to_vector(state), env_score, d, info

    @staticmethod
    def tuple_to_vector(s):
        obs = []
        for i in range(len(s[0])):
            obs.append(s[0][i])
        return obs

    def construct_state(self, vector_obs, matrix_obs):
        one_hot = self.one_hot_encode(matrix_obs, 7)
        flattened_matrix_obs = [vector for sublist in one_hot for item in sublist for vector in item]
        combined_observations = list(vector_obs[2:]) + list(flattened_matrix_obs)
        return combined_observations

    @staticmethod
    def one_hot_encode(matrix_obs, num_categories):
        one_hot_encoded = np.zeros((matrix_obs.shape[0], matrix_obs.shape[1], num_categories))
        for i in range(matrix_obs.shape[0]):
            for j in range(matrix_obs.shape[1]):
                one_hot_encoded[i, j, int(matrix_obs[i][j])-1] = 1
        return one_hot_encoded

    def reset(self):
        self.steps = 0
        state = self.env.reset()
        position = state[1][0]
        if len(state) == 2:
            state = self.construct_state(state[1], state[0])
        return state, position

    def create_and_send_message(self, contents):
        message = OutgoingMessage()
        message.write_string(contents)
        self.customSideChannel.queue_message_to_send(message)

    def load_environment(self, path, identifier, graphics, args):
        try:
            env = UnityEnvironment(f"{path}",
                                   side_channels=[self.engineConfigChannel, self.customSideChannel],
                                   worker_id=identifier,
                                   no_graphics=not graphics,
                                   additional_args=args)
        except UnityEnvironmentException:
            print("Path not found! Please specify the right environment path.")
            return None
        except:
            return self.load_environment(path, identifier + 1, graphics, args)
        return env

    def handle_level_end(self):
        pass
