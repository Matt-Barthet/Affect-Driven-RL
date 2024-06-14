import sys
from abc import ABC
import numpy as np
import torch
from stable_baselines3 import PPO
from BaseEnvironment import BaseEnvironment
from SurrogateModel import KNNSurrogateModel
from Utils.Tensorboard_Callbacks import TensorboardCallback


class PPO_Environment(BaseEnvironment, ABC):

    def __init__(self, id_number, graphics, obs_space, path, arousal_model, weight, args=None):
        super().__init__(id_number=id_number, graphics=graphics, obs_space=obs_space, path=path, args=args, capture_fps=5, time_scale=1, arousal_model=arousal_model, weight=weight)

    def calculate_reward(self, state):
        rotation_component = 1 if (180 - state[-1]) / 180 < 90 else -1
        speed_component = np.linalg.norm([state[0], state[1], state[2]]) / 80
        self.current_reward = (self.current_score - self.previous_score) + rotation_component * speed_component
        self.current_reward /= 2
        # print(self.current_reward)

    def reset_condition(self):
        if self.episode_length > 600:
            self.episode_length = 0
            self.reset()

    def reset(self, **kwargs):
        state = super().reset()
        return self.tuple_to_vector(state[0])

    def step(self, action):
        transformed_action = np.asarray([tuple([action[0] - 1, action[1] - 1])])
        state, env_score, arousal, d, info = super().step(transformed_action)
        state = self.tuple_to_vector(state[0])
        self.calculate_reward(state)
        self.cumulative_reward += self.current_reward
        self.reset_condition()
        final_reward = self.current_reward * (1 - self.weight) + (arousal * self.weight)
        return state, final_reward, d, info


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=6)  # precision is optional

    if len(sys.argv) != 3:
        print(f"Incorrect number of arguments specified, was expecting 7, found {len(sys.argv)}")
        print("Using default arguments")
        cluster = 0
        run = 1
        target_name = "Maximize"
        target_signal_str = "np.ones"
        preference_task = True
        classification_task = False
        weight = 0.5
    else:
        cluster = 0
        run = int(sys.argv[1])
        weight = float(sys.argv[2])
        target_name = "Maximize"
        target_signal_str = "np.ones"
        preference_task = True
        classification_task = False

        # target_name = sys.argv[3]
        # target_signal_str = sys.argv[4]
        # preference_task = True if sys.argv[5] == "True" else False
        # classification_task = True if sys.argv[6] == "True" else False
        # target_signal_str = "np.ones"

    function_mapping = {
        'np.ones': np.ones,
        'np.zeros': np.zeros,
        'imitate': None
    }

    model = KNNSurrogateModel(5, classification_task, preference_task, cluster, "Solid")

    env = PPO_Environment(id_number=run,
                          graphics=True, obs_space={"low": -np.inf, "high": np.inf, "shape": (49,)},
                          path="./Builds/Solid_GoBlend/Racing.exe", arousal_model=model, weight=weight,
                         )

    sideChannel = env.customSideChannel
    env.targetSignal = function_mapping[target_signal_str]

    model = PPO("MlpPolicy", env=env, tensorboard_log="../Tensorboard", device='cpu')
    model.learn(total_timesteps=1000000, progress_bar=True, callback=TensorboardCallback(), tb_log_name="PPO", )

    if weight == 0:
        label = 'optimize'
    elif weight == 0.5:
        label = 'blended'
    else:
        label = 'arousal'
    model.save(f"ppo_solid_{label}_{run}")
