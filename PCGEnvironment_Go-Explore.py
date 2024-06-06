import configparser
import sys
import uuid
from abc import ABC

import numpy as np
from similaritymeasures import similaritymeasures
from torch.utils.tensorboard import SummaryWriter

from BaseEnvironment import BaseEnvironment
from GoBlend.Archive import Archive
from GoBlend.Cell import Cell
from PCGRL.Dijkstra import dijkstra_with_path
from SurrogateModel import KNNSurrogateModel
from Utils.Tensorboard_Callbacks import TensorboardEDPCGRLGO


# noinspection DuplicatedCode
class PCGEnvironmentGoExplore(BaseEnvironment, ABC):

    def __init__(self, id_number, graphics, path, shape, model, scaler, UUID, name=""):

        self.target_signal = None
        self.reset_dijkstra = False
        self.arousal_list, self.path, self.fixed_grid, self.currentSurrogate, self.previousSurrogate = [], [], [], [], []
        self.current_index, self.previous_index, self.facing = (), (), ()
        self.tracks, self.indices = {}, {}
        self.arousal_model, self.scaler = model, scaler
        self.counter, self.cluster_coverage, self.track_length = 0, 0, 0
        self.current_cell = None
        self.name = name

        args = f"-socketID {UUID}"
        obs_space = {"low": -np.inf, "high": np.inf, "shape": shape}
        super().__init__(id_number, graphics, obs_space, path, args=args)

        self.config = configparser.ConfigParser()
        self.config.read('./GoBlend/config_files/pcg_baseline.config')
        self.archive = Archive(self.config)

        self.callback = TensorboardEDPCGRLGO(self)
        self.writer = SummaryWriter(log_dir=f'./Tensorboard/PCGRL-{name}')  # Logger for tensorboard

        self.total_timesteps = int(self.config['Cells']['max_trajectories'])  # total number of "rollouts" to explore
        self.explore_length = int(self.config['Cells']['explore_steps'])  # length of each rollout during exploration
        self.max_trajectory_length = int(self.config['Cells']['max_trajectory_size'])

        self.lambdaValue = float(self.config['Rewards']['lambda'])
        self.behavior_target = self.config['Rewards']['behavior_target']
        self.behavior_function = self.calculate_reward
        self.normalize_behavior = True
        # self.normalize_behavior = self.config['Rewards']['normalize_behavior']
        self.arousal_target = self.config['Rewards']['arousal_target']
        self.arousal_function = self.calculate_reward

        self.dijkstra_length = 0

    def reset(self, **kwargs):
        _ = self.env.reset()
        self.current_index = (50, 52)
        self.previous_index = (50, 51)
        self.fixed_grid = np.ones((100, 100))*7
        self.fixed_grid[int(self.previous_index[0])][int(self.previous_index[1])] = 5
        self.previousSurrogate = None
        self.currentSurrogate = None
        return None

    def update_grid(self, action):
        # print(self.current_index, self.previous_index)
        x_delta = np.sign(int(self.current_index[0] - self.previous_index[0]))
        y_delta = np.sign(int(self.current_index[1] - self.previous_index[1]))
        if x_delta != 0:
            for x_diff in range(int(self.previous_index[0]), int(self.current_index[0]), x_delta):
                self.fixed_grid[int(x_diff)][int(self.current_index[1])] = action
        elif y_delta != 0:
            for y_diff in range(int(self.previous_index[1]), int(self.current_index[1]), y_delta):
                self.fixed_grid[int(self.current_index[0])][int(y_diff)] = action
        self.previous_index = (self.current_index[0], self.current_index[1])

    def reset_to_state(self, actions):
        self.reset()
        for action in actions:
            state, reward, done, _ = self.env.step(action)
            self.current_index = (state[0][0], state[0][2])
            # print(f"Resetting with action: {action} at position {self.current_index}")
            self.update_grid(action)

    def simulate_race(self):
        self.arousal_list.clear()
        counter = 0
        while not self.customSideChannel.race_ended:

            if counter >= 30000:
                break

            self.previousSurrogate = []
            if self.currentSurrogate is not None:
                self.previousSurrogate = np.array(self.currentSurrogate.copy(), dtype=np.float32)

            for i in range(150):
                _ = self.env.step(-1)
                counter += 1
                if i == 147:
                    self.create_and_send_message("Send Vector")

            self.currentSurrogate = np.array(self.customSideChannel.arousal_vector.copy(), dtype=np.float32)

            if len(self.previousSurrogate) == 0:
                self.previousSurrogate = np.zeros((29,))

            if len(self.currentSurrogate) == 0:
                self.currentSurrogate = np.zeros((29,))

            scaled_obs = np.array(self.scaler.transform(self.currentSurrogate.reshape(1, -1))[0])

            if preference_task:
                previous_scaler = np.array(self.scaler.transform(self.previousSurrogate.reshape(1, -1))[0])
                scaled_obs = list(previous_scaler) + list(scaled_obs)

            scaled_obs = np.clip(scaled_obs, 0, 1)
            arousal, indices = self.arousal_model(scaled_obs)

            for index in indices:
                if index not in self.indices:
                    self.indices.update({index: 1})
                else:
                    self.indices[index] += 1

            self.cluster_coverage = len(self.indices) / len(self.arousal_model.x_train) * 100
            self.arousal_list.append(arousal)
            self.previousSurrogate = self.currentSurrogate.copy()

        self.customSideChannel.race_ended = False
        return self.arousal_list

    def close_circuit(self, ga_actions, generation):

        action_list = list(ga_actions.copy())

        print(f"PATH FROM: {self.current_index}")
        self.fixed_grid[50][50] = 0
        _, self.path = dijkstra_with_path(self.fixed_grid,
                                          (self.current_index[0], self.current_index[1]),
                                          (50, 49))
        self.path.append((50, 50))

        self.path = self.path[1:]
        self.dijkstra_length = len(self.path)
        self.reset_dijkstra = True
        self.counter += 1

        while len(self.path) > 0:
            # print(self.path)
            previous_len = len(self.path)
            for new_action in range(5):
                if self.reset_dijkstra:
                    # print(f"Recreating track with actions {action_list.copy()}")
                    self.reset_to_state(action_list.copy())
                    # print(self.current_index)
                next_state, reward, done, info = self.env.step(new_action)
                self.current_index = (next_state[0][0], next_state[0][2])
                # print(self.current_index)
                self.update_grid(new_action)
                if (self.current_index[0], self.current_index[1]) == self.path[0]:
                    action_list.append(new_action)
                    self.path = self.path[1:]
                    self.reset_dijkstra = False
                    if len(self.path) == 0:
                        self.tracks.update({f"Generation-{generation}-Track-{len(self.tracks)}": {
                            "Actions": action_list.copy(), "Arousals": self.arousal_list.copy()}})
                        return True
                    break
                self.reset_dijkstra = True
            if len(self.path) == previous_len:
                return False

    def step(self, action):
        state, env_score, done, info = self.env.step(action)
        if np.array_equal(self.current_index, (49, 50)):
            return state, False
        self.current_index = (state[0][0], state[0][2])
        self.update_grid(action)
        if self.customSideChannel.collision:
            self.customSideChannel.collision = False
            return state, False
        return state, True

    def create_cell(self, action, state):
        if self.current_cell is not None:
            self.current_cell.trajectory_dict['behavior_trajectory'].append(action)
            self.current_cell.trajectory_dict['state_trajectory'].append(state)
            self.current_cell.update_key(state)
            if len(self.current_cell.trajectory_dict['behavior_trajectory']) >= 10:
                self.current_cell.final = True
        else:
            self.current_cell = Cell(state, {"state_trajectory": [state],
                                             "behavior_trajectory": [action],
                                             "arousal_trajectory": [],
                                             "uncertainty_trajectory": [],
                                             "arousal_vectors": [],
                                             "score_trajectory": []})

    def add_arousal(self, arousal_vector, score, dijkstra_length):
        self.current_cell.trajectory_dict['arousal_trajectory'] = arousal_vector
        self.current_cell.trajectory_dict['score_trajectory'].append(score)
        self.current_cell.trajectory_dict['state_trajectory'][-1].append(dijkstra_length)
        pass

    def explore_actions(self):

        if len(self.archive.archive) != 0:
            self.current_cell = self.archive.select_cell()
            self.reset_to_state(self.current_cell.trajectory_dict['behavior_trajectory'].copy())

        action = self.env.action_space.sample()
        state, collision = self.step(action)

        if not collision:
            return False

        if self.current_cell is None:
            action_list = [action]
        else:
            action_list = self.current_cell.trajectory_dict['behavior_trajectory'].copy()
            action_list.append(action)

        turns, straights, bridges, loops = 0, 0, 0, 0

        for action in action_list:
            if action == 0:
                straights += 1
            elif action == 1 or action == 2:
                turns += 1
            elif action == 3:
                loops += 1
            elif action == 4:
                bridges += 1

        state = [turns, straights, bridges, loops]

        self.create_cell(action, state)
        if self.close_circuit(self.current_cell.trajectory_dict['behavior_trajectory'].copy(), 0):
            arousal_values = env.simulate_race()
            self.add_arousal(arousal_values, len(self.current_cell.trajectory_dict['behavior_trajectory'].copy()), self.dijkstra_length)
            self.current_cell.assess_cell(self.lambdaValue, self.normalize_behavior == "True", self.arousal_function)
            self.archive.store_cell(self.current_cell)
            return True
        else:
            return False

    def explore(self):
        for _ in range(self.total_timesteps):
            self.reset()
            self.explore_actions()
            self.callback.update_board()
            results = {"Tracks": self.tracks, "Coverage": self.cluster_coverage, "Index_Counts": self.indices}
            np.save(f"./{self.name} Results.npy", results)
        self.writer.close()

    def calculate_reward(self):
        arousal_signal = self.current_cell.trajectory_dict['arousal_trajectory']
        target_signal = self.target_signal(len(arousal_signal))
        arousal_signal = np.expand_dims(np.array(arousal_signal), axis=1)
        target_signal = np.expand_dims(np.array(target_signal), axis=1)
        target_x = np.expand_dims(np.arange(len(target_signal)), axis=1)
        arousal_x = np.expand_dims(np.arange(len(arousal_signal)), axis=1)
        arousal_signal = np.concatenate([arousal_x, arousal_signal.copy()], axis=1)
        target_signal = np.concatenate([target_x, target_signal], axis=1)
        area = similaritymeasures.area_between_two_curves(target_signal, arousal_signal)
        return -area


def sample_target_signal(length):
    start_point = int(length/3)
    end_point = start_point * 2
    signal = np.ones((length,))
    for value in range(start_point, end_point):
        signal[value] = 0
    return signal


if __name__ == "__main__":

    if len(sys.argv) != 7:
        print(f"Incorrect number of arguments specified, was expecting 7, found {len(sys.argv)}")
        print("Using default arguments")
        cluster = 0
        run = 1
        target_name = "Maximize"
        target_signal_str = "np.ones"
        preference_task = True
        classification_task = False
    else:
        cluster = int(sys.argv[1])
        run = int(sys.argv[2])
        target_name = sys.argv[3]
        target_signal_str = sys.argv[4]
        preference_task = True if sys.argv[5] == "True" else False
        classification_task = True if sys.argv[6] == "True" else False

    function_mapping = {
        'np.ones': np.ones,
        'np.zeros': np.zeros,
        'imitate': sample_target_signal
    }

    arousal_type = "Changes" if preference_task else "Arousal"

    if target_signal_str in function_mapping:
        target_signal = function_mapping[target_signal_str]
    else:
        print(f"Error: Unknown target_signal function '{target_signal_str}'.")
        sys.exit(1)

    experiment_name = f"{target_name} {arousal_type} Cluster {cluster} Run {run}"
    surrogate_model = KNNSurrogateModel(5, classification_task, preference_task, cluster)

    socketID = uuid.uuid4()
    vector_length = (2527,)
    env = PCGEnvironmentGoExplore(0, graphics=True, path="./Builds/Solid_Optimized/Racing.exe",
                                  shape=vector_length, model=surrogate_model, scaler=surrogate_model.scaler,
                                  name=experiment_name, UUID=socketID)

    env.target_signal = target_signal
    max_length = 10
    env.explore()
