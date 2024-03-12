import random
import uuid
from abc import ABC

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter

from BaseEnvironment import BaseEnvironment
from PCGRL.Dijkstra import dijkstra_with_path
from PCGRL.PCGUtility import construct_state
from Utils.Tensorboard_Callbacks import TensorboardEDPCG
from similaritymeasures import similaritymeasures
from statistics import mode

preference_label = False

def KNN(input_tuple):
    x_train = input_tuple[0][0].values
    y_train = np.array(input_tuple[0][1])

    state = input_tuple[1]
    k = 5

    distances = np.array(np.sqrt(np.sum((state - x_train) ** 2, axis=1)))
    k_indices = np.array(np.argsort(distances)[:k])
    k_labels = np.array(y_train[k_indices])

    # print(k_indices)
    # print(distances[k_indices])

    if k == 1:
        return y_train[k_indices][0]

    if preference_label:
        predicted_class = mode(k_labels)
    else:
        weights = 1 / (distances[k_indices] + 1e-5)
        weighted_sum = np.sum(weights * k_labels)
        total_weights = np.sum(weights)
        predicted_class = weighted_sum / total_weights

    # print(predicted_class)
    return predicted_class, k_indices


# noinspection DuplicatedCode
class PCGEnvironmentGA(BaseEnvironment, ABC):

    def __init__(self, id_number, graphics, path, shape, model, scaler, UUID, name="", dataset=None):

        self.reset_dijkstra = False
        self.arousal_list = []
        self.path = []
        self.fixed_grid = []
        self.name = name

        args = f"-socketID {UUID}"

        obs_space = {"low": -np.inf, "high": np.inf, "shape": shape}
        super().__init__(id_number, graphics, obs_space, path, args=args)

        self.current_index = ()
        self.facing = ()

        self.archive = {}
        self.track_length = 0

        self.previousSurrogate = []
        self.currentSurrogate = []
        self.arousal_model = model
        self.scaler = scaler

        self.increase, self.decrease = 0, 0
        self.tracks = {}
        self.population_size = 50
        self.counter = 0
        self.dataset = dataset

        self.indices = {}
        self.cluster_coverage = 0

        self.min_fitness, self.avg_fitness, self.max_fitness, self.avg_length, self.best_length = 0, 0, 0, 0, 0
        self.callback = TensorboardEDPCG(self)
        self.writer = SummaryWriter(log_dir=f'./Tensorboard/EDPCG-{name}')  # Logger for tensorboard

    def reset(self, **kwargs):
        # print("Resetting...")
        state = self.env.reset()
        new_state, self.fixed_grid, self.current_index, self.facing = construct_state(state, self.one_hot_encode)
        self.previousSurrogate = None
        self.currentSurrogate = None
        return self.tuple_to_vector(new_state)

    def send_load_message(self):
        self.action_string = "[Load]:"
        self.action_string += ','.join(f"{action}" for action in self.action_list)
        self.create_and_send_message(self.action_string)

    def reset_to_state(self, actions):
        next_state = self.env.reset()
        for action in actions:
            next_state, reward, done, _ = self.env.step(action)  # Execute the action
        self.current_state, self.fixed_grid, self.current_index, self.facing = construct_state(next_state, self.one_hot_encode)

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
                print("Setting surrogate to zero")
                self.currentSurrogate = np.zeros((29,))

            scaled_obs = np.array(self.scaler.transform(self.currentSurrogate.reshape(1, -1))[0])

            if preference_task:
                previous_scaler = np.array(self.scaler.transform(self.previousSurrogate.reshape(1, -1))[0])
                scaled_obs = list(previous_scaler) + list(scaled_obs)

            # combined_scaler = np.concatenate((self.previousSurrogate.copy(), self.currentSurrogate.copy()))
            # scaled_obs = self.scaler.transform(np.array(combined_scaler).reshape(1, -1))

            if np.max(scaled_obs) > 1:
                print(np.max(scaled_obs), np.argmax(scaled_obs) % 29)
            if np.min(scaled_obs) < 0:
                print(np.min(scaled_obs), np.argmin(scaled_obs) % 29)

            scaled_obs = np.clip(scaled_obs, 0, 1)
            arousal, indices = self.arousal_model((self.dataset, scaled_obs))

            for index in indices:
                if index not in self.indices:
                    self.indices.update({index: 1})
                else:
                    self.indices[index] += 1

            self.cluster_coverage = len(self.indices)/len(self.dataset[0]) * 100
            self.arousal_list.append(arousal)
            self.previousSurrogate = self.currentSurrogate.copy()
            print(arousal)
        self.customSideChannel.race_ended = False
        return self.arousal_list

    def close_circuit(self, ga_actions, generation):

        action_list = list(ga_actions.copy())
        _, self.path = dijkstra_with_path(self.fixed_grid,
                                          (self.current_index[1], self.current_index[0]),
                                          (49, 50))
        self.path.append((50, 50))
        self.reset_dijkstra = True
        self.path = self.path[1:]

        copied_grid = self.fixed_grid.copy()
        copied_grid[int(self.current_index[1])][int(self.current_index[0])] = -2

        for i in range(len(self.path)):
            copied_grid[int(self.path[i][0])][int(self.path[i][1])] = -1

        self.counter += 1

        while len(self.path) > 0:
            previous_len = len(self.path)
            for new_action in range(5):

                if self.reset_dijkstra:
                    self.reset_to_state(action_list.copy())

                next_state, reward, done, info = self.env.step(new_action)
                new_state, self.fixed_grid, self.current_index, self.facing = construct_state(next_state,
                                                                                              self.one_hot_encode)

                if (self.current_index[1], self.current_index[0]) == self.path[0]:
                    action_list.append(new_action)
                    self.path = self.path[1:]
                    self.reset_dijkstra = False
                    if len(self.path) == 0:
                        # self.create_and_send_message(f"Screenshot: /{self.name}/Generation {generation} Track-{len(self.tracks)}.png")
                        self.tracks.update({f"Generation-{generation}-Track-{len(self.tracks)}": { "Actions": action_list.copy(), "Arousals": self.arousal_list.copy()}})
                        return True
                    break
                self.reset_dijkstra = True

            if len(self.path) == previous_len:
                # print(f"{self.counter} Fail")
                # self.create_and_send_message(
                #     f"Screenshot:FAIL {self.name} Generation {generation} Track {self.counter - 1}.png")
                return False

    def step(self, action):
        if self.current_index == (49, 50):
            return False
        state, env_score, done, info = self.env.step(action)
        new_state, self.fixed_grid, self.current_index, self.facing = construct_state(state, self.one_hot_encode)
        if self.customSideChannel.collision:
            self.customSideChannel.collision = False
            return False
        return True


def load_and_clean(filename, cluster):
    data = pd.read_csv(filename)
    data = data.loc[:, data.apply(pd.Series.nunique) != 1]

    if not preference_task:
        if cluster !=0 :
            data = data[data['Cluster'] == cluster]
        arousals = data['[output]arousal'].values
        data = data.drop(columns=['[control]player_id', '[output]arousal', 'botRespawn'])
    elif preference_task:
        if cluster != 0:
            data = data[data['Cluster_R'] == cluster]
        data = data[data['Ranking'] != "stable"]
        arousals = data['Ranking'].values
        data = data.drop(columns=['Player', 'Ranking'])
        # Encode the ranking labels
        label_mapping = {"decrease": 0.0, "increase": 1.0}
        arousals = [label_mapping[label] for label in arousals]

    # data = data[data['Ranking'] != "stable"]
    data = data[data.columns[~data.columns.str.startswith("Cluster")]]
    data = data[data.columns[~data.columns.str.startswith("Time_Index")]]
    data = data[data.columns[~data.columns.str.contains("arousal")]]
    data = data[data.columns[~data.columns.str.contains("botRespawn")]]
    data = data[data.columns[~data.columns.str.contains("Score")]]

    return data, arousals


def load_data_knn(cluster):
    if preference_task:
        data, arousals = load_and_clean(f'./Datasets/solid_3000ms_downsampled_pair_data.csv', cluster)
    else:
        data, arousals = load_and_clean(f'./Datasets/Solid_3000ms_minmax_with_clusters.csv', cluster)
    unscaled_data = pd.read_csv(f'./Datasets/Solid_3000ms_nonorm_with_clusters.csv')
    unscaled_data = unscaled_data.drop(columns=['[control]player_id', '[output]arousal', 'botRespawn'])
    unscaled_data = unscaled_data.loc[:, unscaled_data.apply(pd.Series.nunique) != 1]
    unscaled_data = unscaled_data[unscaled_data.columns[~unscaled_data.columns.str.startswith("Cluster")]]
    unscaled_data = unscaled_data[unscaled_data.columns[~unscaled_data.columns.str.startswith("Time_Index")]]
    unscaled_data = unscaled_data[unscaled_data.columns[~unscaled_data.columns.str.contains("arousal")]]
    unscaled_data = unscaled_data[unscaled_data.columns[~unscaled_data.columns.str.contains("Score")]]
    scaler = MinMaxScaler().fit(unscaled_data.values)
    return scaler, data, arousals


def sample_target_signal(length):
    pass


def calculate_reward(target_signal, arousal_signal):
    arousal_signal = np.expand_dims(np.array(arousal_signal), axis=1)
    target_signal = np.expand_dims(np.array(target_signal), axis=1)
    target_x = np.expand_dims(np.arange(len(target_signal)), axis=1)
    arousal_x = np.expand_dims(np.arange(len(arousal_signal)), axis=1)
    arousal_signal = np.concatenate([arousal_x, arousal_signal.copy()], axis=1)
    target_signal = np.concatenate([target_x, target_signal], axis=1)
    area = similaritymeasures.area_between_two_curves(target_signal, arousal_signal)
    return area

def evaluate_fitness(individual, generation):
    for action in individual:
        if not env.step(action):
            return -1000, []
    if not env.close_circuit(individual, generation):
        return -1000, []
    arousals = env.simulate_race()
    return -calculate_reward(target_signal(len(arousals)), arousals), arousals

def roulette_wheel_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    selection_probabilities = [total_fitness/fitness+0.0001 for fitness in fitness_scores]
    return random.choices(population, weights=selection_probabilities, k=population_size)


if __name__ == "__main__":

    import sys
    import random
    import numpy as np

    # Check if the correct number of arguments are passed
    if len(sys.argv) != 6:
        print("Usage: python your_script.py <cluster> <run> <target_name> <target_signal>")
        sys.exit(1)

    # Mapping of string argument to actual function
    function_mapping = {
        'np.ones': np.ones,
        'np.zeros': np.zeros,
        'imitate': sample_target_signal  # Assume sample_target_signal is defined elsewhere
    }

    cluster = int(sys.argv[1])
    run = int(sys.argv[2])
    target_name = sys.argv[3]
    target_signal_str = sys.argv[4]
    preference_task = True if sys.argv[5] == "True" else False

    arousal_type = "Changes" if preference_task else "Arousal"

    # Validate and convert the target_signal string to an actual function
    if target_signal_str in function_mapping:
        target_signal = function_mapping[target_signal_str]
    else:
        print(f"Error: Unknown target_signal function '{target_signal_str}'.")
        sys.exit(1)

    # Assume load_data_knn is defined elsewhere
    scaler, x_train, y_train = load_data_knn(cluster)

    experiment_name = f"{target_name} {arousal_type} Cluster {cluster} Run {run}"

    # Assume PCGEnvironmentGA and other dependencies are defined elsewhere
    socketID = uuid.uuid4()
    vector_length = (2527,)
    env = PCGEnvironmentGA(0, graphics=True, path="./Builds/SolidRallyPCGRL/Racing.exe",
                           shape=vector_length, model=KNN, scaler=scaler, name=experiment_name, UUID=socketID, dataset=[x_train, y_train])

    offspring_size = 50  # λ
    string_length = 10
    mutation_rate = 0.1
    num_generations = 100
    mu = 10  # Number of parents
    elite_count = 1  # Number of elite individuals to carry over
    population = np.load("./seed_tracks.npy", allow_pickle=True).item()[f'Seed_{run}']

    def one_point_crossover(parent1, parent2):
        crossover_point = random.randint(1, string_length - 1)
        child1 = list(parent1[:crossover_point]) + list(parent2[crossover_point:])
        child2 = list(parent2[:crossover_point]) + list(parent1[crossover_point:])
        return child1, child2

    for generation in range(num_generations):
        fitness_scores = []
        lengths = []

        # Evaluate fitness for each individual
        for individual in population:
            env.reset()
            fitness, arousals = evaluate_fitness(individual, generation)  # Assume evaluate_fitness is defined elsewhere
            fitness_scores.append(fitness if fitness != -1000 else -1000)
            lengths.append(len(arousals) if fitness != -1000 else 0)

        # Update environment with generation statistics
        env.min_fitness = np.nanmin(fitness_scores)
        env.max_fitness = np.nanmax(fitness_scores)
        env.avg_fitness = np.nanmean(fitness_scores)
        env.avg_length = np.nanmean(lengths)
        env.callback.update_board()  # Assume this is part of the environment's functionality

        # Logging the results for this generation
        results = {"Tracks": env.tracks, "Coverage": env.cluster_coverage, "Index_Counts": env.indices}
        np.save(f"./{env.name} Results.npy", results)

        print(f"Generation {generation}: Min Fitness: {env.min_fitness}, "
              f"Avg Fitness: {env.avg_fitness}, "
              f"Max Fitness: {env.max_fitness}")

        # Selection of the mu best offspring plus elites from current generation
        sorted_indices = np.argsort(-np.array(fitness_scores))
        elites = [population[i] for i in sorted_indices[:elite_count]]
        sorted_population = [population[i] for i in sorted_indices]
        mating_pool = sorted_population[:mu]

        # Generate new offspring through crossover and mutation
        new_generation = elites[:]  # Start new generation with the elites
        while len(new_generation) < offspring_size:
            parent1, parent2 = random.sample(mating_pool, 2)
            child1, child2 = one_point_crossover(parent1, parent2)

            child1_unique, child2_unique = True, True

            for child in new_generation:
                if np.array_equal(np.array(child), np.array(child1)):
                    child1_unique = False
                    break

            for child in new_generation:
                if np.array_equal(np.array(child), np.array(child2)):
                    child2_unique = False
                    break

            if child1_unique:
                new_generation.extend([child1])
            if child2_unique:
                new_generation.extend([child2])

        # Mutation
        for i in range(len(new_generation)):
            new_generation[i] = [gene if random.random() > mutation_rate else random.randint(0, 4) for gene in new_generation[i]]

        # The next generation replaces the old one, up to the population size
        population = new_generation[:offspring_size]

    env.writer.close()
    evolved_population = population
