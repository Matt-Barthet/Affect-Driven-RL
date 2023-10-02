from Cell import Cell
from RewardFunctions import reward_functions
import copy
import pickle
import numpy as np
import random


class Archive:

    def __init__(self, config):
        self.max_trajectories = int(config['Cells']['max_trajectories'])
        self.max_trajectory_size = int(config['Cells']['max_trajectory_size'])
        self.selection_method = config['Cells']['cell_selection']
        self.explore_steps = int(config['Cells']['explore_steps'])
        self.selectionLambda = config['Cells']['cell_selection_lambda']

        self.epsilon = float(config['Rewards']['epsilon'])
        self.lambdaValue = float(config['Rewards']['lambda'])
        self.behavior_target = config['Rewards']['behavior_target']
        self.behavior_function = reward_functions[config['Rewards']['behavior_reward']]
        self.arousal_target = config['Rewards']['arousal_target']
        self.arousal_function = reward_functions[config['Rewards']['arousal_reward']]

        self.kNN = int(config['Human Model']['kNN'])

        self.archive = {}
        self.currentCell = None
        self.bestCell = None
        self.bestKey = 0
        self.currentKey = 0
        self.currentStep = 0
        self.iterations = 0

        self.cellSelectionFunctions = {"Random": self.select_random_cell, "Roulette": self.select_cell_roulette}

    def step(self, env, iteration, state):
        if iteration != 0:
            state_dict = self.return_to_cell()
            _, _ = env.reset()
            # set_world_state(state_dict, env.world) LOAD CELL
        else:
            _, _ = env.reset()

        for j in range(self.explore_steps + 100):
            action = env.action_space.weighted_sample()
            observation, reward, done, info = env.step(action)

            bonus = self.create_cell(action, observation, env.player.hp == 0, state)
            self.assess_cell(reward)

            if env.player.hp == 0:
                print("Died, resetting")
                break

            env.render()

        self.save_best_cells()

    def create_cell(self, action, state, final, world):
        bonus = 0
        try:
            self.currentCell.trajectory_dict['behavior_trajectory'].append(action)
            if state not in self.currentCell.trajectory_dict['state_trajectory']:
                bonus += 1
            self.currentCell.trajectory_dict['state_trajectory'].append(state)
            self.currentCell.update_key(state)
            self.currentKey = self.currentCell.key
        except AttributeError:
            self.currentCell = Cell(state, {"state_trajectory": [state], "behavior_trajectory": [action],
                                            "arousal_trajectory": [], "uncertainty_trajectory": []}, {})
            self.currentKey = self.currentCell.key

        self.currentCell.generate_human_arousal(self.kNN)
        self.currentCell.final = final
        self.currentCell.state_dict = world
        return bonus

    def assess_cell(self, score):
        self.currentCell.generate_human_arousal(self.kNN)
        self.currentCell.blended_reward = self.currentCell.calculate_blended_reward(self.lambdaValue,
                                                                                    self.behavior_target == "Imitate",
                                                                                    self.behavior_function,
                                                                                    self.arousal_function)
        self.store_cell()
        self.update_best_cell()

    def select_random_cell(self):
        key, cell = random.choice(list(self.archive.items()))
        if cell.final:
            self.select_random_cell()
        else:
            self.currentKey = copy.copy(key)
            self.currentCell = copy.deepcopy(cell)

    def select_cell_roulette(self):
        weights = [cell.calculate_blended_reward(self.selectionLambda, self.behavior_target == "Imitate") for cell in
                   self.archive]
        weights = np.asarray(weights) / np.sum(weights)
        np.random.choice(list(self.archive.items()), size=1, replace=False, p=weights)
        if not self.currentCell.final:
            self.select_cell_roulette()

    def store_cell(self):
        if self.currentCell.get_cell_length() > self.max_trajectory_size or self.currentCell.blended_reward < 0 or \
                not self.currentCell.legalState:
            return
        if self.store_cell_condition():
            self.archive.update({self.currentKey: copy.deepcopy(self.currentCell)})

    def store_cell_condition(self):
        if self.currentKey not in self.archive:
            return True
        if self.currentCell.blended_reward < self.archive[self.currentKey].blended_reward:
            return False
        if self.currentCell.blended_reward >= self.archive[self.currentKey].blended_reward + self.epsilon:
            return True
        return self.currentCell.get_cell_length() < self.archive[self.currentKey].get_cell_length()

    def return_to_cell(self):
        self.currentStep = 0
        self.cellSelectionFunctions[self.selection_method]()
        self.currentCell.visited += 1
        self.iterations += 1
        print("Archive Size: {} - Iterations: {} - Best Reward: {} - Best Length {} - Best Score {}".format(
            len(self.archive),
            self.iterations,
            self.bestCell.calculate_blended_reward(self.lambdaValue, self.behavior_target == "Imitate"),
            self.bestCell.get_cell_length(),
            self.bestCell.score)
        )
        return self.currentCell.state_dict

    def update_best_cell(self):
        if self.bestCell is None or self.currentCell.blended_reward > self.bestCell.blended_reward:
            self.bestCell = copy.deepcopy(self.currentCell)

    def save_best_cells(self):
        best_reward = 0
        best_cell = None
        for cell in self.archive.values():
            if cell.blended_reward > best_reward:
                best_reward = cell.blended_reward
                best_cell = cell
        pickle.dump(best_cell, open('./Best_Cell.pkl', 'wb'))
        pickle.dump(self.archive, open('./Archive.pkl', 'wb'))
