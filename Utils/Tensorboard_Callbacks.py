from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class PPOHackCallback(BaseCallback):
    def __init__(self, target_arousal_per_component, verbose: int = 0):
        super().__init__(verbose)
        self.arousal_change_threshold = 0.05
        self.target_signal = target_arousal_per_component

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        self.training_env.envs[0].action_list.clear()
        pass

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record('reward/cumulative reward', np.sum(self.model.rollout_buffer.rewards))
        self.logger.record('reward/maximum track length', self.training_env.get_attr('max_score')[0])
        # self.logger.record('reward/normalized reward (based on track length)', normalize_reward / len(self.training_env.envs[0].action_list))

    def _on_training_end(self) -> None:
        pass


class TensorboardCallback(BaseCallback):
    def __init__(self):
        super(TensorboardCallback, self).__init__()

    def _on_step(self) -> bool:
        # self.logger.record('reward/reward', self.training_env.get_attr('current_reward')[0])
        # self.logger.record('reward/cumulative reward', self.training_env.get_attr('cumulative_reward')[0])
        # self.logger.record('reward/maximum environment score', self.training_env.get_attr('best_score')[0])
        # self.logger.record('reward/max reward', self.training_env.get_attr('best_reward')[0])
        return True


class TensorboardGoExplore:
    def __init__(self, env, archive):
        self.env = env
        self.step_count = 0
        self.archive = archive

    def size(self):
        return len(self.archive.archive)

    def best_cell_length(self):
        return self.archive.bestCell.get_cell_length()

    def best_cell_behavior(self):
        return self.archive.bestCell.behavior_reward

    def best_cell_affect(self):
        return self.archive.bestCell.arousal_reward

    def best_cell_lambda(self):
        return self.archive.bestCell.blended_reward

    def on_step(self):
        self.env.writer.add_scalar('archive/archive size', self.size(), self.step_count)
        self.env.writer.add_scalar('best cell/trajectory length', self.best_cell_length(), self.step_count)
        self.env.writer.add_scalar('best cell/behavior reward', self.best_cell_behavior(), self.step_count)
        self.env.writer.add_scalar('best cell/affect reward', self.best_cell_affect(), self.step_count)
        self.env.writer.add_scalar('best cell/blended reward', self.best_cell_lambda(), self.step_count)
        self.step_count += 1


class TensorboardEDPCGRLGO:
    def __init__(self, env):
        self.env = env
        self.step_count = 0

    def update_board(self):
        self.env.writer.add_scalar('Archive/Archive Size', len(self.env.archive.archive), self.step_count)
        if self.env.archive.bestCell is None:
            self.env.writer.add_scalar('Reward/Best Cell Reward', -1000, self.step_count)
            self.env.writer.add_scalar('Reward/Best Cell Length', 0, self.step_count)
        else:
            self.env.writer.add_scalar('Reward/Best Cell Reward', self.env.archive.bestCell.blended_reward, self.step_count)
            self.env.writer.add_scalar('Reward/Best Cell Length', self.env.archive.bestCell.get_cell_length(), self.step_count)
        self.env.writer.add_scalar('KNN/Cluster Coverage', self.env.cluster_coverage, self.step_count)
        self.step_count += 1


class TensorboardEDPCG:
    def __init__(self, env):
        self.env = env
        self.step_count = 0

    def update_board(self):
        self.env.writer.add_scalar('GA Metrics/Minimum Fitness', self.env.min_fitness, self.step_count)
        self.env.writer.add_scalar('GA Metrics/Average Fitness', self.env.avg_fitness, self.step_count)
        self.env.writer.add_scalar('GA Metrics/Maximum Fitness', self.env.max_fitness, self.step_count)
        self.env.writer.add_scalar('GA Metrics/Average Simulation Length', self.env.avg_length, self.step_count)
        self.env.writer.add_scalar('GA Metrics/Best Indv. Simulation Length', self.env.best_length, self.step_count)
        self.env.writer.add_scalar('KNN Metrics/Cluster Coverage', self.env.cluster_coverage, self.step_count)
        self.step_count += 1