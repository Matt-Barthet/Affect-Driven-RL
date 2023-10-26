from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    def __init__(self):
        super(TensorboardCallback, self).__init__()

    def _on_step(self) -> bool:
        self.logger.record('reward/reward', self.training_env.get_attr('reward')[0])
        self.logger.record('reward/cumulative reward', self.training_env.get_attr('cumulative_reward')[0])
        self.logger.record('reward/maximum environment score', self.training_env.get_attr('max_score')[0])
        self.logger.record('reward/max reward', self.training_env.get_attr('max_reward')[0])
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

