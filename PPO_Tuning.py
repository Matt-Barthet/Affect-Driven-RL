import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from PPOEnvironment import PPO_Environment
import joblib  # Add this import

from Utils.Tensorboard_Callbacks import TensorboardCallback


def evaluate_agent(model, eval_env, n_episodes=10):
    """
    Evaluate the given agent on the environment for a certain number of episodes.

    :param model: (BaseAlgorithm) The trained agent.
    :param eval_env: (gym.Env) The evaluation environment.
    :param n_episodes: (int) Number of episodes to evaluate the agent.
    :return: (float) Mean reward for the given number of episodes.
    """
    all_episode_rewards = []

    for i in range(n_episodes):
        episode_rewards = []
        obs = eval_env.reset()
        for _ in range(4 * 140):
            action, _ = model.predict(obs, deterministic=True)  # Use deterministic actions
            obs, reward, done, _ = eval_env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print(mean_episode_reward)
    eval_env.close()
    return mean_episode_reward


def objective(trial):
    # Sample hyperparameters
    lr = trial.suggest_float('lr', 1e-4, 1e-2)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1)
    ent_coef = trial.suggest_float('ent_coef', 1e-4, 0.1, log=True)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
    n_epochs = trial.suggest_int('n_epochs', 1, 10)
    vector_length = (68,)
    env = DummyVecEnv([lambda: PPO_Environment(counter, graphics=True, scaler=None, include_affect=False,
                                               obs_space={"low": -np.inf, "high": np.inf, "shape": vector_length},
                                               path="./Builds/Platformer.app") for counter in [0]
                      ])

    # Create the environment and agent
    model = PPO("MlpPolicy", env, learning_rate=lr, gamma=gamma, gae_lambda=gae_lambda,
                ent_coef=ent_coef, clip_range=clip_range, n_epochs=n_epochs, verbose=1, tensorboard_log="./Tensorboard",
                device='cpu')

    model.learn(total_timesteps=150000, progress_bar=True, callback=TensorboardCallback(), tb_log_name="PPO")
    mean_reward = evaluate_agent(model, env, n_episodes=10)
    return -mean_reward


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print("Best hyperparameters:", study.best_params)
