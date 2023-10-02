import random
from Archive import Archive
import configparser
import gym
import numpy as np
import copy


def round_to(x, base=50):
    return base * round(x / base)


if __name__ == "__main__":

    config_reader = configparser.ConfigParser()
    config_reader.read('./config_files/baseline.config')
    archive = Archive(config_reader)

    exit()
    env = Pedro_Env("Level3", saving_data=False)
    observation, info = env.reset()

    for i in range(archive.max_trajectories):

        if i != 0:
            state_dict = archive.return_to_cell()
            observation, info = env.reset()
            kill_all(env.world)
            set_world_state(state_dict, env.world)

        # Explore for a number of actions.
        for j in range(archive.explore_steps + 100):

            events = pygame.event.get()
            action = "n"
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = "a"
                    if event.key == pygame.K_RIGHT:
                        action = "d"
                    if event.key == pygame.K_UP:
                        action = "w"
                    if event.key == pygame.K_DOWN:
                        action = "s"

            action = env.action_space.weighted_sample()
            observation, reward, done, info = env.step(action)

            bonus = archive.create_cell(action, observation, env.player.hp == 0,
                                        get_world_state(env.perceptor, env.world))
            archive.assess_cell(reward)

            if env.player.hp == 0:
                print("Died, resetting")
                break

            env.render()

        archive.save_best_cells()

    env.close()
