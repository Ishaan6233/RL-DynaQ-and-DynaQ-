from agents.dyna_q_agent import DynaQAgent
from envs.shortcut_maze_env import ShortcutMazeEnvironment
from utils.rl_glue import RLGlue

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def run_experiment(env, agent, env_parameters, agent_parameters, exp_parameters):
    num_runs = exp_parameters['num_runs']
    num_episodes = exp_parameters['num_episodes']
    planning_steps_all = agent_parameters['planning_steps']
    env_info = env_parameters
    all_averages = np.zeros((len(planning_steps_all), num_runs, num_episodes))
    agent_info = {
        "num_states": agent_parameters["num_states"],
        "num_actions": agent_parameters["num_actions"],
        "epsilon": agent_parameters["epsilon"],
        "discount": env_parameters["discount"],
        "step_size": agent_parameters["step_size"]
    }

    for idx, planning_steps in enumerate(planning_steps_all):
        print('Planning steps:', planning_steps)
        agent_info["planning_steps"] = planning_steps

        for i in tqdm(range(num_runs)):
            agent_info['random_seed'] = i
            agent_info['planning_random_seed'] = i

            rl_glue = RLGlue(env, agent)
            rl_glue.rl_init(agent_info, env_info)

            for j in range(num_episodes):
                rl_glue.rl_start()
                is_terminal = False
                num_steps = 0
                while not is_terminal:
                    reward, _, action, is_terminal = rl_glue.rl_step()
                    num_steps += 1
                all_averages[idx][i][j] = num_steps

    return {"all_averages": all_averages, "planning_steps_all": planning_steps_all}


def plot_steps_per_episode(data):
    all_averages = data['all_averages']
    planning_steps_all = data['planning_steps_all']
    for i, planning_steps in enumerate(planning_steps_all):
        plt.plot(np.mean(all_averages[i], axis=0), label=f'Planning steps = {planning_steps}')
    plt.legend(loc='upper right')
    plt.xlabel('Episodes')
    plt.ylabel('Steps per episode')
    plt.axhline(y=16, linestyle='--', color='grey', alpha=0.4)
    plt.show()


if __name__ == "__main__":
    experiment_parameters = {
        "num_runs": 30,
        "num_episodes": 40,
    }
    environment_parameters = {
        "discount": 0.95,
    }
    agent_parameters = {
        "num_states": 54,
        "num_actions": 4,
        "epsilon": 0.1,
        "step_size": 0.125,
        "planning_steps": [0, 5, 50]
    }

    current_env = ShortcutMazeEnvironment
    current_agent = DynaQAgent

    data = run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)
    plot_steps_per_episode(data)
