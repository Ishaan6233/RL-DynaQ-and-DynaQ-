# Dyna-Q and Dyna-Q+ Reinforcement Learning

This project implements Dyna-Q and Dyna-Q+ agents to test learning performance in a shortcut maze environment.

## Structure

```
├── agents/
│ ├── dyna_q_agent.py # Implements the Dyna-Q algorithm
│ └── dyna_q_plus_agent.py # Implements the Dyna-Q+ algorithm
│
├── envs/
│ └── shortcut_maze_env.py # Gridworld environment with a changing layout
│
├── experiments/
│ ├── init.py
│ ├── config.py # Experiment configuration
│ ├── plotting_utils.py # Plotting helpers
│ ├── run_dyna_q.py # Run Dyna-Q experiment
│ └── run_dyna_q_plus.py # Run Dyna-Q+ experiment
│
├── utils/
│ ├── base_agent.py # Abstract BaseAgent class
│ └── rl_glue.py # RL-Glue logic for agent-env interaction
│
├── main.py # Optional entrypoint (not required)
├── requirements.txt # Project dependencies
└── README.md # This file
```

- `agents/`: Contains agent implementations (`DynaQAgent` and `DynaQPlusAgent`).
- `environments/`: Placeholder for the `ShortcutMazeEnvironment`.
- `utils/`: Base agent and RL utilities.
- `experiments/`: Scripts for running experiments and generating plots.

## How to Run

1. Install dependencies:
    ```
    pip install numpy matplotlib
    ```
2. Add environment and RLGlue logic as needed.

3. Run experiments from `experiments/`.

## Algorithms Overview
### Dyna-Q
- Combines real experience and simulated (model-generated) experience.
- Planning is done through simulated updates from a learned model.
- Uses an ε-greedy policy for action selection.
### Dyna-Q+
- Enhances Dyna-Q with an exploration bonus for actions not taken recently.
- Helps agents adapt to changes in environment dynamics (e.g., a shortcut appearing).
- Bonus: 𝑟_bonus = 𝜅(𝜏(𝑠,𝑎))^(1/2)
- where 𝜏(𝑠,𝑎) is the time since action 𝑎 was taken in state 𝑠.

## Visualizations
- The experiment scripts will generate: Line plots comparing average steps per episode across planning steps.
- Cumulative reward over time.
- State visitation heatmaps before and after environment changes.

## References

- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd Edition.
