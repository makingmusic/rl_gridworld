# Documentation Index

Topic guides for `rl_gridworld`. Start with the project [README](../README.md) for narrative context, and [AGENTS.md](../AGENTS.md) if you're an AI coding agent.

## Setup & workflow
- [development.md](development.md) — install, run, devcontainer, debugging tips
- [architecture.md](architecture.md) — module map, who calls who, data flow
- [configs.md](configs.md) — every tunable knob in `main.py` / `main_nn.py`

## Algorithms
- [tabular.md](tabular.md) — tabular Q-learning details
- [nn.md](nn.md) — Deep Q-Network: architecture, adaptive sizing, training loop
- [rewards.md](rewards.md) — reward shaping, tiered scaling, penalties

## Environment
- [environment.md](environment.md) — `GridWorld2D`, actions, obstacles, maze generation, A*

## Tooling
- [wandb.md](wandb.md) — optional Weights & Biases experiment tracking

## Forward-looking
- [future-todo.md](future-todo.md) — roadmap, memory-augmented DQN, side quests

## Sibling experiments
- [`balance/`](../balance/README.md) — classical control (PID) on balancing problems. Inverted pendulum + TUI today; cartpole planned. Plan: [`plans/balance.md`](../plans/balance.md).
