## Weights & Biases (wandb) Integration

[Weights & Biases (wandb)](https://wandb.ai/). By default, wandb logging is enabled.

### Setting up wandb

**Step 1: Install wandb**

```bash
pip install wandb
```

**Step 2: Create a wandb account and login**

1. Go to [wandb.ai](https://wandb.ai/) and create a free account
2. Login to wandb in your terminal:

```bash
wandb login
```

3. Find your API key at [wandb.ai/authorize](https://wandb.ai/authorize)

**Step 3: Configure the project**

- By default, wandb logging is enabled (`USE_WANDB = True` in `main.py`).
- If you do not wish to use wandb, set `USE_WANDB = False` in `main.py`.
- Projects are logged separately:
  - Tabular Q-learning: `rl-gridworld-qlearning`
  - Deep Q-Network: `rl-gridworld-dqn`

**Note:** If you prefer not to use wandb, simply set `USE_WANDB = False` in `main.py` and the program will run without any online logging. The local terminal visualizations using Rich are actually quite comprehensive and may be more useful for understanding the learning process than the wandb dashboard.
