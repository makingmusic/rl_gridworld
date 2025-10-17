# RL Gridworld

This is a Reinforcement Learning implementation of Q-Learning in both 1D and 2D gridworld environments. Why you ask? Well, the reason is to learn. I started this in spring 2025 and whenever I get time i do a bit of progress. I put checkpoints on the progress using Releases that you can see on this page: https://github.com/makingmusic/rl_gridworld/releases : you can see all major progression through those releases.

This has been a "build/learn in public" kind of work for me. When I began, Cursor was not as good at writing long running tasks and I am shocked to see how much better it has become in this timeframe itself. From providing it instructions to write a function, now I give it instructions that are much deeper (as of Oct 2025) such as "make this NN have obstacles".

A lot has been just vibe-coded here and i have selfishly used claude, codex and cursor to get increasing chunks of work done for me. It made the visualization trivial for me to do - because it wouldve taken me down a learning path i wasn't particularly interested in. 
Same for setting up configurations for all my parameters in one place was done by an agent - made it so much easier for me because i could all my configs in one place. 

I used wandb in the earlier runs because it does offer a lot of viz and all. But I dropped it after the first release because going to wandb to see the results was a pain. I realized it is only for people who are doing training at scale. Someday I will clean up that part of the code too.


In the latest release, i am now using uv and switched to python 3.14 - good performance improvement - like 44 sec run became a 17 sec run - had something to do with dropping GIL. Don't understand, but whatever, it is a good thing. Also, uv is so much better than than pip install crap. Switching python versions is ultra trivial - Can't believe installing python 3.14 was 107 ms.


## Quick Start

To get started 
* checkout the repo
* ./run.sh


This will set up a Python virtual environment (uv based now, running on python 3.14 or more)

After setup, you can run:

- `uv python main.py` for tabular Q-learning in the 2D grid world
- `uv python main_nn.py` for Deep Q-Network (DQN) in the 2D grid world

## Configurations
- Look for all configurations in main.py and main_nn.py. There are tons of variables to play around with and their names and documentation is mostly self-explanatory.
- Some documentation for the configurations you can play around with are in configs.md

## The Environment

### 2D Gridworld

A 1D/2D world where the agent can move up, down, left, or right. The goal is to reach the top-right corner. If it is setup with just one row, then it can simulate a 1D world.

Rules:

- Agent can move in four directions: up, down, left, right
- Cannot move outside the grid boundaries
- Goal state is at the top-right corner (grid_size_x-1, grid_size_y-1)
- Default grid size is 20x5

## Implementation Approaches

This project offers two different Q-learning implementations:

### 1. Tabular Q-Learning (`main.py`)
- **Traditional approach** using lookup tables
- **Fast convergence** for small state spaces
- **Exact Q-values** stored in memory
- **Best for**: Small grids (≤ 20x20), learning RL fundamentals

### 2. Deep Q-Network - DQN (`main_nn.py`)
- **Neural network** approximates Q-values
- **Adaptive sizing** automatically scales network capacity based on grid size
- **Experience replay** for stable learning
- **Target network** for stable Q-value estimation
- **Best for**: Large grids (≥ 50x50), advanced RL techniques

For detailed comparison and usage guide, see [NEURAL_NETWORK_GUIDE.md](NEURAL_NETWORK_GUIDE.md).

## Reward Structure & Grid Size Scaling

Reward shaping is complete different animal. Moving all this discussion to a new file: rewards.md 


## Adaptive Neural Network Sizing
Why ? Because it helped me play around with different grid sizes efficiently. There is no real science here but a basic way to move around config to support variations in the kinds of run i was doing. Mostly , this was to save me time between different runs.

### The Problem: Fixed Network Size Limitations

The original DQN implementation used a fixed neural network architecture regardless of grid size:

**Previous Fixed Architecture:**
- **Hidden layers**: 3 layers of 128 neurons each
- **Total parameters**: ~33,000 parameters
- **Buffer size**: 10,000 experiences (fixed)
- **Batch size**: 64 (fixed)

**Why This Failed for Large Grids:**
- **50×50 grid**: 2,500 states → 13.3 parameters per state (adequate)
- **100×100 grid**: 10,000 states → 3.3 parameters per state (insufficient!)
- **200×200 grid**: 40,000 states → 0.8 parameters per state (severely inadequate!)

The fixed network became increasingly under-capacity as grid size increased, leading to poor learning performance and convergence issues.

### The Solution: Adaptive Neural Network Sizing

**New Adaptive Architecture** automatically scales all parameters based on grid dimensions:

#### 1. **Adaptive Hidden Layer Size**
```python
def compute_optimal_nn_size(grid_x, grid_y, min_hidden=128, max_hidden=1024):
    total_states = grid_x * grid_y
    # Target 15 parameters per state for optimal learning capacity
    optimal_hidden = int(np.sqrt(7.5 * total_states))
    # Apply bounds and round to power of 2 for efficiency
    optimal_hidden = max(min_hidden, min(optimal_hidden, max_hidden))
    optimal_hidden = 2 ** int(np.log2(optimal_hidden) + 0.5)
    return optimal_hidden
```

#### 2. **Adaptive Buffer Size**
```python
def compute_adaptive_buffer_size(grid_x, grid_y, base_size=10000):
    total_states = grid_x * grid_y
    # Scale with square root of grid area for diverse experiences
    adaptive_size = min(base_size * (total_states / 2500) ** 0.5, 100000)
    return int(adaptive_size)
```

#### 3. **Adaptive Batch Size**
```python
def compute_adaptive_batch_size(grid_x, grid_y, base_size=64):
    total_states = grid_x * grid_y
    # Scale with grid area to power 0.25 for stable training
    adaptive_size = min(base_size * (total_states / 2500) ** 0.25, 256)
    return int(adaptive_size)
```

### Results: 

| Grid Size | States | Hidden Size | Buffer Size | Batch Size | NN Parameters | Params/State |
|-----------|--------|-------------|-------------|------------|---------------|--------------|
| 10×10     | 100    | 128         | 2,000       | 28         | 33,280        | 332.8        |
| 25×25     | 625    | 128         | 5,000       | 45         | 33,280        | 53.2         |
| 50×50     | 2,500  | 128         | 10,000      | 64         | 33,280        | 13.3         |
| 100×100   | 10,000 | 256         | 20,000      | 90         | 132,096       | 13.2         |
| 200×200   | 40,000 | 512         | 40,000      | 128        | 526,336       | 13.2         |


**Implementation Location**: The adaptive sizing functions are defined in `main_nn.py` (lines 53-107) and automatically applied during agent initialization.


# Future Improvements
I have put them all under future-todo.md
I do find myself spreading thin across so many rabbit holes in this problem. Often I will see a youtube video that will kick off a sidequest in my head and it is (mostly) logged in that file for me to come back to at some point. I am hardly good at doing this with good discipline.


## License

I don't even understand how licensing works across MIT and whatnot. I wrote this with a large amount of help from ChatGPT and if you find it useful, please use it in any way you feel like without any obligations to me. And I welcome your feedback if any. Personally I learnt so much that I could not have kept it hidden in my laptop so here you go !
