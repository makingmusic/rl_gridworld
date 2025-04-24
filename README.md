# RL Gridworld

This is a Reinforcement Learning implementation of Q-Learning in a 1D gridworld environment. Why you ask? Well, the reason is learn. Like really learn what is going on underneath.

I did learn a ton and I find it much more useful to learn through this method than reading a book. It is the reason I am keeping this repo public. May more people find this usefulness!

## The setup

The setup is the simplest RL implemention that I could find to learn on.It is a 1D world. Simple as that.

At the beginning you just define a) the end goal - which is to reach state N and b) the rules of the game.

Rules are simple: You can either go left or right (falling out at the edges not ok :) - obviously).

## The agent

It keeps a python list against each state. This list contains a dict with two items:

1. left
2. right
   Both of these are numbers that are the "expected reward" on taking that action. The agent takes the action that has the greatest reward.

Well, if only life was so simple. It is not.

For reasons known as "epsilon-greedy learning", the agent doesn't just take the higher reward of the two. Instead it generates a random number between 0 and 1 and if the number is < epsilon, then it will chose a random action out of left or right. Otherwise it will chose the larger of the two options. (Search for the text "xplore or exploit" in the function choose_action in the file q_agent.py).

Now if the world was so simple, it would be awesome. It is not. epsilon, it turns out starts at a high value (0.95) and slowly reduces down to 0.01 in small increments. This means that it will nearly always random in the beginning and will lean on the learnt data as it progresses. Makes sense, doesn't it ?

The cool thing I like about my default values is that it NEVER STOPS LEARNING. See how the learning parameters are setup in main.py by searching for the text "Configuration Variables"

The agent tries to learn this over many "episodes". For larger grid sizes, you need more episodes.

# Why is there so much code?

The real deal is in main.py in the for loop annotated as the "# Training loop". That's it. It is a super small implementation, made a bit more readable by the use of classes for the environment (grid1Dworld.py) and the agent (q_agent.py).

Nearly half of the code in main.py (and 100% of the code in plots.py) is for me to really understand what is happenning inside by plotting the changing data in curves and by displaying the changing q-values in the command line. Everything that has to do with the python module "rich" and "matlabplot" is merely to show me things. It has nothing to do with the real work being carried out.

## Display choices

- There is a table that shows the Q-values against each state, action tuple and it is updated on the fly at the end of every episode.
- To really see this table changing and the values converging, you will need to add some non-zero sleep in the variable "sleep_time" in main.py
- At the end, it will show the Q-values for every 5th state in a nice little matlab plot.
- It will also show the decline of number of steps required in each episode. It is fun to watch the required number of steps go down drastically after the first few episodes.
- Uncomment code under "# Plot epsilon decay" in main.py to see the epsilon values decline. It is stating the obvious - no surprise there.

## Configuration

You can modify the following parameters in `main.py`:

- `num_episodes`: Number of training episodes (default: 1000)
- `grid1DSize`: Size of the 1D grid (default: 100)
- `startState`: Starting position (default: 0)
- `goalState`: Goal position (default: grid1DSize - 1)
- `sleep_time`: Time to pause between episodes (default: 0)

## Things of note:

- Training beyond 100 states is just a pain and I only do it for fun
- Adding a reward of -0.01 at every step was a MAJOR breakthrough (See "Determine reward" in gridworld1d.py). Think about why - this was super fun for me. Without this reward, I just couldn't train past 50 states in reasonable cpu time.
- You will find a python notebook file (main.ipynb). It is a failed attempt to make this run in a notebook.

## Future

I want to play with:

- starting at different points in the state machine
- having the goal be another point than the extreme state on the right.
- allow "jumps", which may be X number of steps that can be taken together. Would be fun to see how learning improves (or becomes worse) by higher values of X.
- Add obstacles or forbidden states so the only way to reach the end would be jump over them.
- Changing the tradeoffs between exploitation vs exploration. There are many algorithsm avaiable include the infamous softmax that are used in modern LLMs too. I want to get there.
 
## GPT Says I should try the following:

1. Implement Multiple RL Algorithms
	•	Value Iteration & Policy Iteration: Understand the differences between model-based and model-free approaches.
	•	Monte Carlo Methods: Explore how sampling can be used for policy evaluation and improvement.
	•	Temporal Difference Learning: Implement SARSA and Q-learning to compare on-policy and off-policy methods.

2. Introduce Function Approximation
	•	Linear Function Approximation: Replace tabular methods with linear approximators to handle larger state spaces.
	•	Neural Networks: Begin with simple feedforward networks before moving to more complex architectures.

3. Experiment with Exploration Strategies
	•	Epsilon-Greedy: Analyze how varying epsilon affects learning.
	•	Softmax Action Selection: Implement and compare with epsilon-greedy.
	•	Upper Confidence Bound (UCB): Explore how optimism in the face of uncertainty can drive exploration.

4. Incorporate Stochasticity
	•	Action Noise: Introduce randomness in action outcomes to simulate real-world unpredictability.
	•	Reward Noise: Add variability to rewards to study robustness.

5. Visualize Learning Progress
	•	Heatmaps of State-Value Functions: Visualize how the agent’s understanding of the environment evolves.
	•	Policy Arrows: Display the agent’s preferred action in each state.

## Requirements / Installation etc...

- Python 3.x
- Required packages (install using `pip install -r requirements.txt`):
  - numpy
  - matplotlib
  - rich

## Installation

1. Clone the repository:

```bash
git clone https://github.com/makingmusic/rl_gridworld.git
cd rl_gridworld
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script to start the training:

```bash
python main.py
```

The program will:

1. Initialize a 1D gridworld environment
2. Create a Q-Learning agent
3. Train the agent for the specified number of episodes
4. Display real-time progress using Rich
5. Show final Q-values and learning curves

## License

I don't even understand how licensing works across MIT and whatnot. I wrote this with a large amount of help from ChatGPT and if you find it useful, please use it in any way you feel like without any obligations to me. And I welcome your feedback if any. Personally I learnt so much that I could not have kept it hidden in my laptop so here you go !
