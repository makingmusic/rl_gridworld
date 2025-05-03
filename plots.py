from rich.live     import Live
from rich.table    import Table
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
from rich.console  import Group
from rich.panel    import Panel


def initDisplayTable():
    table = Table()
    table.add_column("State")
    table.add_column("Left")
    table.add_column("Right")
    table.add_column("Decision")
    return table
#end def initDisplayTable

def updateDisplayTableFromQTable(display_table, qtable, max_rows=20):
    # Calculate the step size to show only max_rows rows
    total_states = len(qtable)
    if total_states <= max_rows:
        step_size = 1
    else:
        step_size = total_states // max_rows
        if step_size < 1:
            step_size = 1

    # Get the states we want to show
    states_to_show = sorted(qtable.keys())[::step_size]
    
    # Update existing rows or add new ones if needed
    for i, state in enumerate(states_to_show):
        actions = qtable[state]
        leftValue = float(actions['left'])
        rightValue = float(actions['right'])
        decision = "stay" if leftValue == rightValue else ("<<Left<<" if leftValue > rightValue else ">>Right>>")
        stateValueStr = str(state)
        leftValueStr = str(round(leftValue, 4))
        rightValueStr = str(round(rightValue, 4))
        
        if i < len(display_table.rows):
            # Update existing row
            display_table.columns[0]._cells[i] = stateValueStr
            display_table.columns[1]._cells[i] = leftValueStr
            display_table.columns[2]._cells[i] = rightValueStr
            display_table.columns[3]._cells[i] = decision
        else:
            # Add new row if needed
            display_table.add_row(stateValueStr, leftValueStr, rightValueStr, decision)
    
    # If we have more rows than needed, remove the extra ones
    while len(display_table.rows) > len(states_to_show):
        display_table.rows.pop()
    
    return display_table
#end def updateDisplayTableFromQTable

def plotQTableValues(matlibplotpointer, qtable_data, state_numbers):
    # Plot Q-table values
    # Extract Q-values for specified states across all episodes
    filtered_episode_data = []
    
    # Create a dictionary to store Q-values for each state
    state_values = {}
    for state in state_numbers:
        state_values[state] = {
            'left': [],
            'right': []
        }
    
    # Filter out integer values and only process dictionary values
    for i in range(len(qtable_data)):
        qtable = qtable_data[i]
        
        # Only process dictionary values (Q-tables)
        if isinstance(qtable, dict):
            # Add the corresponding episode number
            filtered_episode_data.append(i // 2)  # Divide by 2 because every other item is an integer
            
            # Process each state in the state_numbers array
            for state in state_numbers:
                # Check if specified state exists in this Q-table
                if state in qtable:
                    state_values[state]['left'].append(qtable[state]['left'])
                    state_values[state]['right'].append(qtable[state]['right'])
                else:
                    # If state doesn't exist in this episode, append 0.0 as default
                    state_values[state]['left'].append(0.0)
                    state_values[state]['right'].append(0.0)

    # Create a new figure with larger size
    matlibplotpointer.figure(figsize=(15, 10))
    
    # Define color palette for better differentiation
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Plot Q-values for each state with improved styling
    for idx, state in enumerate(state_numbers):
        color = colors[idx % len(colors)]
        
        # Plot left action
        left_line, = matlibplotpointer.plot(filtered_episode_data, state_values[state]['left'], 
                                          linestyle='-', color=color, alpha=0.7)
        # Plot right action
        right_line, = matlibplotpointer.plot(filtered_episode_data, state_values[state]['right'], 
                                           linestyle='--', color=color, alpha=0.7)
        
        # Add direct labels at the end of each line
        last_episode = filtered_episode_data[-1]
        left_value = state_values[state]['left'][-1]
        right_value = state_values[state]['right'][-1]
        
        # Add labels with slight offset for better visibility
        matlibplotpointer.annotate(f'State {state} Left', 
                                 xy=(last_episode, left_value),
                                 xytext=(10, 0), textcoords='offset points',
                                 va='center', color=color)
        matlibplotpointer.annotate(f'State {state} Right',
                                 xy=(last_episode, right_value),
                                 xytext=(10, 0), textcoords='offset points',
                                 va='center', color=color)
    
    matlibplotpointer.xlabel('Episode', fontsize=12)
    matlibplotpointer.ylabel('Q-Value', fontsize=12)
    matlibplotpointer.title('Q-Values Evolution During Training', fontsize=14, pad=20)
    
    # Improve grid appearance
    matlibplotpointer.grid(True, linestyle='--', alpha=0.7)
    matlibplotpointer.minorticks_on()
    matlibplotpointer.grid(True, which='minor', linestyle=':', alpha=0.4)
    
    # Remove the legend since we're using direct labels
    # Adjust layout to prevent label cutoff
    matlibplotpointer.tight_layout()
# end def plotQTableValues

def initStepCounterTable():
    stepCounterTable = Table()
    stepCounterTable.add_column("Episode", style="bold cyan")
    stepCounterTable.add_column("Steps Taken", style="bold cyan")
    return stepCounterTable
#end def initStepCounterTable

# Function to update step counter
def updateStepCounter(stepTable, episode, steps):
    # Check if row exists for this episode
    existing_row = None
    for i, row in enumerate(stepTable.rows):
        if stepTable.columns[0]._cells[i] == str(episode):
            existing_row = i
            break
    
    if existing_row is not None:
        # Update existing row
        stepTable.columns[1]._cells[existing_row] = str(steps)
    else:
        # Add new row if episode doesn't exist
        stepTable.add_row(str(episode), str(steps))
#end def updateStepCounter

def plotStepsPerEpisode(pltPointer, episode_data, step_data, num_first_episodes=10, num_last_episodes=200):
    """
    Plot steps per episode showing the first n episodes and last m episodes.
    
    Args:
        pltPointer: Matplotlib pointer for plotting
        episode_data: List of episode numbers
        step_data: List of steps taken per episode
        num_first_episodes: Number of episodes to show at the beginning (default: 10)
        num_last_episodes: Number of episodes to show at the end (default: 200)
    """
    # Create a new figure with increased width
    pltPointer.figure(figsize=(15, 8))
    
    # Get the first n episodes and last m episodes
    first_episodes = episode_data[:num_first_episodes]
    first_steps = step_data[:num_first_episodes]
    last_episodes = episode_data[-num_last_episodes:]
    last_steps = step_data[-num_last_episodes:]
    
    # Create two subplots with a broken axis
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 2, width_ratios=[1, 2], wspace=0.05)
    
    # First subplot (first n episodes)
    ax1 = pltPointer.subplot(gs[0])
    ax1.plot(first_episodes, first_steps, 'b-', label='Steps')
    
    # Add labels for first n episodes
    for i in range(len(first_episodes)):
        ax1.annotate(f'{first_steps[i]}', 
                    xy=(first_episodes[i], first_steps[i]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom')
    
    # Second subplot (last m episodes)
    ax2 = pltPointer.subplot(gs[1])
    ax2.plot(last_episodes, last_steps, 'b-', label='Steps')
    
    # Add labels for last m episodes
    for i in range(len(last_episodes)):
        ax2.annotate(f'{last_steps[i]}', 
                    xy=(last_episodes[i], last_steps[i]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom')
    
    # Set the same y-axis limits for both subplots
    y_min = min(min(first_steps), min(last_steps))
    y_max = max(max(first_steps), max(last_steps))
    y_margin = (y_max - y_min) * 0.1  # 10% margin
    ax1.set_ylim(y_min - y_margin, y_max + y_margin)
    ax2.set_ylim(y_min - y_margin, y_max + y_margin)
    
    # Add break marks
    d = .015  # size of diagonal lines
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    
    # Set labels and title
    ax1.set_xlabel('Episode', fontsize=12)
    ax2.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Steps', fontsize=12)
    pltPointer.suptitle(f'Steps per Episode (First {num_first_episodes} and Last {num_last_episodes} Episodes)', fontsize=14, y=1.05)
    
    # Add grid to both subplots
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Hide the right spine of the first subplot and the left spine of the second subplot
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    # Adjust layout
    pltPointer.tight_layout()
# end def plotStepsPerEpisode

def plotEpsilonDecayPerEpisode(pltPointer, episode_data, epsilon_data):
    pltPointer.figure(figsize=(pltPointer.rcParams['figure.figsize'][0] * 0.8, pltPointer.rcParams['figure.figsize'][1] * 0.8))
    pltPointer.plot(episode_data, epsilon_data, 'r-', label='Epsilon')
    pltPointer.xlabel('Episode')
    pltPointer.ylabel('Epsilon')
    pltPointer.title('Epsilon Decay per Episode')
    pltPointer.grid(True)
# end def plotEpsilonDecayPerEpisode

def plotTemperatureDecayPerEpisode(pltPointer, episode_data, temperature_data):
    pltPointer.figure(figsize=(pltPointer.rcParams['figure.figsize'][0] * 0.8, pltPointer.rcParams['figure.figsize'][1] * 0.8))
    pltPointer.plot(episode_data, temperature_data, 'g-', label='Temperature')
    pltPointer.xlabel('Episode')
    pltPointer.ylabel('Temperature')
    pltPointer.title('Temperature Decay per Episode')
    pltPointer.grid(True)
# end def plotTemperatureDecayPerEpisode

