from rich.live     import Live
from rich.table    import Table
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
from rich.console  import Group
from rich.panel    import Panel


def initDisplayTable():
    table = Table()
    table.add_column("State")
    table.add_column("Up")
    table.add_column("Down")
    table.add_column("Left")
    table.add_column("Right")
    table.add_column("Decision")
    return table
#end def initDisplayTable

def updateDisplayTableFromQTable(display_table, qtable, max_rows=20):
    # Create a new table
    new_table = Table()
    new_table.add_column("State")
    
    # Check if this is a 1D or 2D world by looking at the first state's actions
    first_state = next(iter(qtable.values()))
    is_2d = 'up' in first_state
    
    # Add appropriate columns based on world type
    if is_2d:
        new_table.add_column("Up")
        new_table.add_column("Down")
        new_table.add_column("Left")
        new_table.add_column("Right")
    else:
        new_table.add_column("Left")
        new_table.add_column("Right")
    
    new_table.add_column("Decision")

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
    
    # Add rows to the new table
    for state in states_to_show:
        actions = qtable[state]
        
        if is_2d:
            upValue = float(actions['up'])
            downValue = float(actions['down'])
            leftValue = float(actions['left'])
            rightValue = float(actions['right'])
            
            # Find the best action
            action_values = {'up': upValue, 'down': downValue, 'left': leftValue, 'right': rightValue}
            best_action = max(action_values.items(), key=lambda x: x[1])
            
            # Create decision string
            if all(v == 0.0 for v in action_values.values()):
                decision = "stay"
            else:
                decision = f">>{best_action[0]}>>"
            
            stateValueStr = str(state)
            upValueStr = str(round(upValue, 4))
            downValueStr = str(round(downValue, 4))
            leftValueStr = str(round(leftValue, 4))
            rightValueStr = str(round(rightValue, 4))
            
            new_table.add_row(stateValueStr, upValueStr, downValueStr, leftValueStr, rightValueStr, decision)
        else:
            leftValue = float(actions['left'])
            rightValue = float(actions['right'])
            
            # Find the best action
            action_values = {'left': leftValue, 'right': rightValue}
            best_action = max(action_values.items(), key=lambda x: x[1])
            
            # Create decision string
            if all(v == 0.0 for v in action_values.values()):
                decision = "stay"
            else:
                decision = f">>{best_action[0]}>>"
            
            stateValueStr = str(state)
            leftValueStr = str(round(leftValue, 4))
            rightValueStr = str(round(rightValue, 4))
            
            new_table.add_row(stateValueStr, leftValueStr, rightValueStr, decision)
    
    return new_table
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

def plotStepsPerEpisode(pltPointer, episode_data, step_data, num_episodes=20):
    pltPointer.figure(figsize=(pltPointer.rcParams['figure.figsize'][0] * 0.8, pltPointer.rcParams['figure.figsize'][1] * 0.8))
    
    # Determine which episodes to display
    if len(episode_data) <= 2 * num_episodes:
        # If total episodes is less than or equal to 2*num_episodes, show all of them
        display_episodes = episode_data
        display_steps = step_data
        
        # Plot all episodes
        pltPointer.plot(display_episodes, display_steps, 'b-', label='Steps')
        
        # Add labels for all points
        for i in range(len(display_episodes)):
            pltPointer.annotate(f'{display_steps[i]}', 
                               xy=(display_episodes[i], display_steps[i]),
                               xytext=(0, 10), textcoords='offset points',
                               ha='center', va='bottom')
    else:
        # Otherwise, show first num_episodes and last num_episodes episodes with a broken axis
        first_episodes = episode_data[:num_episodes]
        first_steps = step_data[:num_episodes]
        last_episodes = episode_data[-num_episodes:]
        last_steps = step_data[-num_episodes:]
        
        # Create two subplots with a broken axis
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.05)
        
        # First subplot (first num_episodes episodes)
        ax1 = pltPointer.subplot(gs[0])
        ax1.plot(first_episodes, first_steps, 'b-', label='Steps')
        
        # Add labels for first num_episodes episodes
        for i in range(len(first_episodes)):
            ax1.annotate(f'{first_steps[i]}', 
                        xy=(first_episodes[i], first_steps[i]),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom')
        
        # Second subplot (last num_episodes episodes)
        ax2 = pltPointer.subplot(gs[1])
        ax2.plot(last_episodes, last_steps, 'b-', label='Steps')
        
        # Add labels for last num_episodes episodes
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
        ax1.set_xlabel('Episode')
        ax2.set_xlabel('Episode')
        ax1.set_ylabel('Steps')
        pltPointer.suptitle('Steps per Episode', y=1.05)
        
        # Add grid to both subplots
        ax1.grid(True)
        ax2.grid(True)
        
        # Hide the right spine of the first subplot and the left spine of the second subplot
        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        
        # Adjust layout
        #pltPointer.tight_layout() #todo: revisit
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

