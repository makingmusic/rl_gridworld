from rich.live     import Live
from rich.table    import Table
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
from rich.console  import Group
from rich.panel    import Panel
from rich.console import Console
from rich.text import Text

def create_grid_display(grid_size, start_pos, goal_pos, qtable):
    """Create a text-based grid display showing Q-values and decisions."""
    console = Console()
    grid_display = []
    
    # Create the grid
    for row in range(grid_size):
        grid_row = []
        for col in range(grid_size):
            state = (col, row)
            if state in qtable:
                actions = qtable[state]
                # Find best action
                action_values = {
                    'up': actions['up'],
                    'down': actions['down'],
                    'left': actions['left'],
                    'right': actions['right']
                }
                best_action = max(action_values.items(), key=lambda x: x[1])
                
                # Create cell content
                if all(v == 0.0 for v in action_values.values()):
                    cell = "-"
                else:
                    # Map actions to arrow symbols
                    arrows = {
                        'up': '↑',
                        'down': '↓',
                        'left': '←',
                        'right': '→'
                    }
                    cell = arrows[best_action[0]]
                
                # Create text with appropriate color
                text = Text(cell)
                if state == start_pos:
                    text.stylize('bold green')
                elif state == goal_pos:
                    text.stylize('bold red')
                grid_row.append(text)
            else:
                grid_row.append(Text("-"))
        grid_display.append(grid_row)
    
    return grid_display

def update_grid_display(grid_display, qtable, start_pos, goal_pos):
    """Update the existing grid display with new Q-values."""
    console = Console()
    
    # Create new grid display
    return create_grid_display(len(grid_display), start_pos, goal_pos, qtable)

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

def plotEpsilonDecayPerEpisode(pltPointer, episode_data, epsilon_data):
    pltPointer.figure(figsize=(pltPointer.rcParams['figure.figsize'][0] * 0.8, pltPointer.rcParams['figure.figsize'][1] * 0.8))
    pltPointer.plot(episode_data, epsilon_data, 'r-', label='Epsilon')
    pltPointer.xlabel('Episode')
    pltPointer.ylabel('Epsilon')
    pltPointer.title('Epsilon Decay per Episode')
    pltPointer.grid(True)

def plotTemperatureDecayPerEpisode(pltPointer, episode_data, temperature_data):
    pltPointer.figure(figsize=(pltPointer.rcParams['figure.figsize'][0] * 0.8, pltPointer.rcParams['figure.figsize'][1] * 0.8))
    pltPointer.plot(episode_data, temperature_data, 'g-', label='Temperature')
    pltPointer.xlabel('Episode')
    pltPointer.ylabel('Temperature')
    pltPointer.title('Temperature Decay per Episode')
    pltPointer.grid(True)

def grid_to_table(grid_display):
    from rich.table import Table
    table = Table(show_header=False, box=None, pad_edge=False)
    if not grid_display:
        return table
    for _ in range(len(grid_display[0])):
        table.add_column()
    for row in grid_display:
        table.add_row(*row)
    return table

def display_actual_path(grid_size, start_pos, goal_pos, qtable):
    """Display the actual path that the model would take from start to goal, with clearer visuals.
    The path is limited to 2 * grid_size steps to prevent infinite loops."""
    console = Console()
    grid_display = []
    for _ in range(grid_size):
        grid_display.append(['·'] * grid_size)  # Use middle dot for non-path

    # Start from the start position
    current_pos = start_pos
    path = [current_pos]
    path_arrows = {}
    max_steps = 2 * grid_size
    steps_taken = 0

    # Follow the best actions until we reach the goal or hit step limit
    while current_pos != goal_pos and steps_taken < max_steps:
        # Check if current position exists in Q-table and has valid actions
        if current_pos not in qtable or not qtable[current_pos]:
            break
            
        actions = qtable[current_pos]
        action_values = {
            'up': actions.get('up', 0.0),
            'down': actions.get('down', 0.0),
            'left': actions.get('left', 0.0),
            'right': actions.get('right', 0.0)
        }
        
        # If all actions have zero value, break the path
        if all(v == 0.0 for v in action_values.values()):
            break
            
        best_action = max(action_values.items(), key=lambda x: x[1])[0]
        arrows = {
            'up': '↑',
            'down': '↓',
            'left': '←',
            'right': '→'
        }
        x, y = current_pos
        path_arrows[(x, y)] = arrows[best_action]
        if best_action == 'up':
            current_pos = (x, y - 1)
        elif best_action == 'down':
            current_pos = (x, y + 1)
        elif best_action == 'left':
            current_pos = (x - 1, y)
        elif best_action == 'right':
            current_pos = (x + 1, y)
        path.append(current_pos)
        steps_taken += 1

    # Create the final grid display
    final_display = []
    for row in range(grid_size):
        display_row = []
        for col in range(grid_size):
            pos = (col, row)
            if pos == start_pos:
                display_row.append(Text('S', style='bold green'))
            elif pos == goal_pos:
                display_row.append(Text('G', style='bold red'))
            elif pos in path_arrows:
                display_row.append(Text(path_arrows[pos], style='bold cyan'))
            else:
                display_row.append(Text('·', style='dim'))
        final_display.append(display_row)

    table = Table(show_header=False, box=None, pad_edge=False)
    for _ in range(grid_size):
        table.add_column()
    for row in final_display:
        table.add_row(*row)
    return table

def initDisplayTable():
    """Initialize the display table for 1D Q-learning visualization."""
    table = Table(title="Q-Values")
    table.add_column("State", justify="right", style="cyan")
    table.add_column("Left", justify="right", style="green")
    table.add_column("Right", justify="right", style="green")
    table.add_column("Best Action", justify="center", style="yellow")
    return table

def updateDisplayTableFromQTable(table, qtable, max_rows=10):
    """Update the display table with current Q-values."""
    # Clear existing rows
    table.rows = []
    
    # Add rows for each state
    for state in range(max_rows):
        if state in qtable:
            actions = qtable[state]
            left_value = f"{actions['left']:.3f}"
            right_value = f"{actions['right']:.3f}"
            
            # Determine best action
            if actions['left'] > actions['right']:
                best_action = "←"
            elif actions['right'] > actions['left']:
                best_action = "→"
            else:
                best_action = "-"
                
            table.add_row(str(state), left_value, right_value, best_action)
        else:
            table.add_row(str(state), "0.000", "0.000", "-")
    
    return table

def display_1d_grid(grid_size, start_pos, goal_pos, qtable):
    """Display a 1D grid showing the best actions using arrows."""
    console = Console()
    grid_display = []
    
    # Create the grid row
    for pos in range(grid_size):
        if pos in qtable:
            actions = qtable[pos]
            # Find best action
            if actions['left'] > actions['right']:
                cell = "←"
            elif actions['right'] > actions['left']:
                cell = "→"
            else:
                cell = "-"
            
            # Create text with appropriate color
            text = Text(cell)
            if pos == start_pos:
                text.stylize('bold green')
            elif pos == goal_pos:
                text.stylize('bold red')
            grid_display.append(text)
        else:
            grid_display.append(Text("-"))
    
    # Create table
    table = Table(show_header=False, box=None, pad_edge=False)
    for _ in range(grid_size):
        table.add_column()
    table.add_row(*grid_display)
    return table

def plotQTableValues(pltPointer, qtable_data, states_of_interest):
    """Plot Q-values for specific states over episodes."""
    pltPointer.figure(figsize=(pltPointer.rcParams['figure.figsize'][0] * 0.8, pltPointer.rcParams['figure.figsize'][1] * 0.8))
    
    # Create a plot for each state of interest
    for state in states_of_interest:
        # Extract Q-values for this state over episodes
        left_values = []
        right_values = []
        episodes = []
        
        for episode, qtable in enumerate(qtable_data):
            if isinstance(qtable, dict) and state in qtable:
                left_values.append(qtable[state]['left'])
                right_values.append(qtable[state]['right'])
                episodes.append(episode)
        
        if episodes:  # Only plot if we have data for this state
            pltPointer.plot(episodes, left_values, 'b-', label=f'State {state} Left')
            pltPointer.plot(episodes, right_values, 'r-', label=f'State {state} Right')
    
    pltPointer.xlabel('Episode')
    pltPointer.ylabel('Q-Value')
    pltPointer.title('Q-Values Over Time')
    pltPointer.legend()
    pltPointer.grid(True)

