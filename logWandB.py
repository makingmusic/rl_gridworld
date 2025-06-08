import wandb
import copy

def initWandB(project_name, config = None):
    wandb.init(project=project_name, config=config)


def logEpisode(config = None, step = None):
    wandb.log(config, step=step)

def logEpisodeWithImageControl(config=None, step=None, episode=None, total_episodes=None, N=10):
    """
    Log episode to wandb, but only include image data for the first, last, and N evenly spaced episodes.
    - config: dict, must contain 'q_table_heatmap_img' if image is to be logged
    - step: int, the current episode number
    - episode: int, the current episode number (same as step)
    - total_episodes: int, total number of episodes
    - N: int, number of intermediate episodes to log with image (default 10)
    """
    if config is None or episode is None or total_episodes is None:
        # If 'q_table_heatmap_img' is not in config, also remove 'q_table' if present
        if config is not None and 'q_table_heatmap_img' not in config and 'q_table' in config:
            config = copy.copy(config)
            del config['q_table']
        wandb.log(config, step=step)
        return

    # Always log image for first and last episode
    log_image = (episode == 0) or (episode == total_episodes - 1)
    # For N evenly spaced episodes in between
    if not log_image and N > 0 and total_episodes > 2:
        interval = (total_episodes - 2) / (N + 1)
        # Compute the N evenly spaced episode indices (excluding first and last)
        selected_indices = [round((i + 1) * interval) for i in range(N)]
        selected_indices = [idx + 1 for idx in selected_indices]  # shift by 1 to skip first
        if episode in selected_indices:
            log_image = True
    # If not logging image, remove the key
    if not log_image and config is not None:
        if 'q_table_heatmap_img' in config:
            config = copy.copy(config)
            del config['q_table_heatmap_img']
        # Also remove 'q_table' if 'q_table_heatmap_img' is not present
        if 'q_table_heatmap_img' not in config and 'q_table' in config:
            if not isinstance(config, dict):
                config = dict(config)
            else:
                config = copy.copy(config)
            del config['q_table']
    wandb.log(config, step=step)

def closeWandB():
    wandb.finish()

